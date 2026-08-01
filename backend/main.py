"""
FastAPI application entry point for the Multimodal Study Assistant.

Endpoints:
  POST /ingest  — upload a study file and ingest it into ChromaDB
  POST /query   — ask a question and get an LLM-grounded answer
  GET  /health  — check server + Ollama availability
"""

import os
import shutil
import tempfile
import requests

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from backend.ingest import ingest_file
from backend.retrieve import query_rag, ask_ollama

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multimodal Study Assistant",
    description="Local RAG pipeline for lecture notes, slides, and audio.",
    version="0.1.0",
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    session_id: str = None
    n_results: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list  # list of metadata dicts from ChromaDB


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Return server status and whether Ollama is reachable."""
    ollama_ok = False
    try:
        # A quick GET to the Ollama root tells us if the server is up
        resp = requests.get(OLLAMA_BASE_URL, timeout=3)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    return {"status": "ok", "ollama": ollama_ok}


@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(default="default"),
):
    """Accept a multipart file upload and ingest it into ChromaDB.

    The file is saved to a temp directory, processed, then deleted.
    Returns the number of chunks that were stored.
    """
    # Save the uploaded file to a temp location so model code can read it
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n_chunks = ingest_file(
            tmp_path,
            metadata={"session_id": session_id, "source_file": file.filename},
        )
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Always clean up the temp file even if ingestion fails
        os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "session_id": session_id,
        "chunks_stored": n_chunks,
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Retrieve relevant chunks and ask Ollama to answer the question.

    Steps:
    1. Embed the question and query ChromaDB (query_rag)
    2. Pass retrieved chunks to Ollama as context (ask_ollama)
    3. Return the answer and the source metadata
    """
    rag_result = query_rag(
        question=req.question,
        session_id=req.session_id,
        n_results=req.n_results,
    )

    if not rag_result["chunks"]:
        raise HTTPException(
            status_code=404,
            detail="No relevant study material found. Please ingest some files first.",
        )

    try:
        answer = ask_ollama(req.question, rag_result["chunks"])
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )

    return QueryResponse(
        answer=answer,
        sources=rag_result["metadatas"],
    )
