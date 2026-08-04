"""
FastAPI application entry point for the Multimodal Study Assistant.

Endpoints:
  POST /ingest   — upload a study file and ingest it into ChromaDB
  POST /query    — ask a question and get an LLM-grounded answer
  GET  /sessions — list study modules present in the store, with chunk counts
  GET  /health   — check server + Ollama availability
"""

import os
import shutil
import tempfile
import time
import requests

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from backend.ingest import ingest_file
from backend.retrieve import query_rag, ask_ollama
# Reuse the retrieval module's existing collection handle for /sessions so we
# don't open a second ChromaDB client against the same store.
from backend.retrieve import _collection
from backend.models.sentiment_model import SentimentModel
from backend.signals import log_signal

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multimodal Study Assistant",
    description="Local RAG pipeline for lecture notes, slides, and audio.",
    version="0.1.0",
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Shared singleton — SentimentModel loads its weights lazily on first predict(),
# so constructing it here is free at import time but keeps the pipeline in
# memory for the server's lifetime instead of reloading it per request.
# Same arrangement as _embedder in retrieve.py.
_sentiment = SentimentModel()


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


@app.get("/sessions")
def list_sessions():
    """Return every session_id present in the collection with its chunk count.

    e.g. [{"session_id": "CM3060", "chunks": 36}]

    NOTE: this pulls all chunk metadata into memory to tally the counts.
    ChromaDB has no group-by, and at project scale (hundreds of chunks) the
    cost is negligible. It would need a real aggregate query at large scale.
    """
    records = _collection.get(include=["metadatas"])

    counts: dict = {}
    for meta in records["metadatas"]:
        sid = meta.get("session_id")
        if sid:
            counts[sid] = counts.get(sid, 0) + 1

    return [
        {"session_id": sid, "chunks": n}
        for sid, n in sorted(counts.items())
    ]


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
    # query_rag treats a falsy session_id as "no filter", which would search
    # every module at once and break isolation. Require an explicit one here
    # rather than silently searching everything.
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required. Select a module before asking a "
                   "question — an empty session_id would search all modules.",
        )

    rag_result = query_rag(
        question=req.question,
        session_id=session_id,
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

    # --- Learning signal: how the student sounds when asking ---------------
    # Entirely best-effort. The answer is already generated at this point, so
    # nothing here is allowed to fail the request: a broken model download, a
    # bad prediction, or an unwritable signals file must all degrade to a
    # logged warning. Hence the deliberately broad except.
    try:
        t_sent = time.perf_counter()
        sentiment = _sentiment.predict(req.question)
        sent_elapsed = time.perf_counter() - t_sent
        print(f"[TIMER] Sentiment classification: {sent_elapsed:.2f}s")

        # Distinct source files behind the answer, order preserved.
        sources = list(dict.fromkeys(
            meta.get("source_file")
            for meta in rag_result["metadatas"]
            if meta.get("source_file")
        ))

        log_signal({
            "session_id": session_id,
            # Mirrors session_id today, but kept as its own field so a
            # finer-grained topic can replace it without a schema change.
            "topic": session_id,
            "signal_type": "sentiment",
            "value": sentiment["label"],
            "score": round(float(sentiment["score"]), 4),
            "question": req.question,
            "retrieved_sources": sources,
        })
        print(f"[SIGNAL] sentiment={sentiment['label']} "
              f"({sentiment['score']:.4f}) logged for session '{session_id}'")
    except Exception as exc:  # noqa: BLE001 — signal logging must never fail a query
        print(f"[WARN] Signal logging failed ({type(exc).__name__}: {exc}) — "
              f"returning the answer anyway.")

    return QueryResponse(
        answer=answer,
        sources=rag_result["metadatas"],
    )
