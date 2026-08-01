"""
RAG retrieval and Ollama query logic.

Two public functions:
- query_rag()   — embed a question and retrieve similar chunks from ChromaDB
- ask_ollama()  — build a RAG prompt and call the local Ollama LLM
"""

import json
import os
import time
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import chromadb

from backend.models.embedder import Embedder

# ---------------------------------------------------------------------------
# Shared singletons — lazy model loading is handled inside Embedder
# ---------------------------------------------------------------------------
_embedder = Embedder()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./vectorstore/chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Reuse the same client / collection object across calls
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_or_create_collection(
    name="study_materials",
    metadata={"hnsw:space": "cosine"},
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def query_rag(
    question: str,
    session_id: str = None,
    n_results: int = 5,
) -> dict:
    """Retrieve the most relevant study chunks for a question.

    Embeds the question with the same sentence-transformer model used at
    ingest time, queries ChromaDB by cosine similarity, and optionally
    filters results to a single study session.

    Args:
        question:   The student's query string.
        session_id: If provided, only chunks from this session are returned.
        n_results:  Maximum number of chunks to retrieve.

    Returns:
        A dict with keys:
          - "chunks"    : list of text strings (the retrieved passages)
          - "metadatas" : list of metadata dicts parallel to chunks
          - "distances" : cosine similarity scores (lower = more similar
                          when using cosine space in ChromaDB)
    """
    t_rag = time.perf_counter()

    question_embedding = _embedder.embed(question)[0]  # timed inside Embedder

    where_filter = {"session_id": session_id} if session_id else None

    t_chroma = time.perf_counter()
    results = _collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    print(f"[TIMER] ChromaDB query: {time.perf_counter() - t_chroma:.2f}s")
    print(f"[TIMER] query_rag total: {time.perf_counter() - t_rag:.2f}s")

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    debug_path = Path("query_debug.json")
    try:
        existing = json.loads(debug_path.read_text(encoding="utf-8")) if debug_path.exists() else []
    except (json.JSONDecodeError, ValueError):
        # File exists but is empty or corrupt — start fresh rather than crashing
        existing = []
    existing.append({
        "asked_at": datetime.now().isoformat(),
        "question": question,
        "session_id": session_id,
        "retrieved": [
            {"rank": i + 1, "distance": distances[i], "metadata": metadatas[i], "text": chunks[i]}
            for i in range(len(chunks))
        ],
    })
    debug_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[CHUNKS] {len(chunks)} retrieved chunk(s) saved to query_debug.json")

    return {
        "chunks": chunks,
        "metadatas": metadatas,
        "distances": distances,
    }


def ask_ollama(
    question: str,
    context_chunks: list,
    model: str = None,
) -> str:
    """Send a RAG-augmented prompt to the local Ollama server and return the answer.

    The prompt instructs the model to answer ONLY from the provided notes,
    which prevents hallucination beyond the student's own study materials.

    Args:
        question:       The student's question.
        context_chunks: List of text strings retrieved from ChromaDB.
        model:          Ollama model name. Defaults to OLLAMA_MODEL env var.

    Returns:
        The model's answer as a plain string.

    Raises:
        requests.exceptions.ConnectionError: If Ollama is not running.
    """
    model = model or OLLAMA_MODEL

    notes_block = "\n\n---\n\n".join(context_chunks)

    prompt = (
        "You are a study assistant. Answer the question using ONLY the notes "
        "from the student's lectures provided below. Do not use any knowledge "
        "outside these notes, and do not invent details.\n\n"
        "Be thorough and detailed: include all relevant facts, definitions, "
        "examples, and explanations found in the notes. Preserve specific terms, "
        "names, numbers, and wording from the notes rather than paraphrasing them "
        "away. Do NOT over-summarise or condense — it is better to be complete "
        "than brief. If the notes contain the information, report it in full. "
        "Only state that the notes do not cover something if it is genuinely absent.\n\n"
        f"Notes:\n{notes_block}\n\n"
        f"Question: {question}\n\n"
        "Detailed answer based on the notes:"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    t_ollama = time.perf_counter()
    print(f"[TIMER] Ollama ({model}) generating...")
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=600,
    )
    response.raise_for_status()

    full_answer = []
    token_count = 0
    for line in response.iter_lines():
        if not line:
            continue
        import json
        chunk = json.loads(line)
        full_answer.append(chunk.get("response", ""))
        token_count += 1
        if chunk.get("done"):
            break

    elapsed = time.perf_counter() - t_ollama
    print(f"[TIMER] Ollama total: {elapsed:.2f}s (~{token_count} tokens, {token_count/elapsed:.1f} tok/s)")
    return "".join(full_answer).strip()
