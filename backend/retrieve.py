"""
RAG retrieval and Ollama query logic.

Three public functions:
- query_rag()   — embed a question and retrieve similar chunks from ChromaDB
- ask_ollama()  — build a RAG prompt and call the local Ollama LLM
- log_answer()  — attach a generated answer to the debug record query_rag wrote
"""

import json
import os
import time
import uuid
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import chromadb

from backend.models.embedder import Embedder, collection_name
from backend.paths import resolve_path

# ---------------------------------------------------------------------------
# Shared singletons — lazy model loading is handled inside Embedder
# ---------------------------------------------------------------------------
_embedder = Embedder()

CHROMA_PATH = resolve_path("CHROMA_PATH", "./vectorstore/chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Ollama reports its internal durations in nanoseconds.
NS = 1e9

# ---------------------------------------------------------------------------
# Query debug log
#
# One record per retrieval, appended to query_debug.json:
#
#   id             stable uuid, so the answer can be attached to this exact record
#   asked_at       when the retrieval ran
#   question       the query string
#   session_id     the module searched
#   answer         what the model replied, filled in by log_answer() after
#                  generation; stays null if generation failed or the caller
#                  never generated (e.g. /quiz and /flashcards, which retrieve
#                  through query_rag but produce JSON items rather than an answer)
#   answered_at    when the answer was attached
#   retrieved      the chunks passed to the model, with rank/distance/metadata
#
# NOTE: answers were NOT persisted before this change — every record written
# prior to it has no "answer" key at all, not a null one. Absence of the key
# means "never recorded"; a null value means "recorded, but no answer arrived".
# Historical records therefore cannot show whether an ingest-stage error (an
# ASR mis-transcription, say) actually reached the student in an answer; only
# that the flawed chunk was retrieved into the model's context.
# ---------------------------------------------------------------------------
DEBUG_PATH = Path("query_debug.json")


def _read_debug() -> list:
    """Read the debug log, tolerating a missing, empty or corrupt file."""
    try:
        records = json.loads(DEBUG_PATH.read_text(encoding="utf-8")) if DEBUG_PATH.exists() else []
    except (json.JSONDecodeError, ValueError):
        # File exists but is empty or corrupt — start fresh rather than crashing
        return []
    return records if isinstance(records, list) else [records]


# Reuse the same client / collection object across calls.
# The collection is derived from the active embedding model — a 384-dim and a
# 768-dim model cannot share one collection, so they must not share a name.
COLLECTION_NAME = collection_name()
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
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

    entry_id = str(uuid.uuid4())
    existing = _read_debug()
    existing.append({
        "id": entry_id,
        "asked_at": datetime.now().isoformat(),
        "question": question,
        "session_id": session_id,
        # Filled in later by log_answer(), once the model has actually replied.
        # query_rag only retrieves, so the answer does not exist at this point.
        "answer": None,
        "answered_at": None,
        "retrieved": [
            {"rank": i + 1, "distance": distances[i], "metadata": metadatas[i], "text": chunks[i]}
            for i in range(len(chunks))
        ],
    })
    DEBUG_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[CHUNKS] {len(chunks)} retrieved chunk(s) saved to {DEBUG_PATH}")

    return {
        "chunks": chunks,
        "metadatas": metadatas,
        "distances": distances,
        # Lets the caller attach the model's answer to this exact record.
        "debug_id": entry_id,
    }


def log_answer(debug_id: str, answer: str) -> bool:
    """Attach the model's answer to the debug record query_rag wrote.

    Called by the endpoint after generation succeeds, because query_rag is a
    retrieval function and the answer does not exist while it runs.

    Debug logging must never break a query that otherwise worked, so every
    failure here is reported and swallowed rather than raised.

    Returns:
        True if the record was found and updated, False otherwise.
    """
    if not debug_id:
        return False
    try:
        records = _read_debug()
        for record in reversed(records):     # newest first: it is almost always the last
            if record.get("id") == debug_id:
                record["answer"] = answer
                record["answered_at"] = datetime.now().isoformat()
                DEBUG_PATH.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                return True
        print(f"[WARN] no debug record with id {debug_id}; answer not logged")
        return False
    except Exception as exc:                 # noqa: BLE001 - never fail the query
        print(f"[WARN] could not log answer to {DEBUG_PATH}: "
              f"{type(exc).__name__}: {exc}")
        return False


def build_rag_prompt(question: str, context_chunks: list) -> str:
    """Build the grounded study-assistant prompt from retrieved chunks.

    Extracted from ask_ollama() so evaluation harnesses can send the exact
    production prompt to other models without copying the wording — a copy
    would silently drift the day this is edited, and any model comparison
    built on a stale prompt measures the wrong thing.

    The instructions restrict the model to the supplied notes, which is what
    keeps answers grounded in the student's own material.
    """
    notes_block = "\n\n---\n\n".join(context_chunks)
    return (
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

    prompt = build_rag_prompt(question, context_chunks)

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
    final_chunk = {}
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        full_answer.append(chunk.get("response", ""))
        if chunk.get("done"):
            final_chunk = chunk
            break

    elapsed = time.perf_counter() - t_ollama

    # Throughput comes from Ollama's own counters on the final frame, as
    # verification/eval_llm.py does: eval_count is the number of tokens actually
    # generated and eval_duration the nanoseconds spent generating them.
    # Dividing those isolates generation speed from model load, prompt
    # evaluation and connection time, all of which sit inside the wall clock.
    # Counting streamed frames against wall time instead understated the rate
    # by roughly threefold (~3 tok/s reported where Ollama measured 10-13).
    eval_count = final_chunk.get("eval_count") or 0
    eval_duration = final_chunk.get("eval_duration") or 0

    if eval_count and eval_duration:
        gen_seconds = eval_duration / NS
        print(
            f"[TIMER] Ollama total: {elapsed:.2f}s wall | generation "
            f"{eval_count} tokens in {gen_seconds:.2f}s "
            f"({eval_count / gen_seconds:.1f} tok/s)"
        )
    else:
        # Ollama omits these on some terminal frames (cancelled or load-only
        # responses). Report the wall time rather than divide by zero.
        print(
            f"[TIMER] Ollama total: {elapsed:.2f}s wall | "
            f"generation counters unavailable"
        )

    return "".join(full_answer).strip()
