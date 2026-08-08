"""
FastAPI application entry point for the Multimodal Study Assistant.

Endpoints:
  POST /ingest        — upload a study file and ingest it into ChromaDB
  POST /query         — ask a question and get an LLM-grounded answer
  POST /quiz          — generate an MCQ quiz from a module's material
  POST /quiz/signals  — record per-question quiz outcomes as learning signals
  GET  /sessions      — list study modules present in the store, with chunk counts
  GET  /health        — check server + Ollama availability
"""

import json
import os
import re
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

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


class QuizRequest(BaseModel):
    session_id: str = None
    topic: str = ""
    n_questions: int = 5


class QuizResultItem(BaseModel):
    """One marked question, as scored by the frontend."""
    question: str
    correct: bool
    source_file: str = None


class QuizSignalsRequest(BaseModel):
    session_id: str = None
    topic: str = ""
    results: list[QuizResultItem] = []


# ---------------------------------------------------------------------------
# Quiz generation helpers
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, model: str = None) -> str:
    """Send a raw prompt to Ollama and return the completion.

    Deliberately not ask_ollama(): that function wraps the prompt in the
    study-assistant answering instructions ("answer using ONLY the notes"),
    which fight against an instruction to emit JSON. Quiz generation needs
    full control of the prompt, so it posts directly.

    Non-streaming — there is nothing to display incrementally, and the whole
    payload has to be parsed as one unit anyway.
    """
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Low temperature: this is a structure-following task, not a creative
        # one. Higher values make the model wander out of the JSON shape.
        "options": {"temperature": 0.2},
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=900
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _build_quiz_prompt(topic: str, chunks: list, n_questions: int) -> str:
    """Build the structured MCQ prompt, with the source chunks numbered.

    The chunks are numbered so the model can cite which one each question came
    from, which is what gives the frontend its provenance display.
    """
    numbered = "\n\n".join(
        f"[CHUNK {i}]\n{text}" for i, text in enumerate(chunks)
    )

    return (
        "You are writing a multiple-choice quiz from a student's own lecture "
        "notes. Use ONLY the numbered notes below. Do not invent facts that "
        "are not stated in them.\n\n"
        f"{numbered}\n\n"
        f'Write exactly {n_questions} multiple-choice questions about "{topic}".\n\n'
        "Return ONLY a JSON array. No prose before or after it, no markdown "
        "code fences. Each element must be an object with exactly these keys:\n"
        '  "question": string\n'
        '  "options": array of exactly 4 distinct strings\n'
        '  "correct_index": integer 0-3, indexing into "options"\n'
        '  "source_chunk_index": integer, the [CHUNK n] the question came from\n\n'
        "Required shape:\n"
        '[{"question": "...", "options": ["A", "B", "C", "D"], '
        '"correct_index": 2, "source_chunk_index": 0}]\n\n'
        "Rules:\n"
        "- exactly 4 options per question\n"
        "- exactly one option is correct\n"
        "- the three wrong options must be plausible but wrong per the notes\n"
        "- do not repeat a question\n"
    )


def _loads_lenient(text: str):
    """json.loads, retried once with trailing commas stripped."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Trailing commas before a closing brace/bracket are the single most
    # common thing a small model gets wrong; strip them and retry.
    try:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))
    except (json.JSONDecodeError, ValueError):
        return None


def _validate_item(obj, n_chunks: int):
    """Coerce one raw object into a quiz item, or return (None, reason)."""
    if not isinstance(obj, dict):
        return None, "not a JSON object"

    question = obj.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, "missing or empty 'question'"

    options = obj.get("options")
    if not isinstance(options, list) or len(options) != 4:
        got = len(options) if isinstance(options, list) else type(options).__name__
        return None, f"'options' must be a list of 4, got {got}"
    options = [str(o) for o in options]

    try:
        correct_index = int(obj.get("correct_index"))
    except (TypeError, ValueError):
        return None, f"'correct_index' not an integer: {obj.get('correct_index')!r}"
    if not 0 <= correct_index <= 3:
        return None, f"'correct_index' out of range: {correct_index}"

    # Provenance is nice-to-have, not load-bearing: a bad or missing chunk
    # index degrades to "no source shown" rather than dropping the question.
    try:
        source_index = int(obj.get("source_chunk_index"))
    except (TypeError, ValueError):
        source_index = None
    if source_index is None or not 0 <= source_index < n_chunks:
        source_index = None

    return {
        "question": question.strip(),
        "options": options,
        "correct_index": correct_index,
        "source_chunk_index": source_index,
    }, None


def _parse_quiz_items(raw: str, n_chunks: int):
    """Parse the model's output into quiz items as leniently as possible.

    Handles, in order: markdown fences, prose preamble/postamble around the
    array, trailing commas, and — if the array as a whole cannot be parsed —
    salvaging individual objects so a single malformed item does not lose the
    rest of the quiz.

    Returns (items, warnings).
    """
    warnings = []
    text = raw.strip()

    # 1. Drop markdown code fences wherever they appear.
    if "```" in text:
        warnings.append("output was wrapped in markdown fences")
        # Re-strip: the removed fences leave newlines that would otherwise look
        # like prose to the offset check below.
        text = re.sub(r"```(?:json)?", "", text).strip()

    # 2. Trim prose either side of the outermost array.
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        warnings.append("no JSON array found in model output")
        candidates = []
    else:
        if start > 0 or end < len(text.strip()) - 1:
            warnings.append("prose around the JSON array was stripped")
        array_text = text[start:end + 1]

        parsed = _loads_lenient(array_text)
        if isinstance(parsed, list):
            candidates = parsed
        else:
            # 3. Array-level parse failed — salvage whatever objects we can.
            warnings.append(
                "array did not parse; recovered individual objects instead"
            )
            candidates = [
                obj for obj in (
                    _loads_lenient(m.group(0))
                    for m in re.finditer(r"\{[^{}]*\}", array_text, re.DOTALL)
                )
                if obj is not None
            ]

    items = []
    for i, obj in enumerate(candidates):
        item, reason = _validate_item(obj, n_chunks)
        if item is None:
            warnings.append(f"item {i} rejected: {reason}")
        else:
            items.append(item)

    return items, warnings


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


@app.post("/quiz")
def quiz_endpoint(req: QuizRequest):
    """Generate a multiple-choice quiz from one module's material.

    Steps:
    1. Retrieve chunks on the topic (query_rag — same retrieval as /query)
    2. One structured prompt to Ollama asking for the questions as JSON
    3. Parse leniently, keeping whatever items are valid
    4. Attach each item's source chunk text and metadata for provenance
    """
    # Same guard as /query: a blank session_id would quiz across every module.
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required. Select a module before generating "
                   "a quiz — an empty session_id would search all modules.",
        )

    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(
            status_code=400,
            detail="topic is required — it is what the questions are retrieved "
                   "and written about.",
        )

    # Keep the request sane: n_questions drives both the prompt and how much
    # context we retrieve, so an absurd value would produce an absurd prompt.
    n_questions = max(1, min(int(req.n_questions or 5), 20))

    # Retrieve at least as many chunks as questions, so there is some chance of
    # each question having its own source rather than all sharing one.
    rag_result = query_rag(
        question=topic,
        session_id=session_id,
        n_results=max(5, n_questions),
    )

    chunks = rag_result["chunks"]
    metadatas = rag_result["metadatas"]

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No study material found for this module. Ingest some files first.",
        )

    prompt = _build_quiz_prompt(topic, chunks, n_questions)

    t_gen = time.perf_counter()
    try:
        raw = _ollama_generate(prompt)
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )
    gen_elapsed = time.perf_counter() - t_gen
    print(f"[TIMER] Quiz generation (Ollama): {gen_elapsed:.2f}s "
          f"({len(raw)} chars)")
    # Logged in full so a parse failure can always be diagnosed after the fact.
    print(f"[QUIZ RAW] ---- begin model output ----\n{raw}\n"
          f"[QUIZ RAW] ---- end model output ----")

    t_parse = time.perf_counter()
    items, warnings = _parse_quiz_items(raw, len(chunks))
    parse_elapsed = time.perf_counter() - t_parse
    print(f"[TIMER] Quiz parsing: {parse_elapsed:.4f}s")

    # Attach provenance: the chunk each question was drawn from, plus where
    # that chunk originally came from, so the frontend can show it on marking.
    for item in items:
        idx = item["source_chunk_index"]
        meta = metadatas[idx] if idx is not None else {}
        item["source_chunk_text"] = chunks[idx] if idx is not None else None
        item["source_file"] = meta.get("source_file")
        item["page_or_slide"] = meta.get("page_or_slide")

    print(f"[QUIZ] requested={n_questions} parsed={len(items)} "
          f"warnings={len(warnings)}")

    return {
        "session_id": session_id,
        "topic": topic,
        "requested": n_questions,
        "parsed": len(items),
        "chunks_used": len(chunks),
        "generation_time_s": round(gen_elapsed, 2),
        "parse_time_s": round(parse_elapsed, 4),
        "warnings": warnings,
        "items": items,
    }


@app.post("/quiz/signals")
def quiz_signals_endpoint(req: QuizSignalsRequest):
    """Record one learning signal per marked quiz question.

    Marking itself happens in the frontend (a plain index comparison, never
    the LLM); this endpoint only records the outcomes. It exists so the
    frontend stays an HTTP client of the backend rather than importing
    backend.signals and writing the signals file behind the API's back.
    """
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required to record quiz signals.",
        )

    # Best-effort, exactly like the sentiment signal on /query: the student has
    # already seen their score by the time this is called, so a logging failure
    # must never surface as an error. Each record is isolated so one bad item
    # cannot lose the rest.
    logged, failed = 0, 0
    for result in req.results:
        try:
            log_signal({
                "session_id": session_id,
                # topic mirrors session_id here exactly as it does for the
                # sentiment records. Both signal types must mean the same thing
                # by "topic" so the confidence tracker can aggregate across
                # them; the quiz's own subject goes in quiz_topic below.
                "topic": session_id,
                "signal_type": "quiz",
                "value": "correct" if result.correct else "incorrect",
                # 1.0/0.0 keeps the field numeric and comparable with the
                # sentiment confidence score sharing this column.
                "score": 1.0 if result.correct else 0.0,
                "question": result.question,
                "retrieved_sources": [result.source_file] if result.source_file else [],
                # Signal-type-specific extra: what the quiz was actually about.
                # Kept out of `topic` so that field stays comparable across
                # every signal type.
                "quiz_topic": (req.topic or "").strip() or None,
            })
            logged += 1
        except Exception as exc:  # noqa: BLE001 — logging must never fail marking
            failed += 1
            print(f"[WARN] Quiz signal logging failed "
                  f"({type(exc).__name__}: {exc}) — continuing.")

    print(f"[SIGNAL] quiz: {logged} logged, {failed} failed "
          f"for session '{session_id}'")
    return {"logged": logged, "failed": failed}
