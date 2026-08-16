"""
FastAPI application entry point for the Multimodal Study Assistant.

Endpoints:
  POST /ingest        — upload a study file and ingest it into ChromaDB
  POST /query         — ask a question and get an LLM-grounded answer
  POST /quiz          — generate an MCQ quiz from a module's material
  POST /quiz/signals  — record per-question quiz outcomes as learning signals
  POST /flashcards    — generate a term/definition flashcard deck
  POST /flashcards/signals — record per-card self-ratings as learning signals
  GET  /confidence    — per-signal-type aggregates for one module
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
from backend.retrieve import query_rag, ask_ollama, log_answer
# Reuse the retrieval module's existing collection handle for /sessions so we
# don't open a second ChromaDB client against the same store.
from backend.retrieve import _collection
from backend.models.sentiment_model import SentimentModel
from backend.signals import log_signal, read_signals

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

# NOTE on `str | None = None` throughout these models.
#
# Under Pydantic v2, `x: str = None` declares a field of type str whose default
# happens to be None. Defaults are not validated, so OMITTING the key works —
# but sending it explicitly as JSON null is a type error and FastAPI rejects the
# whole request with 422 before the endpoint body runs.
#
# The frontend builds payloads with dict literals, so optional keys are always
# present and carry null rather than being absent. That combination silently
# lost 10 of 11 flashcard ratings: every card without provenance sent
# "source_file": null and was rejected at the HTTP boundary, where none of the
# endpoints' per-item failure isolation could see it. Any field the frontend may
# send as null must therefore be Optional in the annotation, not merely defaulted.

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    n_results: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list  # list of metadata dicts from ChromaDB


class QuizRequest(BaseModel):
    session_id: str | None = None
    topic: str = ""
    n_questions: int = 5


class QuizResultItem(BaseModel):
    """One marked question, as scored by the frontend."""
    question: str
    correct: bool
    source_file: str | None = None


class QuizSignalsRequest(BaseModel):
    session_id: str | None = None
    topic: str = ""
    results: list[QuizResultItem] = []


class FlashcardRequest(BaseModel):
    session_id: str | None = None
    topic: str = ""
    n_cards: int = 10


class FlashcardResultItem(BaseModel):
    """One card as self-rated by the student."""
    term: str
    known: bool
    source_file: str | None = None


class FlashcardSignalsRequest(BaseModel):
    session_id: str | None = None
    topic: str = ""
    results: list[FlashcardResultItem] = []


# ---------------------------------------------------------------------------
# Confidence aggregation
# ---------------------------------------------------------------------------

# A sentiment record only counts as evidence at or above this confidence, and
# the rule applies to EVERY label, not just negative ones. Below it the model is
# not telling us anything usable: one terse factual question ("what is
# lemmatization") scored neutral 0.597 while another scored negative 0.375 —
# both are weak readings of the same kind of input, so treating the neutral one
# as a finding while discarding the negative one would be arbitrary. Anything
# under the threshold is reported as inconclusive.
SENTIMENT_SCORE_THRESHOLD = 0.6


def _pct(part: int, whole: int):
    """Percentage to 1dp, or None when there is nothing to divide by."""
    return round(100.0 * part / whole, 1) if whole else None


def _aggregate_quiz(records: list) -> dict:
    correct = sum(1 for r in records if r.get("value") == "correct")
    return {
        "count": len(records),
        "correct": correct,
        "incorrect": len(records) - correct,
        "accuracy_pct": _pct(correct, len(records)),
    }


def _aggregate_flashcard(records: list) -> dict:
    known = sum(1 for r in records if r.get("value") == "known")
    return {
        "count": len(records),
        "known": known,
        "unknown": len(records) - known,
        "known_pct": _pct(known, len(records)),
    }


def _aggregate_sentiment(records: list) -> dict:
    """Tally sentiment, holding every low-confidence reading back.

    The threshold applies symmetrically: a weak neutral is no more informative
    than a weak negative, so both land in `inconclusive`.
    """
    counts = {"negative": 0, "neutral": 0, "positive": 0, "inconclusive": 0}
    for r in records:
        label = r.get("value")
        score = r.get("score") or 0.0
        if label in ("negative", "neutral", "positive") \
                and score >= SENTIMENT_SCORE_THRESHOLD:
            counts[label] += 1
        else:
            # Below threshold, or a label this version does not recognise.
            counts["inconclusive"] += 1
    return {
        "count": len(records),
        **counts,
        "score_threshold": SENTIMENT_SCORE_THRESHOLD,
    }


def _breakdown(records: list, field: str, aggregate) -> list:
    """Group records by a type-specific topic field and aggregate each group.

    `topic` is module-level for every signal type, so the finer-grained subject
    a quiz or deck was actually about lives in its own field. Records without
    that field are skipped rather than bucketed under a placeholder.

    Subjects are matched case-insensitively after trimming, so "NLP" and "nlp"
    are one bucket. LIMITATION: that is the whole of the matching. Subjects are
    free text typed per quiz or deck, so "natural language processing" and
    "NLP terminology" remain separate buckets even though a person would call
    them one subject. Merging those needs semantic comparison, which is
    deliberately not attempted — a wrong merge would silently misreport
    mastery, which is worse than an obviously split breakdown.
    """
    groups: dict = {}
    for r in records:
        subject = (r.get(field) or "").strip().lower()
        if subject:
            groups.setdefault(subject, []).append(r)
    return [
        {"subject": subject, **aggregate(group)}
        for subject, group in sorted(groups.items())
    ]


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


# Delimiter for the numbered source chunks in generation prompts.
#
# It must NOT be bracketed. The original "[CHUNK 3]" collided with IEEE-style
# citation markers in academic PDFs: given a survey paper whose retrieved text
# carried 77 bracketed references spanning [104]-[279], the model cited those
# instead of the chunk labels and returned source indices of 211, 212, 214, 215,
# 216, 218 and 237 for a 10-chunk prompt. Every one of those is a literal
# citation marker in the notes. _coerce_source_index rejected them all, which is
# correct, but the result was flashcards with no provenance — 10 of 11 in one
# deck. Corpora without bracketed citations were unaffected, which is why it
# looked intermittent.
SOURCE_LABEL = "=== SOURCE {i} ==="


def _number_chunks(chunks: list) -> str:
    """Label each chunk with a delimiter no source document is likely to use."""
    return "\n\n".join(
        f"{SOURCE_LABEL.format(i=i)}\n{text}" for i, text in enumerate(chunks)
    )


def _source_index_rules(n_chunks: int) -> str:
    """Shared wording constraining source_index to the labels actually present."""
    return (
        f"- source_index must be one of the SOURCE numbers above: an integer "
        f"from 0 to {n_chunks - 1} inclusive\n"
        "- never copy a bracketed number such as [211] out of the note text. "
        "Those are the source document's own citation markers, not source "
        "numbers, and they are not valid here\n"
    )


def _build_quiz_prompt(topic: str, chunks: list, n_questions: int) -> str:
    """Build the structured MCQ prompt, with the source chunks numbered.

    The chunks are numbered so the model can cite which one each question came
    from, which is what gives the frontend its provenance display.
    """
    numbered = _number_chunks(chunks)
    last = len(chunks) - 1

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
        f'  "source_index": integer from 0 to {last}, the number of the '
        "=== SOURCE n === section the question came from\n\n"
        "Required shape:\n"
        '[{"question": "...", "options": ["A", "B", "C", "D"], '
        '"correct_index": 2, "source_index": 0}]\n\n'
        "Rules:\n"
        "- exactly 4 options per question\n"
        "- exactly one option is correct\n"
        "- the three wrong options must be plausible but wrong per the notes\n"
        "- do not repeat a question\n"
        + _source_index_rules(len(chunks))
    )


def _build_flashcard_prompt(topic: str, chunks: list, n_cards: int) -> str:
    """Build the structured flashcard prompt, with the source chunks numbered.

    Fronts are terms, not questions — that keeps flashcards distinct from the
    quiz, which is the feature that asks questions.
    """
    numbered = _number_chunks(chunks)
    last = len(chunks) - 1

    return (
        "You are making revision flashcards from a student's own lecture "
        "notes. Use ONLY the numbered notes below. Do not invent anything "
        "that is not stated in them.\n\n"
        f"{numbered}\n\n"
        f'Make exactly {n_cards} flashcards about "{topic}".\n\n'
        "Each card has a TERM on the front and its DEFINITION on the back.\n"
        "The term must be a concept, term, or name — a short noun phrase. It "
        "must NOT be a question: do not start it with what, why, how, when, "
        "which, or who, and do not end it with a question mark.\n\n"
        "Return ONLY a JSON array. No prose before or after it, no markdown "
        "code fences. Each element must be an object with exactly these keys:\n"
        '  "term": string, the front of the card\n'
        '  "definition": string, the back of the card\n'
        f'  "source_index": integer from 0 to {last}, the number of the '
        "=== SOURCE n === section the card came from\n\n"
        "Required shape:\n"
        '[{"term": "Tokenization", "definition": "Breaking a string of text '
        'into individual chunks called tokens.", "source_index": 0}]\n\n'
        "Rules:\n"
        "- the term is a noun phrase, never a question\n"
        "- the definition is one or two sentences, drawn from the notes\n"
        "- do not repeat a term\n"
        + _source_index_rules(len(chunks))
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


def _coerce_source_index(obj, n_chunks: int):
    """Read source_index, or None if absent/unusable.

    Provenance is nice-to-have, not load-bearing: a bad or missing source index
    degrades to "no source shown" rather than dropping the whole item. The range
    check is what stands between a hallucinated index and an IndexError in
    _attach_provenance, so it must stay.

    "source_chunk_index" is accepted as a fallback: it is the name earlier
    prompts used, and a model that has seen the old phrasing may still emit it.
    """
    raw = obj.get("source_index", obj.get("source_chunk_index"))
    try:
        source_index = int(raw)
    except (TypeError, ValueError):
        return None
    return source_index if 0 <= source_index < n_chunks else None


def _validate_quiz_item(obj, n_chunks: int):
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

    return {
        "question": question.strip(),
        "options": options,
        "correct_index": correct_index,
        "source_index": _coerce_source_index(obj, n_chunks),
    }, None


def _validate_flashcard(obj, n_chunks: int):
    """Coerce one raw object into a flashcard, or return (None, reason).

    A card is a term on the front and its definition on the back — deliberately
    not a question, which is the quiz's job.
    """
    if not isinstance(obj, dict):
        return None, "not a JSON object"

    term = obj.get("term")
    if not isinstance(term, str) or not term.strip():
        return None, "missing or empty 'term'"

    definition = obj.get("definition")
    if not isinstance(definition, str) or not definition.strip():
        return None, "missing or empty 'definition'"

    return {
        "term": term.strip(),
        "definition": definition.strip(),
        "source_index": _coerce_source_index(obj, n_chunks),
    }, None


def _parse_items(raw: str, validate):
    """Parse a JSON array out of the model's output as leniently as possible.

    Handles, in order: markdown fences, prose preamble/postamble around the
    array, trailing commas, and — if the array as a whole cannot be parsed —
    salvaging individual objects so one malformed entry does not lose the rest.

    The recovery is format-agnostic; `validate` decides what a valid item looks
    like, so quizzes and flashcards share this without duplicating it.

    Args:
        raw:      The model's unedited completion.
        validate: Callable taking one decoded object and returning
                  (item, None) on success or (None, reason) on rejection.

    Returns:
        (items, warnings)
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

    complete_array = start != -1 and end > start
    if complete_array:
        if start > 0 or end < len(text.strip()) - 1:
            warnings.append("prose around the JSON array was stripped")
        array_text = text[start:end + 1]
        parsed = _loads_lenient(array_text)
    else:
        # No closing bracket: the model was cut off mid-array, or never opened
        # one. Any objects written before the cut are still perfectly good, so
        # fall through to salvage rather than discarding the whole response.
        array_text = text[start:] if start != -1 else text
        parsed = None

    if isinstance(parsed, list):
        candidates = parsed
    else:
        # 3. Array-level parse failed or was impossible — salvage whatever
        #    complete objects we can. An object truncated mid-write simply does
        #    not match the regex, so it is dropped while its siblings survive.
        warnings.append(
            "array did not parse; recovered individual objects instead"
            if complete_array else
            "no complete JSON array (unterminated or absent); "
            "recovered individual objects instead"
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
        item, reason = validate(obj)
        if item is None:
            warnings.append(f"item {i} rejected: {reason}")
        else:
            items.append(item)

    return items, warnings


def _attach_provenance(items: list, chunks: list, metadatas: list) -> None:
    """Attach each item's source chunk text and metadata, in place.

    Shared by /quiz and /flashcards so both show provenance the same way.
    """
    for item in items:
        idx = item["source_index"]
        meta = metadatas[idx] if idx is not None else {}
        item["source_chunk_text"] = chunks[idx] if idx is not None else None
        item["source_file"] = meta.get("source_file")
        item["page_or_slide"] = meta.get("page_or_slide")


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


@app.get("/confidence")
def confidence_endpoint(session_id: str = ""):
    """Aggregate this module's learning signals, per signal type.

    The three types are returned SEPARATELY and never combined into a single
    score. They are not equally trustworthy: quiz results are measured,
    flashcard ratings are self-reported, and sentiment is inferred by a model
    that is domain-mismatched. Averaging them would produce a number with no
    defensible meaning.

    Returns zeroed aggregates for a module with no signals — an empty history
    is a normal state, not an error.
    """
    session_id = (session_id or "").strip()
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required. Select a module — an empty "
                   "session_id would aggregate across all modules.",
        )

    # read_signals() absorbs a missing/empty/corrupt file into an empty list.
    mine = [r for r in read_signals() if r.get("session_id") == session_id]

    by_type: dict = {"quiz": [], "flashcard": [], "sentiment": []}
    for r in mine:
        by_type.setdefault(r.get("signal_type"), []).append(r)

    return {
        "session_id": session_id,
        "total_signals": len(mine),
        "quiz": _aggregate_quiz(by_type["quiz"]),
        "flashcard": _aggregate_flashcard(by_type["flashcard"]),
        "sentiment": _aggregate_sentiment(by_type["sentiment"]),
        "by_quiz_topic": _breakdown(by_type["quiz"], "quiz_topic", _aggregate_quiz),
        "by_flashcard_topic": _breakdown(
            by_type["flashcard"], "flashcard_topic", _aggregate_flashcard
        ),
    }


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

    # Record what the model actually replied, next to the chunks it was given.
    # query_rag wrote the record but could not fill this in — it retrieves and
    # returns before generation happens. log_answer never raises.
    log_answer(rag_result.get("debug_id"), answer)

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
    items, warnings = _parse_items(
        raw, lambda obj: _validate_quiz_item(obj, len(chunks))
    )
    parse_elapsed = time.perf_counter() - t_parse
    print(f"[TIMER] Quiz parsing: {parse_elapsed:.4f}s")

    # The model sometimes over-produces (e.g. 11 items for a 10-item request).
    # Cap to what was asked for so the frontend never renders more than the
    # student requested; "parsed" below then reports what was actually
    # returned, not what the model happened to emit before the cut.
    if len(items) > n_questions:
        warnings.append(
            f"model produced {len(items)} valid items; truncated to the "
            f"requested {n_questions}"
        )
        items = items[:n_questions]

    # Attach provenance: the chunk each question was drawn from, plus where
    # that chunk originally came from, so the frontend can show it on marking.
    _attach_provenance(items, chunks, metadatas)

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


@app.post("/flashcards")
def flashcards_endpoint(req: FlashcardRequest):
    """Generate a flashcard deck from one module's material.

    Same shape as /quiz — retrieve, prompt, parse leniently, attach provenance
    — but each card is a term and its definition rather than a question.
    """
    # Same guard as /quiz: a blank session_id would draw on every module.
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required. Select a module before generating "
                   "flashcards — an empty session_id would search all modules.",
        )

    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(
            status_code=400,
            detail="topic is required — it is what the cards are retrieved "
                   "and written about.",
        )

    n_cards = max(1, min(int(req.n_cards or 10), 30))

    rag_result = query_rag(
        question=topic,
        session_id=session_id,
        n_results=max(5, n_cards),
    )

    chunks = rag_result["chunks"]
    metadatas = rag_result["metadatas"]

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No study material found for this module. Ingest some files first.",
        )

    prompt = _build_flashcard_prompt(topic, chunks, n_cards)

    t_gen = time.perf_counter()
    try:
        raw = _ollama_generate(prompt)
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )
    gen_elapsed = time.perf_counter() - t_gen
    print(f"[TIMER] Flashcard generation (Ollama): {gen_elapsed:.2f}s "
          f"({len(raw)} chars)")
    print(f"[FLASHCARD RAW] ---- begin model output ----\n{raw}\n"
          f"[FLASHCARD RAW] ---- end model output ----")

    t_parse = time.perf_counter()
    cards, warnings = _parse_items(
        raw, lambda obj: _validate_flashcard(obj, len(chunks))
    )
    parse_elapsed = time.perf_counter() - t_parse
    print(f"[TIMER] Flashcard parsing: {parse_elapsed:.4f}s")

    # A topic with fewer distinct concepts than n_cards makes the model repeat
    # a term with a near-identical definition rather than admit it has run
    # out of material. Dedupe on the term (case-insensitive), keeping the
    # first occurrence, before capping — otherwise a deck that is genuinely
    # smaller than requested would get padded back up with repeats.
    seen_terms = set()
    deduped_cards = []
    for card in cards:
        key = card["term"].lower()
        if key in seen_terms:
            continue
        seen_terms.add(key)
        deduped_cards.append(card)
    if len(deduped_cards) < len(cards):
        warnings.append(
            f"removed {len(cards) - len(deduped_cards)} duplicate term(s)"
        )
    cards = deduped_cards

    # Same over-production guard as /quiz, applied after dedup so it only
    # trims genuine excess rather than cutting into distinct concepts.
    if len(cards) > n_cards:
        warnings.append(
            f"model produced {len(cards)} distinct cards; truncated to the "
            f"requested {n_cards}"
        )
        cards = cards[:n_cards]

    _attach_provenance(cards, chunks, metadatas)

    print(f"[FLASHCARDS] requested={n_cards} parsed={len(cards)} "
          f"warnings={len(warnings)}")

    return {
        "session_id": session_id,
        "topic": topic,
        "requested": n_cards,
        "parsed": len(cards),
        "chunks_used": len(chunks),
        "generation_time_s": round(gen_elapsed, 2),
        "parse_time_s": round(parse_elapsed, 4),
        "warnings": warnings,
        "cards": cards,
    }


@app.post("/flashcards/signals")
def flashcard_signals_endpoint(req: FlashcardSignalsRequest):
    """Record one learning signal per self-rated flashcard.

    Mirrors /quiz/signals: the frontend decides known/unknown (here the student
    does, by pressing a button) and this endpoint only records the outcome.
    """
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required to record flashcard signals.",
        )

    # Best-effort, exactly like the quiz and sentiment signals: the student has
    # already seen the card by the time this runs, so a logging failure must
    # never surface as an error or block moving to the next card.
    logged, failed = 0, 0
    for result in req.results:
        try:
            log_signal({
                "session_id": session_id,
                # topic mirrors session_id, the same invariant every signal
                # type holds to so they stay aggregatable — see the schema
                # docstring in signals.py. The deck's subject goes in
                # flashcard_topic, matching how quiz_topic is handled.
                "topic": session_id,
                "signal_type": "flashcard",
                "value": "known" if result.known else "unknown",
                "score": 1.0 if result.known else 0.0,
                # The card's front stands in for "what was being assessed",
                # the same slot the question fills for quiz and sentiment.
                "question": result.term,
                "retrieved_sources": [result.source_file] if result.source_file else [],
                "flashcard_topic": (req.topic or "").strip() or None,
            })
            logged += 1
        except Exception as exc:  # noqa: BLE001 — logging must never block the deck
            failed += 1
            print(f"[WARN] Flashcard signal logging failed "
                  f"({type(exc).__name__}: {exc}) — continuing.")

    print(f"[SIGNAL] flashcard: {logged} logged, {failed} failed "
          f"for session '{session_id}'")
    return {"logged": logged, "failed": failed}
