# Multimodal Study Assistant — Project Documentation

> A fully-local, multimodal Retrieval-Augmented Generation (RAG) system that ingests
> lecture materials (PDF, PPTX, audio, images), embeds them into a vector store, and
> answers questions using a local LLM. **No external API keys are required** — every
> model runs on the user's own machine.

For setup and usage, see `README.md`. This file explains how the system works and why
it is built the way it is.

---

## 1. What This Project Does

The system lets a student turn their own study material into a searchable,
question-answerable knowledge base, and then practise against it. The workflow is:

1. **Upload** a file — PDF notes, PowerPoint slides, an audio recording of a lecture,
   or an image/diagram.
2. The system **extracts text** from it, using OCR, speech-to-text, and image
   captioning where needed.
3. The text is **chunked, embedded, and stored** in a local vector database (ChromaDB).
4. The student **asks questions**, **generates quizzes**, or **generates flashcards**.
   All three retrieve from the student's own material and feed it to a local LLM (via
   Ollama), which is instructed to answer **using only those notes** — preventing the
   model from inventing facts beyond the source material.
5. The system records **learning signals** — quiz results, flashcard self-ratings, and
   the tone of the questions asked — so the student can see which topics are going badly.

Everything is local: no cloud, no API keys, no data leaving the machine.

---

## 2. High-Level Architecture

```
┌──────────────┐     HTTP      ┌────────────────────────┐
│  Streamlit   │ ◀──────────▶  │      FastAPI backend   │
│  frontend    │   9 endpoints │   (backend/main.py)    │
│  (app.py)    │               │                        │
└──────────────┘               └───────────┬────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
          ┌──────────────────┐   ┌────────────────────┐  ┌─────────────────┐
          │   ingest.py      │   │    retrieve.py     │  │   signals.py    │
          │ (file → chunks)  │   │ (query → answer)   │  │ (learning log)  │
          └────────┬─────────┘   └─────────┬──────────┘  └────────┬────────┘
                   │                       │                      │
     ┌─────┬───────┼───────┬─────┐         │                      ▼
     ▼     ▼       ▼       ▼     ▼         ▼                 signals.json
  Whisper BLIP   OCR  Embedder ChromaDB  ChromaDB + Ollama
  (audio)(image)(image)(text) (vectors) (retrieval + LLM)
```

**Two pipelines** share one ChromaDB collection (`study_materials`, cosine similarity):

- **Ingest** (`ingest.py`) — converts a file into embedded chunks and stores them.
- **Query** (`retrieve.py`) — embeds a question, retrieves similar chunks, asks the LLM.

Both are partitioned by `session_id`, a study-module namespace. Quiz and flashcard
generation reuse the query pipeline: they retrieve chunks with `query_rag()` and then
ask the LLM for structured JSON instead of prose.

The frontend talks to the backend over HTTP only. Every `requests.*` call in the
project lives in `frontend/app.py` and runs server-side, so the browser only ever
contacts Streamlit.

---

## 3. Directory Layout

```
multimodal-study-assistant/
├── backend/
│   ├── main.py                  # FastAPI app — 9 endpoints
│   ├── ingest.py                # Ingestion pipeline (PDF/PPTX/audio/image → ChromaDB)
│   ├── retrieve.py              # RAG retrieval + Ollama LLM call
│   ├── signals.py               # Learning-signal log (append-only JSON array)
│   ├── paths.py                 # Resolves relative paths against the project root
│   ├── models/
│   │   ├── embedder.py          # sentence-transformers all-MiniLM-L6-v2 (384-dim)
│   │   ├── whisper_model.py     # openai-whisper speech-to-text
│   │   ├── blip_model.py        # BLIP image captioning
│   │   ├── ocr_model.py         # pytesseract OCR
│   │   └── sentiment_model.py   # RoBERTa three-class question-tone classifier
│   └── tests/                   # 47 pytest unit tests (all models mocked)
│       ├── test_ingest.py
│       ├── test_retrieve.py
│       ├── test_quiz_flashcards.py
│       ├── test_signal_endpoints.py
│       ├── test_blip.py
│       ├── test_ocr.py
│       ├── test_whisper.py
│       └── test_sentiment.py
├── frontend/
│   └── app.py                   # Streamlit UI — Chat, Quiz, Flashcards, Progress
├── verification/                # Standalone evaluation scripts + findings documents
├── sample_files/                # Test material (gitignored)
├── vectorstore/                 # ChromaDB persistence (gitignored)
├── models_cache/                # HuggingFace weight cache (gitignored)
├── clear_db.py                  # Utility to reset the vector store
├── conftest.py                  # Adds project root to sys.path so `backend.*` resolves
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Configuration template
├── README.md                    # Setup & usage instructions
└── DOCUMENTATION.md             # (this file)
```

---

## 4. The Backend in Detail

### 4.1 `backend/main.py` — FastAPI Application

The HTTP entry point. Nine endpoints:

| Method & Path | Purpose | Key behaviour |
|---|---|---|
| `GET /health` | Liveness check | `{"status": "ok", "ollama": <bool>}`; pings Ollama to confirm it is reachable. |
| `GET /sessions` | List study modules | Every `session_id` in the store with its chunk count. Pulls all chunk metadata to tally — ChromaDB has no group-by, and at project scale the cost is negligible. |
| `GET /confidence` | Learning-signal aggregates | Per-module quiz / flashcard / sentiment aggregates, returned **separately** (see §11). |
| `POST /ingest` | Upload & ingest a file | Multipart file + `session_id`. Saves to a temp file, calls `ingest_file()`, deletes the temp file in a `finally`. Returns `{filename, session_id, chunks_stored}`. |
| `POST /query` | Ask a question | `{question, session_id, n_results}`. Retrieves via `query_rag()`, generates via `ask_ollama()`, logs the answer and a sentiment signal. Returns `{answer, sources}`. |
| `POST /quiz` | Generate a quiz | `{topic, session_id, n_questions}`. Retrieves on the topic, asks the LLM for JSON, keeps whatever validates, truncates to the requested count, attaches source-chunk provenance. |
| `POST /quiz/signals` | Record quiz results | Client-side marking reported back for logging. |
| `POST /flashcards` | Generate flashcards | Same shape as `/quiz`, with term/definition pairs and deduplication. |
| `POST /flashcards/signals` | Record card ratings | Known/unknown self-ratings reported back for logging. |

**Every endpoint requires a non-empty `session_id`.** A blank one would silently search
or aggregate across every module at once, so it is rejected with a `400` rather than
treated as "no filter".

**Error handling:**
- Unsupported file type → `415`
- Other ingest failures → `500`
- Missing or empty `session_id` → `400`
- No relevant chunks found → `404` ("ingest some files first")
- Ollama not running → `503` ("Start it with: ollama serve")

**Pydantic optional fields.** Every frontend-facing optional field is declared
`str | None = None`, not merely defaulted. Under Pydantic v2 a field typed plainly
`str = None` still rejects an explicit JSON `null` with a `422`, and the Streamlit
frontend's dict-literal payloads always send optional keys with `null` rather than
omitting them. This previously dropped 10 of 11 flashcard ratings silently.

### 4.2 `backend/ingest.py` — Ingestion Pipeline

Detects file type by extension and routes to a specialised extractor. All extractors
return `(chunks, metadatas)`, which `_store_chunks()` then embeds and upserts.

**Module-level singletons** (lazily loaded) are created once and reused: `_whisper`,
`_blip`, `_ocr`, `_embedder`, plus the ChromaDB `_collection` (cosine similarity,
`metadata={"hnsw:space": "cosine"}`).

| Type | Handling |
|---|---|
| **PDF** | `pdfplumber` text extraction, with `_fix_doubled_chars()` repairing the "EEvveerr"-style doubled-glyph artifact. Pages yielding under 50 characters fall back to rendering the page as an image (`pdf2image`) and running BLIP captioning + OCR. Chunked paragraph-aware, ~220 words with 40-word overlap. |
| **PPTX** | One chunk per slide: title + body + speaker notes + embedded-image text. `_collect_shapes()` recurses into group shapes, OCRs picture shapes (BLIP fallback when OCR is thin), and scrapes `<a:t>` runs directly from the DrawingML XML to recover SmartArt, tables and charts that have no text frame. |
| **Audio** | Whisper transcription, 200-word sliding window with 50-word overlap. |
| **Image** | A single chunk of BLIP caption + OCR text. |

Every ingest appends a record (chunk text + metadata) to `chunks_debug.json`.

### 4.3 `backend/retrieve.py` — RAG Retrieval & LLM

- **`query_rag(question, session_id=None, n_results=5)`** — embeds the question with
  the *same* embedder used at ingest time (critical for vector compatibility), queries
  ChromaDB by cosine distance filtered to `session_id`, appends a record to
  `query_debug.json`, and returns `{chunks, metadatas, distances, debug_id}`.

- **`ask_ollama(question, context_chunks, model=None)`** — builds a strict grounded
  prompt (*"answer using ONLY the following notes…"*) and streams from Ollama's
  `POST /api/generate`. Raises `ConnectionError` when Ollama is unreachable; the
  endpoint translates that to a `503`.

- **`log_answer(debug_id, answer)`** — backfills the debug record with the generated
  answer, since generation happens after retrieval returns. It never raises.

  Note that `log_answer` is called only from `/query`. Quiz and flashcard generation
  go through `query_rag` but produce structured items rather than an answer, so their
  debug records carry `"answer": null` by design.

**Throughput reporting.** The `[TIMER]` line reports generation speed from Ollama's
own `eval_count` and `eval_duration` counters on the final streamed frame, alongside
total wall time. The two differ substantially: wall time includes model load, prompt
evaluation and connection overhead, so counting streamed frames against wall time
understates throughput by roughly threefold.

### 4.4 `backend/models/` — Lazy-Loaded Model Wrappers

Every wrapper follows the same pattern: **the model is not loaded at import time**,
only on first use (`_load()`), keeping startup fast. Weights are cached to
`./models_cache/` so they survive venv recreation.

| Wrapper | Backing model | Notes |
|---|---|---|
| `embedder.py` | `all-MiniLM-L6-v2` | 384-dim embeddings; accepts a string or list, always returns a list of vectors. |
| `whisper_model.py` | `openai-whisper` | Size from `WHISPER_MODEL_SIZE` (default `base`). `fp16=False` for CPU. |
| `blip_model.py` | `Salesforce/blip-image-captioning-base` | Caption, max 50 new tokens. Accepts a PIL image or a path. |
| `ocr_model.py` | `pytesseract` | No weights; relies on the OS Tesseract install. |
| `sentiment_model.py` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Three-class (negative / neutral / positive), `device=-1` (CPU). Returns `{label, score}`. |

**Why three-class sentiment.** The earlier model
(`distilbert-base-uncased-finetuned-sst-2-english`) was binary and had nowhere to put
affectless text: plain factual questions such as "what is lemmatization" came back
NEGATIVE at over 0.99 confidence, indistinguishable from genuine frustration. With a
neutral class, those land on neutral instead, so a negative label carries information.

### 4.5 `backend/paths.py` — Path Anchoring

`CHROMA_PATH`, `MODELS_CACHE` and `SIGNALS_PATH` are documented as relative paths, and
a relative path resolves against `os.getcwd()` at process start. Depending on whether
uvicorn, Streamlit, or a script under `verification/` was launched from the project
root or from inside `backend/`, that silently pointed at two different directories —
`backend/vectorstore/chroma_db` ended up holding chunks the root store never saw, with
no error. `resolve_path()` anchors every relative value to the project root, while an
absolute value in the environment is still honoured as-is.

---

## 5. The Frontend — `frontend/app.py`

A Streamlit application talking to the FastAPI backend over HTTP.

**Sidebar:**
- Live **health indicators** for the API and Ollama, with actionable messages if either is down.
- **Study module switcher** — a dropdown of every module in the store with its chunk
  count, plus a create box for new modules. Dev/evaluation modules listed in
  `HIDDEN_MODULES` are filtered out so they cannot be selected.
- **File uploader** (pdf/pptx/mp3/wav/png/jpg/jpeg) with an ingest button and spinner.
- A running list of ingested files with their chunk counts.

**Participant links.** When the URL carries `?p=<id>`, the module is pinned to that id
for the whole session and the switcher and create controls are replaced by static
text. This was added for moderated user testing, so a participant cannot reach another
participant's material or a hidden evaluation module. Without `?p=`, the sidebar
behaves normally.

**Main area — four tabs:**

| Tab | Contents |
|---|---|
| 💬 **Chat** | Chat interface preserving history per module in `st.session_state`. Each answer carries an expandable **Sources** panel listing source file, type and page/slide for every retrieved chunk. |
| 📝 **Quiz** | Generates multiple-choice questions on a topic, marks them **client-side**, and reports outcomes to `/quiz/signals`. Each question shows the source chunk it came from. |
| 🗂️ **Flashcards** | Generates term/definition cards, flip-to-reveal, rated known/unknown by the student and reported to `/flashcards/signals`. |
| 📊 **Progress** | Reads `/confidence` and renders the three signal types separately — quiz (measured), flashcards (self-reported), question tone (inferred). |

Marking is always done client-side and never re-derived from the LLM; the backend is
told the outcome purely so it can be logged.

---

## 6. Data Flow End-to-End

**Ingestion:**
```
File upload → temp file → ingest_file()
  → type-specific extractor (PDF / PPTX / audio / image)
      → (OCR / BLIP / Whisper as needed)
  → chunking (sliding window or paragraph-aware)
  → Embedder.embed() → 384-dim vectors
  → ChromaDB.upsert()  [keyed by session_id]
  → returns chunks_stored
```

**Query:**
```
Question → query_rag()
  → Embedder.embed(question)
  → ChromaDB.query() [cosine, filtered by session_id] → top-N chunks
  → ask_ollama(): grounded prompt → Ollama /api/generate (streamed)
  → log_answer() backfills the debug record
  → sentiment signal logged (best-effort)
  → answer + source metadata returned to the UI
```

**Quiz / flashcards:** identical up to retrieval, then one structured prompt asking for
JSON, lenient parsing that keeps whatever validates, truncation to the requested count,
and source-chunk provenance attached for display.

---

## 7. Configuration

Configured via `.env` (copy from `.env.example`), loaded with `python-dotenv`.

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `OLLAMA_MODEL` | `llama3.2` | Local LLM tag (must be `ollama pull`ed) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama REST endpoint |
| `CHROMA_PATH` | `./vectorstore/chroma_db` | Persistent vector store location |
| `MODELS_CACHE` | `./models_cache` | HuggingFace / sentence-transformers weight cache |
| `SIGNALS_PATH` | `./signals.json` | Learning-signal log |

Prefer `127.0.0.1` over `localhost` for `OLLAMA_BASE_URL`. On some Windows machines
`localhost` resolves to `::1` first and, because Ollama listens on IPv4 only, every
request waits for the IPv6 attempt to fail — roughly two seconds per call.

---

## 8. System Dependencies (installed at OS level, not via pip)

- **Ollama** — local LLM server on `127.0.0.1:11434` (`ollama pull llama3.2`).
- **Tesseract OCR** — binary required by `pytesseract`, must be on `PATH`.
- **Poppler** — required by `pdf2image` to render PDF pages, must be on `PATH`.

Python dependencies are pinned in `requirements.txt`. **PyTorch is installed
separately first**, from the CPU-only wheel index and at exact pinned versions;
otherwise pip resolves `torch` against PyPI and pulls the CUDA build. See `README.md` §2.

---

## 9. Testing

All tests live in `backend/tests/` and run with `pytest`. **Every heavy dependency is
mocked** (BLIP, OCR, Whisper, ChromaDB, pdfplumber, python-pptx, the Ollama HTTP call),
so the suite is fast, deterministic, and needs no GPU, no Ollama, and no real model
downloads. 47 tests.

- **`test_ingest.py`** — unsupported extension raises `ValueError`; missing file raises
  `FileNotFoundError`; PDF chunk counts; long pages split; sliding-window overlap;
  image yields exactly one chunk.
- **`test_retrieve.py`** — `query_rag` return structure; session `where`-filter applied
  and omitted; `ask_ollama` assembles streamed tokens; prompt contains notes and
  question; `ConnectionError` propagates.
- **`test_quiz_flashcards.py`** — truncation when the model overproduces; rejection of
  items with duplicate or case-variant options; degenerate items dropped and reported
  in warnings; flashcard deduplication running before truncation.
- **`test_signal_endpoints.py`** — optional fields accept an explicit JSON `null`, a
  real value, or omission; the `session_id` guard returns `400` rather than `422`.
- **`test_blip.py` / `test_ocr.py` / `test_whisper.py` / `test_sentiment.py`** — wrapper
  behaviour with mocked backends, including that weights are not loaded at import.

`conftest.py` at the project root inserts that root into `sys.path` so `import
backend.*` works without an editable install. Tests are isolated from the real debug
logs, so running the suite does not pollute `query_debug.json` or `chunks_debug.json`.

```powershell
pytest backend/tests/ -v
```

---

## 10. Debugging & Observability

- **`chunks_debug.json`** — appended on every ingest; each stored chunk's text and metadata.
- **`query_debug.json`** — appended on every retrieval; question, session, ranked chunks
  with cosine distances, and the generated answer where one exists.
- **`signals.json`** — append-only JSON array of learning signals (see §11).
- **`[TIMER]` markers** — throughout the pipeline: model load, embed, OCR, BLIP,
  Whisper, ChromaDB upsert/query, and Ollama generation with tokens/sec. Useful for
  profiling on CPU.
- **`[QUIZ RAW]` / `[FLASHCARD RAW]` blocks** — the model's unparsed output, logged in
  full so a parse failure can be diagnosed after the fact. These print through
  `_print_raw()`, which replaces characters the output stream cannot encode rather
  than raising: when stdout is redirected to a file it falls back to the locale
  encoding, and a `UnicodeEncodeError` from a debug print would otherwise fail a
  request whose generation had already succeeded.

---

## 11. Key Design Decisions

- **Fully local and private** — no API keys; all inference (embeddings, STT,
  captioning, OCR, LLM) runs on-device.
- **Lazy model loading with caching** — fast startup; the first inference pays the load
  cost; weights persist on disk.
- **Cosine similarity** for text embeddings, which suits sentence-transformer vectors
  better than L2.
- **Module isolation** — the collection is partitioned by `session_id`, and an empty
  one is rejected rather than treated as "search everything".
- **Path anchoring** — relative paths resolve against the project root, not the working
  directory, so the store cannot silently fork in two (§4.5).
- **Grounded prompting** — the LLM is told to answer *only* from the retrieved notes,
  reducing hallucination and keeping answers traceable to source files.
- **Multimodal fallbacks** — image-heavy PDF pages, embedded slide images, SmartArt and
  table XML, and standalone images are recovered via OCR and captioning rather than
  being silently dropped.
- **Client-side marking** — quiz correctness and flashcard "known" are decided in the
  UI and never re-derived from the LLM; the backend only logs the outcome.
- **Signal types are never averaged.** `/confidence` returns quiz, flashcard and
  sentiment aggregates separately, because they are not equally trustworthy: quiz
  results are measured, flashcard ratings are self-reported, and sentiment is inferred
  by a model trained on a different domain. Combining them would produce a number with
  no defensible meaning.
- **Signal logging is always best-effort.** The student-facing result is produced
  before logging runs, so a logging failure degrades to a printed warning and never
  fails the request.

---

## 12. Current Status & Notes

- The backend (9 endpoints), the full ingest pipeline (4 file types), RAG retrieval and
  generation, the four-tab Streamlit UI, and the learning-signal tracker are all
  **implemented and unit-tested** (47 tests).
- `sentiment_model.py` **is** wired into the query flow: `/query` classifies the
  question's tone and logs it as a signal, which `/confidence` then aggregates. It runs
  on CPU (`device=-1`).
- **Known limitation — quiz key quality is bounded by ingest quality.** Quiz items
  generated from clean text are reliably keyed; items generated from OCR-degraded
  sources (scanned slides, screenshots embedded in decks) are frequently mis-keyed or
  ambiguous, because the generator inherits whatever the OCR produced. Nothing
  downstream can detect this — marking is faithful to the key, and it is the key that
  is wrong. The mitigation is transparency rather than correction: the source chunk is
  displayed next to every marked question so a student who disagrees can check the
  material. See `verification/BUCKET_B_FINDINGS.md`.
- **Known limitation — no conversation history.** Each query is independent. Only the
  retrieved chunks are sent to the model, never prior turns, so a follow-up such as
  "explain that again" has no antecedent.
- `requirements.txt` lists `sqlalchemy` for possible future session/metadata tracking;
  it is not currently used.
- `jiwer` is present only for `verification/eval_whisper.py`.
- `test_pipeline.ipynb` and `make_figures.ipynb` at the repo root are for manual
  exploration and report-figure generation respectively; neither is imported by the app.
