# Multimodal Study Assistant — Project Documentation

> A fully-local, multimodal Retrieval-Augmented Generation (RAG) system that ingests
> lecture materials (PDF, PPTX, audio, images), embeds them into a vector store, and
> answers questions using a local LLM. **No external API keys are required** — every
> model runs on the user's own machine.

---

## 1. What This Project Does

The system lets a student turn their own study material into a searchable, question-answerable
knowledge base. The workflow is:

1. **Upload** a file (PDF notes, PowerPoint slides, an audio recording of a lecture, or an image/diagram).
2. The system **extracts text** from it — using OCR, speech-to-text, and image captioning where needed.
3. The text is **chunked, embedded, and stored** in a local vector database (ChromaDB).
4. The student **asks questions**. The system retrieves the most relevant chunks and feeds them
   to a local LLM (via Ollama), which answers **using only the student's own notes** — preventing
   the model from inventing facts beyond the source material.

Everything is local: no cloud, no API keys, no data leaving the machine.

---

## 2. High-Level Architecture

```
┌──────────────┐     HTTP      ┌────────────────────────┐
│  Streamlit   │ ◀──────────▶  │      FastAPI backend   │
│  frontend    │  /ingest      │   (backend/main.py)    │
│ (app.py)     │  /query       │                        │
└──────────────┘  /health      └───────────┬────────────┘
                                            │
                        ┌───────────────────┼────────────────────┐
                        ▼                                          ▼
              ┌──────────────────┐                     ┌────────────────────┐
              │  ingest.py       │                     │   retrieve.py      │
              │  (file → chunks) │                     │  (query → answer)  │
              └────────┬─────────┘                     └─────────┬──────────┘
                       │                                         │
       ┌───────────────┼──────────────┐                         │
       ▼      ▼        ▼      ▼        ▼                         ▼
   Whisper  BLIP     OCR   Embedder  ChromaDB  ◀──────────▶  ChromaDB + Ollama
   (audio) (image) (image) (text)   (vectors)               (retrieval + LLM)
```

**Two distinct pipelines** share the same ChromaDB collection (`study_materials`):

- **Ingest pipeline** (`ingest.py`) — converts a file into embedded chunks and stores them.
- **Query pipeline** (`retrieve.py`) — embeds a question, retrieves similar chunks, and asks the LLM.

---

## 3. Directory Layout

```
multimodal-study-assistant/
├── backend/
│   ├── main.py                  # FastAPI app — 3 endpoints (/ingest, /query, /health)
│   ├── ingest.py                # File ingestion pipeline (PDF/PPTX/audio/image → ChromaDB)
│   ├── retrieve.py              # RAG retrieval + Ollama LLM call
│   ├── models/
│   │   ├── embedder.py          # sentence-transformers all-MiniLM-L6-v2 (384-dim)
│   │   ├── whisper_model.py     # openai-whisper speech-to-text
│   │   ├── blip_model.py        # BLIP image captioning
│   │   ├── ocr_model.py         # pytesseract OCR
│   │   └── sentiment_model.py   # DistilBERT sentiment (auxiliary / experimental)
│   └── tests/                   # pytest unit tests (all models mocked)
│       ├── test_ingest.py
│       ├── test_retrieve.py
│       ├── test_blip.py
│       ├── test_ocr.py
│       ├── test_whisper.py
│       └── test_sentiment.py
├── frontend/
│   └── app.py                   # Streamlit chat UI calling the FastAPI backend
├── sample_files/                # Sample PDFs, PPTX decks, and an MP3 for testing
├── conftest.py                  # Adds project root to sys.path so `backend.*` imports resolve
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Configuration template
├── README.md                    # Setup & usage instructions
└── DOCUMENTATION.md             # (this file)
```

---

## 4. The Backend in Detail

### 4.1 `backend/main.py` — FastAPI Application

The HTTP entry point. Exposes three endpoints:

| Method & Path | Purpose | Key behaviour |
|---|---|---|
| `GET /health` | Liveness check | Returns `{"status": "ok", "ollama": <bool>}`; pings the Ollama server to confirm it's reachable. |
| `POST /ingest` | Upload & ingest a file | Accepts a multipart file + `session_id` form field. Saves to a temp file, calls `ingest_file()`, then deletes the temp file. Returns `{filename, session_id, chunks_stored}`. |
| `POST /query` | Ask a question | Accepts JSON `{question, session_id, n_results}`. Retrieves chunks via `query_rag()`, then generates an answer via `ask_ollama()`. Returns `{answer, sources}`. |

**Error handling:**
- Unsupported file type → `415 Unsupported Media Type`
- Other ingest failures → `500`
- No relevant chunks found → `404` ("ingest some files first")
- Ollama not running → `503` ("Start it with: ollama serve")

The temp file is always cleaned up in a `finally` block, even when ingestion raises.

### 4.2 `backend/ingest.py` — Ingestion Pipeline

The heart of the multimodal handling. Detects file type by extension and routes to a specialised
extractor. All extractors return `(chunks, metadatas)` which are then embedded and upserted.

**Module-level singletons** (lazy-loaded models) are created once and reused:
`_whisper`, `_blip`, `_ocr`, `_embedder`, plus the persistent ChromaDB `_collection`
(configured with cosine similarity: `metadata={"hnsw:space": "cosine"}`).

**Per-file-type extraction:**

- **PDF — `_process_pdf()`**
  - Opens with `pdfplumber`, iterates pages.
  - Runs `_fix_doubled_chars()` to repair the "EEvveerr"-style doubled-glyph artifact common
    in scanned/photocopied PDFs (detected by measuring the ratio of consecutive identical char pairs).
  - **Vision fallback:** if a page yields < 50 characters of text, it is rendered to an image
    (`pdf2image` at 150 DPI) and processed with **BLIP captioning + OCR**.
  - Splits text via `_split_text_for_pdf()` (prefers paragraph boundaries, ~220 words/chunk,
    40-word overlap, falling back to a word window for oversized paragraphs).
  - One chunk's metadata records `source_file`, `source_type="pdf"`, `page_or_slide`, `chunk_index`, `session_id`.

- **PPTX — `_process_pptx()`**
  - One chunk per slide, combining: title + body text + speaker notes + embedded-image text.
  - `_collect_shapes()` recursively walks shapes, handling:
    - **Group shapes** (recurse into children)
    - **Picture shapes** → OCR (and BLIP caption if OCR yields little)
    - **Text frames** → text directly
    - **SmartArt / charts / tables** (GraphicFrames with no text frame) → scrapes `<a:t>` runs
      straight from the underlying DrawingML XML, so diagram/table text isn't lost.

- **Audio — `_process_audio()`**
  - Transcribes with **Whisper**, then chunks the transcript with a 200-word sliding window
    (50-word overlap). `page_or_slide` is `None`.

- **Image — `_process_image()`**
  - Single chunk combining **BLIP caption + OCR text** for a standalone image file.

**Chunking helpers:**
- `_sliding_window(words, window=200, overlap=50)` — generic overlapping word windows.
- `_split_text_for_pdf(...)` — paragraph-aware splitter for PDF text.

**Storage — `_store_chunks()`:**
- Appends a debug record to `chunks_debug.json` (timestamp + every chunk's text & metadata).
- Embeds all chunks via `_embedder.embed()`.
- Generates UUID ids and `upsert`s documents + embeddings + metadata into ChromaDB.
- Returns the number of chunks stored.

**Public API — `ingest_file(file_path, metadata)`:**
- Validates the file exists (`FileNotFoundError`) and the extension is supported (`ValueError`).
- Routes to the correct extractor, stores chunks, returns the count.
- Supported extensions: `.pdf`, `.pptx`, `.mp3`, `.wav`, `.png`, `.jpg`, `.jpeg`.

### 4.3 `backend/retrieve.py` — RAG Retrieval & LLM

Two public functions, sharing the same `_embedder` and ChromaDB `_collection` as ingest.

- **`query_rag(question, session_id=None, n_results=5)`**
  - Embeds the question with the *same* sentence-transformer used at ingest time (critical for
    vector compatibility).
  - Queries ChromaDB by cosine distance, optionally filtered to a single `session_id`
    (via a `where` filter).
  - Appends a debug record to `query_debug.json` (question, session, ranked retrieved chunks with distances).
  - Returns `{chunks, metadatas, distances}`.

- **`ask_ollama(question, context_chunks, model=None)`**
  - Joins retrieved chunks into a notes block and builds a strict RAG prompt:
    *"Using ONLY the following notes … Do not use knowledge outside these notes."* — this grounds
    the answer and reduces hallucination.
  - Calls the Ollama REST API (`POST /api/generate`) with **streaming enabled**, assembling tokens
    as they arrive.
  - Returns the assembled answer string. Raises `ConnectionError` if Ollama is unreachable (the
    endpoint translates this to a `503`).

### 4.4 `backend/models/` — Lazy-Loaded Model Wrappers

Every wrapper follows the same pattern: **the model is not loaded at import time**, only on first
use (`_load()`), keeping startup fast. Weights are cached to `./models_cache/` (configurable) so
they survive venv recreation.

| Wrapper | Backing model | Notes |
|---|---|---|
| `embedder.py` | `all-MiniLM-L6-v2` | 384-dim text embeddings; accepts a string or list, always returns a list of vectors. |
| `whisper_model.py` | `openai-whisper` | Size from `WHISPER_MODEL_SIZE` env (default `base`). `fp16=False` for CPU. |
| `blip_model.py` | `Salesforce/blip-image-captioning-base` | Generates a caption (max 50 new tokens). Accepts PIL image or path. |
| `ocr_model.py` | `pytesseract` (Tesseract binary) | No weights to download; relies on the OS Tesseract install. Optional custom binary path. |
| `sentiment_model.py` | `distilbert-base-uncased-finetuned-sst-2-english` | Auxiliary/experimental — flags emotionally charged passages. Returns `{label, score}`. **Note:** currently hardcoded to `device=0` (GPU); set to `-1` for CPU-only machines. |

---

## 5. The Frontend — `frontend/app.py`

A **Streamlit** chat application that talks to the FastAPI backend at `http://localhost:8000`.

**Sidebar:**
- Live **health indicators** for the API and Ollama (with actionable error messages if either is down).
- **Session ID** input — each study topic can have its own isolated namespace.
- **File uploader** (pdf/pptx/mp3/wav/png/jpg/jpeg) with an "Ingest File" button and progress spinner.
- A running list of ingested files (with chunk counts) and a "Clear Chat" button.

**Main area:**
- A chat interface (`st.chat_message` / `st.chat_input`) preserving history in `st.session_state`.
- Each assistant answer includes an expandable **Sources** panel listing the source file, type,
  and page/slide for every retrieved chunk — giving traceability back to the original material.
- Robust error handling for HTTP errors, timeouts, and unexpected exceptions.

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
  → ask_ollama(): build grounded prompt → Ollama /api/generate (streamed)
  → answer + source metadata returned to the UI
```

---

## 7. Configuration

Configured via a `.env` file (copy from `.env.example`), loaded with `python-dotenv`:

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | Whisper size: `tiny`/`base`/`small`/`medium`/`large` |
| `OLLAMA_MODEL` | `llama3` | Local LLM tag (must be `ollama pull`ed) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama REST endpoint |
| `CHROMA_PATH` | `./vectorstore/chroma_db` | Persistent vector store location |
| `MODELS_CACHE` | `./models_cache` | HuggingFace / sentence-transformers weight cache |

---

## 8. System Dependencies (installed at OS level, not via pip)

- **Ollama** — local LLM server on `localhost:11434` (run `ollama pull llama3`).
- **Tesseract OCR** — binary required by `pytesseract`.
- **Poppler** — required by `pdf2image` to render PDF pages.

Python dependencies are pinned in `requirements.txt`. **PyTorch is installed separately first**
using the CPU-only wheel index, to avoid downloading the ~2 GB CUDA build.

---

## 9. Testing

All tests live in `backend/tests/` and run with `pytest`. **Every heavy dependency is mocked**
(BLIP, OCR, Whisper, ChromaDB, pdfplumber, python-pptx, the Ollama HTTP call), so the suite is
fast, deterministic, and needs no GPU, no Ollama, and no real model downloads.

Coverage highlights:

- **`test_ingest.py`** — unsupported extension raises `ValueError`; missing file raises
  `FileNotFoundError`; PDF returns correct chunk count; long pages split into multiple chunks;
  `_sliding_window` overlap correctness; image yields exactly one chunk.
- **`test_retrieve.py`** — `query_rag` return structure; session `where`-filter applied / omitted;
  `ask_ollama` assembles streamed tokens; prompt contains both notes and question; `ConnectionError`
  propagates when Ollama is down.
- **`test_blip.py`, `test_ocr.py`, `test_whisper.py`, `test_sentiment.py`** — model wrapper behaviour
  with mocked backends.

`conftest.py` inserts the project root into `sys.path` so `import backend.*` works without an
editable install.

Run:
```powershell
pytest backend/tests/ -v
```

---

## 10. Debugging & Observability

- **`chunks_debug.json`** — written on every ingest; records each stored chunk's text and metadata.
- **`query_debug.json`** — written on every query; records the question, session, and ranked
  retrieved chunks with their cosine distances.
- **`[TIMER]` log markers** — throughout the pipeline (model load, embed, OCR, BLIP, Whisper,
  ChromaDB upsert/query, Ollama generation with tokens/sec). Useful for profiling on CPU; left in place.

---

## 11. Key Design Decisions

- **Fully local & private** — no API keys; all inference (embeddings, STT, captioning, OCR, LLM) runs on-device.
- **Lazy model loading + caching** — fast startup; first inference pays the load cost; weights cached to disk.
- **Cosine similarity** for text embeddings (better than L2 for sentence-transformer vectors).
- **Session isolation** — ChromaDB collection is filtered by `session_id`, so each study topic
  searches only its own material.
- **Grounded prompting** — the LLM is explicitly told to answer *only* from the retrieved notes,
  reducing hallucination and keeping answers traceable to source files.
- **Multimodal fallbacks** — image-heavy PDF pages, embedded slide images, SmartArt/table XML, and
  standalone images are all recovered via OCR + captioning rather than being silently dropped.

---

## 12. Current Status & Notes

- Backend (3 endpoints), full ingest pipeline (4 file types), RAG retrieval + Ollama, and the
  Streamlit UI are all **implemented and unit-tested**.
- `sentiment_model.py` is present as an **auxiliary/experimental** component and is not yet wired
  into the ingest or query flow. Note its `device=0` (GPU) default — change to `device=-1` to run on CPU.
- `requirements.txt` lists `sqlalchemy` for *future* session/metadata tracking; it is not yet used.
- A `test_pipeline.ipynb` notebook exists at the repo root for manual/exploratory testing.
```
