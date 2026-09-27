# Multimodal Study Assistant — Local RAG Pipeline

A fully local, multimodal Retrieval-Augmented Generation (RAG) system for studying
lecture material — PDFs, PowerPoint decks, audio recordings, and images.
**No API keys required.** Every model runs on your own machine.

Ask questions about your own notes, generate quizzes and flashcards from them, and
track which topics you are struggling with.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | `python --version` to check |
| [Ollama](https://ollama.com) | latest | Local LLM server |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | 5.x | OCR for images and scanned pages |
| [Poppler](https://poppler.freedesktop.org/) | latest | Required by `pdf2image` to render PDF pages |
| Git | any | For cloning |

Tesseract and Poppler must both be on your `PATH`.

### Installing Tesseract

**Windows:** download the installer from https://github.com/UB-Mannheim/tesseract/wiki
then add `C:\Program Files\Tesseract-OCR` to your `PATH`.

```bash
brew install tesseract          # macOS
sudo apt-get install tesseract-ocr   # Ubuntu/Debian
```

### Installing Poppler

**Windows:** download from https://github.com/oschwartz10612/poppler-windows/releases
and add the `bin/` folder to your `PATH`.

```bash
brew install poppler            # macOS
sudo apt-get install poppler-utils   # Ubuntu/Debian
```

---

## 2. Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/kxboon/Multi-Model-Study-Assistant.git
cd Multi-Model-Study-Assistant

# 2. Create and activate a virtual environment
python -m venv venv

venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS / Linux
```

### 3. Install PyTorch first, with these exact versions

PyTorch **must** be installed before `requirements.txt`, from the CPU wheel index,
with these pins. Installing it unpinned, or letting `requirements.txt` pull it in,
makes pip resolve `torch` against PyPI and download the multi-gigabyte CUDA build.

```bash
pip install torch==2.11.0+cpu torchvision==0.26.0+cpu torchaudio==2.11.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install everything else

```bash
pip install -r requirements.txt
```

### 5. Create your config

```bash
cp .env.example .env            # macOS / Linux
copy .env.example .env          # Windows
```

Every value has a working default, so the file runs unedited. See
[section 7](#7-configuration) for what each setting does.

### 6. Pull the LLM

```bash
ollama pull llama3.2
```

Use whatever tag you set as `OLLAMA_MODEL` in `.env`. The default is `llama3.2`
(about 2 GB).

---

## 3. Running the application

Two processes, in two terminals, both started **from the project root**.

```bash
# Terminal 1 — the API
uvicorn backend.main:app --reload

# Terminal 2 — the UI (the API must already be running)
streamlit run frontend/app.py
```

- API: `http://localhost:8000` — interactive docs at `/docs`
- UI: `http://localhost:8501`

Run `uvicorn` from the project root, not from inside `backend/`. Relative paths for
the vector store and the model cache are anchored to the project root by
`backend/paths.py`, but `query_debug.json` still resolves against the working
directory, so a server started elsewhere writes its debug log somewhere else.

Ollama must be running as well (`ollama serve`, or the background service if your
install starts one).

### First run

Model weights download lazily, on first use, not at startup. The first ingest pulls
Whisper, BLIP and the embedder; the first question also pulls the sentiment model,
around 3 GB in total, once. Expect the first ingest and the first question of a fresh
install to be noticeably slow, and everything after that to be fast. Weights are cached
in `models_cache/` and survive recreating the venv.

### Verify the setup

```bash
pytest backend/tests/ -v          # 47 tests, no Ollama or downloads needed
curl http://localhost:8000/health # {"status":"ok","ollama":true}
```

`"ollama": false` means the API is up but Ollama is not reachable, ingest will work,
questions will not.

---

## 4. Running the tests

```bash
pytest backend/tests/ -v
```

47 tests. Every heavy dependency is mocked: BLIP, OCR, Whisper, ChromaDB,
pdfplumber, python-pptx, and the Ollama HTTP call so the suite is fast,
deterministic, and needs no GPU, no Ollama, and no model downloads.

---

## 5. Using the API (curl examples)

### Health

```bash
curl http://localhost:8000/health
# {"status":"ok","ollama":true}
```

### List study modules

```bash
curl http://localhost:8000/sessions
# [{"session_id":"CM3060 NLP","chunks":65}]
```

### Ingest a file

`session_id` is the study module the material belongs to. Every endpoint requires a
non-empty one, a blank value would search or aggregate across every module at once.

`sample_files/` is gitignored and therefore **empty on a fresh clone**, the material
developed against was licensed course content that cannot be redistributed. Put any PDF,
PPTX, MP3/WAV or PNG/JPG of your own there and substitute its name below; a few pages of
lecture notes is enough to try every feature.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_files/test_notes.pdf" \
  -F "session_id=lecture_week1"

# {"filename":"test_notes.pdf","session_id":"lecture_week1","chunks_stored":12}
```

PPTX, audio and image files use the same call:

```bash
curl -X POST http://localhost:8000/ingest -F "file=@slides.pptx"  -F "session_id=lecture_week1"
curl -X POST http://localhost:8000/ingest -F "file=@lecture.mp3"  -F "session_id=lecture_week1"
curl -X POST http://localhost:8000/ingest -F "file=@diagram.png"  -F "session_id=lecture_week1"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is backpropagation?", "session_id": "lecture_week1"}'

# {
#   "answer": "Backpropagation is an algorithm for training neural networks ...",
#   "sources": [
#     {"source_file": "test_notes.pdf", "source_type": "pdf", "page_or_slide": 3}
#   ]
# }
```

### Generate a quiz

```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{"topic": "backpropagation", "session_id": "lecture_week1", "n_questions": 5}'
```

### Generate flashcards

```bash
curl -X POST http://localhost:8000/flashcards \
  -H "Content-Type: application/json" \
  -d '{"topic": "backpropagation", "session_id": "lecture_week1", "n_cards": 5}'
```

### Read a module's learning signals

```bash
curl "http://localhost:8000/confidence?session_id=lecture_week1"
```

Quiz, flashcard and sentiment aggregates are returned **separately and never
averaged into a single score**, they are not equally trustworthy. See
`DOCUMENTATION.md` §11.

---

## 6. Expected disk usage

| Component | Download size | Cached in |
|---|---|---|
| Whisper `base` | ~74 MB | `~/.cache/whisper/` |
| BLIP image captioning | ~450 MB | `./models_cache/` |
| Sentiment (`twitter-roberta-base-sentiment-latest`) | ~500 MB | `./models_cache/` |
| Sentence-transformers `all-MiniLM-L6-v2` | ~90 MB | `./models_cache/` |
| `llama3.2` via Ollama | ~2 GB | `~/.ollama/` |
| **Total, first run** | **~3.1 GB** | Later runs reuse the cache |

---

## 7. Configuration

All settings live in `.env` (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `OLLAMA_MODEL` | `llama3.2` | Model tag Ollama has pulled |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Use `127.0.0.1`, not `localhost` — see `.env.example` |
| `CHROMA_PATH` | `./vectorstore/chroma_db` | Vector store location |
| `MODELS_CACHE` | `./models_cache` | HuggingFace weight cache |
| `SIGNALS_PATH` | `./signals.json` | Learning-signal log |

---

## 8. Project structure

```
Multi-Model-Study-Assistant/
├── backend/
│   ├── main.py              # FastAPI app — 9 endpoints
│   ├── ingest.py            # File → chunks → embeddings → ChromaDB
│   ├── retrieve.py          # Question → retrieval → grounded prompt → Ollama
│   ├── signals.py           # Learning-signal log (quiz / flashcard / sentiment)
│   ├── paths.py             # Anchors relative paths to the project root
│   ├── models/
│   │   ├── embedder.py          # sentence-transformers all-MiniLM-L6-v2
│   │   ├── whisper_model.py     # Whisper speech-to-text
│   │   ├── blip_model.py        # BLIP image captioning
│   │   ├── ocr_model.py         # pytesseract OCR
│   │   └── sentiment_model.py   # RoBERTa question-tone classifier
│   └── tests/               # 47 pytest unit tests, all mocked
├── frontend/
│   └── app.py               # Streamlit UI — Chat, Quiz, Flashcards, Progress
├── verification/            # Standalone evaluation scripts and their findings
├── sample_files/            # Your own material to ingest (gitignored, empty on clone)
├── vectorstore/             # ChromaDB persists here (auto-created, gitignored)
├── models_cache/            # HuggingFace weight cache (gitignored)
├── clear_db.py              # Reset the vector store
├── run_usertest.ps1         # Launches the full stack + an ngrok tunnel (user study)
├── test_pipeline.ipynb      # Scratch notebook for manual exploration
├── conftest.py              # Puts the project root on sys.path for tests
├── requirements.txt
├── .env.example
├── README.md
└── DOCUMENTATION.md         # Architecture and design decisions
```

For how any of it works, see `DOCUMENTATION.md`.

---

## 9. Evaluation scripts

`verification/` holds the standalone harnesses behind the findings documents
(retrieval recall, LLM answer quality, Whisper model comparison, module isolation).
They are not imported by the app.

Run them as modules from the repository root, not as file paths:

```bash
python -m verification.verify_query
python -m verification.verify_isolation
```

The `verify_*` scripts import `backend.*` and have no path shim, so
`python verification/verify_query.py` fails with `ModuleNotFoundError: No module
named 'backend'`. The `eval_*` scripts set `sys.path` themselves and run either way.

Most need Ollama running and material already ingested.

---

## 10. Resetting

```bash
python clear_db.py            # empty the active collection, keep the directory
python clear_db.py --all      # empty every study_materials* collection
python clear_db.py --purge    # delete the persisted store from disk entirely
```

An evaluation run under a different embedding model writes to its own suffixed
collection, so the default clears only the active one and `--all` clears those too.
All three are irreversible; re-ingest your files afterwards. The debug logs
(`chunks_debug.json`, `query_debug.json`, `signals.json`) are written at the project
root, are gitignored, and are safe to delete, they regenerate on the next ingest or
question.
