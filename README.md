# Multimodal Study Assistant — Local RAG Pipeline

A fully local, multimodal Retrieval-Augmented Generation (RAG) system for
studying lecture notes, slides, and audio recordings.  
**No API keys required.** Everything runs on your machine.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Use `python --version` to check |
| [Ollama](https://ollama.com) | latest | Acts as the local LLM server |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | 5.x | Required for OCR on images |
| [Poppler](https://poppler.freedesktop.org/) | latest | Required by `pdf2image` to render PDF pages |
| Git | any | For cloning |

### Installing Tesseract

**Windows:**
Download the installer from https://github.com/UB-Mannheim/tesseract/wiki  
After install, add `C:\Program Files\Tesseract-OCR` to your `PATH`.

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

### Installing Poppler (for pdf2image)

**Windows:** Download from https://github.com/oschwartz10612/poppler-windows/releases  
Add the `bin/` folder to your `PATH`.

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

---

## 2. Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd multimodal-study-assistant

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install CPU-only PyTorch first (avoids downloading a 2 GB CUDA build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Copy the environment template and edit if needed
cp .env.example .env

# 6. Pull the LLM model into Ollama (downloads ~4 GB on first run)
ollama pull llama3
```

---

## 3. Running the API Server

```bash
# From the project root (not the backend/ folder)
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## 4. Running the Tests

```bash
# Run all unit tests (no Ollama, no real model weights needed)
pytest backend/tests/ -v
```

All tests mock out the heavy models and ChromaDB, so they are fast and
can run offline.

---

## 5. Using the API — curl examples

### Check server health
```bash
curl http://localhost:8000/health
# {"status":"ok","ollama":true}
```

### Ingest a file
```bash
# Ingest a PDF into session "lecture_week1"
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_files/test_notes.pdf" \
  -F "session_id=lecture_week1"

# Expected response:
# {"filename":"test_notes.pdf","session_id":"lecture_week1","chunks_stored":12}
```

### Ask a question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is backpropagation?", "session_id": "lecture_week1"}'

# Expected response:
# {
#   "answer": "Backpropagation is an algorithm for training neural networks ...",
#   "sources": [
#     {"source_file": "test_notes.pdf", "source_type": "pdf", "page_or_slide": 3, ...}
#   ]
# }
```

### Ingest a PPTX slide deck
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_files/test_slides.pptx" \
  -F "session_id=lecture_week1"
```

### Ingest an audio recording
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_files/test_audio.mp3" \
  -F "session_id=lecture_week1"
```

---

## 6. Expected Disk Usage

| Component | Download size | Notes |
|---|---|---|
| Whisper `base` | ~74 MB | Cached in `~/.cache/whisper/` |
| BLIP captioning | ~450 MB | Cached in `./models_cache/` |
| Sentiment (DistilBERT) | ~260 MB | Cached in `./models_cache/` |
| Sentence-transformers `all-MiniLM-L6-v2` | ~90 MB | Cached in `./models_cache/` |
| Llama 3 (via Ollama) | ~4.7 GB | Stored by Ollama (`~/.ollama/`) |
| **Total (first run)** | **~5.7 GB** | Subsequent runs use cache |

---

## Project Structure

```
multimodal-study-assistant/
├── backend/
│   ├── main.py              # FastAPI app (3 endpoints)
│   ├── ingest.py            # File ingestion pipeline
│   ├── retrieve.py          # RAG retrieval + Ollama query
│   ├── models/
│   │   ├── whisper_model.py # Whisper STT wrapper
│   │   ├── blip_model.py    # BLIP image captioning wrapper
│   │   ├── ocr_model.py     # pytesseract OCR wrapper
│   │   ├── sentiment_model.py # HuggingFace sentiment wrapper
│   │   └── embedder.py      # sentence-transformers wrapper
│   ├── vectorstore/         # ChromaDB persists here (auto-created)
│   └── tests/               # pytest unit tests (all mocked)
├── sample_files/            # Drop your test files here
├── models_cache/            # HuggingFace weight cache
├── requirements.txt
├── .env.example
└── README.md
```
