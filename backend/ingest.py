"""
File ingestion pipeline for the Multimodal Study Assistant.

Supports: PDF, PPTX, MP3/WAV audio, PNG/JPG images.
Each file is broken into semantically meaningful chunks, embedded with
sentence-transformers, then stored in a persistent ChromaDB collection.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import pdfplumber
from pdf2image import convert_from_path
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
import chromadb

from models.whisper_model import WhisperModel
from models.blip_model import BLIPModel
from models.ocr_model import OCRModel
from models.embedder import Embedder

# ---------------------------------------------------------------------------
# Shared model instances — module-level singletons with lazy loading baked in
# ---------------------------------------------------------------------------
_whisper = WhisperModel()
_blip = BLIPModel()
_ocr = OCRModel()
_embedder = Embedder()

# ---------------------------------------------------------------------------
# ChromaDB persistent client — keeps the vector store between restarts
# ---------------------------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PATH", "./vectorstore/chroma_db")
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_or_create_collection(
    name="study_materials",
    # Cosine similarity is better than L2 for text embeddings
    metadata={"hnsw:space": "cosine"},
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sliding_window(words: list, window: int = 200, overlap: int = 50) -> list:
    """Split a word list into overlapping chunks.

    Args:
        words:   Tokenised words (split on whitespace).
        window:  Maximum words per chunk.
        overlap: Words shared between consecutive chunks to preserve context.

    Returns:
        List of chunk strings.
    """
    chunks = []
    step = window - overlap  # how far we advance the window each time
    for start in range(0, len(words), step):
        chunk_words = words[start : start + window]
        chunks.append(" ".join(chunk_words))
        if start + window >= len(words):
            break
    return chunks


def _fix_doubled_chars(text: str) -> str:
    """Remove the doubled-character scanning artifact common in photocopied PDFs.

    When a PDF is produced by scanning a physical book, every glyph can appear
    twice (e.g. "EEvveerr" instead of "Ever").  We detect this by measuring the
    fraction of consecutive identical character pairs and, if the ratio is high
    enough, collapse them.
    """
    import re
    if not text:
        return text
    # Count consecutive identical character pairs (excluding newlines)
    pairs = re.findall(r'(.)\1', text.replace("\n", ""))
    if len(pairs) / max(len(text) - text.count("\n"), 1) > 0.30:
        text = re.sub(r'(.)\1', r'\1', text)
    return text


def _split_text_for_pdf(text: str, max_words: int = 220, overlap: int = 40) -> list:
    """Split PDF text into page-sized chunks with light overlap.

    Prefers paragraph boundaries, then falls back to word windows when a
    paragraph is still too large.
    """
    normalized = "\n".join(line.strip() for line in text.splitlines())
    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n") if paragraph.strip()]
    chunks = []
    current_words = []

    def flush_current() -> None:
        if current_words:
            chunks.append(" ".join(current_words).strip())
            current_words.clear()

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if not paragraph_words:
            continue

        if len(paragraph_words) > max_words:
            flush_current()
            chunks.extend(_sliding_window(paragraph_words, window=max_words, overlap=overlap))
            continue

        if current_words and len(current_words) + len(paragraph_words) > max_words:
            flush_current()

        current_words.extend(paragraph_words)

    flush_current()
    return [chunk for chunk in chunks if chunk.strip()]


def _store_chunks(chunks: list, metadatas: list) -> int:
    """Embed a list of text chunks and upsert them into ChromaDB.

    Args:
        chunks:    List of text strings to embed and store.
        metadatas: Parallel list of metadata dicts for each chunk.

    Returns:
        Number of chunks successfully stored.
    """
    if not chunks:
        return 0

    print(f"[CHUNKS] {len(chunks)} chunk(s) — saving to chunks_debug.json")
    debug_path = Path("chunks_debug.json")
    try:
        existing = json.loads(debug_path.read_text(encoding="utf-8")) if debug_path.exists() else []
    except (json.JSONDecodeError, ValueError):
        existing = []
    existing.append({
        "saved_at": datetime.now().isoformat(),
        "chunks": [
            {"index": i, "metadata": metadatas[i], "text": chunk}
            for i, chunk in enumerate(chunks)
        ],
    })
    debug_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    embeddings = _embedder.embed(chunks)  # timed inside Embedder

    # Re-ingesting a file must REPLACE its previous chunks, not append a second
    # copy. IDs are random uuid4s, so upsert never collides and would duplicate.
    # Delete any existing chunks for this (source_file, session_id) pair first.
    # No-op when there's nothing to delete (first-time ingest).
    identity = {
        "$and": [
            {"source_file": metadatas[0]["source_file"]},
            {"session_id": metadatas[0]["session_id"]},
        ]
    }
    t_del = time.perf_counter()
    _collection.delete(where=identity)
    print(f"[TIMER] ChromaDB delete (prior chunks for this file+session): {time.perf_counter() - t_del:.2f}s")

    ids = [str(uuid.uuid4()) for _ in chunks]

    t0 = time.perf_counter()
    _collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[TIMER] ChromaDB upsert ({len(chunks)} chunks): {time.perf_counter() - t0:.2f}s")
    return len(chunks)


# ---------------------------------------------------------------------------
# File-type-specific extraction functions
# ---------------------------------------------------------------------------

def _process_pdf(file_path: str, filename: str, session_id: str) -> tuple:
    """Extract text from a PDF, falling back to BLIP+OCR for image-heavy pages.

    Returns:
        (chunks, metadatas) lists ready for _store_chunks().
    """
    t_pdf = time.perf_counter()
    chunks, metadatas = [], []
    vision_pages = 0
    total_pages = 0

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            total_pages += 1
            t_page = time.perf_counter()
            text = page.extract_text() or ""

            text = _fix_doubled_chars(text)

            if len(text.strip()) < 50:
                vision_pages += 1
                t_conv = time.perf_counter()
                images = convert_from_path(
                    file_path,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    dpi=150,
                )
                print(f"[TIMER]   PDF page {page_num+1} pdf2image convert: {time.perf_counter() - t_conv:.2f}s")
                if images:
                    img = images[0]
                    caption = _blip.caption(img)   # timed inside BLIPModel
                    ocr_text = _ocr.extract_text(img)  # timed inside OCRModel
                    text = _fix_doubled_chars(f"[Image caption: {caption}] {ocr_text}")

            page_chunks = _split_text_for_pdf(text)
            for page_chunk in page_chunks:
                if page_chunk.strip():
                    chunks.append(page_chunk.strip())
                    metadatas.append(
                        {
                            "source_file": filename,
                            "source_type": "pdf",
                            "page_or_slide": page_num + 1,
                            "chunk_index": len(chunks) - 1,
                            "session_id": session_id,
                        }
                    )
            print(f"[TIMER]   PDF page {page_num+1} total: {time.perf_counter() - t_page:.2f}s")

    print(f"[TIMER] _process_pdf total ({total_pages} pages, {vision_pages} vision pages, {len(chunks)} chunks): {time.perf_counter() - t_pdf:.2f}s")
    return chunks, metadatas


_DRAWINGML_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _collect_shapes(shape, title_shape):
    """Recursively yield (kind, payload) from a shape and any grouped children.

    Handles three cases:
    - Group shapes (shape_type 6): recurse into children
    - Picture shapes (shape_type 13): yield for OCR/BLIP processing
    - Everything else with a text_frame: yield the text directly
    - GraphicFrame / SmartArt (no text_frame): scrape <a:t> runs from raw XML
    """
    if shape.shape_type == 6:  # MSO_SHAPE_TYPE.GROUP
        for child in shape.shapes:
            yield from _collect_shapes(child, title_shape)
    elif shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
        try:
            yield ("picture", shape.image.blob)
        except Exception:
            pass
    elif shape.has_text_frame and shape != title_shape:
        text = shape.text_frame.text.strip()
        if text:
            yield ("text", text)
    else:
        # SmartArt, charts, tables, and other GraphicFrame objects don't expose
        # a text_frame, but their text runs are still in the XML as <a:t> elements.
        if hasattr(shape, "_element"):
            runs = [
                el.text
                for el in shape._element.iter(f"{_DRAWINGML_NS}t")
                if el.text and el.text.strip()
            ]
            if runs:
                yield ("text", " ".join(runs))


def _process_pptx(file_path: str, filename: str, session_id: str) -> tuple:
    """Extract text and embedded images from each slide of a PPTX file.

    One chunk = one slide (title + body + notes + image captions).

    Returns:
        (chunks, metadatas) lists ready for _store_chunks().
    """
    t_pptx = time.perf_counter()
    chunks, metadatas = [], []
    prs = Presentation(file_path)

    for slide_num, slide in enumerate(prs.slides, start=1):
        t_slide = time.perf_counter()
        parts = []

        if slide.shapes.title and slide.shapes.title.text:
            parts.append(f"Title: {slide.shapes.title.text.strip()}")

        title_shape = slide.shapes.title
        for kind, payload in (item for s in slide.shapes for item in _collect_shapes(s, title_shape)):
            if kind == "text":
                parts.append(payload)
            elif kind == "picture" and isinstance(payload, bytes):
                try:
                    import io
                    pil_img = Image.open(io.BytesIO(payload)).convert("RGB")
                    ocr_text = _ocr.extract_text(pil_img)
                    if len(ocr_text) >= 20:
                        parts.append(f"[Embedded image text: {ocr_text}]")
                    else:
                        caption = _blip.caption(pil_img)
                        parts.append(f"[Embedded image: {caption}]")
                except Exception:
                    pass

        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                parts.append(f"Speaker notes: {notes_text}")

        combined = "\n".join(parts)
        if combined.strip():
            chunks.append(combined)
            metadatas.append(
                {
                    "source_file": filename,
                    "source_type": "pptx",
                    "page_or_slide": slide_num,
                    "chunk_index": len(chunks) - 1,
                    "session_id": session_id,
                }
            )
        print(f"[TIMER]   PPTX slide {slide_num}: {time.perf_counter() - t_slide:.2f}s")

    print(f"[TIMER] _process_pptx total ({len(prs.slides)} slides, {len(chunks)} chunks): {time.perf_counter() - t_pptx:.2f}s")
    return chunks, metadatas


def _process_audio(file_path: str, filename: str, session_id: str) -> tuple:
    """Transcribe audio and chunk the transcript with a sliding window.

    Returns:
        (chunks, metadatas) lists ready for _store_chunks().
    """
    t_audio = time.perf_counter()
    transcript = _whisper.transcribe(file_path)  # timed inside WhisperModel
    words = transcript.split()

    raw_chunks = _sliding_window(words, window=200, overlap=50)

    chunks, metadatas = [], []
    for i, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip():
            chunks.append(chunk_text)
            metadatas.append(
                {
                    "source_file": filename,
                    "source_type": "audio",
                    # No page/slide concept for audio — omit the key entirely.
                    # ChromaDB rejects None metadata values.
                    "chunk_index": i,
                    "session_id": session_id,
                }
            )

    print(f"[TIMER] _process_audio total ({len(words)} words, {len(chunks)} chunks): {time.perf_counter() - t_audio:.2f}s")
    return chunks, metadatas


def _process_image(file_path: str, filename: str, session_id: str) -> tuple:
    """Run BLIP captioning + OCR on a standalone image file.

    Returns a single chunk combining both sources of text.
    """
    t_img = time.perf_counter()
    img = Image.open(file_path).convert("RGB")
    caption = _blip.caption(img)    # timed inside BLIPModel
    ocr_text = _ocr.extract_text(img)  # timed inside OCRModel

    combined = f"[Image caption: {caption}] {ocr_text}".strip()

    if not combined:
        return [], []

    chunks = [combined]
    metadatas = [
        {
            "source_file": filename,
            "source_type": "image",
            # No page/slide concept for a standalone image — omit the key
            # entirely. ChromaDB rejects None metadata values.
            "chunk_index": 0,
            "session_id": session_id,
        }
    ]
    print(f"[TIMER] _process_image total: {time.perf_counter() - t_img:.2f}s")
    return chunks, metadatas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(file_path: str, metadata: dict) -> int:
    """Ingest a study file into the ChromaDB vector store.

    Detects file type by extension, routes to the appropriate extraction
    pipeline, embeds all resulting chunks, and persists them.

    Args:
        file_path: Absolute or relative path to the file to ingest.
        metadata:  Dict that must include at least ``session_id``.
                   Additional keys are ignored.

    Returns:
        Number of chunks stored in ChromaDB.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If file_path does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Prefer a caller-supplied original filename. HTTP uploads land in a random
    # temp file, so path.name would record e.g. "tmpqzkuxxqr.pptx" — wrong for
    # citations, and it defeats the source_file+session_id dedup on re-upload.
    # Direct callers that pass no source_file fall back to the real path name.
    filename = metadata.get("source_file") or path.name
    session_id = metadata.get("session_id", "default")
    ext = path.suffix.lower()

    t_total = time.perf_counter()
    print(f"\n[TIMER] === ingest_file START: {filename} ===")

    if ext == ".pdf":
        chunks, metadatas = _process_pdf(file_path, filename, session_id)
    elif ext == ".pptx":
        chunks, metadatas = _process_pptx(file_path, filename, session_id)
    elif ext in (".mp3", ".wav"):
        chunks, metadatas = _process_audio(file_path, filename, session_id)
    elif ext in (".png", ".jpg", ".jpeg"):
        chunks, metadatas = _process_image(file_path, filename, session_id)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            "Supported: .pdf, .pptx, .mp3, .wav, .png, .jpg"
        )

    stored = _store_chunks(chunks, metadatas)
    print(f"[TIMER] === ingest_file TOTAL: {time.perf_counter() - t_total:.2f}s ({stored} chunks stored) ===\n")
    return stored
