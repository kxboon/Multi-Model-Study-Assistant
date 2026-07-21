"""
Unit tests for the ingest pipeline.

All heavy dependencies (BLIP, OCR, Whisper, ChromaDB, pdf2image, pdfplumber,
python-pptx) are mocked so the tests are fast and self-contained.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_ingest_mocks():
    """Return a dict of patches needed to isolate ingest.py from real models.

    patch.multiple("backend.ingest", **mocks) expects bare attribute names,
    not the full dotted path.
    """
    return {
        "_whisper": MagicMock(),
        "_blip": MagicMock(),
        "_ocr": MagicMock(),
        "_embedder": MagicMock(),
        "_collection": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ingest_unsupported_extension(tmp_path):
    """ingest_file() should raise ValueError for unsupported file types."""
    fake_file = tmp_path / "notes.docx"
    fake_file.write_text("content")

    # We need to patch the collection to avoid ChromaDB setup issues
    mocks = _make_ingest_mocks()
    with patch.multiple("backend.ingest", **mocks):
        from backend.ingest import ingest_file

        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_file(str(fake_file), metadata={"session_id": "s1"})


def test_ingest_file_not_found():
    """ingest_file() should raise FileNotFoundError for missing files."""
    mocks = _make_ingest_mocks()
    with patch.multiple("backend.ingest", **mocks):
        from backend.ingest import ingest_file

        with pytest.raises(FileNotFoundError):
            ingest_file("/nonexistent/file.pdf", metadata={"session_id": "s1"})


def test_ingest_pdf_returns_chunk_count(tmp_path):
    """ingest_file() on a PDF should return the number of stored chunks."""
    pdf_file = tmp_path / "notes.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake content")  # minimal fake PDF bytes

    mocks = _make_ingest_mocks()

    # Mock pdfplumber to return one page with enough text (>50 chars)
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "A" * 100  # well above the 50-char threshold
    fake_pdf_ctx = MagicMock()
    fake_pdf_ctx.__enter__ = MagicMock(return_value=fake_pdf_ctx)
    fake_pdf_ctx.__exit__ = MagicMock(return_value=False)
    fake_pdf_ctx.pages = [fake_page]

    # Embedder returns a list with one vector per chunk
    mocks["_embedder"].embed.return_value = [[0.1] * 384]

    with (
        patch.multiple("backend.ingest", **mocks),
        patch("backend.ingest.pdfplumber.open", return_value=fake_pdf_ctx),
    ):
        from backend.ingest import ingest_file

        n = ingest_file(str(pdf_file), metadata={"session_id": "session_abc"})

    # We fed 1 page → should get 1 chunk back
    assert n == 1


def test_ingest_pdf_splits_long_pages(tmp_path):
    """Long PDF pages should be split into multiple chunks."""
    pdf_file = tmp_path / "long_notes.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake content")

    mocks = _make_ingest_mocks()

    fake_page = MagicMock()
    fake_page.extract_text.return_value = " ".join([f"word{i}" for i in range(400)])
    fake_pdf_ctx = MagicMock()
    fake_pdf_ctx.__enter__ = MagicMock(return_value=fake_pdf_ctx)
    fake_pdf_ctx.__exit__ = MagicMock(return_value=False)
    fake_pdf_ctx.pages = [fake_page]

    mocks["_embedder"].embed.return_value = [[0.1] * 384, [0.2] * 384]

    with (
        patch.multiple("backend.ingest", **mocks),
        patch("backend.ingest.pdfplumber.open", return_value=fake_pdf_ctx),
    ):
        from backend.ingest import ingest_file

        n = ingest_file(str(pdf_file), metadata={"session_id": "session_long"})

    assert n > 1


def test_sliding_window_chunks():
    """_sliding_window() should produce overlapping text chunks correctly."""
    from backend.ingest import _sliding_window

    words = list(range(300))  # 300 fake "words"
    words = [str(w) for w in words]
    chunks = _sliding_window(words, window=200, overlap=50)

    # First chunk should contain words 0–199
    assert chunks[0].startswith("0")
    # Second chunk overlaps, starting at word 150
    assert chunks[1].startswith("150")
    # All words are accounted for — last chunk covers the tail
    assert chunks[-1].endswith("299")


def test_ingest_image_returns_one_chunk(tmp_path):
    """ingest_file() on a .png image should store exactly one chunk."""
    from PIL import Image

    img_path = tmp_path / "diagram.png"
    Image.new("RGB", (50, 50)).save(str(img_path))

    mocks = _make_ingest_mocks()
    mocks["_blip"].caption.return_value = "a diagram"
    mocks["_ocr"].extract_text.return_value = "some text"
    mocks["_embedder"].embed.return_value = [[0.0] * 384]

    with patch.multiple("backend.ingest", **mocks):
        from backend.ingest import ingest_file

        n = ingest_file(str(img_path), metadata={"session_id": "s2"})

    assert n == 1
