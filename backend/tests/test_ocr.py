"""
Unit tests for the OCRModel wrapper.

Mocks pytesseract so no Tesseract binary is required to run these tests.
"""

from unittest.mock import MagicMock, patch
from PIL import Image


def _make_tiny_image():
    return Image.new("RGB", (10, 10), color=(200, 200, 200))


def test_extract_text_returns_string():
    """OCRModel.extract_text() should return the OCR string from pytesseract."""
    with patch("backend.models.ocr_model.pytesseract") as mock_tess:
        mock_tess.image_to_string.return_value = "  Hello OCR  "

        from backend.models.ocr_model import OCRModel

        ocr = OCRModel()
        result = ocr.extract_text(_make_tiny_image())

        assert result == "Hello OCR"
        mock_tess.image_to_string.assert_called_once()


def test_extract_text_accepts_file_path(tmp_path):
    """OCRModel.extract_text() should open a file path and run OCR on it."""
    img_path = str(tmp_path / "test.png")
    _make_tiny_image().save(img_path)

    with patch("backend.models.ocr_model.pytesseract") as mock_tess:
        mock_tess.image_to_string.return_value = "file path test"

        from backend.models.ocr_model import OCRModel

        ocr = OCRModel()
        result = ocr.extract_text(img_path)

        assert result == "file path test"


def test_empty_image_returns_empty_string():
    """A blank image should return an empty string after stripping."""
    with patch("backend.models.ocr_model.pytesseract") as mock_tess:
        mock_tess.image_to_string.return_value = "   \n\n   "

        from backend.models.ocr_model import OCRModel

        ocr = OCRModel()
        result = ocr.extract_text(_make_tiny_image())

        assert result == ""
