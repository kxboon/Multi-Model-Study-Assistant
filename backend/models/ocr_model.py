"""
OCR wrapper using pytesseract (Tesseract engine).
Tesseract must be installed on the OS — see README for instructions.
"""

import time
import pytesseract
from PIL import Image


class OCRModel:
    """Thin wrapper around pytesseract for extracting text from images.

    No model weights to download — Tesseract is a rule-based OCR engine
    that ships as a system binary. Lazy init still guards against the
    import-time cost of PIL operations.
    """

    def __init__(self, tesseract_cmd: str = None):
        # Override the tesseract binary path if it's not on PATH
        # e.g. on Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image: "Image.Image | str") -> str:
        """Run OCR on an image and return the extracted text.

        Args:
            image: A PIL Image or a file path string.

        Returns:
            The raw OCR string (may contain newlines and artefacts).
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        t0 = time.perf_counter()
        text = pytesseract.image_to_string(image, lang="eng")
        print(f"[TIMER] OCR extract_text: {time.perf_counter() - t0:.2f}s")
        return text.strip()
