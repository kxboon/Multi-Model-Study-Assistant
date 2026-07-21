"""
BLIP image-captioning wrapper.
Uses Salesforce/blip-image-captioning-base via HuggingFace Transformers.
Runs on CPU by default so no GPU is required for the FYP experiments.
"""

import os
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class BLIPModel:
    """Lazy-loading wrapper around BLIP image captioning.

    The processor and model weights are only downloaded / loaded into RAM
    the first time caption() is called, keeping imports instant.
    """

    MODEL_ID = "Salesforce/blip-image-captioning-base"

    def __init__(self):
        self._processor = None
        self._model = None
        # Cache directory keeps weights local so they survive re-installs
        self._cache_dir = os.getenv("MODELS_CACHE", "./models_cache")

    def _load(self):
        """Download (once) and load the BLIP processor + model weights."""
        if self._model is None:
            t0 = time.perf_counter()
            self._processor = BlipProcessor.from_pretrained(
                self.MODEL_ID, cache_dir=self._cache_dir
            )
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.MODEL_ID, cache_dir=self._cache_dir
            )
            self._model.eval()
            print(f"[TIMER] BLIP model load: {time.perf_counter() - t0:.2f}s")

    def caption(self, image: "Image.Image | str") -> str:
        """Generate a natural-language caption for an image.

        Args:
            image: A PIL Image object or a file path string.

        Returns:
            A caption string such as "a diagram showing a neural network".
        """
        self._load()

        # Accept both PIL images and file paths for flexibility
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        t0 = time.perf_counter()
        output_ids = self._model.generate(**inputs, max_new_tokens=50)
        caption = self._processor.decode(output_ids[0], skip_special_tokens=True)
        print(f"[TIMER] BLIP caption: {time.perf_counter() - t0:.2f}s")
        return caption.strip()
