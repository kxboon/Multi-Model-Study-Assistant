"""
Sentiment analysis wrapper using HuggingFace Transformers pipeline.
Uses distilbert-base-uncased-finetuned-sst-2-english by default —
a lightweight model that runs well on CPU.

In the study assistant context, sentiment can flag emotionally charged
passages in lecture notes (e.g. strong warnings, key insights).
"""

import os
from transformers import pipeline


class SentimentModel:
    """Lazy-loading sentiment classifier.

    The pipeline (tokeniser + weights) is downloaded once and cached
    to ./models_cache/ to avoid repeated downloads across sessions.
    """

    MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        self._pipe = None
        self._cache_dir = os.getenv("MODELS_CACHE", "./models_cache")

    def _load(self):
        """Initialise the HuggingFace text-classification pipeline."""
        if self._pipe is None:
            self._pipe = pipeline(
                "text-classification",
                model=self.MODEL_ID,
                model_kwargs={"cache_dir": self._cache_dir},
                # device=-1 forces CPU; 0 would use the first CUDA GPU
                device=0,
            )

    def predict(self, text: str) -> dict:
        """Return sentiment label and confidence score for a text snippet.

        Args:
            text: The input string (truncated internally to 512 tokens).

        Returns:
            {"label": "POSITIVE" | "NEGATIVE", "score": float}
        """
        self._load()

        # Truncate to 512 chars as a safety measure before tokenisation
        result = self._pipe(text[:512])
        # pipeline returns a list; we always pass one item so take index 0
        return result[0]
