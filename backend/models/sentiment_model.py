"""
Sentiment analysis wrapper using HuggingFace Transformers pipeline.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest, which returns three
labels — negative / neutral / positive.

The three-way split matters here. The previous model
(distilbert-base-uncased-finetuned-sst-2-english) was binary, so it had
nowhere to put affectless text: plain factual questions like "what is
lemmatization" came back NEGATIVE at >0.99, indistinguishable from genuine
frustration. With a neutral class those questions land on neutral instead,
so a negative label actually means something.

In the study assistant context, sentiment flags how a student sounds when
asking a question, which is logged as a learning signal (see backend/signals.py).

NOTE: this model's labels are lowercase ("negative", not "NEGATIVE").
"""

import os
from transformers import pipeline


class SentimentModel:
    """Lazy-loading sentiment classifier.

    The pipeline (tokeniser + weights) is downloaded once and cached
    to ./models_cache/ to avoid repeated downloads across sessions.
    """

    MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

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
                device=-1,
            )

    def predict(self, text: str) -> dict:
        """Return sentiment label and confidence score for a text snippet.

        Args:
            text: The input string (truncated internally to 512 tokens).

        Returns:
            {"label": "negative" | "neutral" | "positive", "score": float}
        """
        self._load()

        # Truncate to 512 chars as a safety measure before tokenisation
        result = self._pipe(text[:512])
        # pipeline returns a list; we always pass one item so take index 0
        return result[0]
