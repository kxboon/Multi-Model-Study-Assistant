"""Sentiment wrapper over cardiffnlp/twitter-roberta-base-sentiment-latest.

Three labels: negative / neutral / positive, lowercase. The neutral class is why
this replaced the binary distilbert-sst-2 model, which had nowhere to put
affectless text and scored plain questions like "what is lemmatization" NEGATIVE
at >0.99, indistinguishable from real frustration.

Used to flag how a student sounds when asking, logged via backend/signals.py.
"""

from transformers import pipeline

from backend.paths import resolve_path


class SentimentModel:
    """Lazy-loading sentiment classifier, cached to ./models_cache/."""

    MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self):
        self._pipe = None
        self._cache_dir = resolve_path("MODELS_CACHE", "./models_cache")

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
        """Return {"label": negative|neutral|positive, "score": float}."""
        self._load()

        # Truncate before tokenisation; the pipeline returns a one-item list.
        return self._pipe(text[:512])[0]
