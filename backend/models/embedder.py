"""
Sentence-transformers embedding wrapper.
Uses all-MiniLM-L6-v2 — a small but high-quality model that produces
384-dimensional embeddings and runs fast on CPU.
"""

import os
import time
from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazy-loading text embedder backed by sentence-transformers.

    Weights are cached to ./models_cache/ so subsequent runs skip the
    download step even if the venv is recreated.
    """

    MODEL_ID = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._cache_dir = os.getenv("MODELS_CACHE", "./models_cache")

    def _load(self):
        """Download (once) and load the sentence-transformer model."""
        if self._model is None:
            t0 = time.perf_counter()
            self._model = SentenceTransformer(
                self.MODEL_ID, cache_folder=self._cache_dir
            )
            print(f"[TIMER] Embedder model load: {time.perf_counter() - t0:.2f}s")

    def embed(self, texts: "list[str] | str") -> list:
        """Convert one or more text strings into embedding vectors.

        Args:
            texts: A single string or a list of strings.

        Returns:
            A list of embedding vectors (each a list of floats).
            If a single string is passed, returns a list containing one vector.
        """
        self._load()

        # Normalise input so callers can pass either a string or a list
        if isinstance(texts, str):
            texts = [texts]

        t0 = time.perf_counter()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        print(f"[TIMER] Embedder encode ({len(texts)} chunk(s)): {time.perf_counter() - t0:.2f}s")
        return embeddings.tolist()
