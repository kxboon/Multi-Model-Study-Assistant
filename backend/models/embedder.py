"""Sentence-transformers embedding wrapper.

Defaults to all-MiniLM-L6-v2 (384-dim, fast on CPU). EMBED_MODEL swaps it for an
evaluation run, e.g. all-mpnet-base-v2 (768-dim).

collection_name() lives here because a ChromaDB collection's dimensionality is
pinned by its first write, so the model and the collection are not independent
choices. Both ingest and retrieve derive the collection from here, so a process
cannot mix models and an EMBED_MODEL run gets its own collection.
"""

import os
import re
import time
from sentence_transformers import SentenceTransformer

from backend.paths import resolve_path

# The default model keeps this name, so the swap mechanism never orphans data.
DEFAULT_COLLECTION = "study_materials"


class Embedder:
    """Lazy-loading text embedder backed by sentence-transformers.

    Weights cache to ./models_cache/ and survive recreating the venv. The model
    is the ``model_id`` argument, else EMBED_MODEL, else MODEL_ID. The ingest and
    retrieve singletons take no argument, so EMBED_MODEL applies to both.
    """

    MODEL_ID = "all-MiniLM-L6-v2"

    def __init__(self, model_id: str = None):
        self.model_id = self.resolve_model_id(model_id)
        self._model = None
        self._cache_dir = resolve_path("MODELS_CACHE", "./models_cache")

    @classmethod
    def resolve_model_id(cls, model_id: str = None) -> str:
        """Return the model name an Embedder(model_id) would use.

        A classmethod so callers can label output files without loading anything.
        Reading MODEL_ID directly reports the default, not the model in use.
        """
        return model_id or os.getenv("EMBED_MODEL") or cls.MODEL_ID

    def _load(self):
        """Download (once) and load the sentence-transformer model."""
        if self._model is None:
            t0 = time.perf_counter()
            self._model = SentenceTransformer(
                self.model_id, cache_folder=self._cache_dir
            )
            print(f"[TIMER] Embedder model load ({self.model_id}): "
                  f"{time.perf_counter() - t0:.2f}s")

    def embed(self, texts: "list[str] | str") -> list:
        """Convert one or more strings into embedding vectors.

        Accepts a string or a list; always returns a list of vectors.
        """
        self._load()

        # Normalise input so callers can pass either a string or a list
        if isinstance(texts, str):
            texts = [texts]

        t0 = time.perf_counter()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        print(f"[TIMER] Embedder encode ({len(texts)} chunk(s)): {time.perf_counter() - t0:.2f}s")
        return embeddings.tolist()


def collection_name(model_id: str = None) -> str:
    """Return the ChromaDB collection that holds vectors for a given model.

    The default model keeps the original collection name, so switching this
    mechanism on does not move or orphan any existing data. Any other model
    gets its own suffixed collection, because dimensionality is fixed per
    collection (see the module docstring).

    Chroma restricts collection names to [a-zA-Z0-9._-], so anything else in
    the model id (notably the "/" in fully-qualified HuggingFace names) is
    replaced with a hyphen.
    """
    resolved = Embedder.resolve_model_id(model_id)
    if resolved == Embedder.MODEL_ID:
        return DEFAULT_COLLECTION
    safe = re.sub(r"[^a-zA-Z0-9._-]", "-", resolved)
    return f"{DEFAULT_COLLECTION}__{safe}"
