"""
Sentence-transformers embedding wrapper.

Defaults to all-MiniLM-L6-v2 — a small but high-quality model that produces
384-dimensional embeddings and runs fast on CPU. The active model can be
swapped for an evaluation run by setting the EMBED_MODEL environment
variable, e.g. all-mpnet-base-v2 (768-dim, larger and slower).

Why the collection name lives in this file
------------------------------------------
A ChromaDB collection's dimensionality is pinned by its FIRST write and can
never hold two sizes — writing a 768-dim vector into a 384-dim collection
raises InvalidDimensionException. So the embedding model and the collection
are not independent choices: changing one without the other is always a bug.

Keeping collection_name() next to the model resolution makes that impossible
to get wrong. Both backend.ingest and backend.retrieve derive their collection
from here, so a single process cannot mix models, and a run under EMBED_MODEL
automatically reads and writes its own collection.
"""

import os
import re
import time
from sentence_transformers import SentenceTransformer

# Base name for the vector store collection. The default model keeps this name
# unchanged so existing data is never orphaned by the swap mechanism.
DEFAULT_COLLECTION = "study_materials"


class Embedder:
    """Lazy-loading text embedder backed by sentence-transformers.

    Weights are cached to ./models_cache/ so subsequent runs skip the
    download step even if the venv is recreated.

    The model is chosen, in order of precedence:
      1. the ``model_id`` constructor argument,
      2. the EMBED_MODEL environment variable,
      3. MODEL_ID, the default below.

    Module-level singletons (backend.ingest, backend.retrieve) are constructed
    with no arguments at import time, so the environment variable is the only
    way to swap them — which is deliberate: it means one setting applies to
    ingest and query alike for the whole process.
    """

    MODEL_ID = "all-MiniLM-L6-v2"

    def __init__(self, model_id: str = None):
        self.model_id = self.resolve_model_id(model_id)
        self._model = None
        self._cache_dir = os.getenv("MODELS_CACHE", "./models_cache")

    @classmethod
    def resolve_model_id(cls, model_id: str = None) -> str:
        """Return the model name that an Embedder(model_id) would actually use.

        Exposed as a classmethod so callers can label output files with the
        active model without instantiating (and therefore loading) anything.
        Reading MODEL_ID directly is wrong once EMBED_MODEL is in play — it
        reports the default rather than the model in use.
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
