"""Resolves on-disk paths for persistent state (vector store, model cache,
signals log) against the project root rather than the current working
directory.

CHROMA_PATH / MODELS_CACHE / SIGNALS_PATH are documented as relative paths
(e.g. "./vectorstore/chroma_db"), and a relative path resolves against
os.getcwd() at the moment the process starts. Depending on whether uvicorn,
streamlit, or a script under verification/ was launched from the project
root or from inside backend/, that silently pointed at two different,
diverging directories on disk with no error — backend/vectorstore/chroma_db
ended up holding chunks the root vectorstore/chroma_db never saw. Anchoring
every relative value to PROJECT_ROOT here makes the store the same
regardless of launch directory; an absolute value in the environment is
still honoured as-is.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(env_var: str, default: str) -> str:
    """Read env_var (or fall back to default), anchoring a relative result
    to PROJECT_ROOT instead of the current working directory."""
    value = os.getenv(env_var, default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)
