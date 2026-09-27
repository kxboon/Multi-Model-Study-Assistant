"""Resolve persistent-state paths against the project root, not the CWD.

CHROMA_PATH / MODELS_CACHE / SIGNALS_PATH default to relative paths, and a
relative path resolves against os.getcwd() at process start. Launching uvicorn
from backend/ rather than the root therefore created a second, diverging store
at backend/vectorstore/chroma_db with no error. Anchoring to PROJECT_ROOT fixes
that; an absolute value in the environment is still honoured as-is.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(env_var: str, default: str) -> str:
    """Read env_var, anchoring a relative result to PROJECT_ROOT."""
    value = os.getenv(env_var, default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)
