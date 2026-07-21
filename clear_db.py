"""
Clear the ChromaDB vector store.

Two modes:
  * Default  — drop the "study_materials" collection via the ChromaDB API
               (keeps the store directory, just empties the data).
  * --purge  — delete the entire persisted store directory from disk.

Usage:
    python clear_db.py            # empty the collection
    python clear_db.py --purge    # wipe the whole CHROMA_PATH directory

The CHROMA_PATH is read from .env (falls back to ./vectorstore/chroma_db).
IMPORTANT: stop the FastAPI/uvicorn server first — it holds the DB open.
"""

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./vectorstore/chroma_db")
COLLECTION_NAME = "study_materials"


def purge_directory() -> None:
    """Delete the entire persisted store directory."""
    path = Path(CHROMA_PATH)
    if path.exists():
        shutil.rmtree(path)
        print(f"[PURGED] Deleted store directory: {path.resolve()}")
    else:
        print(f"[SKIP] Nothing to delete — {path.resolve()} does not exist.")


def clear_collection() -> None:
    """Drop and recreate the collection so it ends up empty."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        count = client.get_collection(COLLECTION_NAME).count()
        client.delete_collection(COLLECTION_NAME)
        print(f"[CLEARED] Dropped collection '{COLLECTION_NAME}' ({count} chunks removed).")
    else:
        print(f"[SKIP] Collection '{COLLECTION_NAME}' not found. Existing: {existing or 'none'}")

    # Recreate it empty with the same cosine config the app expects
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[OK] Recreated empty collection '{COLLECTION_NAME}'.")


def main() -> None:
    if "--purge" in sys.argv:
        purge_directory()
    else:
        clear_collection()


if __name__ == "__main__":
    main()
