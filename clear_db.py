"""
Clear the ChromaDB vector store.

Two modes:
  * Default  — drop the active collection via the ChromaDB API
               (keeps the store directory, just empties the data).
  * --purge  — delete the entire persisted store directory from disk.

Usage:
    python clear_db.py            # empty the active collection
    python clear_db.py --all      # empty every study_materials* collection
    python clear_db.py --purge    # wipe the whole CHROMA_PATH directory

Which collection is "active" depends on the embedding model: an evaluation run
under EMBED_MODEL writes to its own suffixed collection (see
backend.models.embedder.collection_name). Clearing only the default would
therefore leave those behind as orphans, so this script reports every
study_materials* collection it finds and --all clears the lot.

The CHROMA_PATH is read from .env (falls back to ./vectorstore/chroma_db),
resolved relative to the project root regardless of the current working
directory (see backend/paths.py).
IMPORTANT: stop the FastAPI/uvicorn server first — it holds the DB open.
"""

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from backend.models.embedder import DEFAULT_COLLECTION, collection_name
from backend.paths import resolve_path

CHROMA_PATH = resolve_path("CHROMA_PATH", "./vectorstore/chroma_db")
COLLECTION_NAME = collection_name()


def purge_directory() -> None:
    """Delete the entire persisted store directory."""
    path = Path(CHROMA_PATH)
    if path.exists():
        shutil.rmtree(path)
        print(f"[PURGED] Deleted store directory: {path.resolve()}")
    else:
        print(f"[SKIP] Nothing to delete — {path.resolve()} does not exist.")


def clear_collection(clear_all: bool = False) -> None:
    """Drop and recreate collections so they end up empty.

    Clears the active collection by default, or every study_materials*
    collection when clear_all is set. Either way it lists what else is in the
    store, so an evaluation collection left over from an EMBED_MODEL run is
    visible rather than silently orphaned.
    """
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    existing = [c.name for c in client.list_collections()]
    related = [n for n in existing if n == DEFAULT_COLLECTION
               or n.startswith(f"{DEFAULT_COLLECTION}__")]
    targets = related if clear_all else [COLLECTION_NAME]

    for name in targets:
        if name in existing:
            count = client.get_collection(name).count()
            client.delete_collection(name)
            print(f"[CLEARED] Dropped collection '{name}' ({count} chunks removed).")
        else:
            print(f"[SKIP] Collection '{name}' not found.")

        # Recreate it empty with the same cosine config the app expects
        client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[OK] Recreated empty collection '{name}'.")

    untouched = [n for n in related if n not in targets]
    if untouched:
        print(f"\n[NOTE] Left alone: {', '.join(untouched)}")
        print("       These hold vectors from other embedding models. "
              "Use --all to clear them too.")
    other = [n for n in existing if n not in related]
    if other:
        print(f"[NOTE] Unrelated collections in this store: {', '.join(other)}")


def main() -> None:
    if "--purge" in sys.argv:
        purge_directory()
    else:
        clear_collection(clear_all="--all" in sys.argv)


if __name__ == "__main__":
    main()
