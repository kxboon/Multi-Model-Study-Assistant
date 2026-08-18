"""Copy the `eval_set` corpus into a second collection, re-embedding only.

Why copy rather than re-ingest
------------------------------
Re-running ingest_file() under a different embedding model would re-run
Whisper, and Whisper is the one non-deterministic step in the pipeline. If the
transcript shifted by a single word, _sliding_window would move every boundary
after it and the audio chunk indices — 13 of the 20 ground-truth entries —
would silently stop meaning what verification/eval_chunks.md says they mean.
The comparison would then be measuring two corpora, not two models.

Chunking is purely textual and happens entirely before embedding, so lifting
the stored chunks across and re-embedding them reproduces exactly what a
re-ingest would produce, but guaranteed rather than hoped for. The embedding
model becomes the only variable that differs between the two runs.

Usage (the model must be set, or source and target would be the same):
    EMBED_MODEL=all-mpnet-base-v2 python verification/build_eval_set_mpnet.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import chromadb

from backend.models.embedder import Embedder, collection_name
from backend.paths import resolve_path

SESSION = "eval_set"
CHROMA_PATH = resolve_path("CHROMA_PATH", "./vectorstore/chroma_db")
BATCH = 32


def fetch_session(collection, session_id: str) -> list:
    """Return the session's rows as (id, document, metadata), index-ordered."""
    got = collection.get(where={"session_id": session_id},
                         include=["documents", "metadatas"])
    rows = list(zip(got["ids"], got["documents"], got["metadatas"]))
    rows.sort(key=lambda r: (r[2].get("source_file") or "",
                             r[2].get("chunk_index", 0)))
    return rows


def identity(rows: list) -> list:
    """The (source_file, chunk_index, text) triples that must survive the copy."""
    return [(m.get("source_file"), m.get("chunk_index"), doc)
            for _, doc, m in rows]


def main() -> None:
    target_model = Embedder.resolve_model_id()
    source_name = collection_name(Embedder.MODEL_ID)   # the default, 384-dim
    target_name = collection_name()                    # derived from EMBED_MODEL

    print("=" * 78)
    print("COPY eval_set CORPUS TO A NEW EMBEDDING MODEL")
    print("=" * 78)
    print(f"source collection : {source_name}  (model {Embedder.MODEL_ID})")
    print(f"target collection : {target_name}  (model {target_model})")

    if target_name == source_name:
        print("\n[FAIL] EMBED_MODEL is unset or set to the default, so the target "
              "is the source.\n       Set EMBED_MODEL to the model you are "
              "evaluating, e.g.:\n"
              "       EMBED_MODEL=all-mpnet-base-v2 python "
              "verification/build_eval_set_mpnet.py")
        raise SystemExit(1)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    source = client.get_or_create_collection(
        name=source_name, metadata={"hnsw:space": "cosine"})
    target = client.get_or_create_collection(
        name=target_name, metadata={"hnsw:space": "cosine"})

    rows = fetch_session(source, SESSION)
    if not rows:
        print(f"\n[FAIL] session '{SESSION}' is empty in '{source_name}'.")
        raise SystemExit(1)
    print(f"\nread {len(rows)} chunk(s) from the source collection")

    # Idempotent: a re-run replaces this session rather than duplicating it.
    stale = len(target.get(where={"session_id": SESSION}, include=[])["ids"])
    if stale:
        target.delete(where={"session_id": SESSION})
        print(f"cleared {stale} pre-existing chunk(s) from the target")

    ids = [r[0] for r in rows]
    documents = [r[1] for r in rows]
    metadatas = [r[2] for r in rows]

    embedder = Embedder()
    t0 = time.perf_counter()
    embeddings = []
    for start in range(0, len(documents), BATCH):
        embeddings.extend(embedder.embed(documents[start:start + BATCH]))
    embed_seconds = time.perf_counter() - t0

    dims = {len(v) for v in embeddings}
    assert len(dims) == 1, f"ragged embedding widths: {sorted(dims)}"
    dim = dims.pop()
    print(f"\nembedded {len(embeddings)} chunk(s) in {embed_seconds:.1f}s "
          f"-> {dim}-dim vectors")

    target.upsert(ids=ids, documents=documents,
                  embeddings=embeddings, metadatas=metadatas)

    # ---- Verify the copy against the source, after the write ----------------
    print("\n" + "-" * 78)
    print("VERIFICATION")
    print("-" * 78)

    copied = fetch_session(target, SESSION)
    src_identity, dst_identity = identity(rows), identity(copied)

    print(f"  chunk count      source {len(rows)} / target {len(copied)}")
    assert len(copied) == len(rows), "chunk count differs"

    mismatches = [
        (s[0], s[1]) for s, d in zip(src_identity, dst_identity) if s != d
    ]
    if mismatches:
        for source_file, idx in mismatches[:10]:
            print(f"  [MISMATCH] {source_file} #{idx}")
        raise AssertionError(f"{len(mismatches)} chunk(s) differ from the source")
    print("  chunk text       identical on all "
          f"{len(copied)} chunk(s) (exact string comparison)")

    src_keys = [(s[0], s[1]) for s in src_identity]
    dst_keys = [(d[0], d[1]) for d in dst_identity]
    assert src_keys == dst_keys, "chunk indices differ"
    print("  (source_file, chunk_index) pairs identical — ground truth valid")

    stored = target.get(where={"session_id": SESSION}, limit=1,
                        include=["embeddings"])
    stored_dim = len(stored["embeddings"][0])
    print(f"  stored vectors   {stored_dim}-dim")
    assert stored_dim == dim, "stored dimensionality differs from what we wrote"

    src_dim = len(source.get(where={"session_id": SESSION}, limit=1,
                             include=["embeddings"])["embeddings"][0])
    print(f"  source vectors   {src_dim}-dim (unchanged)")
    assert src_dim != stored_dim, (
        "source and target have the same dimensionality — did the model "
        "actually change?"
    )

    per_file = {}
    for _, _, m in copied:
        per_file[m.get("source_file")] = per_file.get(m.get("source_file"), 0) + 1
    print("\n  per source file:")
    for fname, n in sorted(per_file.items()):
        print(f"    {fname:34s} {n:>3}")

    print("\n" + "=" * 78)
    print(f"OK — '{target_name}' holds {len(copied)} chunks, {stored_dim}-dim, "
          "text identical to baseline")
    print("=" * 78)


if __name__ == "__main__":
    main()
