"""Bucket C retrieval evaluation — recall@5 over the fixed `eval_set` corpus.

Reads verification/eval_questions.json, runs each question through the real
retrieval path (backend.retrieve.query_rag), and checks whether any of the
question's acceptable (source_file, chunk_index) pairs came back in the top-k.

Ground truth is a PAIR, never a bare index: chunk_index restarts at 0 for every
source file, so `chunk 5` on its own matches three different chunks in this
corpus. Every comparison here is on the pair.

This script is read-only with respect to the vector store. It asserts the
`eval_set` chunk count is unchanged at the end — if the corpus drifts, the
recorded ground-truth indices silently stop meaning anything.

Run from the repo root:
    python verification/eval_recall.py
"""

import io
import json
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from backend.models.embedder import Embedder
import backend.retrieve as retrieve

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
N_RESULTS = 5
# Resolved, not Embedder.MODEL_ID — under EMBED_MODEL the class constant still
# reports the default, which would make an mpnet run overwrite the MiniLM
# results file with results that are not MiniLM's.
EMBEDDING_MODEL = Embedder.resolve_model_id()
OUT_PATH = Path(__file__).parent / f"eval_results_{EMBEDDING_MODEL}.json"


def load_questions() -> tuple:
    spec = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return spec["session_id"], spec["questions"]


def session_chunk_count(session_id: str) -> int:
    """Read-only count of chunks in the session, used as a tamper check."""
    return len(retrieve._collection.get(where={"session_id": session_id},
                                        include=[])["ids"])


def stored_vector_dim(session_id: str) -> "int | None":
    """Dimensionality of the vectors actually stored for this session.

    Recorded so a results file proves which model produced it rather than
    merely claiming so: a 768-dim reading cannot have come from MiniLM.
    """
    got = retrieve._collection.get(where={"session_id": session_id},
                                   limit=1, include=["embeddings"])
    vectors = got.get("embeddings")
    return len(vectors[0]) if vectors is not None and len(vectors) else None


def retrieve_quietly(question: str, session_id: str) -> tuple:
    """Run query_rag, swallowing its [TIMER]/[CHUNKS] chatter.

    Returns (retrieved_pairs, elapsed_seconds). The pairs are
    (source_file, chunk_index) tuples in rank order, rank 1 first.
    """
    sink = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(sink):
        result = retrieve.query_rag(question, session_id=session_id,
                                    n_results=N_RESULTS)
    elapsed = time.perf_counter() - t0

    pairs = [
        (m.get("source_file"), m.get("chunk_index"))
        for m in result["metadatas"]
    ]
    return pairs, elapsed, result["distances"]


def evaluate(session_id: str, questions: list) -> list:
    """Run every question and record hit/miss plus the rank of the first hit."""
    rows = []
    for q in questions:
        accepted = {(q["source_file"], idx) for idx in q["ground_truth"]}
        pairs, elapsed, distances = retrieve_quietly(q["question"], session_id)

        # Rank is 1-indexed; the first accepted pair encountered wins.
        rank = next((i + 1 for i, p in enumerate(pairs) if p in accepted), None)

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "source_file": q["source_file"],
            "ground_truth": q["ground_truth"],
            "single_gt": q["single_gt"],
            "hit": rank is not None,
            "rank": rank,
            "retrieval_seconds": round(elapsed, 3),
            "retrieved": [
                {
                    "rank": i + 1,
                    "source_file": sf,
                    "chunk_index": ci,
                    "distance": round(distances[i], 4),
                    "is_ground_truth": (sf, ci) in accepted,
                }
                for i, (sf, ci) in enumerate(pairs)
            ],
        })
    return rows


def recall(rows: list) -> "float | None":
    return round(sum(r["hit"] for r in rows) / len(rows), 3) if rows else None


def summarise(rows: list) -> dict:
    single = [r for r in rows if r["single_gt"]]
    multi = [r for r in rows if not r["single_gt"]]

    by_file = {}
    for r in rows:
        by_file.setdefault(r["source_file"], []).append(r)

    times = [r["retrieval_seconds"] for r in rows]
    return {
        "recall_at_5": recall(rows),
        "recall_at_5_single_gt": recall(single),
        "recall_at_5_multi_gt": recall(multi),
        "counts": {
            "total": len(rows),
            "hits": sum(r["hit"] for r in rows),
            "single_gt": len(single),
            "single_gt_hits": sum(r["hit"] for r in single),
            "multi_gt": len(multi),
            "multi_gt_hits": sum(r["hit"] for r in multi),
        },
        "by_source_file": {
            fname: {
                "questions": len(rs),
                "hits": sum(r["hit"] for r in rs),
                "recall_at_5": recall(rs),
            }
            for fname, rs in sorted(by_file.items())
        },
        "mean_retrieval_seconds": round(sum(times) / len(times), 3),
        "min_retrieval_seconds": min(times),
        "max_retrieval_seconds": max(times),
    }


def print_report(rows: list, summary: dict, session_id: str, chunks: int,
                 dim: int, query_dim: int) -> None:
    bar = "=" * 78
    print(bar)
    print(f"BUCKET C RETRIEVAL EVAL — recall@{N_RESULTS}")
    print(f"embedding model : {EMBEDDING_MODEL}")
    print(f"collection      : {retrieve.COLLECTION_NAME}")
    print(f"vector dims     : {dim} stored / {query_dim} query")
    print(f"session         : {session_id}  ({chunks} chunks)")
    print(bar)

    for r in rows:
        flag = "HIT " if r["hit"] else "MISS"
        rank = f"rank {r['rank']}" if r["rank"] else "rank none"
        gt = ", ".join(str(i) for i in r["ground_truth"])
        print(f"\n[{flag}] Q{r['id']:02d}  {rank:9s}  {r['retrieval_seconds']:.3f}s")
        print(f"        {r['question']}")
        print(f"        ground truth: {r['source_file']} #[{gt}]"
              f"   single_gt={r['single_gt']}")
        for hit in r["retrieved"]:
            mark = " <-- GT" if hit["is_ground_truth"] else ""
            print(f"          {hit['rank']}. {hit['source_file']:32s} "
                  f"#{hit['chunk_index']:<3} d={hit['distance']:.4f}{mark}")

    c = summary["counts"]
    print("\n" + bar)
    print("SUMMARY")
    print(bar)
    print(f"  overall recall@{N_RESULTS}          {summary['recall_at_5']:.3f}"
          f"   ({c['hits']}/{c['total']})")
    print(f"  single_gt recall@{N_RESULTS}        {summary['recall_at_5_single_gt']:.3f}"
          f"   ({c['single_gt_hits']}/{c['single_gt']})   <-- the discriminating subset")
    # Two distinct causes put a question in this bucket, so the label names the
    # category rather than one cause: the audio chunker's ~50% overlap (seven
    # questions) and a PDF passage split mid-argument (Q19).
    print(f"  multi_gt  recall@{N_RESULTS}        {summary['recall_at_5_multi_gt']:.3f}"
          f"   ({c['multi_gt_hits']}/{c['multi_gt']})   (easier: >1 acceptable chunk)")
    print("\n  by source file:")
    for fname, s in summary["by_source_file"].items():
        print(f"    {fname:34s} {s['recall_at_5']:.3f}  "
              f"({s['hits']}/{s['questions']})")
    print(f"\n  mean retrieval time      {summary['mean_retrieval_seconds']:.3f}s"
          f"   (min {summary['min_retrieval_seconds']:.3f}s, "
          f"max {summary['max_retrieval_seconds']:.3f}s)")

    misses = [r for r in rows if not r["hit"]]
    if misses:
        print(f"\n  misses ({len(misses)}):")
        for r in misses:
            print(f"    Q{r['id']:02d}  {r['question']}")
    print(bar)


def main() -> None:
    session_id, questions = load_questions()

    before = session_chunk_count(session_id)
    if before == 0:
        print(f"[FAIL] session '{session_id}' is empty — nothing to evaluate.")
        raise SystemExit(1)

    # Warm-up: the first embed pays the model load, which would otherwise land
    # entirely on Q1 and distort both its timing and the mean.
    warm = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(warm):
        probe = retrieve._embedder.embed("warm up the embedder")[0]
        retrieve.query_rag("warm up the embedder", session_id=session_id,
                           n_results=N_RESULTS)
    print(f"[warm-up query: {time.perf_counter() - t0:.2f}s — excluded from results]\n")

    # A query vector of a different width than the stored vectors cannot
    # retrieve anything — Chroma raises — so agreement here is load-bearing.
    query_dim = len(probe)
    dim = stored_vector_dim(session_id)
    assert dim == query_dim, (
        f"query vectors are {query_dim}-dim but stored vectors are {dim}-dim"
    )

    rows = evaluate(session_id, questions)
    summary = summarise(rows)
    print_report(rows, summary, session_id, before, dim, query_dim)

    after = session_chunk_count(session_id)
    assert after == before, (
        f"eval_set changed during the run ({before} -> {after}); "
        "ground-truth indices are no longer valid"
    )
    print(f"corpus unchanged: {before} chunks before, {after} after")

    OUT_PATH.write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL,
        "collection": retrieve.COLLECTION_NAME,
        "embedding_dim": dim,
        "session_id": session_id,
        "n_results": N_RESULTS,
        "corpus_chunks": before,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "results": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
