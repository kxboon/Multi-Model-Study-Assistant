"""
Verification harness — SESSION ISOLATION (end-to-end).

Ingests two clearly distinct files into two different sessions, then proves a
query against one session can never retrieve the other's chunks — even when the
better semantic match lives in the other session.

    iso_a  <- Natural_Language_Processing.mp3   (audio: NLP, spam, chatbots)
    iso_b  <- CM3060_L6_slide7_bayes.jpg        (image: Bayes, prior/posterior)

Checks (each asserted explicitly, pass/fail reported):
    1. query iso_a with an iso_a question   -> all chunks session_id == iso_a
    2. query iso_a with an iso_b-only question -> still zero iso_b chunks
    3. query iso_b with an iso_b question   -> all chunks session_id == iso_b
    4. query iso_b with an iso_a-only question -> still zero iso_a chunks
    5. POST /query with missing/blank session_id -> HTTP 400, not a cross-session
       search. Requires the FastAPI server to be running; reported as SKIP if not.

Every retrieved chunk's session_id and source_file is printed so the filter can
be inspected directly. Results are appended to bucket_b_verification.json.

This harness does NOT modify any project file, and does NOT fix anything.

Run from the repository root:

    python -m verification.verify_isolation
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests

import backend.ingest as ing
from backend.retrieve import query_rag

RESULTS_FILE = Path(__file__).resolve().parent / "bucket_b_verification.json"
API_URL = "http://127.0.0.1:8000"

SESSION_A = "iso_a"
SESSION_B = "iso_b"
FILE_A = "sample_files/Natural_Language_Processing.mp3"
FILE_B = "sample_files/CM3060_L6_slide7_bayes.jpg"

# Questions each session's own material answers, and which the other's does not.
Q_A = "what is natural language processing used for"
Q_B = "what is the difference between prior and posterior probability"

N_RESULTS = 5


def run_query(label: str, question: str, session_id: str, forbidden: str) -> dict:
    """Query one session and assert no chunk leaked from `forbidden`.

    Prints session_id + source_file for every retrieved chunk as evidence.
    """
    print("-" * 70)
    print(f"[{label}]")
    print(f"  question   : {question!r}")
    print(f"  session_id : {session_id}   (must NOT return any {forbidden} chunk)")

    t0 = time.perf_counter()
    rag = query_rag(question=question, session_id=session_id, n_results=N_RESULTS)
    elapsed = time.perf_counter() - t0

    metas = rag["metadatas"]
    dists = rag["distances"]

    print(f"  retrieved  : {len(metas)} chunk(s) in {elapsed:.2f}s")
    chunks_evidence = []
    for i, (meta, dist) in enumerate(zip(metas, dists), 1):
        sid = meta.get("session_id")
        src = meta.get("source_file")
        print(f"    {i}. session_id={sid!r}  source_file={src!r}  distance={dist:.4f}")
        chunks_evidence.append(
            {"rank": i, "session_id": sid, "source_file": src, "distance": round(dist, 4)}
        )

    sids = [m.get("session_id") for m in metas]
    leaked = [s for s in sids if s != session_id]
    forbidden_hits = [s for s in sids if s == forbidden]

    passed = (len(forbidden_hits) == 0) and (len(leaked) == 0)
    print(f"  own-session chunks : {sum(1 for s in sids if s == session_id)}/{len(sids)}")
    print(f"  {forbidden} chunks leaked : {len(forbidden_hits)}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "check": label,
        "question": question,
        "queried_session": session_id,
        "forbidden_session": forbidden,
        "retrieval_time_s": round(elapsed, 2),
        "chunks_returned": len(metas),
        "own_session_chunks": sum(1 for s in sids if s == session_id),
        "forbidden_chunks_returned": len(forbidden_hits),
        "foreign_chunks_returned": len(leaked),
        "chunks": chunks_evidence,
        "passed": passed,
    }


def check_endpoint_guard() -> dict:
    """Confirm /query rejects a missing/blank session_id with 400."""
    print("-" * 70)
    print("[CHECK 5] endpoint guard — /query must reject missing/blank session_id")

    cases = [
        ("missing session_id", {"question": "test"}),
        ("blank session_id", {"question": "test", "session_id": "   "}),
        ("empty session_id", {"question": "test", "session_id": ""}),
    ]
    results = []
    reachable = True

    for label, payload in cases:
        try:
            r = requests.post(f"{API_URL}/query", json=payload, timeout=30)
            status = r.status_code
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:200]
        except requests.exceptions.RequestException as exc:
            reachable = False
            print(f"  [{label}] server unreachable at {API_URL}: {type(exc).__name__}")
            results.append({"case": label, "status": None, "detail": None, "passed": None})
            continue

        ok = status == 400
        print(f"  [{label}] -> HTTP {status}  {'PASS' if ok else 'FAIL (expected 400)'}")
        print(f"      detail: {detail}")
        results.append({"case": label, "status": status, "detail": detail, "passed": ok})

    if not reachable:
        print("  RESULT: SKIP — start the server first:")
        print("      python -m uvicorn backend.main:app --port 8000")
        return {"check": "endpoint_guard", "passed": None, "skipped": True, "cases": results}

    passed = all(c["passed"] for c in results)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return {"check": "endpoint_guard", "passed": passed, "skipped": False, "cases": results}


def session_count(session_id: str) -> int:
    return len(ing._collection.get(where={"session_id": session_id}, include=[])["ids"])


def main() -> int:
    print("=" * 70)
    print("VERIFICATION — session isolation")
    print(f"  {SESSION_A} <- {FILE_A}")
    print(f"  {SESSION_B} <- {FILE_B}")
    print("=" * 70)

    for f in (FILE_A, FILE_B):
        if not Path(f).exists():
            print(f"[FAIL] file not found: {Path(f).resolve()}")
            return 1

    record = {
        "timestamp": datetime.now().isoformat(),
        "verification": "session_isolation",
        "sessions": [SESSION_A, SESSION_B],
        "files": {SESSION_A: FILE_A, SESSION_B: FILE_B},
        "passed": False,
        "failed_stage": None,
        "error": None,
        "checks": [],
    }

    t_total = time.perf_counter()
    checks = []
    try:
        # ---- Setup: ingest both files into their own sessions ----
        print("\n### SETUP — ingesting both files\n")
        record["chunks_ingested"] = {
            SESSION_A: ing.ingest_file(FILE_A, {"session_id": SESSION_A}),
            SESSION_B: ing.ingest_file(FILE_B, {"session_id": SESSION_B}),
        }
        print(f"\n  {SESSION_A}: {session_count(SESSION_A)} chunks in store")
        print(f"  {SESSION_B}: {session_count(SESSION_B)} chunks in store")

        # ---- Checks 1-4: retrieval isolation, both directions ----
        print("\n### RETRIEVAL ISOLATION CHECKS\n")
        checks.append(run_query(
            "CHECK 1 — iso_a asked about its OWN content",
            Q_A, SESSION_A, forbidden=SESSION_B))
        checks.append(run_query(
            "CHECK 2 — iso_a asked about iso_b-ONLY content",
            Q_B, SESSION_A, forbidden=SESSION_B))
        checks.append(run_query(
            "CHECK 3 — iso_b asked about its OWN content (swapped)",
            Q_B, SESSION_B, forbidden=SESSION_A))
        checks.append(run_query(
            "CHECK 4 — iso_b asked about iso_a-ONLY content (swapped)",
            Q_A, SESSION_B, forbidden=SESSION_A))

        # ---- Check 5: endpoint guard ----
        print("\n### ENDPOINT GUARD CHECK\n")
        checks.append(check_endpoint_guard())

        scored = [c["passed"] for c in checks if c["passed"] is not None]
        record["passed"] = all(scored) and len(scored) > 0
    except Exception as exc:  # noqa: BLE001 — report the failing stage
        record["failed_stage"] = "isolation_checks"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print("\n[FAIL] exception during isolation verification:")
        traceback.print_exc()
    finally:
        record["checks"] = checks
        record["total_time_s"] = round(time.perf_counter() - t_total, 2)

    # ---- Append to the shared results file ----
    try:
        existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if RESULTS_FILE.exists() else []
        if not isinstance(existing, list):
            existing = [existing]
    except (json.JSONDecodeError, ValueError):
        existing = []
    existing.append(record)
    RESULTS_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("ISOLATION VERIFICATION SUMMARY")
    print("=" * 70)
    for c in checks:
        verdict = "SKIP" if c["passed"] is None else ("PASS" if c["passed"] else "FAIL")
        print(f"  [{verdict}] {c['check']}")
    print(f"  total time : {record['total_time_s']} s")
    print(f"  RESULT     : {'PASS' if record['passed'] else 'FAIL'}")
    print(f"  appended to: {RESULTS_FILE}")

    # ---- Cleanup: drop the throwaway sessions ----
    print("\n### CLEANUP — dropping test sessions")
    for sess in (SESSION_A, SESSION_B):
        before = session_count(sess)
        ing._collection.delete(where={"session_id": sess})
        print(f"  {sess}: {before} -> {session_count(sess)} chunks")
    print("=" * 70)

    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
