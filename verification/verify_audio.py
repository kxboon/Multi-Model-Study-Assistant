"""
Bucket B verification harness — audio ingest path (end-to-end).

Calls backend.ingest.ingest_file() DIRECTLY (not the HTTP endpoint) on an
audio file and records the key metrics for each stage of the chain:

    Whisper transcribe -> sliding-window chunk -> embed -> ChromaDB upsert

Metrics captured:
    - transcribe time (s)
    - transcript character length
    - chunk count
    - embed time (s)
    - whether the ChromaDB upsert succeeded

Results are appended to bucket_b_verification.json.

This harness does NOT modify any project file. It wraps the module-level
model singletons in backend.ingest with thin timing shims so the real
pipeline runs unchanged and its own [TIMER] lines still print live.

Run from the repository root:
    python verify_audio.py
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import backend.ingest as ing

AUDIO_PATH = "sample_files/Natural_Language_Processing.mp3"
SESSION_ID = "verify_audio"
# Write results next to this script (verification/), not the cwd it's run from.
RESULTS_FILE = Path(__file__).resolve().parent / "bucket_b_verification.json"

# ---------------------------------------------------------------------------
# Instrumentation: wrap the ingest module's singletons to capture metrics.
# We record into `m` and delegate to the originals so the pipeline is unchanged.
# ---------------------------------------------------------------------------
m = {
    "transcribe_time_s": None,
    "transcript_char_len": None,
    "chunk_count": None,
    "embed_time_s": None,
    "chroma_upsert_succeeded": False,
    "stage_reached": "start",
}

_orig_transcribe = ing._whisper.transcribe
_orig_embed = ing._embedder.embed
_orig_upsert = ing._collection.upsert


def _wrapped_transcribe(audio_path, *args, **kwargs):
    m["stage_reached"] = "whisper"
    t0 = time.perf_counter()
    transcript = _orig_transcribe(audio_path, *args, **kwargs)
    m["transcribe_time_s"] = round(time.perf_counter() - t0, 2)
    m["transcript_char_len"] = len(transcript)
    m["stage_reached"] = "chunk"
    return transcript


def _wrapped_embed(texts, *args, **kwargs):
    m["stage_reached"] = "embed"
    # texts is the list of chunks handed to the embedder
    try:
        m["chunk_count"] = len(texts)
    except TypeError:
        pass
    t0 = time.perf_counter()
    embeddings = _orig_embed(texts, *args, **kwargs)
    m["embed_time_s"] = round(time.perf_counter() - t0, 2)
    m["stage_reached"] = "chroma"
    return embeddings


def _wrapped_upsert(*args, **kwargs):
    result = _orig_upsert(*args, **kwargs)   # raises if the upsert fails
    m["chroma_upsert_succeeded"] = True
    m["stage_reached"] = "done"
    return result


ing._whisper.transcribe = _wrapped_transcribe
ing._embedder.embed = _wrapped_embed
ing._collection.upsert = _wrapped_upsert


# ---------------------------------------------------------------------------
# Run the verification
# ---------------------------------------------------------------------------
def main() -> int:
    audio = Path(AUDIO_PATH)
    print("=" * 70)
    print("BUCKET B VERIFICATION — audio ingest path")
    print(f"audio file : {audio}  (exists={audio.exists()})")
    print(f"session_id : {SESSION_ID}")
    print("=" * 70)

    if not audio.exists():
        print(f"[FAIL] audio file not found: {audio.resolve()}")
        return 1

    record = {
        "timestamp": datetime.now().isoformat(),
        "audio_file": str(audio),
        "session_id": SESSION_ID,
        "passed": False,
        "failed_stage": None,
        "error": None,
    }

    t_total = time.perf_counter()
    try:
        chunks_stored = ing.ingest_file(AUDIO_PATH, {"session_id": SESSION_ID})
        record["chunks_stored"] = chunks_stored
        record["passed"] = m["chroma_upsert_succeeded"] and chunks_stored > 0
    except Exception as exc:  # noqa: BLE001 — we want the failing stage reported
        record["failed_stage"] = m["stage_reached"]
        record["error"] = f"{type(exc).__name__}: {exc}"
        print("\n[FAIL] exception during ingest_file:")
        traceback.print_exc()
    finally:
        record["total_time_s"] = round(time.perf_counter() - t_total, 2)
        record["transcribe_time_s"] = m["transcribe_time_s"]
        record["transcript_char_len"] = m["transcript_char_len"]
        record["chunk_count"] = m["chunk_count"]
        record["embed_time_s"] = m["embed_time_s"]
        record["chroma_upsert_succeeded"] = m["chroma_upsert_succeeded"]

    # Append to the results file (kept as a JSON array of runs)
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
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  transcribe time        : {record['transcribe_time_s']} s")
    print(f"  transcript char length : {record['transcript_char_len']}")
    print(f"  chunk count            : {record['chunk_count']}")
    print(f"  embed time             : {record['embed_time_s']} s")
    print(f"  chroma upsert succeeded: {record['chroma_upsert_succeeded']}")
    print(f"  chunks stored (return) : {record.get('chunks_stored')}")
    print(f"  total time             : {record['total_time_s']} s")
    print(f"  RESULT                 : {'PASS' if record['passed'] else 'FAIL'}")
    if not record["passed"]:
        print(f"  failed stage           : {record['failed_stage']}")
        print(f"  error                  : {record['error']}")
    print(f"  appended to            : {RESULTS_FILE}")
    print("=" * 70)

    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
