"""
Verification harness — standalone IMAGE ingest path (end-to-end).

Calls backend.ingest.ingest_file() DIRECTLY (not the HTTP endpoint) on an
image file and records the key outputs/metrics of each stage:

    BLIP caption + OCR -> combine -> embed -> ChromaDB upsert

Metrics/outputs captured:
    - routed to _process_image? (BLIP + OCR)
    - raw OCR output
    - BLIP caption
    - chunk count
    - embed time (s)
    - whether the ChromaDB upsert succeeded  (the code path the
      page_or_slide: None fix touched — confirm it stores cleanly)

This harness does NOT modify any project file. It wraps the module-level
model singletons in backend.ingest with thin shims so the real pipeline runs
unchanged and its own [TIMER] lines still print live.

Run from the repository root:
    python -m verification.verify_image
"""

import sys
import time
import traceback
from pathlib import Path

import backend.ingest as ing

IMAGE_PATH = "sample_files/CM3060_L6_slide7_bayes.jpg"
SESSION_ID = "verify_image"

# ---------------------------------------------------------------------------
# Instrumentation: wrap the ingest module's singletons to capture outputs.
# Record into `m` and delegate to the originals so the pipeline is unchanged.
# ---------------------------------------------------------------------------
m = {
    "routed_to_image": False,
    "blip_caption": None,
    "ocr_text": None,
    "chunk_count": None,
    "embed_time_s": None,
    "chroma_upsert_succeeded": False,
    "stage_reached": "start",
}

_orig_caption = ing._blip.caption
_orig_ocr = ing._ocr.extract_text
_orig_embed = ing._embedder.embed
_orig_upsert = ing._collection.upsert


def _wrapped_caption(img, *args, **kwargs):
    # Only _process_image calls _blip.caption — reaching here proves the route.
    m["routed_to_image"] = True
    m["stage_reached"] = "blip"
    caption = _orig_caption(img, *args, **kwargs)
    m["blip_caption"] = caption
    return caption


def _wrapped_ocr(img, *args, **kwargs):
    m["stage_reached"] = "ocr"
    ocr_text = _orig_ocr(img, *args, **kwargs)
    m["ocr_text"] = ocr_text
    return ocr_text


def _wrapped_embed(texts, *args, **kwargs):
    m["stage_reached"] = "embed"
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


ing._blip.caption = _wrapped_caption
ing._ocr.extract_text = _wrapped_ocr
ing._embedder.embed = _wrapped_embed
ing._collection.upsert = _wrapped_upsert


def safe_print(s: str = "") -> None:
    """Print without crashing on a non-UTF-8 console (e.g. Windows gbk).

    OCR/BLIP text can contain non-ASCII glyphs (¢, curly quotes) that the
    active console codec can't encode; round-trip through that codec with
    errors='replace' so the summary never raises. Harness hygiene only.
    """
    enc = sys.stdout.encoding or "utf-8"
    print(s.encode(enc, errors="replace").decode(enc))


def main() -> int:
    image = Path(IMAGE_PATH)
    print("=" * 70)
    print("VERIFICATION — standalone image ingest path")
    print(f"image file : {image}  (exists={image.exists()})")
    print(f"session_id : {SESSION_ID}")
    print("=" * 70)

    if not image.exists():
        print(f"[FAIL] image file not found: {image.resolve()}")
        return 1

    stored = None
    t_total = time.perf_counter()
    try:
        stored = ing.ingest_file(IMAGE_PATH, {"session_id": SESSION_ID})
    except Exception:  # noqa: BLE001 — we want the failing stage reported
        print("\n[FAIL] exception during ingest_file:")
        traceback.print_exc()
    total_time = round(time.perf_counter() - t_total, 2)

    print("\n" + "=" * 70)
    print("IMAGE INGEST SUMMARY")
    print("=" * 70)
    print(f"  routed to _process_image : {m['routed_to_image']}")
    print(f"  ---- BLIP caption ----")
    safe_print(f"  {m['blip_caption']!r}")
    print(f"  ---- raw OCR output ----")
    safe_print(f"  {m['ocr_text']!r}")
    print(f"  ------------------------")
    print(f"  chunk count              : {m['chunk_count']}")
    print(f"  embed time               : {m['embed_time_s']} s")
    print(f"  chroma upsert succeeded  : {m['chroma_upsert_succeeded']}")
    print(f"  chunks stored (return)   : {stored}")
    print(f"  stage reached            : {m['stage_reached']}")
    print(f"  total time               : {total_time} s")
    print("=" * 70)

    return 0 if (m["chroma_upsert_succeeded"] and (stored or 0) > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
