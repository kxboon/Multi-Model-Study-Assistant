"""
Bucket B verification — QUERY half of the audio path.

The audio (session_id="verify_audio") is already ingested (9 chunks in Chroma).
This harness exercises the retrieval + generation half directly (not the HTTP
endpoint):

    query_rag(question, session_id="verify_audio", n_results=5)
        -> inspect retrieved chunks (source must be audio, not PDF/PPTX)
    ask_ollama(question, chunks)
        -> full grounded answer

It logs retrieval time, which chunks came back (source + distance + preview),
and the full answer text. It does NOT judge groundedness — that's for the human.

Run from the repository root:
    python verify_query.py
"""

import sys
import time

from backend.retrieve import query_rag, ask_ollama

# Default is a question the transcript covers well. Pass a different question as
# the first CLI arg to test other cases (e.g. a refusal check on absent content).
DEFAULT_QUESTION = "What is the difference between stemming and lemmatization?"
QUESTION = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
SESSION_ID = "verify_audio"
N_RESULTS = 5


def preview(text: str, n: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def main() -> int:
    print("=" * 70)
    print("BUCKET B VERIFICATION — query half of audio path")
    print(f"question   : {QUESTION!r}")
    print(f"session_id : {SESSION_ID}")
    print(f"n_results  : {N_RESULTS}")
    print("=" * 70)

    # ---- Retrieval ----
    t0 = time.perf_counter()
    rag = query_rag(question=QUESTION, session_id=SESSION_ID, n_results=N_RESULTS)
    retrieval_time = time.perf_counter() - t0

    chunks = rag["chunks"]
    metadatas = rag["metadatas"]
    distances = rag["distances"]

    print(f"\n[RETRIEVAL] {len(chunks)} chunk(s) in {retrieval_time:.2f}s\n")

    source_types = []
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances), 1):
        stype = meta.get("source_type")
        source_types.append(stype)
        print(f"--- chunk {i} ---")
        print(f"  source_type : {stype}")
        print(f"  source_file : {meta.get('source_file')}")
        print(f"  chunk_index : {meta.get('chunk_index')}")
        print(f"  session_id  : {meta.get('session_id')}")
        print(f"  distance    : {dist:.4f}")
        print(f"  text        : {preview(chunk)}")
        print()

    # ---- Objective (non-judgemental) retrieval facts ----
    audio_count = sum(1 for s in source_types if s == "audio")
    other = [s for s in source_types if s != "audio"]
    print("[RETRIEVAL FACTS]")
    print(f"  returned         : {len(chunks)} chunk(s)")
    print(f"  audio-sourced    : {audio_count}/{len(chunks)}")
    print(f"  non-audio sources: {other if other else 'none'}")
    print(f"  retrieval time   : {retrieval_time:.2f}s")

    if not chunks:
        print("\n[RETRIEVAL] returned NO chunks — skipping ask_ollama.")
        return 1

    # ---- Generation ----
    print("\n" + "=" * 70)
    print("ASK_OLLAMA — full answer")
    print("=" * 70)
    answer = ask_ollama(QUESTION, chunks)
    print("\n---ANSWER---")
    print(answer)
    print("---END ANSWER---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
