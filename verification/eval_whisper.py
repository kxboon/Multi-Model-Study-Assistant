"""Whisper model sweep (tiny/base/small/medium) — accuracy on the NLP lecture audio.

Transcription measurement ONLY. This script never touches ChromaDB: eval_set's
chunk indices depend on the transcript already stored there, so re-ingesting
would invalidate every audio ground-truth index in eval_questions.json.

Two metrics are reported, and they are not interchangeable:

  * WER (word error rate) — the standard ASR measure. Levenshtein distance over
    words, (S + D + I) / N. Lower is better; 0.0 is perfect. Computed with jiwer.

  * Similarity ratio — difflib.SequenceMatcher(...).ratio() over word lists with
    autojunk disabled. This is what verification/smoke/smoke_whisper.py used to
    produce the recorded 97.27% figure, reproduced here verbatim so the old
    number stays comparable. It counts matching blocks rather than edits, so it
    is structurally more forgiving than WER: a high ratio and a non-trivial WER
    are not in contradiction.

Normalisation (applied identically to both metrics, and to both sides):
    lowercase -> replace every non-[a-z0-9\\s] character with a space -> split
    on whitespace. That folds away casing, punctuation and repeated spaces, so
    Whisper is not penalised for capitalising a sentence or adding a comma.
    WER without a stated normalisation is meaningless, so it is printed in the
    report and recorded in the JSON.

The reference transcript is the YouTube one at verification/smoke/. It has
twelve chapter headings interleaved into the prose which were never spoken
aloud; left in, they score as deletions no model can avoid. Both scorings are
reported — "raw" against the file as-is (traceable to the 97.27% lineage) and
"cleaned" with the headings removed (the number that actually measures
transcription).

Usage:
    python verification/eval_whisper.py
"""

import difflib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jiwer
import whisper

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "sample_files" / "Natural_Language_Processing.mp3"
REFERENCE = ROOT / "verification" / "smoke" / "test_smoke_answer.md"
OUT_PATH = ROOT / "verification" / "eval_whisper_results.json"
MODELS = ["tiny", "base", "small", "medium"]

# medium is ~1.5GB of weights and this machine has 7.7GB of shared RAM. If it
# cannot be loaded or run, that is a real finding about what this hardware
# supports, not something to engineer around: the failure is caught, recorded
# verbatim, and the remaining models still report. `large` is not attempted.
TOLERATE_FAILURE = {"medium"}

NORMALISATION = ("lowercase; replace every non-[a-z0-9 whitespace] character "
                 "with a space; split on whitespace")

# Chapter headings from the YouTube transcript UI. These are on-screen captions,
# never spoken, so they are deletions Whisper cannot possibly avoid.
HEADINGS = [
    "Unstructured data",
    "Structured data",
    "Natural Language Understanding (NLU) & Natural Language Generation (NLG)",
    "Machine Translation use case",
    "Virtual Assistance / Chat Bots use case",
    "Sentiment Analysis use case",
    "Spam Detection use case",
    "Tokenization",
    "Stemming & Lemmatization",
    "Part of Speech Tagging",
    "Named Entity Recognition (NER)",
    "Summary",
]

# Known transcription errors that propagated downstream into chat answers, quiz
# items and flashcards. Whether `small` fixes these matters more for the project
# than the aggregate WER does.
#
# Each case names the form the REFERENCE uses and the wrong form base produced.
# The expected form is verified against the reference before scoring: if the
# reference does not actually contain it, the premise is wrong and the case is
# reported as unsupported rather than silently scored as a failure. That check
# earned its place immediately — "lem" was assumed to be a mangling of "lemma",
# but the speaker really does say "lem", so there was never an error to fix.
#   (label, regex for the reference/correct form, regex for the wrong form)
KNOWN_ERRORS = [
    ("'lemmatization' spelt 'Lematization'",
     r"\blemmatization\b", r"\blematization\b"),
    ("'lem' (reference uses the short form, NOT 'lemma')",
     r"\bthe lem\b|\bits lem\b", r"\bits lemma\b|\bthe lemma of\b"),
    ("'make is now a noun' heard as 'make is now now'",
     r"\bmake is now a noun\b", r"\bmake is now now\b"),
]


def normalise(text: str) -> list:
    """Lowercase, strip punctuation, collapse whitespace -> list of words.

    Identical to smoke_whisper.py's normalize(), so the similarity ratio here
    is the same measurement that produced the recorded 97.27%.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def strip_headings(text: str) -> str:
    """Remove the unspoken chapter headings from the reference text.

    The caller derives the removed-word count by differencing the normalised
    word lists rather than summing heading lengths. Summing over-counts by one:
    one heading is fused to the following word in the source file
    ("...use caseand also to things like chatbots"), so its final token and the
    body's first token are a single token before removal and one token after.
    That fused "caseand" is also a guaranteed substitution error for any model
    scored against the raw reference, since nobody said it.
    """
    missing = [h for h in HEADINGS if h not in text]
    if missing:
        raise AssertionError(
            "heading(s) not found in the reference — the file changed, so the "
            f"cleaned scoring would be wrong: {missing}"
        )
    for heading in HEADINGS:
        text = text.replace(heading, " ", 1)
    return text


def score(hypothesis_words: list, reference_words: list) -> dict:
    """WER (via jiwer) plus the smoke test's similarity ratio, on word lists."""
    out = jiwer.process_words(" ".join(reference_words),
                              " ".join(hypothesis_words))
    ratio = difflib.SequenceMatcher(
        None, hypothesis_words, reference_words, autojunk=False).ratio()
    return {
        "wer": round(out.wer, 4),
        "wer_percent": round(out.wer * 100, 2),
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "similarity_ratio": round(ratio, 4),
        "similarity_percent": round(ratio * 100, 2),
    }


def check_known_errors(transcript: str, reference: str) -> list:
    """Score each known error case, having first checked its premise.

    A case is only meaningful if the reference actually contains the form we
    call correct. When it does not, the case is reported as unsupported — the
    assumption behind it was wrong — rather than counted as a model failure.
    """
    low = transcript.lower()
    ref_low = reference.lower()
    findings = []
    for label, right_re, wrong_re in KNOWN_ERRORS:
        supported = re.search(right_re, ref_low) is not None
        right = re.search(right_re, low) is not None
        wrong = re.search(wrong_re, low) is not None

        if not supported:
            status = "PREMISE UNSUPPORTED (reference lacks the assumed form)"
        elif right and not wrong:
            status = "matches reference"
        elif wrong and not right:
            status = "still wrong"
        elif wrong and right:
            status = "both forms present"
        else:
            status = "neither form found"

        findings.append({
            "case": label,
            "premise_supported_by_reference": supported,
            "reference_form_present": right,
            "wrong_form_present": wrong,
            "status": status,
        })
    return findings


def transcribe(size: str) -> dict:
    """Load and run one Whisper model, timing the two phases separately.

    A model in TOLERATE_FAILURE that cannot load or run records the exception
    verbatim and returns ok=False instead of aborting the sweep. Reporting that
    this hardware cannot run the model IS the result for that point on the curve.
    """
    cached = Path.home() / ".cache" / "whisper" / f"{size}.pt"
    was_cached = cached.exists()

    print(f"\n--- whisper '{size}' "
          f"({'cached' if was_cached else 'DOWNLOADING, first use'})")

    load_seconds = None
    try:
        t0 = time.perf_counter()
        model = whisper.load_model(size)
        load_seconds = time.perf_counter() - t0
        print(f"    model load  : {load_seconds:.2f}s")

        t1 = time.perf_counter()
        result = model.transcribe(str(AUDIO), fp16=False)
        transcribe_seconds = time.perf_counter() - t1
        print(f"    transcribe  : {transcribe_seconds:.2f}s")
    except (Exception, MemoryError) as exc:   # noqa: BLE001 - recording it is the point
        phase = "transcribe" if load_seconds is not None else "load"
        detail = f"{type(exc).__name__}: {exc}"
        print(f"    [FAILED during {phase}] {detail}")
        if size not in TOLERATE_FAILURE:
            raise
        return {
            "model": size,
            "ok": False,
            "model_was_cached": was_cached,
            "failed_phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "load_seconds": round(load_seconds, 2) if load_seconds else None,
        }

    text = str(result["text"]).strip()
    print(f"    characters  : {len(text)}")
    print(f"    words       : {len(text.split())}")
    return {
        "model": size,
        "ok": True,
        "model_was_cached": was_cached,
        "load_seconds": round(load_seconds, 2),
        "transcribe_seconds": round(transcribe_seconds, 2),
        "characters": len(text),
        "words": len(text.split()),
        "normalised_words": len(normalise(text)),
        "transcript": text,
    }


def main() -> None:
    for path in (AUDIO, REFERENCE):
        if not path.exists():
            print(f"[FAIL] missing: {path}")
            raise SystemExit(1)

    raw_reference = REFERENCE.read_text(encoding="utf-8")
    cleaned_reference = strip_headings(raw_reference)
    raw_words = normalise(raw_reference)
    cleaned_words = normalise(cleaned_reference)
    removed_words = len(raw_words) - len(cleaned_words)

    # Existing runs are reused rather than regenerated: Whisper proved
    # deterministic here (base and small produced byte-identical transcripts
    # across two runs), the method has not changed, and re-transcribing would
    # burn ~4 minutes to reproduce numbers already recorded. Each run carries a
    # flag saying whether it was measured now or carried over.
    previous = {}
    if OUT_PATH.exists():
        try:
            prior = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            if prior.get("normalisation") == NORMALISATION:
                previous = {r["model"]: r for r in prior.get("runs", [])
                            if r.get("ok", True)}
        except (json.JSONDecodeError, ValueError, KeyError):
            previous = {}

    bar = "=" * 78
    print(bar)
    print(f"WHISPER model sweep ({', '.join(MODELS)}) — transcription accuracy")
    print(bar)
    print(f"audio      : {AUDIO.relative_to(ROOT)}")
    print(f"reference  : {REFERENCE.relative_to(ROOT)}")
    print(f"normalise  : {NORMALISATION}")
    print(f"\nreference words (raw)     : {len(raw_words)}")
    print(f"reference words (cleaned) : {len(cleaned_words)}  "
          f"({removed_words} words of unspoken chapter headings removed, "
          f"{removed_words / len(raw_words) * 100:.1f}%)")
    print("\nMetrics: WER is the standard ASR measure (lower better, via jiwer).")
    print("         Similarity is difflib SequenceMatcher word-level ratio with")
    print("         autojunk=False — the same method smoke_whisper.py used for")
    print("         its recorded 97.27%, kept so that number stays comparable.")

    runs = []
    for size in MODELS:
        if size in previous:
            run = dict(previous[size])
            run["measured_this_run"] = False
            print(f"\n--- whisper '{size}' (carried over from a previous run, "
                  "Whisper is deterministic here)")
            print(f"    model load  : {run['load_seconds']:.2f}s")
            print(f"    transcribe  : {run['transcribe_seconds']:.2f}s")
        else:
            run = transcribe(size)
            run["measured_this_run"] = True

        if not run.get("ok", True):
            runs.append(run)
            continue

        hypothesis = normalise(run["transcript"])
        run["scores"] = {
            "raw_reference": score(hypothesis, raw_words),
            "cleaned_reference": score(hypothesis, cleaned_words),
        }
        run["known_errors"] = check_known_errors(run["transcript"], raw_reference)
        runs.append(run)

    # ---- Report ------------------------------------------------------------
    print("\n" + bar)
    print("RESULTS")
    print(bar)
    ok_runs = [r for r in runs if r.get("ok", True)]
    failed = [r for r in runs if not r.get("ok", True)]

    print(f"{'':10s} {'load':>9s} {'transcribe':>11s} {'chars':>7s} {'words':>7s}"
          f"  {'measured':>9s}")
    for r in runs:
        if not r.get("ok", True):
            print(f"{r['model']:10s} {'FAILED during ' + r['failed_phase']:>40s}")
            continue
        print(f"{r['model']:10s} {r['load_seconds']:>8.2f}s "
              f"{r['transcribe_seconds']:>10.2f}s "
              f"{r['characters']:>7d} {r['words']:>7d}"
              f"  {'now' if r.get('measured_this_run') else 'carried':>9s}")

    for key, label in (("cleaned_reference", "CLEANED reference (headings removed) — headline"),
                       ("raw_reference", "RAW reference (as-is, includes unspoken headings)")):
        print(f"\n  {label}")
        print(f"    {'':8s} {'WER':>8s} {'sub':>6s} {'del':>6s} {'ins':>6s} "
              f"{'hits':>6s} {'similarity':>11s}")
        for r in ok_runs:
            s = r["scores"][key]
            print(f"    {r['model']:8s} {s['wer_percent']:>7.2f}% "
                  f"{s['substitutions']:>6d} {s['deletions']:>6d} "
                  f"{s['insertions']:>6d} {s['hits']:>6d} "
                  f"{s['similarity_percent']:>10.2f}%")

    if failed:
        print("\n" + bar)
        print("FAILED MODELS")
        print(bar)
        for r in failed:
            print(f"  {r['model']}: failed during {r['failed_phase']}")
            print(f"    {r['error_type']}: {r['error']}")

    print("\n" + bar)
    print("KNOWN ERROR CASES (the ones that propagated into answers/quiz/cards)")
    print(bar)
    for r in ok_runs:
        print(f"\n  {r['model']}:")
        for f in r["known_errors"]:
            mark = {"matches reference": "[MATCHES REF]",
                    "still wrong": "[STILL WRONG]"}.get(f["status"], "[N/A]")
            print(f"    {mark:15s} {f['case']}")
            if mark == "[N/A]":
                print(f"                    -> {f['status']}")

    OUT_PATH.write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "audio": str(AUDIO.relative_to(ROOT)).replace("\\", "/"),
        "reference": str(REFERENCE.relative_to(ROOT)).replace("\\", "/"),
        "normalisation": NORMALISATION,
        "metrics": {
            "wer": "word error rate, (S+D+I)/N over normalised words, via jiwer "
                   f"{jiwer.__version__ if hasattr(jiwer, '__version__') else '4.0.0'}",
            "similarity_ratio": "difflib.SequenceMatcher(None, hyp, ref, "
                                "autojunk=False).ratio() over normalised word "
                                "lists — the method used by "
                                "verification/smoke/smoke_whisper.py, which "
                                "recorded 97.27% for base",
        },
        "reference_words_raw": len(raw_words),
        "reference_words_cleaned": len(cleaned_words),
        "heading_words_removed": removed_words,
        "headings_removed": HEADINGS,
        "runs": runs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresults written to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
