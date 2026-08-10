"""Reduced LLM comparison — llama3.2 vs llama3:8b vs mistral:7b on eval_set.

Method
------
Retrieval runs ONCE per question. The same retrieved chunks, and therefore the
same prompt, are sent to all three models, so the only variable is the model.
Re-retrieving per model would let embedding nondeterminism or store state leak
into the comparison.

The prompt comes from backend.retrieve.build_rag_prompt — the exact function
ask_ollama uses in production — rather than a copy, so this measures the system
as it actually runs.

Models are looped OUTER and questions INNER: each model loads once and answers
all seven, which costs three model loads instead of twenty-one. On a machine
where an 8B model needs ~5GB resident and load can fail, that matters.

This script records answers verbatim and does NOT judge them. Quality is
assessed separately against a rubric.

Failure handling
----------------
llama3:8b has been observed failing to load with
"unable to allocate CPU_REPACK buffer" when free memory is tight,a real
constraint of this hardware, not a flake. Each generation retries a few times
and every attempt is recorded, so the retry count is itself a reported result.

Outputs
-------
    verification/eval_llm_results.json    full records, written incrementally
    verification/eval_llm_answers.md      readable, question by question

Usage:
    python verification/eval_llm.py
"""

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv

load_dotenv()

import backend.retrieve as retrieve
from backend.retrieve import build_rag_prompt, query_rag

# Benchmark traffic is not genuine student usage, and query_debug.json is read
# back as evidence for report claims. Keep synthetic queries out of it.
retrieve.DEBUG_PATH = Path(tempfile.mkdtemp()) / "query_debug.json"

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "verification" / "eval_llm_results.json"
OUT_MD = ROOT / "verification" / "eval_llm_answers.md"

BASE = retrieve.OLLAMA_BASE_URL
SESSION = "eval_set"
N_RESULTS = 5
MAX_ATTEMPTS = 4
NS = 1e9

# llama3.2 first (smallest, cheapest to load) and llama3:8b last, so a repeated
# OOM at the end cannot cost us the other two models' results.
MODELS = ["llama3.2", "mistral:7b", "llama3:8b"]

QUESTIONS = [
    ("Q4", "If a word's dictionary root differs from chopping its suffix, "
           "which technique finds the root?"),
    ("Q9", "How can a system tell junk mail from real mail?"),
    ("Q13", "Can the same word be different parts of speech?"),
    ("Q14", "What is the formula for the posterior probability P(A|B)?"),
    ("Q15", "Which probability represents knowledge before observing data?"),
    ("Q20", "What issues does machine learning face?"),
    # Deliberately absent from the corpus: separates refusal from fabrication.
    ("Q-absent", "What is a convolutional neural network?"),
]


def generate(model: str, prompt: str, keep_alive: str = "10m") -> dict:
    """Stream one generation, timing it and recording every attempt.

    Payload mirrors ask_ollama (model/prompt/stream only, no sampling options)
    so the answers are what production would produce. keep_alive is added purely
    to control when the model unloads between sweeps.
    """
    attempts = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.perf_counter()
        ttft = None
        pieces, final = [], {}
        try:
            r = requests.post(f"{BASE}/api/generate", stream=True, timeout=3600,
                              json={"model": model, "prompt": prompt,
                                    "stream": True, "keep_alive": keep_alive})
            if r.status_code != 200:
                attempts.append({"attempt": attempt, "ok": False,
                                 "http_status": r.status_code,
                                 "error": r.text[:500]})
                print(f"        attempt {attempt}: HTTP {r.status_code} — retrying",
                      flush=True)
                time.sleep(5)
                continue
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("response", "")
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                pieces.append(piece)
                if obj.get("done"):
                    final = obj
                    break
        except Exception as exc:                      # noqa: BLE001
            attempts.append({"attempt": attempt, "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"        attempt {attempt}: {type(exc).__name__} — retrying",
                  flush=True)
            time.sleep(5)
            continue

        wall = time.perf_counter() - t0
        ec, ed = final.get("eval_count") or 0, final.get("eval_duration") or 0
        pc, pd = final.get("prompt_eval_count") or 0, final.get("prompt_eval_duration") or 0
        attempts.append({"attempt": attempt, "ok": True})
        return {
            "ok": True,
            "answer": "".join(pieces).strip(),
            "wall_seconds": round(wall, 2),
            "ttft_seconds": round(ttft, 2) if ttft else None,
            "load_seconds": round((final.get("load_duration") or 0) / NS, 2),
            "prompt_tokens": pc,
            "prompt_eval_seconds": round(pd / NS, 2),
            "prompt_tokens_per_sec": round(pc / (pd / NS), 1) if pd else None,
            "output_tokens": ec,
            "eval_seconds": round(ed / NS, 2),
            "tokens_per_sec": round(ec / (ed / NS), 2) if ed else None,
            "done_reason": final.get("done_reason"),
            "attempts": attempts,
            "retries": len(attempts) - 1,
        }

    # Surface the last exception at the top level as well as in `attempts`, so a
    # reader sees why it failed without digging — writing only a generic string
    # here would be the same "silently swallowed the error" problem.
    last = attempts[-1] if attempts else {}
    detail = last.get("error") or (f"HTTP {last.get('http_status')}"
                                   if last.get("http_status") else "unknown")
    return {"ok": False, "answer": None, "attempts": attempts,
            "retries": len(attempts) - 1,
            "error": f"all {len(attempts)} attempts failed; last: {detail}"}


def retrieve_all() -> list:
    """Retrieve once per question and build the shared prompt."""
    print("=" * 78)
    print(f"RETRIEVAL — once per question, shared by all models "
          f"(session '{SESSION}', n_results={N_RESULTS})")
    print("=" * 78)
    items = []
    for qid, question in QUESTIONS:
        res = query_rag(question, session_id=SESSION, n_results=N_RESULTS)
        prompt = build_rag_prompt(question, res["chunks"])
        sources = [f"{m.get('source_file')}#{m.get('chunk_index')}"
                   for m in res["metadatas"]]
        print(f"  {qid:9s} {len(prompt):>6d} chars  {' '.join(sources)}")
        items.append({
            "id": qid, "question": question, "prompt": prompt,
            "prompt_chars": len(prompt),
            "retrieved": [
                {"rank": i + 1, "source_file": m.get("source_file"),
                 "chunk_index": m.get("chunk_index"),
                 "distance": round(res["distances"][i], 4),
                 "text": res["chunks"][i]}
                for i, m in enumerate(res["metadatas"])
            ],
        })
    return items


def cell(answers: dict, model: str, qid: str) -> dict:
    """Return one model x question cell, never a bare null.

    Results are written incrementally so a late crash cannot lose earlier work,
    which means a partial file always contains combinations that have not run
    yet. Those must not look like failures: a null in both cases is ambiguous
    and reads as "the harness silently swallowed an error". Every cell
    therefore carries an explicit status — pending, ok, or failed — and a
    failed one carries its exception.
    """
    res = answers.get((model, qid))
    if res is None:
        return {"status": "pending", "note": "not run yet at the time of writing"}
    return {**res, "status": "ok" if res.get("ok") else "failed"}


def write_json(items: list, answers: dict, started: str) -> None:
    total = len(items) * len(MODELS)
    done = sum(1 for i in items for m in MODELS if (m, i["id"]) in answers)
    failed = sum(1 for i in items for m in MODELS
                 if not (answers.get((m, i["id"])) or {"ok": True}).get("ok"))
    OUT_JSON.write_text(json.dumps({
        "run_at": started,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        # A partial file must say so plainly rather than leaving the reader to
        # infer it from nulls.
        "complete": done == total,
        "progress": {"done": done, "total": total, "failed": failed,
                     "pending": total - done},
        "session_id": SESSION,
        "n_results": N_RESULTS,
        "models": MODELS,
        "prompt_source": "backend.retrieve.build_rag_prompt (production prompt)",
        "method": "retrieval runs once per question; identical chunks and prompt "
                  "sent to every model, so the model is the only variable",
        "questions": [
            {**{k: v for k, v in item.items() if k != "prompt"},
             "answers": {m: cell(answers, m, item["id"]) for m in MODELS}}
            for item in items
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(items: list, answers: dict, started: str) -> None:
    total = len(items) * len(MODELS)
    done = sum(1 for i in items for m in MODELS if (m, i["id"]) in answers)
    L = [
        "# LLM comparison — answers",
        "",
        f"Generated {started}. Session `{SESSION}`, `n_results={N_RESULTS}`.",
        "",
    ]
    if done < total:
        L += [f"> **Incomplete — written mid-sweep.** {done} of {total} "
              f"generations done, {total - done} still pending. Cells marked "
              "_pending_ have not run yet; they are **not** failures.", ""]
    L += [
        "Retrieval ran **once per question**; the identical retrieved chunks and "
        "the identical prompt were sent to all three models, so the model is the "
        "only variable. The prompt is the production one "
        "(`backend.retrieve.build_rag_prompt`), which instructs the model to "
        "answer using only the supplied notes.",
        "",
        "Answers are recorded verbatim and are **not** judged here.",
        "",
        f"Models: {', '.join('`' + m + '`' for m in MODELS)}",
        "",
    ]
    for item in items:
        L += ["---", "", f"## {item['id']} — {item['question']}", ""]
        if item["id"] == "Q-absent":
            L += ["> **Not in the corpus.** Retrieval still returns its five "
                  "nearest chunks; this question separates a model that says the "
                  "notes do not cover it from one that answers anyway.", ""]

        L += ["<details><summary>Retrieved chunks (shared by all models)</summary>", ""]
        for c in item["retrieved"]:
            L += [f"**{c['rank']}. `{c['source_file']}` #{c['chunk_index']}** "
                  f"— distance {c['distance']}", "", "```text", c["text"], "```", ""]
        L += ["</details>", ""]

        L += ["| Model | Wall | TTFT | Prompt tok | Prompt eval | Out tok | tok/s | Retries |",
              "|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            a = answers.get((m, item["id"]))
            if a is None:
                L.append(f"| `{m}` | — | — | — | — | — | — | _pending_ |")
                continue
            if not a.get("ok"):
                L.append(f"| `{m}` | — | — | — | — | — | — | "
                         f"{a.get('retries', 0)} (FAILED) |")
                continue
            L.append(f"| `{m}` | {a['wall_seconds']}s | {a['ttft_seconds']}s | "
                     f"{a['prompt_tokens']} | {a['prompt_eval_seconds']}s | "
                     f"{a['output_tokens']} | {a['tokens_per_sec']} | {a['retries']} |")
        L.append("")

        for m in MODELS:
            a = answers.get((m, item["id"]))
            L += [f"### `{m}`", ""]
            if a is None:
                L += ["_Not run yet — this file was written mid-sweep._", ""]
            elif a.get("ok"):
                L += [a["answer"] or "_(empty answer)_", ""]
            else:
                err = a.get("error", "unknown error")
                attempts = "; ".join(
                    f"attempt {t['attempt']}: "
                    + (f"HTTP {t['http_status']} {t.get('error', '')}"
                       if t.get("http_status") else t.get("error", "?"))
                    for t in a.get("attempts", []) if not t.get("ok"))
                L += [f"_**FAILED** after {a.get('retries', 0)} retries: {err}_", "",
                      f"```text\n{attempts}\n```", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    started = datetime.now().isoformat(timespec="seconds")
    items = retrieve_all()

    answers, total_retries = {}, 0
    for mi, model in enumerate(MODELS):
        print(f"\n{'=' * 78}\n{model}\n{'=' * 78}", flush=True)
        for item in items:
            # Unload after the last question of the last model only; otherwise
            # keep the model warm across its own seven questions.
            last = (item is items[-1])
            ka = "0s" if last else "10m"
            print(f"  [{datetime.now():%H:%M:%S}] {item['id']}", flush=True)
            res = generate(model, item["prompt"], keep_alive=ka)
            answers[(model, item["id"])] = res
            total_retries += res.get("retries", 0)
            if res["ok"]:
                print(f"        {res['wall_seconds']}s  ttft {res['ttft_seconds']}s  "
                      f"prompt {res['prompt_tokens']}tok/{res['prompt_eval_seconds']}s  "
                      f"out {res['output_tokens']}tok  {res['tokens_per_sec']} tok/s"
                      + (f"  ({res['retries']} retries)" if res["retries"] else ""),
                      flush=True)
            else:
                print(f"        FAILED after {res['retries']} retries", flush=True)
            # Write after every answer so a later failure cannot lose earlier work.
            write_json(items, answers, started)
        write_md(items, answers, started)

    write_json(items, answers, started)
    write_md(items, answers, started)

    ok = sum(1 for v in answers.values() if v.get("ok"))
    print(f"\n{'=' * 78}")
    print(f"{ok}/{len(answers)} generations succeeded, {total_retries} retries total")
    for model in MODELS:
        rs = [answers[(model, i['id'])] for i in items if (model, i['id']) in answers]
        good = [r for r in rs if r.get("ok")]
        if good:
            print(f"  {model:12s} mean wall {sum(r['wall_seconds'] for r in good)/len(good):6.1f}s"
                  f"  mean {sum(r['tokens_per_sec'] for r in good)/len(good):5.2f} tok/s"
                  f"  retries {sum(r['retries'] for r in rs)}")
    print(f"\n{OUT_JSON.relative_to(ROOT)}\n{OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
