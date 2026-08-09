# Bucket B — Task 1: Verify untested ingest paths end-to-end

**Scope:** Audio (Whisper) and standalone image (BLIP + OCR) ingest paths.
Both existed in code with passing mocked unit tests, but neither had been
exercised on real material.

**Status:** Complete. Both paths verified end-to-end (ingest → storage →
retrieval → grounded answer).

**Hardware:** HP Victus 16, AMD integrated graphics, CPU-only inference, Windows.
**Models:** Whisper `base`, BLIP `blip-image-captioning-base`, Tesseract OCR,
`all-MiniLM-L6-v2` embeddings, Llama 3.2 3B via Ollama.

---

## 1. Verification criteria

Two tiers were defined before running anything:

1. **Ingest completes** — file routes to the correct extractor, produces text,
   chunks embed, ChromaDB upsert succeeds.
2. **Query returns grounded output** — a question whose answer is in the
   ingested file retrieves chunks from that file, and the generated answer
   draws on those chunks.

"Grounded" here means the answer uses the retrieved chunks — not that it is
flawless. Answer-quality assessment against a pre-stated rubric is deferred to
the systematic evaluation (Bucket C).

---

## 2. Defects found

Four issues surfaced. None were detectable by the existing unit test suite.

| # | Issue | Type | Affected | Resolution |
|---|-------|------|----------|------------|
| 1 | ffmpeg installed but not on the Python process PATH | Environment | Audio | Permanent user PATH fix; no code change |
| 2 | `page_or_slide: None` rejected by ChromaDB (`Expected metadata value to be a str, int, float or bool`) | Code | Audio, image | Key omitted entirely for modalities with no page concept |
| 3 | Re-ingesting a file appended duplicate chunks instead of replacing them (`uuid4` IDs never collide, so `upsert` never replaces) | Code | All modalities | Delete-then-insert on `source_file` + `session_id` before upsert |
| 4 | HTTP uploads stored the temp filename (`tmpqzkuxxqr.pptx`) as `source_file` | Code | `/ingest` endpoint | Original filename passed through in metadata; `path.name` retained as fallback |

### Notes on individual defects

**#1 (ffmpeg)** masked #2 — the pipeline failed at the first stage, so the
metadata defect was not reachable until the environment was fixed.

**#3 (duplication)** was found incidentally during the refusal test: the
top-5 retrieval returned the same chunk at ranks 1–2 and again at ranks 3–4,
because the file had been ingested twice across verification runs. Duplicate
chunks consume top-k slots, degrading retrieval quality rather than merely
wasting storage. The fix repairs existing pollution as well as preventing new:
re-ingesting a duplicated file removes all prior copies (verified 18 → 9).

**#4 (filename provenance)** had two independent consequences: source citations
in the frontend displayed meaningless temp names, and the #3 dedup fix was
silently inert on the HTTP path, since each upload generated a fresh temp name
that the identity filter could never match. Chunks ingested before this fix do
not self-heal and required manual deletion.

---

## 3. Measurements

### Audio path

Test file: 1,278-word self-recorded clip, `.mp3`, single speaker, quiet room.
Transcript: 7,006 characters → 9 chunks.

| Stage | Time |
|-------|------|
| Whisper model load (`base`) | 1.00 s |
| Whisper transcribe | 61.18 s |
| Embedder model load | 3.09 s |
| Embed (9 chunks) | 2.56 s |
| ChromaDB upsert | 0.10 s |
| **Total ingest** | **67.95 s** |

Query (content covered by the transcript):

| Stage | Time |
|-------|------|
| Retrieval total | 4.69 s |
| — ChromaDB query itself | 0.05 s |
| LLM generation (~266 tokens) | 52.55 s (5.1 tok/s) |

Transcription accuracy on this clip measured 97.27% similarity against the
original read script. This is a clean single-speaker control, not a
representative lecture recording; accuracy on noisy and accented material is
a Bucket C question.

### Image path

Test file: rendered lecture slide (Bayes' rule — mixed printed text, handwritten
annotation, and diagram). Produced 1 chunk.

| Stage | Time |
|-------|------|
| BLIP model load | 5.36 s |
| BLIP caption | 13.18 s |
| OCR extract | 1.88 s |
| Embed (1 chunk) | 0.07 s |
| ChromaDB upsert | 0.08 s |
| **Total ingest** | **26.32 s** |

Query: retrieval 4.78 s (ChromaDB query 0.05 s), LLM generation 39.82 s
(~318 tokens).

Consistent with the preliminary prototype findings: retrieval is negligible
relative to generation, and ingest cost is dominated by the vision models for
image material and by Whisper for audio.

---

## 4. Retrieval and answer-quality observations

These are qualitative observations from single test files, not systematic
evaluation.

**Correct refusal on absent content.** Asked about a topic the transcript does
not cover, the model stated that the notes do not address it, quoted the
transcript's actual phrasing, and offered a related concept without claiming it
answered the question. No fabrication. Retrieved distances were uniformly higher
(0.46–0.52) than for covered content (top hit 0.35), so the retrieval signal
itself distinguished the two cases.

**Embellishment on covered content.** The audio answer correctly reproduced the
stemming/lemmatization contrast and its examples, but added a claim about
part-of-speech identification that appears nowhere in the retrieved chunks — the
transcript attributes part-of-speech to a separate tagging stage.

**Chunk-boundary retrieval miss.** The transcript's definition of stemming sat
near a chunk boundary and did not appear in the top-5, so the model reasoned
about stemming without its defining passage retrieved. Note that overlap is
already implemented (audio: 200-word window, 50-word overlap), so the open
question for Bucket C is whether the current overlap is *sufficient*, not
whether overlap exists.

**ASR error propagation.** Whisper transcribed "lemma" as "Lem"; the error
passed through chunking and retrieval and surfaced verbatim in the generated
answer.

**BLIP caption near-useless on a text-heavy slide.** The caption returned was
`a poster with the words raves and probait` — "Bayes" and "probability"
both misread. Retrieval value on this slide came almost entirely from OCR.

**OCR noisy but substantive.** Definitions survived legibly, but mathematical
notation degraded: `P(A|B)=[P(B|A)*P(A)]/P(B)` was rendered as
`P(A|B)=IP(B|A)*P(A)I/P(B)`, "Bayes" became "Baves", and bullet glyphs became `¢`.

**Factual error traceable to degraded input.** The image answer correctly
defined prior and posterior from the OCR'd text but identified the posterior as
`P(B|A)` — that is the likelihood, not the posterior. The mangled notation
plausibly contributed: with the formula's structure degraded, the model
reconstructed the term-to-notation mapping incorrectly rather than flagging
uncertainty.

This is the same failure pattern observed in the preliminary prototype
(Appendix B.2): when retrieved material is incomplete or degraded, the model
produces a confident answer from what it has rather than acknowledging the gap.

---

## 5. Methodological observation

The pytest suite (25 tests) passed throughout and detected none of the four
defects. The suite mocks its backends, and each defect lived precisely in the
boundary that mocking replaces:

- A mocked ChromaDB does not enforce the real metadata type constraint (#2).
- A mocked vector store does not accumulate state across calls, so duplicate
  growth is invisible (#3).
- Mocked file handling does not exercise the temp-file path used by the real
  upload endpoint (#4).
- Mocked audio decoding never invokes the ffmpeg subprocess (#1).

Mocked unit tests verify wiring; they cannot verify integration with real
external systems. This is the argument for scheduling end-to-end verification on
real material as its own task rather than treating a passing test suite as
evidence that a path works.

---

## 6. Limitations of this verification

- One test file per modality. Clean single-speaker audio and one legible,
  high-contrast slide.
- Robustness across realistic input variation (noisy recordings, accented
  speech, low-contrast or dense slides, scanned PDFs) is **not** established
  here — that is RQ3 and belongs to Bucket C.
- Answer quality was assessed informally against the source, not against a
  pre-stated rubric with a larger question set.
- Session isolation was not tested; all verification ran under dedicated
  single-purpose session identifiers. Cross-session isolation is verified as
  part of the module switching task.

---

## 7. Deferred items

Recorded here so they are not lost.

| Item | Where it belongs |
|------|------------------|
| `_process_image` runs BLIP and OCR unconditionally, while `_process_pptx` already runs OCR first and falls back to BLIP only when OCR returns under 20 characters. Applying the same logic to the image path would remove ~13 s from image ingest for text-heavy images. | Bucket B or C |
| Chunk size and overlap as a tuned variable, measured by recall@5 on the 20-question evaluation set. Current: audio 200/50, PDF 220/40, PPTX one chunk per slide. | Bucket C |
| Retrieval scoping by material type (lecture note / exam paper / recording) — test whether type-filtered retrieval improves recall@5 before committing to the schema change. | Bucket C |
| Verification scripts require `python -m verification.<name>` from the repository root; direct execution fails to resolve `backend.*` imports. | Document in README |

---

## 8. Artifacts

- `verification/verify_audio.py`, `verification/verify_image.py`,
  `verification/verify_query.py` — path verification harnesses
- `verification/smoke/` — Whisper smoke test and its transcript comparison
- `verification/bucket_b_verification.json` — run records, including the two
  failed runs preceding the passing one
- Commit history — each defect fixed as a single-purpose commit

`backend/tests/` remains the mocked pytest suite; `verification/` holds all
real-material checks. The separation is deliberate and reflects the
methodological point in section 5.

# Bucket B — Task 2: Module switching

**Scope:** Frontend module selection UI plus backend `session_id` routing, so
each study module has its own isolated knowledge base.

**Status:** Complete. Module switching working, session isolation verified.

---

## 1. What was already in place

Backend routing already worked before this task. `/ingest` accepted `session_id`
as a form field, `/query` accepted it in the request body, and every stored chunk
carried it in metadata. The ChromaDB metadata filter was verified at the
storage layer during the preliminary submission.

What was missing was the user-facing half: the frontend exposed session
selection as a free-text box, with no notion of which modules existed, and no
scoping of interface state to the selected module.

This reduced the task from "build session management" to "expose it correctly
and enforce it", which is a smaller piece of work than the original workplan
estimate assumed.

---

## 2. Changes made

| # | Change | Component |
|---|--------|-----------|
| 1 | `GET /sessions` — returns distinct `session_id` values present in the collection with their chunk counts | `backend/main.py` |
| 2 | `/query` now rejects a missing or blank `session_id` with HTTP 400 | `backend/main.py` |
| 3 | Free-text session box replaced with a dropdown of existing modules plus a separate create-module input | `frontend/app.py` |
| 4 | Chat history scoped per module (dict keyed by module rather than one shared list) | `frontend/app.py` |
| 5 | Ingested-files list scoped per module | `frontend/app.py` |
| 6 | Dropdown chunk count refreshes after a successful ingest | `frontend/app.py` |

**Design decision:** the module name *is* the `session_id`. Switching modules
changes only which key the ChromaDB metadata filter uses; there is one
collection throughout. Consequences accepted deliberately: module names cannot
be renamed (renaming would orphan every chunk stored under the old name), and
names are trimmed on creation so whitespace variants cannot produce distinct
modules by accident.

**Module list source:** derived from ChromaDB rather than maintained separately,
so the dropdown reflects modules that genuinely hold material. Newly created
modules hold no chunks yet, so they are held in frontend state and merged with
the server list until something is ingested into them. Note that
`GET /sessions` reads all chunk metadata into memory to compute counts;
ChromaDB has no group-by. This is negligible at project scale and would need
revisiting at larger volumes.

---

## 3. Defect found and fixed: unenforced isolation on the query path

`QueryRequest.session_id` defaulted to `None`, and `query_rag` builds its
filter as `where_filter = {"session_id": session_id} if session_id else None`.
A falsy `session_id` therefore produced **no filter at all**, causing the query
to search every module simultaneously.

With the previous free-text input this was directly reachable: clearing the box
silently turned a per-module query into a cross-module one, with no error and no
visible indication.

The guard was added at the endpoint rather than inside `query_rag`, leaving the
retrieval function's signature unchanged so the verification harnesses that call
it directly were unaffected. The endpoint also strips whitespace, so a
whitespace-only value cannot pass as a truthy filter.

This distinguishes *isolation works* from *isolation is enforced*. The filter
was always correct when supplied; nothing required it to be supplied.

---

## 4. Session isolation verification

Harness: `verification/verify_isolation.py`. Two sessions were populated with
distinct material (`iso_a` ← 9 audio chunks, `iso_b` ← 1 image chunk) and
queried against each other.

| Check | Result |
|-------|--------|
| 1. `iso_a` asked its own question | PASS — 5/5 own-session chunks, 0 leaked |
| 2. `iso_a` asked a question only `iso_b` can answer | PASS — 0 `iso_b` chunks returned |
| 3. `iso_b` asked its own question | PASS — 1/1 own-session |
| 4. `iso_b` asked a question only `iso_a` can answer | PASS — 0 `iso_a` chunks returned |
| 5. `/query` with missing, blank, and empty `session_id` | PASS — HTTP 400 in all three cases |

Checks 2 and 4 are the load-bearing ones. Check 1 only demonstrates that
retrieval works; check 2 demonstrates that the *filter* is what isolates, because
the semantically better match provably existed in the other session and was
still not returned.

For check 2, the Bayes question scored 0.5239 against `iso_b`'s chunk, but
querying `iso_a` returned only its own chunks at distances 0.8861–0.9653. The
store returned distinctly worse matches rather than crossing the session
boundary. Check 4 showed the same asymmetry in the opposite direction: `iso_a`
held a 0.3206 match for the NLP question and none of it leaked into `iso_b`'s
results.

---

## 5. Manual interface testing

Programmatic checks verify the mechanism; they cannot verify that the interface
behaves coherently. A manual pass was run separately, covering module creation,
switching with history in both modules, ingest scoping, and persistence across
browser refresh.

**Confirmed working:** dropdown labels (rendered as `CM3060 (36 chunks)`)
correctly map back to the bare `session_id` for filtering; per-module chat
history is retained rather than cleared, so switching away and back restores the
earlier conversation; modules and their material persist across refresh once
something has been ingested into them.

**Defect found:** after a successful ingest the dropdown continued to display the
pre-ingest chunk count, so the sidebar reported zero chunks while the ingest
confirmation reported chunks stored — two contradictory figures on screen
simultaneously. Queries worked correctly throughout; the fault was display-only.
Fixed by triggering a rerun after successful ingest.

This defect was invisible to programmatic verification, which confirmed the
endpoint returned correct counts. Only interacting with the running interface
surfaced it.

---

## 6. Answer-behaviour observation: adjacency drives refusal quality

Testing two populated modules produced a fourth data point on grounded-answer
behaviour, and the pattern across all four is now clearer.

Asked about image recognition — a topic absent from the module's material — the
model acknowledged the absence, then constructed an answer anyway, reasoning from
retrieved chunks on supervised learning, evaluation metrics, and neural networks
to speculate about how image recognition "could be related" to concepts the notes
did cover. It hedged throughout, but it did not decline.

This contrasts with the audio-path refusal test, where the model declined
cleanly on absent content. The distinguishing variable appears to be **topical
adjacency of the retrieved chunks**:

| Case | Retrieved material | Behaviour |
|------|-------------------|-----------|
| Audio — "rule-based approach in NLP" | Unrelated (spam detection, NLP intro) | Clean refusal |
| Audio — stemming vs lemmatization | Relevant; definition split across a chunk boundary | Grounded, one unsupported addition |
| Image — prior vs posterior | Single chunk, notation degraded by OCR | Grounded, one factual error |
| Module test — "what is image recognition" | Topically adjacent (supervised learning, neural networks) | Acknowledged gap, then speculated |

When retrieval returns clearly unrelated material, the model refuses. When it
returns *adjacent* material that feels as though it should connect, the model
builds a bridge instead of declining. This is the same failure identified in the
preliminary report's improvements section, now with a second and stronger
instance on different material.

It supports testing the proposed mitigation — prompting the model to refuse
rather than approximate when retrieved chunks do not directly address the
question — as an explicit Bucket C variable rather than an assumed improvement.

**Incidental latency figures** from the same session, consistent with earlier
measurements: roughly 40 s, 10 s, 20 s and 22 s per answer, with the shortest
being a refusal. Generation time continues to scale with response length.

---

## 7. Process observation: a workaround committed before the cause was diagnosed

During manual testing the server failed to start, and the intra-package imports
across `backend/` were flattened (`from models.whisper_model import …` in place
of `from backend.models.whisper_model import …`) to resolve it. This was
committed.

The actual cause was the working directory: `uvicorn backend.main:app` resolves
`backend` as a package and must be run from the repository root, not from inside
`backend/`, where the package is not visible from within itself. The import
error appeared to indicate a path problem in the code, so the apparent fix was
to strip the prefix.

The consequence was that every entry point — the server and all four
verification harnesses — then required `PYTHONPATH=backend` to run at all. For a
project that has to be startable by a supervisor, a marker, and user-study
participants, that is a meaningful usability regression, and it reached the
repository before being noticed.

Reverted across four files (8 import lines). The mocked test suite is the
strongest confirmation: all 25 tests patch their targets by string path
(`patch("backend.ingest", …)`), so they would fail if module paths did not
match. All 25 pass, and all four harnesses run again with plain
`python -m verification.<name>` from the root with no environment variables.

The general point: a change that makes an error message disappear is not
necessarily a change that addresses its cause.

---

## 8. Known limitations

**No transactional isolation between concurrent ingest and query.** The system
assumes single-user, sequential operation, consistent with its local-execution
design. Concurrency is largely prevented by the interface — Streamlit blocks
during ingest and query — but is reachable across two browser tabs or a shared
backend. The specific consequence is that the delete-then-insert deduplication
introduced in task 1 briefly removes a file's existing chunks before inserting
replacements, so a query issued during a re-ingest of the same file may
transiently miss material that exists both before and after. Not addressed:
transactional ingest falls outside the single-user deployment model. Noted so
that user-study participants are given individual local instances rather than a
shared backend.

**Module names cannot be renamed.** Renaming would orphan every chunk stored
under the previous name.

**Chat history is in-memory only.** History is scoped per module but does not
survive a browser refresh. Persistence was not built and is not claimed.

**Interface latency on interaction.** Switching modules or refreshing takes
roughly five seconds, because each Streamlit rerun re-executes the whole script,
issuing a `/health` call (which itself has a three-second Ollama timeout) and a
`/sessions` call that scans all chunk metadata. Caching both calls with a short
TTL is the standard remedy, but it interacts with the post-ingest refresh fix
above and so was deliberately kept separate. Candidate for the refinement slot
following the mid-development user check, since interface friction would
otherwise colour that feedback.

---

## 9. Artifacts

- `verification/verify_isolation.py` — isolation harness
- `verification/bucket_b_verification.json` — record 5 holds the isolation run,
  with every retrieved chunk's `session_id`, `source_file` and distance

# Bucket B — Task 3: Wire sentiment classifier into the query pipeline

**Scope:** Run sentiment classification on student questions and persist a
per-topic affect signal for later aggregation.

**Status:** Complete. The default classifier was found unsuitable on evidence and
replaced; the pipeline itself is model-agnostic.

---

## 1. Defect found before any wiring: `device=0`

`sentiment_model.py` initialised its pipeline with `device=0`, which selects the
first CUDA device. The development machine has integrated AMD graphics and a
CPU-only PyTorch build (`2.11.0+cpu`, `torch.cuda.is_available()` is `False`), so
the call could never have succeeded — it would have failed inside PyTorch on
first use with a "not compiled with CUDA" error. Changed to `device=-1`, which
the file's own comment already documented as the CPU setting.

This is the **fifth** defect in the same pattern (after the ChromaDB
`None`-metadata constraint, chunk duplication, temp-filename provenance, and
ffmpeg PATH). `test_sentiment.py` patches `transformers.pipeline` entirely, so
`device=0` was never evaluated by anything real across four passing tests. The
sentiment path had genuinely never executed outside mocks.

Measured after the fix: warm load 0.40 s, inference 21.4 ms per short string. The
first call took 31.4 s, but that included a one-time ~268 MB weight download and
is not a steady-state figure.

---

## 2. Design decisions

**Classify the question text only**, not the student's reaction to an answer.
Simplest defensible unit, and it is the text the pipeline already has.

**Topic granularity is the module.** `topic` is set to `session_id`. Finer-grained
topic extraction would require either another model or a keyword pass, and was
judged out of scope. `topic` is nonetheless stored as its own field rather than
aliased to `session_id`, so a finer definition can replace it without a schema
change.

**JSON storage, not SQLite.** The confidence tracker is a feature, not a
contribution; aggregation is a `defaultdict` over a few hundred records, and SQL
buys nothing at that volume. The existing `query_debug.json` pattern is reused,
including its corrupt-file guard. The tradeoff accepted: whole-file read and
rewrite per append, which is O(n) and would need revisiting over a term of heavy
use.

**Generic `signal_type` field from the start.** Recorded so that quiz-performance
signals could later be logged without a schema migration — which is exactly what
happened in task 4.

**Classification runs at the endpoint, not inside `query_rag`.** The endpoint has
both the question and the retrieval metadata, and keeping it out of `retrieve.py`
means the verification harnesses that call `query_rag` directly do not emit
signals as a side effect.

Record shape:

```json
{"timestamp": "...", "session_id": "CM3060", "topic": "CM3060",
 "signal_type": "sentiment", "value": "neutral", "score": 0.94,
 "question": "...", "retrieved_sources": ["L6.pptx"]}
```

`retrieved_sources` is stored although nothing currently reads it: the data is
already available from retrieval, and recording it preserves the option of
aggregating at file granularity later without re-running anything.

---

## 3. Failure isolation

A signal-logging failure must never fail a query — the student should get their
answer regardless. This was proven rather than assumed, by injecting three
distinct failures:

| Injected failure | Result |
|------------------|--------|
| `predict()` raises `RuntimeError` | HTTP 200, answer returned |
| `log_signal()` raises `OSError` | HTTP 200, answer returned |
| `predict()` returns a malformed dict | HTTP 200, answer returned |

Each logged a warning and continued.

A module-level singleton loads the model once per server lifetime rather than per
request. Confirmed in live runs: 0.69 s on the first request, 0.03 s thereafter.

---

## 4. Principal finding: the default classifier cannot detect what it was wired in to detect

The first live run exposed the problem immediately.

| Question | SST-2 result |
|----------|-------------|
| "What is natural language processing used for?" (neutral) | NEGATIVE 0.9961 |
| "I still don't understand this at all" (frustrated) | NEGATIVE 0.9990 |

A gap of 0.003 between a plainly neutral factual question and genuine
frustration. There is no signal here to threshold on.

The cause is the model, not the wiring.
`distilbert-base-uncased-finetuned-sst-2-english` is trained on SST-2, a binary
movie-review corpus. It has **no neutral class**, so every input is forced into
POSITIVE or NEGATIVE, and it emits a high-confidence score either way. Factual
questions contain no positive sentiment markers, so they land reliably NEGATIVE.

The practical consequence: a confidence tracker built on this signal would report
every module as low-confidence regardless of how the student actually felt. The
plumbing was correct and the schema was right; the instrument was wrong.

This was not an unanticipated failure. The preliminary report deferred the choice
explicitly — "sentiment classification will use a Hugging Face transformer, with
the specific model deferred pending empirical comparison". This is the first data
point of that comparison, and it rejected the default.

### Replacement: a three-class model

`cardiffnlp/twitter-roberta-base-sentiment-latest` returns
negative / neutral / positive. Tested standalone before adoption, on five inputs
chosen to include the cases the incumbent failed:

| Input | Expected | 3-class | SST-2 |
|-------|----------|---------|-------|
| "What is natural language processing used for?" | neutral | neutral 0.934 | NEGATIVE 0.996 |
| "I still don't understand this at all" | frustrated | negative 0.792 | NEGATIVE 0.999 |
| "That explanation finally made it click, thanks" | positive | positive 0.879 | POSITIVE 1.000 |
| "what is lemmatization" | no affect | neutral 0.597 | NEGATIVE 0.991 |
| "why is this so confusing" | frustrated | negative 0.751 | NEGATIVE 0.999 |

Neutral questions now carry negative mass of 0.036 and 0.375 against 0.75–0.79
for genuine frustration — a gap wide enough to threshold on, where the incumbent
offered 0.003.

The replacement's confidences are lower across the board (0.60–0.93 versus
0.99+). This is the model being calibrated rather than worse: SST-2's uniform
>0.99 was false confidence produced by having only two bins to sort every input
into.

Confirmed end-to-end on two unseen inputs (not among the five it was selected
against): a neutral question recorded `neutral 0.942`, a frustrated one recorded
`negative 0.855`.

### Cost

| | 3-class (RoBERTa-base) | SST-2 (DistilBERT) |
|---|---|---|
| Warm load | 1.27 s | 0.40 s |
| Mean inference | 41.3 ms | 21.4 ms |

Roughly twice the cost, as expected for a model twice the size. Against 40–60 s
of Ollama generation this is under 0.1% of request time — irrelevant in practice.

### Threshold guidance for aggregation

The weakest case is `"what is lemmatization"` — neutral 0.597, negative 0.375.
Terse, affectless text is genuinely ambiguous to this model, and a blunter phrasing
could tip it to negative. Since terse queries are the most common real input, the
aggregation step **must not treat a bare `negative` label as evidence of
struggle**; a score threshold of roughly 0.6 or above should be required.

---

## 5. Secondary finding: a passing test can be testing a fiction

Two of the four sentiment tests asserted `"POSITIVE"` and `"NEGATIVE"` — strings
the replacement model can never emit. Both continued to pass, because they assert
against their own mocked return values. Green, but no longer evidence that the
wrapper handles real output.

One test's docstring also claimed it "should correctly identify NEGATIVE
sentiment". It does no such thing: the pipeline is mocked, so it verifies only
that the wrapper passes a label through untouched. The wording implied
model-quality coverage that does not exist.

Both were corrected to the real vocabulary. This is the same blind spot that
produced the `device=0` defect, in a second form: mocked tests can drift silently
out of correspondence with the thing they claim to cover, and remain green
throughout.

---

## 6. Limitations

- **Sentiment polarity is a proxy for confidence, not a measure of it.** A
  student writing "I still don't get backpropagation" is expressing confusion,
  which is not the same construct as negative sentiment. D'Mello and Graesser
  report 68–78% accuracy for text-based affect classification, and that was on
  affect-rich tutoring dialogue rather than terse database-style queries. The
  signal should be read as a rough indicator.
- **The signal will be sparse.** Most study questions carry no affective content
  at all. Meaningful signal arises only when a student explicitly expresses
  frustration or satisfaction, which is a minority of interactions.
- **The replacement model is trained on tweets**, a domain mismatch with study
  questions — though arguably no worse a mismatch than movie reviews.
- **Module-level granularity.** The tracker can report that a student appears to
  be struggling with a module, not which parts of it.
- Model selection was made against five hand-chosen inputs plus two live
  confirmations. This is a sanity check, not a systematic comparison.

---

## 7. Deferred

| Item | Where |
|------|-------|
| Explicit self-rating control ("Got it / Partly / Still lost") after each answer, as a second channel. Would give direct confidence data rather than an inferred proxy, and comparing the two would test whether text-based affect detection works in a terse query interface at all — a direct test of D'Mello and Graesser's applicability outside tutoring dialogue. | Bucket F if time allows |
| Systematic comparison of sentiment models against a labelled set of real student queries, rather than five hand-chosen examples | Bucket C |
| Audit remaining test docstrings for the overclaim pattern found in section 5 | Any |

# Bucket B — Task 4: Quiz generator

**Scope:** Structured-prompt extension of the RAG path producing multiple-choice
questions from ingested material, with deterministic marking and per-question
signal logging.

**Status:** Complete and working. Question *correctness* is materially limited by
source text quality — see sections 4 and 5.

---

## 1. Design decisions

**Multiple-choice only.** Free-text answers would require the language model to
grade them, which reintroduces exactly the failure mode documented in tasks 1–2:
the model producing confident but incorrect judgements. With MCQ, the model
supplies a `correct_index` at generation time and marking is a plain integer
comparison in Python. Open-ended questions were considered and deferred; they
have pedagogical value as retrieval practice but cannot be marked reliably, and
they would need a second generation path and result flow.

**Configuration form rather than conversational setup.** Generation is a single
request carrying topic, question count and module, rather than a multi-turn
exchange. On CPU-only hardware each model call costs 20–60 seconds, so a
back-and-forth setup would cost minutes before the first question appeared.

**One generation call, not one per question.** Five questions in a single call
takes roughly the same wall-clock time as one, since generation cost scales with
output tokens rather than request count.

**Marking in Python, not by the model.** The model is never asked to grade. This
was the point of choosing MCQ, and it is worth stating explicitly because the
reliability problem did not disappear — it moved upstream (section 4).

**Retrieval reuses `query_rag`.** Quiz generation retrieves on the supplied topic
through the same function the chat path uses, so questions are grounded in the
same chunks a chat answer would draw on, and session isolation applies unchanged.

**Separate generation helper.** Quiz generation does not reuse `ask_ollama`,
whose prompt wraps the request in "answer using ONLY the notes" instructions that
conflict with an instruction to emit JSON. A separate non-streaming call at
temperature 0.2 is used instead.

---

## 2. Structured output reliability

The main technical risk was whether Llama 3.2 3B would emit parseable JSON
consistently. A lenient parser was written to handle failure: strip markdown
fences, tolerate trailing commas and surrounding prose, salvage individual items
from a malformed array, and return whatever validates rather than failing the
whole request.

Tested against nine malformed shapes:

| Input | Result |
|-------|--------|
| Clean array | 2 items |
| Markdown fences | 1 item, warns |
| Prose preamble and postamble | 1 item, warns |
| Trailing comma | 2 items |
| One broken item among good ones | 2 kept, 1 rejected with reason |
| `correct_index` out of range | Bad item rejected, others kept |
| `source_chunk_index` out of range | Item kept, provenance dropped |
| Unparseable array (missing comma) | 2 salvaged individually |
| No JSON at all | 0 items, warns |

A bad `source_chunk_index` degrades to "no source shown" rather than discarding
the question: provenance is desirable, `correct_index` is load-bearing.

**In practice the leniency was not needed.** Every live generation returned a
clean JSON array with no fences or preamble. Parsing cost is negligible
(0.0001 s against 61.5 s generation). The model's structured-output reliability
on this task was better than anticipated; the parser remains as insurance and
because malformed output cannot be ruled out across a larger sample.

---

## 3. Measurements

| Stage | Time |
|-------|------|
| Generation, 5 items (Ollama, Llama 3.2 3B) | 61.50 s |
| Parsing | 0.0001 s |
| Marking | Immediate (integer comparison) |

Generation dominates, consistent with every other measurement in this project.
A five-item quiz costs roughly the same as one long chat answer.

---

## 4. Principal finding: question correctness tracks source text quality

Three quizzes were generated and every item checked by hand against the source
chunk shown in its provenance panel.

| Quiz | Source | Source text quality | Items | Hard keying errors | Ambiguous |
|------|--------|--------------------|-------|--------------------|-----------|
| 1 | NLP lecture audio | Clean transcript | 5 | 1 | 0 |
| 2 | Bayes slide image | OCR-degraded | 5 | 2 | 1 |
| 3 | NLP lecture audio | Clean transcript | 5 | 0 | 1 |

A *hard keying error* means the item's `correct_index` points at an answer the
source does not support, so a student answering correctly is marked wrong. An
*ambiguous* item has more than one defensible answer with only one keyed.

**Clean-source quizzes: 9 of 10 items correctly keyed.**
**OCR-source quiz: 2 of 5 items correctly keyed.**

### The OCR-sourced failures in detail

The retrieved chunk for the Bayes slide reads, in part:
`Baves's rule and probabilistic inference … LIKEL\HOOD | PR [OR …
P(A|B)=IP(B|A)*P(A)I/P(B)`

Two of the three failures are directly attributable to that damage:

- One question asked for a formula the slide never gives, and keyed the answer
  to `IP(B|A)P(A)` — a fragment of the mangled `P(A|B)` formula, with the OCR's
  corrupted bracket characters embedded in it. The question is unanswerable and
  the key is an OCR artifact.
- One question asserted that the posterior probability is "a form of
  likelihood". This is false — the posterior is P(A|B), the likelihood is
  P(B|A), and the slide defines them as distinct terms. The definitions survived
  OCR but their mapping to notation did not, and the model reconstructed the
  relationship incorrectly.

The third (ambiguous) item paraphrased two different slide sentences into two
options, both defensible.

### The compounding failure chain

This is the same degradation documented in task 1, now propagating one stage
further:

> BLIP caption uninformative on a text-heavy slide → the image path depends
> entirely on OCR → OCR mangles mathematical notation → chat answers built on
> those chunks contain factual errors (the prior/posterior conflation recorded in
> task 1) → quiz items generated from the same chunks inherit the damage and are
> mis-keyed.

A single ingest-stage weakness degrades every downstream feature built on it.
This is the clearest demonstration in the project so far that retrieval-grounded
output is bounded by ingest quality, and it argues for treating ingest fidelity
as a first-class evaluation target rather than a preliminary step.

### Why this matters for the confidence tracker

Quiz results were intended as the *objective* signal balancing the noisier
sentiment channel. If a portion of items are mis-keyed, a low score may reflect
bad questions rather than weak understanding — and the tracker cannot distinguish
the two. The clean-source rate (9/10) is usable; the OCR-source rate (2/5) is
not. Aggregation should therefore be read with source modality in mind, and this
limitation stated wherever quiz-derived confidence is reported.

### Not detectable downstream

Marking is faithful to `correct_index`, and `correct_index` is what is wrong.
No validation in the pipeline can catch this, because the pipeline has no
independent notion of the right answer. The mitigation implemented is
transparency rather than correction: the source chunk is displayed alongside
every marked question, so a student who disagrees with a verdict can check the
material themselves.

---

## 5. Secondary finding: questions test recall, not understanding

Across all three quizzes the generated items ask for surface facts — what a stage
is called, which tool derives a word stem, what a term stands for. None ask why a
stage is ordered as it is, when one technique is preferable to another, or what
fails if a step is omitted. A student could answer most items correctly from a
single skim without conceptual grasp.

Two causes:

**The chunk is the wrong unit for conceptual questions.** A ~200-word chunk
contains statements of fact, not arguments spanning a topic. Questions requiring
synthesis across chunks are not available to a generator that sees only the
retrieved set for one topic.

**The prompt does not ask for difficulty.** It requests multiple-choice items
from the supplied chunks, so the model produces the most readily extractable
thing, which is definitions.

This is worth recording because it was **predicted in the literature review
before any code was written**. Chapter 2 notes that Karpicke and Roediger's
retrieval-practice evidence comes from paired-associate vocabulary learning, and
that "retrieving the definition of a term such as 'gradient descent' is closer to
vocabulary recall, but understanding how the algorithm minimizes loss is not".
The generator has landed squarely at the vocabulary-recall end of that gap. The
retrieval-practice benefit claimed for the feature therefore rests on the part of
the literature whose generalisation to conceptual learning the review already
flagged as open.

Whether prompting explicitly for application-level questions improves this is
testable, but improvement could not be verified without a difficulty rubric, and
Llama 3.2 3B on CPU has limited headroom. Recorded as an improvement rather than
attempted.

---

## 6. Signal schema: cross-type aggregation

Quiz results are logged through the same `log_signal` path as sentiment, one
record per question, with `signal_type: "quiz"` and `value: "correct"` /
`"incorrect"`. No schema change was needed — which is what the generic
`signal_type` field was introduced for in task 3.

One correction was required before the schema was sound. Quiz records initially
set `topic` to the quiz's subject string while sentiment records set it to
`session_id`, so the two signal types meant different things by the same field
and could not be grouped together. `topic` now mirrors `session_id` for both, and
the quiz's own subject is kept in a separate `quiz_topic` field so the
granularity is not lost. The invariant — every signal type must mean the same
thing by `topic` — is now documented in `signals.py` alongside the schema.

`quiz_topic` is absent from sentiment records, so any reader must use `.get()`
rather than direct indexing.

Marking happens in the frontend, so a thin `POST /quiz/signals` endpoint exists
for it to record outcomes rather than importing `backend.signals` and writing to
the signals file behind the API.

---

## 7. Defect found in manual testing

Formula text in answer options rendered incorrectly: `P(A|B)=[P(B|A)*P(A)]/P(B)`
displayed as `*P(B|A)P(A)/P(B)`, because Streamlit's markdown renderer consumed
the asterisks as emphasis markers. Cosmetic but misleading on exactly the
material where precision matters most. Fixed.

As in task 2, this was invisible to programmatic verification — the API returned
correct strings throughout; only the rendered interface showed the corruption.

---

## 8. Limitations

- Question correctness was assessed by hand across three quizzes (15 items). This
  is a small sample and the author both knows the source material and built the
  system, so the judgement is not independent.
- Item quality was judged for keying correctness and ambiguity only, not against
  a pedagogical rubric.
- Only two source modalities were sampled (clean audio transcript, OCR-degraded
  image). PDF and PPTX sources were not tested and may fall between the two.
- Difficulty is not controlled. A difficulty selector was considered and left
  out: the model has no reliable basis for self-assessing difficulty, and
  offering the control would imply a calibration that does not exist.
- Generated quizzes are not persisted. Closing the browser loses them.

---

## 9. Deferred

| Item | Where |
|------|-------|
| Open-ended questions as unmarked retrieval practice, with the source chunk revealed for self-assessment | Bucket F if time allows |
| Prompting for application-level rather than definitional questions; would need a rubric to evaluate | Bucket C or F |
| Measuring quiz keying accuracy on PDF and PPTX sources to complete the source-quality picture | Bucket C |
| Applying the PPTX path's OCR-first logic to `_process_image`, which would not fix OCR quality but would stop uninformative BLIP captions entering chunks | Bucket B or C |

# Bucket B — Task 5: Flashcard generator

**Scope:** Term-and-definition flashcards generated from ingested material, with
card-by-card reveal and self-rated recall logged as a third signal type.

**Status:** Complete. Two new failure modes found, and the source-quality finding
from task 4 confirmed on a second feature and a third modality.

---

## 1. Design decisions

**Term on the front, not a question.** Questions are the quiz's job; putting them
on flashcards too would duplicate the feature. Term-front also maps directly onto
the paired-associate paradigm that Karpicke and Roediger's evidence is strongest
for.

**Self-rated recall, two options.** After revealing a definition the student
marks *Got it* or *Didn't know*. Three options were considered and rejected:
"partly" is ambiguous when the answer is a definition the student either recalled
or did not.

This is the third signal type, and deliberately the only one that is neither
inferred nor model-generated. Sentiment is inferred from question wording and is
a weak proxy (task 3); quiz results are objective but depend on the model keying
questions correctly, which task 4 showed is unreliable on degraded source. A
self-rating is reported directly by the student and nothing in the pipeline can
corrupt it. Its limitation — that self-assessment can be inaccurate — is a
property of the construct, not a defect in the system, and is easily stated.

Adding this control does not weaken the model-orchestration premise: the cards
themselves are still generated by the language model from retrieved chunks, and
all five pre-trained models remain in the pipeline. What was added is a feedback
mechanism, not a replacement for a model.

**Spaced repetition remains out of scope**, per the workplan. No scheduling, no
intervals, nothing persisted between decks.

**Parsing helpers refactored rather than duplicated.** The quiz parser had its
validator hard-wired into the middle of it, so it was split into a generic
`_parse_items(raw, validate)` with per-type validators and shared provenance
handling. Quiz parsing was verified byte-identical afterwards by re-running all
nine original malformed-input cases against exact expected item counts.

---

## 2. Defect found: truncated JSON discarded every valid item

The first live generation returned HTTP 200 with zero cards. The model had
written five complete, valid card objects and then been cut off before the
closing `]`.

The parser required both an opening and a closing bracket, so one missing
character discarded five good cards. This directly contradicted the requirement
it was written to satisfy — return the valid items rather than failing the whole
request — and **the same bug was live in the already-committed quiz endpoint**.

Fixed by falling through to object-level salvage when the array is unterminated
or absent. Replaying the exact truncated output now recovers all five cards.
Three regression cases were added: truncated array, truncated mid-object (drops
the partial, keeps the rest), and bare objects with no array at all.

Truncation is not an exotic failure — it occurs whenever generation hits an
output limit, which becomes more likely as decks and quizzes grow. The nine
synthetic cases written in task 4 did not include it, so it passed testing and
would have surfaced in ordinary use.

This is the sixth defect in the "passed testing, failed on real input" pattern,
and it carries a variation worth noting: the earlier cases involved mocked
backends failing to enforce real constraints, whereas here the tests were
hand-written failure cases. A lenient parser tested only against the failures its
author thought to imagine is a parser tested against that author's imagination.

---

## 3. Measurements

| Stage | Time |
|-------|------|
| Generation, 5 cards (Ollama, Llama 3.2 3B) | 17.69 s |
| Parsing | < 0.1 ms |

Substantially faster than the quiz's 61.5 s for five items, because a card is a
term and a definition while a quiz item is a question plus four options plus an
answer key. Generation cost scales with output tokens, as everywhere else in this
project.

---

## 4. Source-quality comparison

Three decks were generated on the three available source modalities and every
card checked by hand against the source chunk shown in its provenance panel.

| Source | Extraction path | Cards requested | Distinct terms | Faithful | Notes |
|--------|----------------|-----------------|----------------|----------|-------|
| PDF (Mitchell ch.1) | `pdfplumber`, direct text | 5 | 5 | 5 | Two definitions vague but not wrong |
| Audio (IBM NLP explainer) | Whisper ASR | 10 | 5 | 5 | 50% duplication; ASR errors carried through |
| Image (Bayes slide) | BLIP + Tesseract OCR | 3 | 5 | 3 | One card built from corrupted OCR fragment |

**This confirms the task 4 finding on a second feature and adds the missing third
modality.** Output quality tracks extraction quality in the same order:
PDF > audio > image. PDF has no transcription layer at all, audio introduces ASR
errors, and OCR introduces character-level corruption.

### The image-sourced failure

Three of the five cards — Prior, Posterior, Likelihood — were faithful. The
remaining two were a near-duplicate with a circular definition ("the posterior
probability of a variable A given prior knowledge about it") and one card headed
*Marginality*, defined as: "A form of its marginality (P(A)) and being given the
data is 0 knowledge of how likely B is to be True, given that A is True."

That is a lightly cleaned-up restatement of the OCR text
`A form of its marginality (P(A)) and being TR E gaven thed Is 0 knowledge of how
likely B is to be True`, including the meaningless `Is 0` fragment. "Marginality"
is not a term the slide defines; it is a fragment of a sentence about P(A) that
the model promoted into a headword.

### ASR errors continue to propagate

The audio deck produced cards headed **"Lematization"** defining the root as the
**"Lem"** — Whisper's mis-transcription of *lemmatization* and *lemma*. These
errors have now surfaced in three separate features: a chat answer (task 1), a
quiz item (task 4), and flashcard terms here. An ingest-stage transcription error
reaches the student verbatim through every downstream feature.

One card also drifted from source: *Entity recognition* was defined as
identifying "names, locations, and organizations", whereas the transcript
describes *named* entity recognition with the examples Arizona → US state and
Ralph → person's name. The definition is generic knowledge about NER rather than
what the source said — the same mild embellishment pattern recorded in tasks 1
and 3.

### Flashcards degrade more gracefully than quizzes

On the same OCR-damaged slide, the quiz produced 2 of 5 correctly keyed items
while flashcards produced 3 of 5 faithful cards, and the flashcard failures were
incoherent rather than actively misleading.

The mechanism is worth stating: a flashcard requires the model to **extract** a
definition already present in the text, while a quiz item requires it to
**invent** three plausible distractors and correctly judge which of four options
the source supports. The quiz task involves judgement; the flashcard task
involves restatement. **Generation tasks requiring judgement degrade faster on
noisy source than tasks requiring extraction.**

This matters practically: a mis-keyed quiz tells a student they are wrong when
they are right, whereas an incoherent flashcard is visibly nonsense and can be
disregarded.

---

## 5. New failure mode: duplication when card count exceeds available concepts

The audio deck was generated with the default ten cards and returned ten, but
only **five distinct terms** — Tokenization ×2, Stemming ×3, Lematization ×3,
Part-of-speech tagging ×1, Entity recognition ×1 — with duplicate definitions
reproduced word for word.

The retrieved chunks for that topic contain roughly five definable concepts. Asked
for ten, the model repeated rather than stopping. Nothing in the pipeline
prevents this: there is no deduplication of generated terms, and the model has no
way to report that the source supports fewer cards than requested.

This is independent of source quality. It is a mismatch between a user-specified
target and what the retrieved material can actually support, and it will occur on
clean sources whenever the requested count exceeds the available concepts.

Image-sourced modules are structurally worst affected: one image yields one
chunk, so a module built from a single slide has very little material for any
deck size.

Two mitigations are available and neither was implemented: deduplicate terms
after parsing and return fewer cards with an honest count, or widen retrieval when
a larger deck is requested. The first is preferable — reporting "5 distinct
concepts found" is more useful than silently padding.

---

## 7. Interface defects found by manual testing

**Every interaction reloaded the page.** Streamlit re-executes the whole script on
each button click, so each reveal re-ran `check_health()` (an HTTP call with a 3 s
Ollama timeout) and `fetch_sessions()` (a full metadata scan). Measured at
~12.3 s per reveal click.

Caching both calls (15 s TTL for health, 30 s for sessions) reduced steady-state
reveal cost to ~46 ms — a 267× improvement. The session cache is cleared
explicitly after a successful ingest, since otherwise the stale-chunk-count
regression fixed in task 2 would return.

**A larger cause was found underneath the cache.** Every `localhost` HTTP call on
this machine cost roughly 2 s of name resolution: IPv6 `::1` is attempted first
and times out before falling back to IPv4, and nothing listens on `::1`.
`/health` paid it twice — once frontend-to-backend, once inside the backend
calling Ollama. Switching to `127.0.0.1` removed it.

This has a consequence for earlier measurements: **every latency figure recorded
before this change includes approximately 2 s per Ollama call of avoidable
name-resolution overhead.** Small relative to a 40–60 s generation, but it means
reported figures were not measuring only what they appeared to.

**Revealing a card jumped the interface back to the Chat tab.** `st.tabs` holds
the active tab client-side, so an explicit `st.rerun()` recreates the tab strip
and drops the user on the first tab. Every reveal and every rating triggered one.
Fixed by rendering into `st.empty()` slots and redrawing in place, which also
removes a server round trip per click. The same pattern was applied to quiz submit
and deck reset; the ingest rerun was deliberately left alone, since it is what
clears the cache and refreshes the dropdown count.

A related bug was caught while fixing this: clearing an `st.empty()` slot from
inside its own container destroys the container mid-write and corrupts the
element tree. Resolved by returning the click from the draw function and
resetting in the caller.

**All three of these were invisible to programmatic verification** — the API
returned correct data throughout. Together with the stale dropdown count (task 2)
and the markdown asterisk corruption (task 4), that is now five interface defects
found only by using the running application. Interface behaviour requires human
testing as a distinct verification activity, not as a supplement to endpoint
checks.

---

## 8. Limitations

- Card quality was assessed by hand across three decks (15 cards) by the author,
  who both knows the source material and built the system. Not an independent
  judgement.
- The audio source used was a professionally produced explainer video: single
  clear speaker, scripted delivery, no background noise. It represents an upper
  bound for ASR quality rather than a typical self-recorded lecture, so the
  audio row of the comparison is optimistic.
- Only one topic was sampled per source.
- Self-rated recall is **self-reported confidence, not measured confidence**.
  Students can believe they know material they cannot reproduce — the
  illusion-of-competence effect that Karpicke and Roediger's work speaks to
  directly. Any confidence figure derived from this signal must be reported as
  self-assessment.
- Decks are not persisted; closing the browser loses them.

---

## 9. Deferred

| Item | Where |
|------|-------|
| Deduplicate generated terms and return an honest count when the source supports fewer cards than requested | Bucket B or F |
| Establish whether the card-count parameter is ignored by the model or not reaching the prompt | Before user study |
| Widen retrieval for larger requested deck sizes | Bucket C |
| Compare self-rated recall against inferred sentiment on the same modules, as a test of whether text-based affect detection transfers to a terse query interface | Bucket C or F |

# Bucket B — Task 6: Per-topic confidence aggregation

**Scope:** Aggregate the stored signals by topic and surface them in the
interface.

**Status:** Complete. Bucket B's build tasks are finished.

---

## 1. Design decision: the three signal types are not blended

The system now records three kinds of signal, and they differ in what they
actually measure:

| Signal | What it is | Reliability |
|--------|-----------|-------------|
| Quiz result | Whether the student picked the option the model keyed as correct | Objective, but only as good as the keying — task 4 measured 9/10 on clean source and 2/5 on OCR-degraded source |
| Flashcard rating | The student's own judgement of whether they recalled a definition | Honest but self-reported; subject to the illusion-of-competence effect |
| Question sentiment | Affect inferred from the wording of a question | Weakest — a proxy for a construct it does not directly measure, and sparse, since most study questions carry no affect |

Combining these into a single confidence figure was considered and rejected.
A number averaging an inferred proxy, a model-dependent measure, and a
self-report would not correspond to anything, and could not be defended in the
report. They are therefore reported separately, each labelled in the interface
with what it is: **measured**, **self-reported**, and **inferred**. The labelling
is deliberate — the distinction has to be made in the write-up anyway, and
building it into the interface means it cannot quietly be dropped.

---

## 2. The sentiment threshold

Task 3 established that terse factual questions sit close to the neutral/negative
boundary: `"what is lemmatization"` scored neutral 0.597, negative 0.375. Since
terse queries are the most common real input, a bare label is not evidence of
anything.

Sentiment records are therefore only counted when the classifier's score is at
least 0.6; below that they are reported as **inconclusive** rather than silently
folded into a category. The threshold is a named constant carrying its
justification in a comment, so the reasoning survives contact with future
editing.

The threshold is applied symmetrically to all labels. It was initially applied
only to negative records, which was inconsistent: `neutral 0.597` is the same
weak-evidence case as `negative 0.375` and is now treated the same way.

Verified directly, since weak negatives do not occur often enough to wait for:

| Input | Result |
|-------|--------|
| negative 0.855 | counted negative |
| negative 0.600 (exactly at threshold) | counted negative |
| negative 0.599 | inconclusive |
| negative 0.375 (the task-3 case) | inconclusive |
| unknown label | inconclusive |

---

## 3. Implementation notes

`read_signals()` was extracted from `log_signal()` rather than duplicated, so
both paths share one definition of how the signals file is read safely, including
the corrupt-file guard. It also filters out non-dict entries, so a hand-edited
file cannot break every consumer. Verified against a missing file, an empty file,
a corrupt file, and a file containing junk entries — all degrade to an empty list
rather than raising.

An empty module returns zeroed aggregates with `accuracy_pct: null` rather than
`0`, so "no data" cannot be misread as "0% accuracy". The interface shows an
explicit no-activity message rather than zeros.

---

## 4. Interface defect found in testing

The Progress tab's Refresh button crashed with `DuplicateWidgetID`: redrawing
into an `st.empty()` slot created a button with a key already registered in that
script run. The quiz and flashcard redraws had avoided this only incidentally,
because their widget keys carry a card or question index.

The generalisable rule: **any redraw into a slot must vary its widget keys, not
just its content.** Fixed with a pass identifier in the key.

This is the sixth interface defect found only by running the application. As in
tasks 2, 4 and 5, the API returned correct data throughout.

---

## 5. Limitations

**Per-subject percentages fragment across phrasings.** Topics are typed freely
when generating a quiz or deck, so `"NLP"`, `"natural language processing"` and
`"NLP terminology"` accumulate as separate buckets for what is arguably one
subject. Case is normalised, which merges the trivial variants; semantic merging
is not attempted and would require the finer-grained topic extraction deliberately
deferred in task 3. Per-subject figures should be read as per-phrasing figures.

**Quiz accuracy is not comparable across source modalities.** Task 4 and task 5
showed keying accuracy tracks extraction quality, so a low quiz percentage on an
image-sourced module may reflect OCR damage rather than the student's
understanding. The aggregate does not distinguish these, and any confidence
figure derived from quiz results should be read with the source modality in mind.

**Sentiment signal is sparse.** Most questions are affectively neutral, so this
channel contributes little unless a student writes something explicitly
frustrated.

**Aggregation is module-level**, per the task 3 decision. The subject breakdowns
give partial finer granularity, but only for quizzes and decks, and only as
accurately as the typed topic strings allow.

**Modules created but not yet ingested into do not survive a refresh.** They live
in browser session state until something is stored under them, so refreshing
makes an empty module disappear. Not data loss, but it reads as such.

---

## 6. Deferred

| Item | Where |
|------|-------|
| Make empty modules survive a refresh, or prevent their creation until material is added | Bucket F |
| Semantic merging of topic strings, dependent on finer-grained topic extraction | Out of scope |
| Compare self-rated recall against inferred sentiment on the same modules, as a test of whether text-based affect detection transfers to terse query interfaces | Bucket C or F |