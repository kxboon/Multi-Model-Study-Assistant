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