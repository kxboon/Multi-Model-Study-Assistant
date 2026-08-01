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
