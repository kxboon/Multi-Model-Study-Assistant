# Bucket C — Empirical model evaluation

**Scope:** Build an evaluation test set and compare alternative models for three
pipeline slots — speech recognition, embedding, and language generation — on the
target hardware. Addresses RQ2.

**Status:** Complete. All four tasks run.

**Hardware:** HP Victus 16, integrated AMD Radeon graphics with no CUDA or
ROCm-compatible acceleration available, CPU-only PyTorch (`2.11.0+cpu`), 7.7 GB
usable shared RAM, Windows.

---

## 1. The evaluation test set

Twenty questions written by hand against a fixed corpus, each with a known
ground-truth chunk, scored by recall@5 — whether an acceptable chunk appears in
the top five retrieved.

**Corpus (`eval_set`, 61 chunks):** a scanned PDF extract (Mitchell, *Machine
Learning*, ch.1), a lecture audio transcript, and a slide image. Mixed
deliberately, so recall can be broken down by source modality.

**Ground truth is a `(source_file, chunk_index)` pair.** Chunk indices restart per
file, so a bare index is ambiguous — an error that would have silently
mis-scored every question.

**Multiple acceptable chunks are permitted.** The audio chunker uses a 200-word
window with 50-word overlap, so most content appears in two consecutive chunks.
Forcing a single ground truth would have scored correct retrievals as misses.
Eight of twenty questions have more than one acceptable chunk.

Questions were graded for difficulty on purpose. Questions that repeat
distinctive source wording test keyword overlap rather than semantic retrieval,
and a set composed only of those would score near 100% for every model and
discriminate nothing. Roughly half are phrased away from the source.

**One ground truth was corrected before any model comparison.** Q19 ("how are
training examples chosen") was bound to PDF chunk 12, but chunk 13 answers it at
least as directly — the PDF chunker does not overlap, so the passage was split
mid-argument. Both chunks are now acceptable. The correction was made after the
baseline run and before any comparison, so every model is scored against the same
set. Note that this changed the denominator, not the retriever: the
single-ground-truth hit count was 10 before and after.

---

## 2. Retrieval baseline (all-MiniLM-L6-v2)

| Metric | recall@5 |
|--------|----------|
| Overall | 0.900 (18/20) |
| Single-ground-truth subset | 0.833 (10/12) |
| Multi-ground-truth subset | 1.000 (8/8) |

**The multi-ground-truth subset scores a perfect 1.000 and carries no
discriminating information.** Where the chunker's overlap gives the retriever two
acceptable targets, it always hits one. The honest headline figure is therefore
**0.833**, not 0.900; the overall number is inflated by chunk redundancy.

This is itself a finding about the chunking strategy rather than the retriever:
overlap materially improves recall, and the two chunkers in this system are
inconsistent — audio overlaps by 50%, PDF does not overlap at all.

By source:

| Source | recall@5 |
|--------|----------|
| Audio | 1.000 (13/13) |
| Image | 1.000 (3/3) |
| PDF | 0.500 (2/4) |

Mean retrieval time 0.054 s, excluding a one-off embedder load. Retrieval is
negligible against generation, consistent with the preliminary prototype
measurements.

---

## 3. Embedding comparison: all-MiniLM-L6-v2 vs all-mpnet-base-v2

### Method

Embedding models produce different vector dimensionalities (384 vs 768), and a
ChromaDB collection's dimensionality is fixed by its first write — verified
empirically rather than assumed. mpnet therefore required its own collection.
The collection name is derived from the resolved model rather than configured
separately, which makes a dimension mismatch structurally impossible instead of
something to remember.

**The corpus was copied and re-embedded, not re-ingested.** Re-ingesting would
re-run Whisper, which is the one non-deterministic step in the pipeline; a
transcript shifting by a single word moves every subsequent chunk boundary and
silently invalidates 13 of the 20 ground-truth indices. The comparison would then
be between two models *and* two corpora, with the difference attributed to the
model. Copying guarantees identical text — verified by SHA-256 over all 61
chunks, identical for both collections — leaving the model as the only variable.

### Results

| Metric | MiniLM | mpnet |
|--------|--------|-------|
| Overall recall@5 | 0.900 (18/20) | 0.850 (17/20) |
| Single-ground-truth subset | 0.833 (10/12) | 0.833 (10/12) |
| Audio | 1.000 (13/13) | 0.923 (12/13) |
| Image | 1.000 (3/3) | 1.000 (3/3) |
| PDF | 0.500 (2/4) | 0.500 (2/4) |
| Mean retrieval time | 0.070 s | 0.121 s |
| Model size on disk | ~90 MB | ~420 MB |

**The discriminating subset ties exactly, and both models fail the same two
questions.** The overall difference of 0.900 versus 0.850 is one question out of
twenty; on a set this size a single item moves overall recall five points. The
defensible claim is **no evidence mpnet helps here**, not that mpnet is worse.

### Why they tie: the bottleneck is extraction, not embedding

Both models miss Q17 ("how do you choose a function approximation algorithm").
The target chunk's section heading is stored as
`1.2.4 Choosining a a F Functitoion A Approximimatitoion Alglgor...` — the
scanned PDF's OCR layer duplicated characters, and the damage is concentrated in
headings, which is precisely the text a section-level question keys on.

**No embedding model can match text that is not in the index.** Doubling the
embedding dimension buys nothing against corrupted source. This is the same
root cause documented throughout Bucket B, now shown to cap retrieval
performance as well as generation quality.

### One instructive regression

Q9 ("how can a system tell junk mail from real mail") is answered directly by the
audio, which discusses spam detection, overused words and poor grammar. MiniLM
returned that chunk at rank 1. mpnet returned five PDF chunks and no audio at
all, apparently matching the abstract task — binary classification — in
machine-learning text over the surface topic.

That is exactly the stronger semantic generalisation mpnet is supposed to
provide, and here it actively hurt: it crossed source files to find a conceptual
match and lost the literal answer. **Stronger semantic matching is not uniformly
better in a mixed-modality corpus where surface topic carries the intent.**

**Recommendation: retain all-MiniLM-L6-v2.** No accuracy gain, 1.7× slower per
query, 4.7× the disk footprint. The mechanism to re-test is in place for after
extraction quality is improved, which is where the available gains actually are.

---

## 4. Speech recognition comparison: Whisper tiny / base / small / medium

### Method

One audio file (a professionally produced explainer, ~10 minutes, 1,278 words)
scored against its official transcript. Two metrics reported: **word error rate**,
the standard ASR measure, and the string-similarity score used in earlier
measurements, so previously recorded figures stay comparable.

Normalisation applied to both sides before scoring: lowercase, replace
non-alphanumeric characters with spaces, collapse whitespace. WER is meaningless
without stating this.

Two references were scored. The raw transcript file contains 41 words of
unspoken chapter headings; the cleaned reference removes them. The headings alone
account for roughly 3 WER points, and the deletion counts close the accounting
exactly (48 → 7 for `base`).

`base` against the raw reference reproduces the previously recorded 97.27%
similarity exactly, confirming the earlier figure and these are the same
measurement.

### Results (cleaned reference)

| Model | WER | Sub | Del | Ins | Similarity | Transcribe | vs base |
|-------|-----|-----|-----|-----|------------|------------|---------|
| tiny | 2.51% | 16 | 8 | 9 | 98.14% | 38.4 s | 0.54× |
| base | 1.59% | 11 | 7 | 3 | 98.78% | 70.7 s | 1.00× |
| small | 1.37% | 4 | 4 | 10 | 99.17% | 179.7 s | 2.54× |
| medium | 0.68% | 5 | 4 | — | 99.47% | 409.1 s | 5.79× |

`medium` loaded and ran successfully despite only 1.4 GB free physical RAM at
start — the memory exhaustion anticipated in the contingency plan did not occur
at this size. `large` was not attempted.

### The aggregate metric and the project-relevant metric disagree

WER falls monotonically, and `medium` more than halves `base`. Following the
aggregate would select `medium`.

**Substitutions tell a different story: 16 → 11 → 4 → 5. They bottom out at
`small`, and `medium` has one more substitution than `small`, not fewer.**

The distinction matters because error types are not equivalent here.
A substitution puts a *wrong word* into the transcript, which flows into
flashcard terms and quiz options as false study material. Insertions are largely
Whisper faithfully capturing filler and repetition that the published reference
tidied away — they add noise, not falsehood. `medium`'s entire WER advantage
comes from insertions and deletions.

On the criterion that governs downstream damage, **`small` captures essentially
all the available benefit at 44% of `medium`'s transcription cost.**

### Known content errors

| Case | tiny | base | small | medium |
|------|------|------|-------|--------|
| "lemmatization" | worse — invented "limb-attization" and "limitation" | "Lematization" | correct | correct |
| "make is now a noun" | wrong | wrong | correct | correct |

`tiny` is worse than its WER suggests. It produced two *different* inventions for
the same word, and a flashcard headed "limitation" is silently plausible in a way
"Lematization" is not — a wrong word that looks right is more dangerous than one
that looks wrong.

**One earlier claim was corrected during this work.** The transcription of the
word root as "lem" had been recorded as an ASR error propagating downstream. It
is not an error: the speaker uses that informal short form and the reference
transcript reads the same way. The mistake was made by pattern-matching — "Lem"
*looks* like a truncation of "lemma" — without checking the reference, and it
survived into three write-ups before being caught by reading the two passages
side by side. It is the same failure this project repeatedly documents in the
model: a plausible inference asserted as observation.

**Recommendation: `small`.** It fixes both genuine content errors, `medium`
fixes none beyond it, and `tiny` should be ruled out. The cost is 2.5× base's
transcription time, paid once per file at ingest, against errors that are read
repeatedly in generated study material.

**Caveat:** this rests on three known error cases from one recording, and that
recording is a scripted studio production — an upper bound for ASR conditions,
not a typical self-recorded lecture. The substitution floor at `small` may be a
property of this audio. Accent robustness, which is the property Whisper was
selected for, remains untested.

---

## 5. Language model comparison: Llama 3.2 3B vs Mistral 7B vs Llama 3 8B

### Method

Seven questions, chosen to exercise the failure modes documented in Bucket B
rather than to sample evenly: two where the retrieved source is corrupted, one
where retrieval is known to fail, one absent from the corpus entirely, and three
ordinary cases.

**Retrieval ran once per question and the identical chunks and prompt were sent
to all three models**, so the model is the only variable. The production grounded
prompt was used unchanged.

A reduced set was used rather than all twenty questions. At 60–185 seconds per
RAG query, twenty questions across three models is roughly three hours of
generation, which the remaining schedule could not absorb. Answers were judged by
hand against the source.

### Feasibility and cost

Both larger models are 4-bit quantised (4.34 GB and 4.07 GB downloads).

`llama3:8b` **failed to load on first attempt**: `unable to allocate CPU_REPACK
buffer`, needing a 3.93 GB contiguous allocation with 1.6 GB physical and 2.2 GB
swap free. Ollama retried automatically half a second later and succeeded. During
its RAG generation, free physical RAM bottomed out at **49 MB** and the pagefile
grew by several gigabytes. Mistral was less severe, bottoming out at ~1 GB free.

| Model | tok/s | TTFT | Prompt eval |
|-------|-------|------|-------------|
| llama3.2 (3B) | 10.6–13.3 | 11–25 s | 11–23 s |
| mistral:7b | 5.4–6.4 | 30–65 s | 30–63 s |
| llama3:8b | 6.0–6.9 | 31–68 s | 30–60 s |

**Time to first token is dominated by prompt processing, not generation.** A
five-chunk RAG prompt is roughly 1,200–1,700 tokens, and ingesting it takes
30–65 seconds on the 7–8B models. The user waits over a minute before the first
word appears, and generation speed itself is comparatively stable.

This reframes the latency picture from the preliminary report, which attributed
query latency to generation. It also means that increasing top-k from five to
ten — proposed there as a retrieval improvement — would roughly double
time-to-first-token at this model scale.

### Answer quality: no model dominates, and the criteria conflict

The clearest result came from **Q14**, which asks for a formula the source
contains only in OCR-corrupted form (`P(A|B)=IP(B|A)*P(A)I/P(B)`):

| Model | Behaviour |
|-------|-----------|
| llama3.2 | Silently corrected the formula, then attributed the clean version to the notes as a quotation |
| mistral:7b | Corrected it and explained each term — the most useful answer, and the term definitions genuinely are in the source |
| llama3:8b | Reproduced the corruption verbatim, including the spurious characters |

**The most faithful answer was the least useful, and the most useful answer was
the least faithful.** Groundedness and helpfulness — two of the three rubric
criteria — pull in opposite directions when the retrieved source is damaged.
`llama3.2`'s answer is the most troubling of the three: it is correct, but it
presents its own prior knowledge as a quotation from the student's notes,
which is the failure the grounded prompt exists to prevent.

**Q20**, where retrieval is known to miss the target chunk and returns topically
adjacent material: only `mistral:7b` stated that the notes do not explicitly list
issues before offering possibilities. Both Llama models asserted that the notes
describe issues and then supplied material not in the retrieved chunks.

**Q-absent** (a concept nowhere in the corpus): `llama3:8b` refused most cleanly.
`llama3.2` declined and then usefully listed adjacent content that was present.
`mistral:7b` declined and then explained the concept from its own training — a
direct grounding violation.

Each model wins a different criterion. There is no consistent quality ordering.

**Recommendation: retain Llama 3.2 3B.** No consistent quality advantage from
either larger model, roughly half the tokens per second, two to three times the
time-to-first-token, and an 8B model that loads only marginally on this hardware
and drives free memory to 49 MB. The 3B default is the defensible choice, and
"the larger model exceeded practical resource limits" is a legitimate answer to
RQ2 rather than a failure to report one.

---

## 6. What Bucket C establishes about RQ2

RQ2 asks how model variants trade off accuracy, latency and resource use on
commodity student hardware without dedicated GPU acceleration.

**The largest model is not the best choice in any of the three slots.** mpnet
ties MiniLM while costing 1.7× per query. Whisper `medium` halves aggregate WER
while fixing no additional content errors at 2.3× `small`'s cost. Neither 7B nor
8B language model shows a consistent quality advantage over 3B, at half the
throughput.

**In two of three slots, the aggregate metric would have selected the wrong
model.** WER favours `medium`; substitution count — the errors that corrupt
generated study material — favours `small`. Overall recall@5 favours MiniLM by
one question; the discriminating subset shows a tie. Reporting only the headline
figure in either case would have produced a defensible-looking but wrong
recommendation.

**The binding constraint is not model capability.** Retrieval fails on the PDF
because OCR destroyed the section headings, and no embedding model can match
absent text. Generation quality is capped by what extraction produced. Prompt
ingestion, not token generation, dominates perceived latency at 7–8B scale.
Model selection is not where the remaining gains are.

---

## 7. Limitations

- Twenty questions is a small evaluation set, written and scored by the author,
  who knows the corpus and built the system. Recall differences of one or two
  questions are within noise.
- The corpus is three files from three modalities. Results by source modality
  rest on 13, 4 and 3 questions respectively.
- Whisper was compared on a single recording, professionally produced and
  scripted. Accent and noise robustness — the properties Whisper was selected
  for — remain untested, and the Gantt task specifying Singaporean-accented audio
  was not fulfilled.
- The language model comparison used seven questions rather than twenty, for
  schedule reasons, and answers were judged by the author against the three
  binary rubric criteria without a second assessor.
- Only three language models were compared, all from the Llama and Mistral
  families. Same-class alternatives such as Phi-3 Mini — named in the
  preliminary report's contingency plan — and Qwen 2.5 were not evaluated.
- Chunk size and overlap were not varied. The natural experiment between the
  overlapping audio chunker and the non-overlapping PDF chunker suggests overlap
  matters, but it was not tested directly.

---

## 8. Correction to previously reported figures

Generation throughput was previously reported as 4.1–4.6 tokens per second,
including in the preliminary report. Those figures were derived by counting
streamed response lines. This evaluation used Ollama's native `eval_count` and
`eval_duration` counters and measured **10.6–13.3 tok/s for the same model on the
same hardware**.

The native counters are authoritative. The earlier figures were produced by a
measurement artefact and should be corrected wherever they appear, along with any
conclusion resting on them — in particular the preliminary report's assessment
that query latency sat at the edge of acceptability for interactive use.

Separately, all latency figures recorded before the `localhost` to `127.0.0.1`
change include roughly two seconds per call of IPv6 name-resolution overhead.