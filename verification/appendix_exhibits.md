# Appendix Exhibits

Verbatim source material for the report appendix. Each exhibit records its module,
source file, the topic/question requested, and whether the text is pulled from a
stored log/artifact or was freshly regenerated for this document. Where regenerated,
the model's non-determinism means the exact wording may differ from whatever the
original run (if any) produced — this is noted per item.

---

## 1. The Bayes slide chain

**Module:** `verify_image` &nbsp;&nbsp; **Source file:** `CM3060_L6_slide7_bayes.jpg`

### 1a. BLIP caption + raw OCR text (stored chunk)

**Status: from log** — `chunks_debug.json`, record `saved_at: 2026-08-01T20:31:52.843332`
(identical text recurs at two later timestamps in the same file, confirming stability).
Metadata: `source_type: image`, `chunk_index: 0`.

The ingest pipeline stores caption and OCR concatenated as
`[Image caption: {caption}] {ocr_text}` (see `backend/ingest.py`, `_process_image`).
Split here for clarity; both halves are verbatim from the stored chunk.

**BLIP caption:**
```
a poster with the words raves and probait
```

**Raw OCR text:**
```
Baves's rule and probabilistic inference

LIKEL\HOOD | PR [OR of ¢ Posterior probability of a variable A
ane ve of 'B" A, pe ue * Given prior knowledge about it in the
; "%" is TRUE A form of its marginality (P(A)) and
being TR E gaven thed Is 0 knowledge of how likely B is to be True,
given that A is True (P(B|A))

P(A) ¢ Prior: Probability distribution representing
P(AIB)= knowledge or uncertainty of a data object

prior or before observing it

P(B) * Posterior: Conditional probability

A distribution representing what parameters
are likely after observing the data object.
gear kh, The ee if i, Likelihood: The probability of falling under
abi lity, of "A" of af a specific category or class.
us UE gun hak "Bis TRUE we, P(A|B)=IP(B|A)*P(A)I/P(B)

@luminousmen.com * Tips on how to remember. AB,BA,AB

Image taken from https://luminousmen.com/media/data-science-bayes-theorem-2.jpg
```

### 1b. Chat answer to "what is the difference between prior and posterior probability"

**Status: not available.** `query_debug.json` contains this exact question logged
twice for this chain (`session_id: verify_image`, `asked_at: 2026-08-01T20:35:21`, and
again under `iso_a`/`iso_b` at `2026-08-01T22:50:43`), plus a related retrieval logged
under the topic `"prior, posterior, likelihood"` (`2026-08-09T11:23:47`). **All four
records have `"answer": null`.** `retrieve.py`'s `log_answer()` — which backfills the
generated answer onto the retrieval record — was added after these runs, so the
question and retrieved chunks were captured but the model's actual reply to the
student was never persisted. `verification/BUCKET_B_FINDINGS.md` independently notes
the same gap: "answers were not persisted during Bucket B, so no artefact records what
the model returned for any [question]." Regenerating a fresh answer would not
reproduce the original, uninspectable one, so none is included here.

### 1c. Quiz item "What is P(B|A) equal to?"

**Status: regenerated — original wording differs.**

The exact question "What is P(B|A) equal to?" is confirmed to have been asked in a
real quiz: `signals.json` has the record
`{"session_id": "verify_image", "topic": "verify_image", "signal_type": "quiz",
"value": "incorrect", "question": "What is P(B|A) equal to?", "quiz_topic": "Baye's
rule", "timestamp": "2026-08-08T22:33:32"}` — but quiz signal records only capture the
question text and correct/incorrect outcome, never the four options or the answer key
(see `backend/signals.py`), so the original item cannot be reconstructed from logs.

Regenerating `/quiz` on `session_id: verify_image`, `topic: "Baye's rule"` (matching
the logged `quiz_topic`) across two attempts (3 and 5 questions) did not reproduce
that exact question string. The closest analog, from the second attempt:

```json
{
  "question": "What is the formula for calculating the posterior probability of a variable B, given that A is true?",
  "options": [
    "P(B|A) = P(A|B) * P(A)",
    "P(B|A) = P(A|B) * P(B)",
    "P(B|A) = P(A) * P(B)",
    "P(B|A) = P(A) / P(B)"
  ],
  "correct_index": 1,
  "keyed_answer": "P(B|A) = P(A|B) * P(B)",
  "source_index": 0,
  "source_file": "CM3060_L6_slide7_bayes.jpg",
  "page_or_slide": null
}
```

Supportedness check: the cited chunk (§1a above) only states a formula for
**P(A|B)** — `P(A|B)=IP(B|A)*P(A)I/P(B)` — it never states a rearranged formula for
**P(B|A)**. The keyed answer `P(B|A) = P(A|B) * P(B)` is also not the correct
rearrangement of Bayes' rule (the correct form omits nothing but should read
`P(B|A) = P(A|B) * P(B) / P(A)`; the keyed option drops the `/P(A)` term entirely).
So this item is both uncited by its own source and mathematically wrong.

### 1d. "Marginality" flashcard

**Status: from log.** Definition text from `verification/BUCKET_B_FINDINGS.md`
(§"The image-sourced failure"); corroborated by the `signals.json` record
`{"session_id": "verify_image", "topic": "verify_image", "signal_type": "flashcard",
"value": "known", "question": "Marginality", "retrieved_sources":
["CM3060_L6_slide7_bayes.jpg"], "flashcard_topic": "prior, posterior, likelihood",
"timestamp": "2026-08-09T11:26:11"}` (signals only store the term, not the
definition, hence the two sources together).

> **Marginality** — "A form of its marginality (P(A)) and being given the data is 0
> knowledge of how likely B is to be True, given that A is True."

Per BUCKET_B_FINDINGS.md, this is "a lightly cleaned-up restatement of the OCR text
`A form of its marginality (P(A)) and being TR E gaven thed Is 0 knowledge of how
likely B is to be True`, including the meaningless `Is 0` fragment. 'Marginality' is
not a term the slide defines; it is a fragment of a sentence about P(A) that the
model promoted into a headword."

---

## 2. Three-model comparison on the Bayes formula question

**Status: from log.** `verification/eval_llm_results.json`, question `id: "Q14"`
(the same item `verification/BUCKET_C_FINDINGS.md` discusses as "the clearest
result"). Question text: **"What is the formula for the posterior probability
P(A|B)?"** Complete, verbatim `answer` fields — not summarized:

### llama3.2
```
According to the notes, the formula for the posterior probability P(A|B) is:

P(A|B) = P(B|A) * P(A) / P(B)

This is stated in the notes as follows:

"P(B|A) * P(A) / P(B)"
```

### mistral:7b
```
According to the notes, the formula for the posterior probability P(A|B) using Bayes' theorem is as follows:

P(A|B) = P(B|A) * P(A) / P(B)

In this formula:
- P(A|B) is the posterior probability of variable A given B
- P(B|A) is the likelihood, which represents the probability of falling under a specific category or class A, given that B is true
- P(A) is the prior probability of A, representing the knowledge or uncertainty of data object A before observing it
- P(B) is the marginal likelihood of B, representing the probability distribution of B without considering A

It's important to note that this formula assumes independence between A and B, which means that knowing A does not affect our understanding of B. The notes also mention a tip for remembering this relationship: AB, BA, AB (A before B, then B before A, and both before combining them).
```

### llama3:8b
```
According to the notes, Bayes' theorem states that:

P(A|B) = IP(B|A)*P(A)/P(B)

This formula represents the conditional probability of "A" given that "B" is true.
```

Note llama3:8b reproduces the OCR corruption verbatim (the stray leading `I` in
`IP(B|A)`), while llama3.2 and mistral:7b both silently correct it to `P(B|A)`.
mistral:7b additionally asserts an independence assumption between A and B that
appears nowhere in the source notes.

---

## 3. The sparse-passage exhibit

**Module:** `Kai_Xiang_Test` &nbsp;&nbsp; **Source file:**
`Attacks_and_Defenses_in_Short-Range_Wireless_Technologies_for_IoT.pdf`

### 3a. Bibliography line naming BlueBorne

**Status: from log.** `chunks_debug.json`, metadata `page_or_slide: 40, chunk_index:
91, session_id: Kai_Xiang_Test`. The full chunk is one continuous OCR/pdfplumber-run
of the reference list (no line breaks preserved); the relevant excerpt, with the two
preceding entries for context showing this is unambiguously a citation list, not
body prose:

```
[243] M.Herfurt.(2004).BluBug.Accessed:Jan.19,2020.[Online].Available:
https://trifinite.org/trifinite_stuff_bluebug.html
[244] A.Laurie.(2013).HeloMotoBluetoothDevicePlanter.[Online].Avail-
able:https://trifinite.org/trifinite_stuff_helomoto.html
[245] Armis-Company. (2017). BlueBorne Cyber Threat Impacts Amazon
Echo and Google Home. [Online]. Available: https://www.
armis.com/blueborne/
```

`BlueBorne` appears exactly once in the source PDF, as the title fragment of
reference [245] — a citation, never a defined or explained term.

### 3b. Generated BlueBorne flashcard

**Status: regenerated — no original run exists to compare against.** No BlueBorne
flashcard appears in `signals.json` or any findings doc, so this is a first
generation, run under the **current (fixed)** flashcard prompt — the one with the
explicit refusal rule ("if a term is named in the notes but never explained there, do
not make a card for it"). `topic: "Bluetooth attacks and named vulnerabilities"`,
`n_cards: 10` (chosen because retrieval-testing confirmed this topic pulls the page-40
bibliography chunk into the top 10 results).

```json
{
  "term": "BlueBorne",
  "definition": "A Bluetooth vulnerability that allows attackers to execute arbitrary code on devices",
  "source_index": 0,
  "source_file": "Attacks_and_Defenses_in_Short-Range_Wireless_Technologies_for_IoT.pdf",
  "page_or_slide": 23
}
```

Two things worth flagging. First, the definition is factually correct in the real
world (BlueBorne is a real 2017 Armis-disclosed Bluetooth RCE vulnerability chain) —
this is the model supplying outside knowledge, not inventing nonsense, which is
exactly the failure mode the refusal rule was meant to block and did not. Second, the
card's own `source_index` points to **page 23**, not page 40 — the chunk it actually
cites is a different passage about Bluetooth social-engineering/interruption attacks
(BlueSpam, BlueBug, Bluejacking) that does not contain the word "BlueBorne" at all,
even though the true page-40 bibliography chunk (§3a) was present among the same 10
retrieved chunks. So the citation is wrong even independent of the content-source
question. The same generation run produced nine other cards (BlueSniper, BlueBug,
Blueprinting, CSKES, Fitbit, etc.) — all real product/attack/tool names lifted from
the same reference-heavy passage, all citing page 23, all with outside-knowledge
definitions.

### 3c. "Inductive Bias" card + its chapter-preview source bullet

**Status: card regenerated (no logged original); source bullet from log.**

The card does not appear in `signals.json` or any findings doc, so it's a fresh
generation under the current prompt: `session_id: Machine Learning`, `topic:
"Chapter overview and inductive bias"`, `n_cards: 5`.

```json
{
  "term": "Inductive Bias",
  "definition": "the need for inductive bias in learning",
  "source_index": 4,
  "source_file": "test_notes.pdf",
  "page_or_slide": 17
}
```

Source bullet — **from log**, `chunks_debug.json`, `page_or_slide: 17, chunk_index:
41, session_id: Machine Learning` (identical text also stored under `C3015 ML`,
`eval_set`, `migration_check`, same source file):

> "• Chapter 2 covers concept learning based on symbolic or logical
> representations. It also discusses the general-to-specific ordering over
> hypotheses, and **the need for inductive bias in learning**."

This is the same failure the prompt fix targeted — the definition is a near-verbatim
lift of the clause the term appears in, not a statement of what inductive bias means,
and it is circular (the word "inductive bias" is used to define "Inductive Bias").
The source bullet only *names* the concept as something Chapter 2 covers; it never
explains it at this point in the book, so the refusal rule should have applied here
too and did not.

---

## 4. Pre-fix Mitchell flashcards (verbatim fragments)

**Module:** `Machine Learning` &nbsp;&nbsp; **Source file:** `test_notes.pdf`
&nbsp;&nbsp; **Topic:** "Learning system design"

**Status: mixed.** Representation / Learning Mechanism / Training Experience are
exactly as originally reported (matches the wording given at the start of this
review). To reproduce them under the actual old prompt rather than relying on
memory alone, the old (pre-fix) `_build_flashcard_prompt` text was reconstructed in a
standalone script — reusing the real `query_rag`, `_ollama_generate`, `_parse_items`,
and `_attach_provenance` production code, without modifying `backend/main.py` — and
run once against the same module/topic. That run reproduced all three fragments
**verbatim**, confirming they are the actual old-prompt behavior and not a one-off:

```json
[
  {"term": "Representation", "definition": "a representation for this target knowledge", "source_index": 0},
  {"term": "Learning Mechanism", "definition": "a learning mechanism Suplied by the British Library", "source_index": 0},
  {"term": "Training Experience", "definition": "that allows users to update data entries would fit our definition of a learning system", "source_index": 2}
]
```

**"Design Choices" could not be reproduced.** It was not among the three problem
cards originally quoted, and two separate reconstruction runs under the exact old
prompt did not produce a card by that name — generation is not deterministic run to
run (a third old-prompt attempt on the same topic returned five *good*, non-fragment
definitions with no fragment cards at all). The two fragment-style cards the
reconstruction runs produced in "Design Choices"'s place:

```json
{"term": "Target Function", "definition": "the true target function V can indeed be represented by a linear combination of these", "source_index": 1}
{"term": "Performance System", "definition": "the performance system, critic, generalizer, and experi­ ment generator", "source_index": 1}
```

If the report needs a literal "Design Choices" exhibit, it will need to come from
wherever the original 5-card run's output was recorded outside this project's logs
(the original run predates any stored artifact); it cannot be regenerated on demand.

**Source chunk containing the stamp** — **from log**, `chunks_debug.json`,
`page_or_slide: 7, chunk_index: 16, session_id: Machine Learning` (cited by both the
Representation and Learning Mechanism cards, `source_index: 0` in the reconstruction):

```
In order to complete the design of the learning system, we must now choose 1. the exact type of knowledge to be learned 2. a representation for this target knowledge 3. a learning mechanism Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

The Training Experience card's source chunk (`page_or_slide: 6, chunk_index: 11`,
also from `chunks_debug.json`):

```
5 CHAPTER 1I INTRODUCTION that allows users to update data entries would fit our definition of a learning system: it improves its performance at answering database queries, based on the experience gained from database updates. Rather than worry about whether this type of activity falls under the usual informal conversational meaning of the word "“learning,”" we wiJlIl simply adopt our technical definition of the class of programs that improve through experience. Within this class we will find many types of problems that require more or less sophisticated solutions. Our concern here is not to analyze the meaning of the English word "“learning”" as it is used in ev­ eryday language. Instead, our goal is to define precisely a class of problems that encompasses interesting forms of learning, to explore algorithms that solve such problems, and to understand the fundamental structure of learning problems and processes. 1.2 DESIGNING A LEARNING SYSTEM In order to illustrate some of the basic design issues and approaches to machine learning, let us consider designing a program to learn to play checkers, with the goal of entering it in the world checkers tournament. We adopt the obvious performance measure: the percent of games it wins in this world tournament. 1.2.1 Choosing the Training Experience
```

Note "a learning mechanism Suplied by the British Library" reads as a single clause
only because the OCR/PDF-extraction pipeline concatenated the numbered list item
directly against a library-stamp watermark with no separator — the stamp is not part
of the book's content.

---

## 5. Multi-answer quiz item on design choices

**Module:** `Machine Learning` &nbsp;&nbsp; **Source file:** `test_notes.pdf`
&nbsp;&nbsp; **Topic:** "Learning system design" &nbsp;&nbsp; **Status: from this
session's live regeneration** (captured verbatim during the quiz before/after audit
earlier in this session, current/unmodified quiz prompt — nothing in the quiz prompt
was changed).

```json
{
  "question": "What is one of the design choices in designing a machine learning approach?",
  "options": [
    "Choosing the type of training experience",
    "Choosing the target function to be learned",
    "Choosing a representation for the target function",
    "Choosing a learning algorithm"
  ],
  "correct_index": 0,
  "keyed_answer": "Choosing the type of training experience",
  "source_index": 4,
  "source_file": "test_notes.pdf",
  "page_or_slide": 18
}
```

Cited source chunk (`page_or_slide: 18`):

> "Designing a machine learning approach involves a number of design choices,
> including choosing the type of training experience, the target function to be
> learned, a representation for this target function, and an algorithm for learning
> the target function from training examples."

All four options are, per this sentence, actually listed as design choices — the
quiz's "exactly one option is correct" requirement is violated by the source material
itself, not by a fabrication. `signals.json` independently confirms a real marked
quiz on the same session/topic asked a differently-worded version of this same
question: `{"session_id": "Machine Learning", "topic": "Machine Learning",
"signal_type": "quiz", "value": "correct", "question": "What are some of the design
choices involved in designing a machine learning approach?", "quiz_topic": "learning
system design", "timestamp": "2026-08-18T21:50:05"}` — again, no options or answer key
are stored in the signal, only the question text and outcome.
