import time, os, re, difflib, whisper

HERE = os.path.dirname(os.path.abspath(__file__))

AUDIO = os.path.join(HERE, "../../sample_files/Natural_Language_Processing.mp3")  # <-- your recorded file (change extension if needed)
ANSWER = os.path.join(HERE, "test_smoke_answer.md")
OUTPUT = os.path.join(HERE, "output.txt")
SIZE  = "base"             # try "tiny" first if base is painfully slow
THRESHOLD = 0.80           # minimum similarity ratio for the smoke test to pass

if not os.path.exists(AUDIO):
    raise SystemExit(f"Audio file not found: {AUDIO} — check the path/filename")

t0 = time.perf_counter()
model = whisper.load_model(SIZE)
print(f"load: {time.perf_counter()-t0:.1f}s  (first run includes model download)")

t1 = time.perf_counter()
result = model.transcribe(AUDIO, fp16=False)
print(f"transcribe: {time.perf_counter()-t1:.1f}s")

print("---TRANSCRIPT---")
print(str(result["text"]).strip())

# Compare the output with the reference transcript from YouTube.
# Whisper's output is never byte-identical to a human/YouTube transcript
# (casing, punctuation, section headers, the occasional word differ), so an
# exact match is the wrong check. Instead we normalize both texts and require
# a high similarity ratio — enough to confirm the pipeline transcribed the
# right audio correctly.

def normalize(text) -> list:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # drop punctuation
    return text.split()                        # -> list of words (collapses whitespace)

with open(ANSWER, "r", encoding="utf-8") as f:
    transcript = f.read()

print("---TRANSCRIPT FROM YOUTUBE---")
print(transcript.strip())

hypothesis = normalize(result["text"])
reference  = normalize(transcript)

# Compare at the WORD level with autojunk disabled. Character-level ratios on
# long text are deflated by difflib's autojunk heuristic (it treats common
# characters like spaces and vowels as junk), so word-level is far more faithful.
ratio = difflib.SequenceMatcher(None, hypothesis, reference, autojunk=False).ratio()
print(f"---SIMILARITY--- {ratio:.2%} (threshold {THRESHOLD:.0%})")

# Save the run result so it can be reviewed later.
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("---TRANSCRIPT---\n")
    f.write(str(result["text"]).strip() + "\n\n")
    f.write(f"---SIMILARITY--- {ratio:.2%} (threshold {THRESHOLD:.0%})\n")
    f.write(f"RESULT: {'PASS' if ratio >= THRESHOLD else 'FAIL'}\n")
print(f"saved output -> {OUTPUT}")

assert ratio >= THRESHOLD, (
    f"Transcript similarity {ratio:.2%} is below the {THRESHOLD:.0%} threshold — "
    f"transcription may be wrong or the wrong audio file was used."
)

print("Success!")
