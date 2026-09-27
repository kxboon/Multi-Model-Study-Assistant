"""Learning-signal logging.

A signal is one observation of how a student is coping with their material:
question sentiment, quiz results, flashcard self-ratings. Records are appended
to a JSON file rather than a database, which is enough at this scale.

    {"timestamp", "session_id", "topic", "signal_type", "value", "score",
     "question", "retrieved_sources"}

`topic` mirrors `session_id` today but is stored separately so a finer-grained
topic can replace it later. Every signal type must mean the same thing by it or
records cannot be aggregated across types, so a type's own subject goes in its
own field instead (`quiz_topic`, `flashcard_topic`).
"""

import json
from datetime import datetime
from pathlib import Path

from backend.paths import resolve_path

SIGNALS_PATH = resolve_path("SIGNALS_PATH", "./signals.json")


def read_signals() -> list:
    """Return every signal record, or [] if there are none to read.

    A missing, empty or corrupt file yields [] rather than raising: signals are
    observational, so a reader should get "no data yet", never a crash.
    """
    path = Path(SIGNALS_PATH)

    try:
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (json.JSONDecodeError, ValueError):
        # File exists but is empty or corrupt — start fresh rather than crashing
        return []

    if not isinstance(records, list):
        records = [records]

    # Hand-edited files: anything that is not a dict is not a record.
    return [r for r in records if isinstance(r, dict)]


def log_signal(record: dict) -> None:
    """Append one signal record to the signals JSON file.

    Read-append-write of the whole file, as query_debug.json does: fine at this
    scale and keeps the file valid JSON rather than JSON-lines. A missing
    `timestamp` is filled in here.
    """
    record.setdefault("timestamp", datetime.now().isoformat())

    existing = read_signals()
    existing.append(record)

    Path(SIGNALS_PATH).write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
