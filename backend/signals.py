"""
Learning-signal logging.

A "signal" is any observation about how a student is coping with their
material — currently the sentiment of the questions they ask, later things
like quiz performance. Signals are appended to a JSON file so they can be
analysed offline without adding a database to the project.

Record shape:

    {
      "timestamp": "2026-08-04T14:03:11.123456",
      "session_id": "CM3060",
      "topic": "CM3060",
      "signal_type": "sentiment",
      "value": "NEGATIVE",
      "score": 0.87,
      "question": "I still don't understand this at all",
      "retrieved_sources": ["CM3060_L6.pptx"]
    }

Two fields are deliberately loose:

- `topic` currently mirrors `session_id`, but it is stored as its own field
  rather than aliased, so a finer-grained topic (a slide range, a concept)
  can replace it later without rewriting existing records.
- `signal_type` names the kind of observation, so quiz-performance or
  time-on-task signals can be appended to the same file with no schema change.
"""

import json
import os
from datetime import datetime
from pathlib import Path

SIGNALS_PATH = os.getenv("SIGNALS_PATH", "./signals.json")


def log_signal(record: dict) -> None:
    """Append one signal record to the signals JSON file.

    Uses the same read-append-write cycle as query_debug.json in retrieve.py:
    the whole file is loaded, extended, and rewritten. That is fine at project
    scale and keeps the file valid JSON rather than JSON-lines.

    A missing `timestamp` is filled in here so every caller does not have to
    remember it. Any other field is stored exactly as given.

    Args:
        record: The signal to append. See the module docstring for the shape.
    """
    record.setdefault("timestamp", datetime.now().isoformat())

    path = Path(SIGNALS_PATH)

    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (json.JSONDecodeError, ValueError):
        # File exists but is empty or corrupt — start fresh rather than crashing
        existing = []

    if not isinstance(existing, list):
        existing = [existing]

    existing.append(record)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
