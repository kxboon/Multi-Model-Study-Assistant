"""
Regression tests for the signal endpoints' request validation.

These exist because of a silent data-loss bug: `source_file: str = None` under
Pydantic v2 declares a str field with a None default, which accepts an OMITTED
key but rejects an explicit JSON null with 422. The frontend builds its payloads
from dict literals, so the key is always present and carries null whenever a
card or question has no provenance. Ten of eleven flashcard ratings were lost
that way, and the rejection happens during FastAPI request validation — before
the endpoint body runs, so the endpoints' own per-item failure isolation never
saw it.

The payload shapes below mirror frontend/app.py's post_flashcard_signals() and
post_quiz_signals() exactly. If those change shape, these should change with them.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    """TestClient with the signals file redirected to a temp path."""
    import backend.signals as signals
    from backend.main import app

    with patch.object(signals, "SIGNALS_PATH", str(tmp_path / "signals.json")):
        yield TestClient(app)


def _flashcard_payload(source_file):
    """Exactly what frontend post_flashcard_signals() builds, one card per call."""
    return {
        "session_id": "s1",
        "topic": "NLP",
        "results": [
            {"term": "Tokenization", "known": True, "source_file": source_file}
        ],
    }


def _quiz_payload(source_file):
    """Exactly what frontend post_quiz_signals() builds."""
    return {
        "session_id": "s1",
        "topic": "NLP",
        "results": [
            {"question": "What is tokenization?", "correct": True,
             "source_file": source_file}
        ],
    }


# ---------------------------------------------------------------------------
# The regression itself: an explicit null must be accepted, not 422'd
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,payload", [
    ("/flashcards/signals", _flashcard_payload(None)),
    ("/quiz/signals", _quiz_payload(None)),
])
def test_signals_accept_explicit_null_source_file(client, path, payload):
    """A card/question without provenance must still record its signal."""
    r = client.post(path, json=payload)
    assert r.status_code == 200, (
        f"{path} rejected an explicit null source_file: {r.text}. "
        "The annotation must be `str | None`, not `str` with a None default."
    )
    assert r.json()["logged"] == 1
    assert r.json()["failed"] == 0


@pytest.mark.parametrize("path,payload", [
    ("/flashcards/signals", _flashcard_payload("notes.pdf")),
    ("/quiz/signals", _quiz_payload("notes.pdf")),
])
def test_signals_accept_real_source_file(client, path, payload):
    """The provenance-carrying case must keep working."""
    r = client.post(path, json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["logged"] == 1


@pytest.mark.parametrize("path,payload", [
    ("/flashcards/signals", {"session_id": "s1", "topic": "NLP",
                             "results": [{"term": "Stemming", "known": False}]}),
    ("/quiz/signals", {"session_id": "s1", "topic": "NLP",
                       "results": [{"question": "Q?", "correct": False}]}),
])
def test_signals_accept_omitted_source_file(client, path, payload):
    """Omitting the key entirely must also work — it did even before the fix."""
    r = client.post(path, json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["logged"] == 1


def test_null_source_file_is_recorded_as_empty_sources(client, tmp_path):
    """A null source_file should log the signal with no sources, not drop it."""
    import backend.signals as signals

    r = client.post("/flashcards/signals", json=_flashcard_payload(None))
    assert r.status_code == 200
    records = json.loads(open(signals.SIGNALS_PATH, encoding="utf-8").read())
    assert len(records) == 1
    assert records[0]["retrieved_sources"] == []
    assert records[0]["signal_type"] == "flashcard"
    assert records[0]["value"] == "known"


# ---------------------------------------------------------------------------
# The same latent pattern on session_id: null must not 422 either. It should
# reach the endpoint's own 400 guard, which gives a usable message.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,payload", [
    ("/flashcards/signals", {"session_id": None, "topic": "NLP", "results": []}),
    ("/quiz/signals", {"session_id": None, "topic": "NLP", "results": []}),
    ("/query", {"question": "hi", "session_id": None}),
    ("/quiz", {"session_id": None, "topic": "NLP"}),
    ("/flashcards", {"session_id": None, "topic": "NLP"}),
])
def test_null_session_id_reaches_the_400_guard(client, path, payload):
    """An explicit null session_id must produce the endpoint's 400, not a 422.

    400 means validation let the request through and the endpoint's own guard
    explained the problem; 422 would mean the model rejected it first and the
    caller gets a schema error instead of "session_id is required".
    """
    r = client.post(path, json=payload)
    assert r.status_code == 400, (
        f"{path} returned {r.status_code}, expected the endpoint's own 400 "
        f"guard: {r.text}"
    )
    assert "session_id" in r.json()["detail"]
