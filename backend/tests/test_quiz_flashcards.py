"""
Regression tests for /quiz and /flashcards post-parse cleanup.

Three bugs motivated this file:
  1. The model sometimes emits more valid items than were requested (a 10-item
     quiz request came back with 11 parsed items, and the frontend rendered
     all of them). /quiz and /flashcards now truncate to the requested count
     after parsing.
  2. A flashcard deck requested from a topic with fewer distinct concepts than
     n_cards came back with the same term repeated 2-3 times under different
     casing, each with a near-identical definition. /flashcards now dedupes
     on term.lower() (first occurrence wins) before truncating.
  3. A live run against a real corpus produced quiz items where all four
     options were the identical string ("Improve performance at a specific
     task" x4) — unanswerable regardless of source quality. _validate_quiz_item
     now rejects any item whose options are not four distinct strings after
     case-insensitive, whitespace-normalised comparison. This is deliberately
     separate from the flashcard term dedup above: that one merges duplicates
     across cards in a deck, this one rejects duplicates within a single
     item's own options.

In all cases "parsed" in the response must match len(items)/len(cards)
exactly — it describes what was actually returned, not what the model
happened to emit before cleanup.

query_rag and _ollama_generate are mocked throughout: these tests exercise
the parsing/truncation/dedup/validation logic in backend/main.py, not
retrieval or the model itself.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


def _mock_rag_result(n_chunks=5):
    """A query_rag()-shaped dict with enough chunks that source_index never
    needs to matter — none of these tests care about provenance."""
    return {
        "chunks": [f"chunk text {i}" for i in range(n_chunks)],
        "metadatas": [
            {"source_file": "notes.pdf", "page_or_slide": i, "session_id": "s1"}
            for i in range(n_chunks)
        ],
        "distances": [0.1] * n_chunks,
    }


def _quiz_item(i):
    return {
        "question": f"Question {i}?",
        "options": ["a", "b", "c", "d"],
        "correct_index": 0,
    }


def _flashcard(term, definition):
    return {"term": term, "definition": definition}


# ---------------------------------------------------------------------------
# /quiz — truncation
# ---------------------------------------------------------------------------

def test_quiz_truncates_when_model_overproduces(client):
    """11 valid items for a 10-item request must come back as exactly 10."""
    raw = json.dumps([_quiz_item(i) for i in range(11)])
    with patch("backend.main.query_rag", return_value=_mock_rag_result()), \
         patch("backend.main._ollama_generate", return_value=raw):
        r = client.post("/quiz", json={
            "session_id": "s1", "topic": "NLP", "n_questions": 10,
        })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requested"] == 10
    assert body["parsed"] == 10
    assert len(body["items"]) == 10
    assert any("truncated" in w for w in body["warnings"])


def test_quiz_parsed_matches_actual_when_under_requested(client):
    """Fewer valid items than requested must be reported honestly, not padded."""
    raw = json.dumps([_quiz_item(i) for i in range(4)])
    with patch("backend.main.query_rag", return_value=_mock_rag_result()), \
         patch("backend.main._ollama_generate", return_value=raw):
        r = client.post("/quiz", json={
            "session_id": "s1", "topic": "NLP", "n_questions": 10,
        })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requested"] == 10
    assert body["parsed"] == 4
    assert len(body["items"]) == 4
    assert not any("truncated" in w for w in body["warnings"])


# ---------------------------------------------------------------------------
# /quiz — degenerate option rejection
# ---------------------------------------------------------------------------

def _quiz_item_with_options(options, correct_index=0):
    return {"question": "Q?", "options": options, "correct_index": correct_index}


def test_validate_quiz_item_rejects_identical_options():
    """Direct unit check: four identical options must be rejected outright."""
    from backend.main import _validate_quiz_item

    item, reason = _validate_quiz_item(
        _quiz_item_with_options(["Same", "Same", "Same", "Same"]), n_chunks=5
    )
    assert item is None
    assert "duplicate" in reason


def test_validate_quiz_item_rejects_case_and_whitespace_variant_duplicates():
    """'Overfitting' / 'overfitting' / ' Overfitting ' must count as one option,
    not four — the comparison is case-insensitive and whitespace-normalised."""
    from backend.main import _validate_quiz_item

    item, reason = _validate_quiz_item(
        _quiz_item_with_options(
            ["Overfitting", "overfitting", " Overfitting ", "OVERFITTING"]
        ),
        n_chunks=5,
    )
    assert item is None
    assert "duplicate" in reason


def test_validate_quiz_item_accepts_four_distinct_options():
    """Regression guard: genuinely distinct options must still pass."""
    from backend.main import _validate_quiz_item

    item, reason = _validate_quiz_item(
        _quiz_item_with_options(["Overfitting", "Underfitting", "Bias", "Variance"]),
        n_chunks=5,
    )
    assert item is not None
    assert reason is None
    assert item["options"] == ["Overfitting", "Underfitting", "Bias", "Variance"]


def test_quiz_drops_degenerate_item_and_reports_it_in_warnings(client):
    """A live run produced items where all four options were the identical
    string ("Improve performance at a specific task" x4) — unanswerable
    regardless of source quality. Mixed in with valid items, only the
    degenerate one should be dropped, and it must be reported in warnings
    the same way any other rejected item is."""
    raw = json.dumps([
        _quiz_item_with_options(["Improve performance at a specific task"] * 4),
        _quiz_item_with_options(["A", "B", "C", "D"], correct_index=1),
        _quiz_item_with_options(["W", "X", "Y", "Z"], correct_index=2),
    ])
    with patch("backend.main.query_rag", return_value=_mock_rag_result()), \
         patch("backend.main._ollama_generate", return_value=raw):
        r = client.post("/quiz", json={
            "session_id": "s1", "topic": "ML", "n_questions": 3,
        })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requested"] == 3
    assert body["parsed"] == 2
    assert len(body["items"]) == 2
    assert [i["options"] for i in body["items"]] == [
        ["A", "B", "C", "D"], ["W", "X", "Y", "Z"],
    ]
    assert any("duplicate" in w for w in body["warnings"])


def test_quiz_rejects_all_items_when_every_item_is_degenerate(client):
    """If every item the model produced is degenerate, /quiz must still
    return 200 with an empty (not padded) item list — the same "honest
    parsed count" contract as the under-production case."""
    raw = json.dumps([
        _quiz_item_with_options(["Same", "Same", "Same", "Same"]),
        _quiz_item_with_options(["X", "x", " X ", "X"]),
    ])
    with patch("backend.main.query_rag", return_value=_mock_rag_result()), \
         patch("backend.main._ollama_generate", return_value=raw):
        r = client.post("/quiz", json={
            "session_id": "s1", "topic": "ML", "n_questions": 2,
        })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requested"] == 2
    assert body["parsed"] == 0
    assert body["items"] == []
    assert sum("duplicate" in w for w in body["warnings"]) == 2


# ---------------------------------------------------------------------------
# /flashcards — dedup and truncation
# ---------------------------------------------------------------------------

def test_flashcards_dedupes_case_variant_terms(client):
    """Case-variant repeats of the same term collapse to one, first kept."""
    raw = json.dumps([
        _flashcard("Tokenization", "def A"),
        _flashcard("tokenization", "def A duplicate"),
        _flashcard("Stemming", "def B"),
        _flashcard("STEMMING", "def B duplicate"),
        _flashcard("Lemmatization", "def C"),
        _flashcard("Tokenization", "def A duplicate 2"),
    ])
    with patch("backend.main.query_rag", return_value=_mock_rag_result()), \
         patch("backend.main._ollama_generate", return_value=raw):
        r = client.post("/flashcards", json={
            "session_id": "s1", "topic": "NLP", "n_cards": 10,
        })

    assert r.status_code == 200, r.text
    body = r.json()
    terms = [c["term"] for c in body["cards"]]
    assert terms == ["Tokenization", "Stemming", "Lemmatization"]
    # The first occurrence's definition survives, not a later duplicate's.
    assert body["cards"][0]["definition"] == "def A"
    assert body["requested"] == 10
    assert body["parsed"] == 3
    assert len(body["cards"]) == 3
    assert any("3 duplicate" in w for w in body["warnings"])


def test_flashcards_truncates_after_dedup(client):
    """More distinct cards than requested must still be capped."""
    raw = json.dumps([_flashcard(f"Term{i}", f"def {i}") for i in range(5)])
    with patch("backend.main.query_rag", return_value=_mock_rag_result()), \
         patch("backend.main._ollama_generate", return_value=raw):
        r = client.post("/flashcards", json={
            "session_id": "s1", "topic": "NLP", "n_cards": 3,
        })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requested"] == 3
    assert body["parsed"] == 3
    assert len(body["cards"]) == 3
    assert not any("duplicate" in w for w in body["warnings"])
    assert any("truncated" in w for w in body["warnings"])


def test_flashcards_dedup_runs_before_truncation(client):
    """Dedup must free up room before the cap is applied, not the reverse.

    8 raw cards with 2 duplicate terms -> 6 distinct -> capped to 4. If
    truncation ran first, the duplicates near the front would eat into the
    cap and leave fewer than 4 distinct cards.
    """
    raw = json.dumps([
        _flashcard("A", "def A"), _flashcard("a", "def A dup"),
        _flashcard("B", "def B"), _flashcard("b", "def B dup"),
        _flashcard("C", "def C"), _flashcard("D", "def D"),
        _flashcard("E", "def E"), _flashcard("F", "def F"),
    ])
    with patch("backend.main.query_rag", return_value=_mock_rag_result()), \
         patch("backend.main._ollama_generate", return_value=raw):
        r = client.post("/flashcards", json={
            "session_id": "s1", "topic": "NLP", "n_cards": 4,
        })

    assert r.status_code == 200, r.text
    body = r.json()
    terms = [c["term"] for c in body["cards"]]
    assert terms == ["A", "B", "C", "D"]
    assert body["requested"] == 4
    assert body["parsed"] == 4
    assert any("2 duplicate" in w for w in body["warnings"])
    assert any("truncated" in w for w in body["warnings"])
