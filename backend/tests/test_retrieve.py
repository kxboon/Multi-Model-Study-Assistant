"""
Unit tests for the retrieve module (query_rag and ask_ollama).

Ollama calls are mocked so tests run without a running LLM server.
ChromaDB is also mocked to avoid filesystem setup.
"""

from unittest.mock import MagicMock, patch
import json


# ---------------------------------------------------------------------------
# query_rag tests
# ---------------------------------------------------------------------------

def test_query_rag_returns_expected_structure():
    """query_rag() should return a dict with chunks, metadatas, and distances."""
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["chunk one", "chunk two"]],
        "metadatas": [[{"source_file": "notes.pdf"}, {"source_file": "notes.pdf"}]],
        "distances": [[0.1, 0.2]],
    }

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.0] * 384]

    with (
        patch("backend.retrieve._collection", fake_collection),
        patch("backend.retrieve._embedder", fake_embedder),
    ):
        from backend.retrieve import query_rag

        result = query_rag("What is machine learning?", session_id="s1")

    assert "chunks" in result
    assert "metadatas" in result
    assert "distances" in result
    assert len(result["chunks"]) == 2
    assert result["chunks"][0] == "chunk one"


def test_query_rag_applies_session_filter():
    """query_rag() with session_id should pass a where-filter to ChromaDB."""
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.0] * 384]

    with (
        patch("backend.retrieve._collection", fake_collection),
        patch("backend.retrieve._embedder", fake_embedder),
    ):
        from backend.retrieve import query_rag

        query_rag("test", session_id="my_session")

    call_kwargs = fake_collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"session_id": "my_session"}


def test_query_rag_no_session_filter():
    """query_rag() without session_id should pass where=None to ChromaDB."""
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.0] * 384]

    with (
        patch("backend.retrieve._collection", fake_collection),
        patch("backend.retrieve._embedder", fake_embedder),
    ):
        from backend.retrieve import query_rag

        query_rag("test")

    call_kwargs = fake_collection.query.call_args.kwargs
    assert call_kwargs["where"] is None


# ---------------------------------------------------------------------------
# ask_ollama tests
# ---------------------------------------------------------------------------

def _fake_ollama_response(tokens: list):
    """Build a mock requests.Response that streams Ollama-style JSON lines."""
    lines = []
    for i, token in enumerate(tokens):
        done = i == len(tokens) - 1
        lines.append(json.dumps({"response": token, "done": done}).encode())
    return lines


def test_ask_ollama_returns_assembled_string():
    """ask_ollama() should concatenate all streamed response tokens."""
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.iter_lines.return_value = _fake_ollama_response(
        ["Hello", " ", "world", "!"]
    )

    with patch("backend.retrieve.requests.post", return_value=fake_response):
        from backend.retrieve import ask_ollama

        answer = ask_ollama(
            question="What is a neural network?",
            context_chunks=["A neural network is a model inspired by the brain."],
        )

    assert answer == "Hello world!"


def test_ask_ollama_prompt_contains_question_and_notes():
    """ask_ollama() should include both the notes and question in the prompt."""
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.iter_lines.return_value = _fake_ollama_response(["ok"])

    with patch("backend.retrieve.requests.post", return_value=fake_response) as mock_post:
        from backend.retrieve import ask_ollama

        ask_ollama(
            question="Define overfitting.",
            context_chunks=["Overfitting is when a model memorises training data."],
        )

    payload = mock_post.call_args.kwargs["json"]
    assert "Define overfitting." in payload["prompt"]
    assert "Overfitting is when a model memorises" in payload["prompt"]
    assert "stream" in payload


def test_ask_ollama_connection_error():
    """ask_ollama() should propagate ConnectionError when Ollama is unreachable."""
    import requests as req_lib

    with patch(
        "backend.retrieve.requests.post",
        side_effect=req_lib.exceptions.ConnectionError("refused"),
    ):
        from backend.retrieve import ask_ollama

        try:
            ask_ollama("question", ["context"])
            assert False, "Expected ConnectionError"
        except req_lib.exceptions.ConnectionError:
            pass  # expected
