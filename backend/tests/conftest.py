"""Shared pytest fixtures for the backend test suite.

The debug logs are project evidence, not scratch space
------------------------------------------------------
query_debug.json and chunks_debug.json record what was retrieved, what the
model answered, and what was chunked at ingest. They are read back as evidence
for claims in the written report — e.g. whether a transcription error actually
reached a student in an answer.

The tests mock ChromaDB and Ollama but used to leave the file paths alone, so
every `pytest` run appended synthetic fixtures to the real files: queries for
sessions "s1" / "my_session" / None, and chunks from throwaway "hello world"
PDFs. Those are indistinguishable from genuine records once written, which
quietly corrupts the evidence.

The autouse fixture below redirects both logs to pytest's per-test tmp_path, so
a test run cannot touch the real files. It is autouse deliberately: an opt-in
fixture would only protect the tests that remembered to ask for it, and the
problem originally arose precisely because nobody thought to.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_debug_logs(tmp_path, monkeypatch):
    """Point both debug logs at a temp directory for the duration of a test.

    monkeypatch restores the originals afterwards, so nothing leaks between
    tests or outlives the run. Imports are inside the fixture to keep test
    collection from paying for the heavy model imports in backend.ingest.
    """
    import backend.ingest as ingest
    import backend.retrieve as retrieve

    monkeypatch.setattr(retrieve, "DEBUG_PATH", tmp_path / "query_debug.json")
    monkeypatch.setattr(ingest, "CHUNKS_DEBUG_PATH", tmp_path / "chunks_debug.json")
    yield
