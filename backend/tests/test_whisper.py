"""
Unit tests for the WhisperModel wrapper.

We mock the underlying whisper library so the test runs without any audio
files or GPU, and without downloading the 74 MB base model weights.
"""

from unittest.mock import MagicMock, patch


def test_transcribe_returns_string():
    """WhisperModel.transcribe() should return a plain string."""
    # Patch whisper.load_model at the module level where it is imported
    with patch("backend.models.whisper_model.whisper") as mock_whisper:
        # Set up a fake model that returns a known transcript
        fake_model = MagicMock()
        fake_model.transcribe.return_value = {"text": "  Hello, world.  "}
        mock_whisper.load_model.return_value = fake_model

        from backend.models.whisper_model import WhisperModel

        model = WhisperModel(model_size="base")
        result = model.transcribe("fake_audio.mp3")

        # Should strip whitespace from the returned text
        assert result == "Hello, world."
        # Verify the model was called with fp16=False (CPU-safe)
        fake_model.transcribe.assert_called_once_with("fake_audio.mp3", fp16=False)


def test_lazy_load_not_called_on_init():
    """The Whisper model should NOT be loaded until transcribe() is called."""
    with patch("backend.models.whisper_model.whisper") as mock_whisper:
        from backend.models.whisper_model import WhisperModel

        model = WhisperModel()
        # load_model should not have been called yet
        mock_whisper.load_model.assert_not_called()


def test_model_size_from_env(monkeypatch):
    """WhisperModel should read model size from WHISPER_MODEL_SIZE env var."""
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "tiny")

    with patch("backend.models.whisper_model.whisper") as mock_whisper:
        fake_model = MagicMock()
        fake_model.transcribe.return_value = {"text": "test"}
        mock_whisper.load_model.return_value = fake_model

        from backend.models.whisper_model import WhisperModel

        model = WhisperModel()  # no explicit size — should read env var
        model.transcribe("fake.mp3")

        mock_whisper.load_model.assert_called_once_with("tiny")
