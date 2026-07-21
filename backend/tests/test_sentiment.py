"""
Unit tests for the SentimentModel wrapper.

Mocks the HuggingFace pipeline so no model weights are downloaded.
"""

from unittest.mock import MagicMock, patch


def test_predict_returns_label_and_score():
    """SentimentModel.predict() should return a dict with label and score."""
    with patch("backend.models.sentiment_model.pipeline") as mock_pipeline_fn:
        fake_pipe = MagicMock()
        # HuggingFace pipelines return a list of result dicts
        fake_pipe.return_value = [{"label": "POSITIVE", "score": 0.98}]
        mock_pipeline_fn.return_value = fake_pipe

        from backend.models.sentiment_model import SentimentModel

        model = SentimentModel()
        result = model.predict("I love studying machine learning!")

        assert result["label"] == "POSITIVE"
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0


def test_predict_negative_sentiment():
    """SentimentModel.predict() should correctly identify NEGATIVE sentiment."""
    with patch("backend.models.sentiment_model.pipeline") as mock_pipeline_fn:
        fake_pipe = MagicMock()
        fake_pipe.return_value = [{"label": "NEGATIVE", "score": 0.91}]
        mock_pipeline_fn.return_value = fake_pipe

        from backend.models.sentiment_model import SentimentModel

        model = SentimentModel()
        result = model.predict("This exam was really difficult and stressful.")

        assert result["label"] == "NEGATIVE"


def test_lazy_load_not_called_on_init():
    """Pipeline should NOT be initialised until predict() is called."""
    with patch("backend.models.sentiment_model.pipeline") as mock_pipeline_fn:
        from backend.models.sentiment_model import SentimentModel

        _ = SentimentModel()
        mock_pipeline_fn.assert_not_called()


def test_long_text_truncated():
    """Texts longer than 512 chars should be truncated before prediction."""
    with patch("backend.models.sentiment_model.pipeline") as mock_pipeline_fn:
        fake_pipe = MagicMock()
        fake_pipe.return_value = [{"label": "POSITIVE", "score": 0.75}]
        mock_pipeline_fn.return_value = fake_pipe

        from backend.models.sentiment_model import SentimentModel

        model = SentimentModel()
        long_text = "word " * 300  # ~1500 chars
        model.predict(long_text)

        # Check that the text passed to the pipe was truncated to 512 chars
        called_text = fake_pipe.call_args[0][0]
        assert len(called_text) <= 512
