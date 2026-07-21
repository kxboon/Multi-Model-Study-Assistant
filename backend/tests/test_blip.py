"""
Unit tests for the BLIPModel image-captioning wrapper.

Mocks out the HuggingFace model download so no weights are needed
and tests run on a tiny synthetic PIL image.
"""

from unittest.mock import MagicMock, patch
from PIL import Image


def _make_tiny_image():
    """Create a 10×10 white RGB image — the smallest valid PIL image."""
    return Image.new("RGB", (10, 10), color=(255, 255, 255))


def test_caption_returns_string():
    """BLIPModel.caption() should return a non-empty string for any image."""
    with (
        patch("backend.models.blip_model.BlipProcessor") as MockProcessor,
        patch("backend.models.blip_model.BlipForConditionalGeneration") as MockModel,
    ):
        # Wire up fake processor and model
        fake_processor = MagicMock()
        fake_processor.return_value = {"pixel_values": MagicMock()}
        # decode() returns the caption string
        fake_processor.decode.return_value = "  a white square  "

        fake_model_instance = MagicMock()
        fake_model_instance.generate.return_value = [[101, 102, 103]]

        MockProcessor.from_pretrained.return_value = fake_processor
        MockModel.from_pretrained.return_value = fake_model_instance

        from backend.models.blip_model import BLIPModel

        blip = BLIPModel()
        result = blip.caption(_make_tiny_image())

        # Should return stripped caption string
        assert isinstance(result, str)
        assert result == "a white square"


def test_caption_accepts_file_path(tmp_path):
    """BLIPModel.caption() should accept a file path string, not just PIL."""
    img_path = str(tmp_path / "test.png")
    _make_tiny_image().save(img_path)

    with (
        patch("backend.models.blip_model.BlipProcessor") as MockProcessor,
        patch("backend.models.blip_model.BlipForConditionalGeneration") as MockModel,
    ):
        fake_processor = MagicMock()
        fake_processor.decode.return_value = "a white image"
        fake_model_instance = MagicMock()
        fake_model_instance.generate.return_value = [[1]]

        MockProcessor.from_pretrained.return_value = fake_processor
        MockModel.from_pretrained.return_value = fake_model_instance

        from backend.models.blip_model import BLIPModel

        blip = BLIPModel()
        result = blip.caption(img_path)

        assert isinstance(result, str)


def test_lazy_load_not_called_on_init():
    """BLIP model should NOT be loaded until caption() is first called."""
    with (
        patch("backend.models.blip_model.BlipProcessor") as MockProcessor,
        patch("backend.models.blip_model.BlipForConditionalGeneration") as MockModel,
    ):
        from backend.models.blip_model import BLIPModel

        _ = BLIPModel()
        MockProcessor.from_pretrained.assert_not_called()
        MockModel.from_pretrained.assert_not_called()
