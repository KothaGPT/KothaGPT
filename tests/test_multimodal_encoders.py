from __future__ import annotations

import pytest
import torch

from ml.models.multimodal.audio import AudioEncoder, AudioProjector
from ml.models.multimodal.document import DocumentEncoder, DocumentProjector
from ml.models.multimodal.vision import VisionEncoder, VisionProjector


def test_vision_encoder_and_projector_shapes():
    encoder = VisionEncoder(hidden_size=24, mm_hidden=12, patch_size=4, image_size=16)

    encoded = encoder.encode(torch.randn(2, 3, 12, 20))
    projected = VisionProjector(12, 24)(torch.randn(2, 5, 12))

    assert encoded.shape == (2, 16, 24)
    assert projected.shape == (2, 5, 24)


@pytest.mark.parametrize("images", [torch.randn(3, 16, 16), torch.ones(2, 3, 16, 16, dtype=torch.uint8)])
def test_vision_encoder_input_types(images):
    encoder = VisionEncoder(hidden_size=16, mm_hidden=8, patch_size=4, image_size=16)
    if images.is_floating_point():
        assert encoder(images).shape == (1, 16, 16)
    else:
        with pytest.raises(TypeError, match="floating-point"):
            encoder(images)


@pytest.mark.parametrize(
    "images, error, message",
    [
        (torch.randn(2, 1, 16, 16), ValueError, "shape"),
        (torch.randn(3, 16), ValueError, "shape"),
        (torch.ones(2, 3, 16, 16, dtype=torch.int64), TypeError, "floating-point"),
    ],
)
def test_vision_encoder_rejects_invalid_images(images, error, message):
    encoder = VisionEncoder(hidden_size=16, mm_hidden=8, patch_size=4, image_size=16)
    with pytest.raises(error, match=message):
        encoder(images)


def test_audio_encoder_and_projector_shapes():
    encoder = AudioEncoder(hidden_size=24, mm_hidden=12, n_fft=32, hop_length=8)

    encoded = encoder.encode(torch.randn(2, 128))
    projected = AudioProjector(12, 24)(torch.randn(2, 7, 12))

    assert encoded.shape == (2, 17, 24)
    assert projected.shape == (2, 7, 24)


def test_audio_encoder_accepts_short_mono_waveform():
    encoder = AudioEncoder(hidden_size=16, mm_hidden=8, n_fft=32, hop_length=8)

    assert encoder(torch.randn(8)).shape == (1, 5, 16)


@pytest.mark.parametrize(
    "audio, error, message",
    [
        (torch.randn(2, 3, 4), ValueError, "shape"),
        (torch.ones(2, 64, dtype=torch.int64), TypeError, "floating-point"),
        (torch.empty(2, 0), ValueError, "at least one sample"),
    ],
)
def test_audio_encoder_rejects_invalid_inputs(audio, error, message):
    encoder = AudioEncoder(hidden_size=16, mm_hidden=8, n_fft=32, hop_length=8)
    with pytest.raises(error, match=message):
        encoder(audio)


def test_document_encoder_includes_layout_pages_and_projects():
    encoder = DocumentEncoder(hidden_size=24, mm_hidden=12, vocab_size=32, max_pages=4)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    boxes = torch.rand(2, 3, 4)
    boxes[..., 2:] = torch.maximum(boxes[..., 2:], boxes[..., :2])
    page_ids = torch.tensor([[0, 0, 1], [1, 2, 2]])

    encoded = encoder.encode(input_ids, boxes, page_ids)
    projected = DocumentProjector(12, 24)(torch.randn(2, 5, 12))

    assert encoded.shape == (2, 3, 24)
    assert projected.shape == (2, 5, 24)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"input_ids": torch.ones(2, 3, 4, dtype=torch.long)}, ValueError, "input_ids"),
        ({"input_ids": torch.tensor([[0, 32]])}, ValueError, "vocabulary"),
        ({"input_ids": torch.tensor([[1, 2]]), "boxes": torch.rand(1, 2, 3)}, ValueError, "boxes"),
        (
            {"input_ids": torch.tensor([[1, 2]]), "boxes": torch.tensor([[[0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0]]])},
            ValueError,
            "normalized",
        ),
        (
            {"input_ids": torch.tensor([[1]]), "boxes": torch.tensor([[[0.8, 0.0, 0.2, 1.0]]])},
            ValueError,
            "x1 >= x0",
        ),
        (
            {"input_ids": torch.tensor([[1, 2]]), "page_ids": torch.tensor([[0, 4]])},
            ValueError,
            "page_ids",
        ),
    ],
)
def test_document_encoder_rejects_invalid_inputs(kwargs, error, message):
    encoder = DocumentEncoder(hidden_size=16, mm_hidden=8, vocab_size=32, max_pages=4)
    with pytest.raises(error, match=message):
        encoder(**kwargs)
