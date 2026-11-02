from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.models.config import ModelConfig, load_config
from ml.models.multimodal import KothaMultimodal


def _config(**overrides) -> ModelConfig:
    options = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 4,
        "intermediate_size": 24,
        "max_position_embeddings": 32,
    }
    options.update(overrides)
    return ModelConfig(**options)


def test_unified_model_fuses_text_vision_audio_and_document_embeddings():
    model = KothaMultimodal(
        _config(),
        mm_hidden=12,
        audio_n_fft=32,
        audio_hop_length=8,
        document_vocab_size=32,
        vision_image_size=16,
        vision_patch_size=4,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    boxes = torch.rand(1, 2, 4)
    boxes[..., 2:] = torch.maximum(boxes[..., 2:], boxes[..., :2])

    output = model(
        input_ids,
        images=torch.randn(1, 3, 16, 16),
        audio=torch.randn(1, 64),
        document_input_ids=torch.tensor([[4, 5]]),
        document_boxes=boxes,
        document_page_ids=torch.tensor([[0, 1]]),
    )

    assert output["logits"].shape == (1, 3 + 16 + 9 + 2, 32)
    assert output["loss"] is None


def test_unified_model_supports_text_labels_and_modality_labels():
    model = KothaMultimodal(
        _config(),
        mm_hidden=8,
        modalities=("vision",),
        vision_image_size=16,
        vision_patch_size=4,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    images = torch.randn(1, 3, 16, 16)
    labels = torch.randint(0, 32, (1, 19))

    result = model(input_ids, labels, images=images)

    assert result["logits"].shape == (1, 19, 32)
    assert result["loss"] is not None
    result["loss"].backward()
    assert model.vision_encoder.patch_embedding.weight.grad is not None


def test_unified_model_rejects_unconfigured_modality():
    model = KothaMultimodal(_config(), modalities=("vision",))

    with pytest.raises(ValueError, match="not configured"):
        model(torch.tensor([[1, 2]]), audio=torch.randn(1, 64))


def test_unified_model_rejects_orphan_document_metadata():
    model = KothaMultimodal(_config())

    with pytest.raises(ValueError, match="require document_input_ids"):
        model(torch.tensor([[1, 2]]), document_boxes=torch.rand(1, 2, 4))


def test_unified_model_rejects_fused_sequence_beyond_context():
    model = KothaMultimodal(
        _config(max_position_embeddings=4),
        modalities=("vision",),
        vision_image_size=16,
        vision_patch_size=4,
    )

    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        model(torch.tensor([[1, 2]]), images=torch.randn(1, 3, 16, 16))


def test_unified_model_requires_labels_for_full_fused_sequence():
    model = KothaMultimodal(
        _config(),
        modalities=("vision",),
        vision_image_size=16,
        vision_patch_size=4,
    )

    with pytest.raises(ValueError, match="labels must match"):
        model(torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]]), images=torch.randn(1, 3, 16, 16))


def test_unified_model_rejects_modality_batch_mismatch():
    model = KothaMultimodal(
        _config(),
        modalities=("vision",),
        vision_image_size=16,
        vision_patch_size=4,
    )

    with pytest.raises(ValueError, match="batch size must match"):
        model(torch.tensor([[1, 2]]), images=torch.randn(2, 3, 16, 16))


@pytest.mark.parametrize(
    "config_name, modalities",
    [
        ("mm-vision.yaml", ("vision",)),
        ("mm-audio.yaml", ("audio",)),
        ("mm-document.yaml", ("document",)),
        ("mm-unified.yaml", ("vision", "audio", "document")),
    ],
)
def test_multimodal_configs_load_and_select_encoders(config_name, modalities):
    config_path = Path(__file__).parents[1] / "ml" / "configs" / config_name
    config = load_config(config_path)
    assert config.model.mm_hidden == 512
    configured_modalities = tuple(
        name
        for name in ("vision", "audio", "document")
        if getattr(config.model, f"{name}_encoder")
    )
    assert configured_modalities == modalities
    model_config = _config(
        vision_encoder="vision" in modalities,
        audio_encoder="audio" in modalities,
        document_encoder="document" in modalities,
    )
    model = KothaMultimodal(model_config)
    assert model.modalities == modalities
