"""KothaGPT decoder integration for vision, audio, and document embeddings."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from ml.models import KothaGPT
from ml.models.config import ModelConfig
from ml.models.multimodal.audio import AudioEncoder
from ml.models.multimodal.document import DocumentEncoder
from ml.models.multimodal.vision import VisionEncoder


class KothaMultimodal(KothaGPT):
    """Append projected modality embeddings to text embeddings before decoding."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        mm_hidden: int | None = None,
        modalities: tuple[str, ...] | None = None,
        audio_n_fft: int = 400,
        audio_hop_length: int = 160,
        document_vocab_size: int | None = None,
        vision_image_size: int = 224,
        vision_patch_size: int = 16,
    ):
        super().__init__(config)
        mm_hidden = mm_hidden if mm_hidden is not None else (config.mm_hidden or 512)
        if mm_hidden <= 0:
            raise ValueError("mm_hidden must be positive")
        if modalities is None:
            configured_modalities = tuple(
                name
                for name, enabled in (
                    ("vision", config.vision_encoder),
                    ("audio", config.audio_encoder),
                    ("document", config.document_encoder),
                )
                if enabled
            )
            modalities = configured_modalities or ("vision", "audio", "document")
        allowed = {"vision", "audio", "document"}
        if not modalities or len(set(modalities)) != len(modalities):
            raise ValueError("modalities must contain unique modality names")
        unknown = set(modalities) - allowed
        if unknown:
            raise ValueError(f"unsupported modalities: {sorted(unknown)}")
        self.mm_hidden = mm_hidden
        self.modalities = modalities
        self.vision_encoder = (
            VisionEncoder(
                config.hidden_size,
                mm_hidden,
                patch_size=vision_patch_size,
                image_size=vision_image_size,
            )
            if "vision" in modalities
            else None
        )
        self.audio_encoder = (
            AudioEncoder(
                config.hidden_size,
                mm_hidden,
                n_fft=audio_n_fft,
                hop_length=audio_hop_length,
            )
            if "audio" in modalities
            else None
        )
        self.document_encoder = (
            DocumentEncoder(
                config.hidden_size,
                mm_hidden,
                vocab_size=document_vocab_size or config.vocab_size,
            )
            if "document" in modalities
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        document_input_ids: torch.Tensor | None = None,
        document_boxes: torch.Tensor | None = None,
        document_page_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        supplied = {
            "vision": images,
            "audio": audio,
            "document": document_input_ids,
        }
        unsupported = [name for name, value in supplied.items() if value is not None and name not in self.modalities]
        if unsupported:
            raise ValueError(f"model was not configured for modalities: {unsupported}")
        if document_boxes is not None and document_input_ids is None:
            raise ValueError("document_boxes require document_input_ids")
        if document_page_ids is not None and document_input_ids is None:
            raise ValueError("document_page_ids require document_input_ids")

        for modality, tensor in supplied.items():
            if tensor is not None and tensor.shape[0] != input_ids.shape[0]:
                raise ValueError(f"{modality} batch size must match input_ids")

        text_embeddings = self.embed_tokens(input_ids)
        embeddings = [text_embeddings]
        if images is not None:
            if self.vision_encoder is None:
                raise RuntimeError("vision encoder is not initialized")
            embeddings.append(self.vision_encoder(images).to(dtype=text_embeddings.dtype))
        if audio is not None:
            if self.audio_encoder is None:
                raise RuntimeError("audio encoder is not initialized")
            embeddings.append(self.audio_encoder(audio).to(dtype=text_embeddings.dtype))
        if document_input_ids is not None:
            if self.document_encoder is None:
                raise RuntimeError("document encoder is not initialized")
            embeddings.append(
                self.document_encoder(
                    document_input_ids,
                    boxes=document_boxes,
                    page_ids=document_page_ids,
                ).to(dtype=text_embeddings.dtype)
            )

        fused_embeddings = torch.cat(embeddings, dim=1)
        if fused_embeddings.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                f"fused sequence length {fused_embeddings.shape[1]} exceeds "
                f"max_position_embeddings {self.config.max_position_embeddings}"
            )
        hidden = fused_embeddings
        cos, sin = self.layers[0].attention.rotary_emb(hidden, seq_len=hidden.shape[1])
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, cos, sin)
            else:
                hidden = layer(hidden, cos, sin)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            if labels.shape != (input_ids.shape[0], fused_embeddings.shape[1]):
                raise ValueError("labels must match the fused input sequence shape")
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}
