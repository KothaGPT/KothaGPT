"""Patch-based image encoder and hidden-space projector."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class VisionProjector(nn.Module):
    """Project image patch features into the decoder hidden dimension."""

    def __init__(self, mm_hidden: int, hidden_size: int):
        super().__init__()
        if mm_hidden <= 0 or hidden_size <= 0:
            raise ValueError("mm_hidden and hidden_size must be positive")
        self.mm_hidden = mm_hidden
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(mm_hidden, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        if patch_features.shape[-1] != self.mm_hidden:
            raise ValueError(f"expected patch features with width {self.mm_hidden}")
        return self.projection(patch_features)


class VisionEncoder(nn.Module):
    """Lightweight patch encoder for RGB images with positional embeddings.

    This is a trainable convolutional encoder, not a pretrained or frozen ViT.
    Images may be provided as [C,H,W] or [B,C,H,W] RGB tensors.
    """

    def __init__(
        self,
        hidden_size: int,
        mm_hidden: int = 512,
        patch_size: int = 16,
        image_size: int = 224,
    ):
        super().__init__()
        if min(hidden_size, mm_hidden, patch_size, image_size) <= 0:
            raise ValueError("hidden_size, mm_hidden, patch_size, and image_size must be positive")
        self.hidden_size = hidden_size
        self.mm_hidden = mm_hidden
        self.patch_size = patch_size
        self.image_size = image_size
        self.patch_embedding = nn.Conv2d(
            3,
            mm_hidden,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, mm_hidden, math.ceil(image_size / patch_size) ** 2)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.projector = VisionProjector(mm_hidden, hidden_size)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return [batch, patch_count, hidden_size] embeddings."""
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if not images.is_floating_point():
            raise TypeError("images must be a floating-point tensor")
        if images.shape[0] == 0:
            raise ValueError("images must contain at least one image")
        if images.shape[-2] <= 0 or images.shape[-1] <= 0:
            raise ValueError("image height and width must be positive")
        if not torch.isfinite(images).all():
            raise ValueError("images must contain only finite values")

        images = F.interpolate(
            images.to(dtype=self.patch_embedding.weight.dtype),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.patch_embedding(images)
        features = features.flatten(2).transpose(1, 2)
        positions = F.interpolate(
            self.position_embedding,
            size=features.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return self.projector(features + positions)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)
