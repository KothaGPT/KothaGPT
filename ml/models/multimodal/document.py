"""OCR text and page-layout encoder for document inputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DocumentProjector(nn.Module):
    """Project document token features into the decoder hidden dimension."""

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

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        if token_features.shape[-1] != self.mm_hidden:
            raise ValueError(f"expected document features with width {self.mm_hidden}")
        return self.projection(token_features)


class DocumentEncoder(nn.Module):
    """Encode token IDs and normalized bounding boxes as page-aware embeddings.

    Inputs are ``input_ids`` [batch, tokens] and optional ``boxes`` [batch,
    tokens, 4], normalized to [0, 1] in (x0, y0, x1, y1) order. Page IDs are
    optional non-negative integers [batch, tokens].
    """

    def __init__(
        self,
        hidden_size: int,
        mm_hidden: int = 512,
        vocab_size: int = 50000,
        max_pages: int = 128,
        coordinate_bins: int = 1001,
    ):
        super().__init__()
        if min(hidden_size, mm_hidden, vocab_size, max_pages) <= 0:
            raise ValueError("hidden_size, mm_hidden, vocab_size, and max_pages must be positive")
        if coordinate_bins < 2:
            raise ValueError("coordinate_bins must be at least 2")
        self.hidden_size = hidden_size
        self.mm_hidden = mm_hidden
        self.max_pages = max_pages
        self.coordinate_bins = coordinate_bins
        self.text_embedding = nn.Embedding(vocab_size, mm_hidden)
        self.layout_embeddings = nn.ModuleList(
            nn.Embedding(coordinate_bins, mm_hidden) for _ in range(4)
        )
        self.page_embedding = nn.Embedding(max_pages, mm_hidden)
        self.projector = DocumentProjector(mm_hidden, hidden_size)

    def encode(
        self,
        input_ids: torch.Tensor,
        boxes: torch.Tensor | None = None,
        page_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [batch, tokens, hidden_size] page- and layout-aware embeddings."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if input_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("input_ids must have an integer dtype")
        if input_ids.numel() == 0:
            raise ValueError("documents must contain at least one token")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.text_embedding.num_embeddings):
            raise ValueError("input_ids contain values outside the configured vocabulary")
        batch, tokens = input_ids.shape
        features = self.text_embedding(input_ids.long())

        if boxes is not None:
            if boxes.shape != (batch, tokens, 4):
                raise ValueError("boxes must have shape [batch, tokens, 4]")
            if not boxes.is_floating_point():
                raise TypeError("boxes must be a floating-point tensor")
            if not torch.isfinite(boxes).all() or torch.any(boxes < 0) or torch.any(boxes > 1):
                raise ValueError("box coordinates must be finite and normalized to [0, 1]")
            if torch.any(boxes[..., 2] < boxes[..., 0]) or torch.any(
                boxes[..., 3] < boxes[..., 1]
            ):
                raise ValueError("box coordinates must satisfy x1 >= x0 and y1 >= y0")
            coordinates = (boxes * (self.coordinate_bins - 1)).round().long()
            for index, embedding in enumerate(self.layout_embeddings):
                features = features + embedding(coordinates[..., index])

        if page_ids is not None:
            if page_ids.shape != (batch, tokens):
                raise ValueError("page_ids must have shape [batch, tokens]")
            if page_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("page_ids must have an integer dtype")
            if torch.any(page_ids < 0) or torch.any(page_ids >= self.max_pages):
                raise ValueError(f"page_ids must be in [0, {self.max_pages})")
            features = features + self.page_embedding(page_ids.long())

        features = F.layer_norm(features, (self.mm_hidden,))
        return self.projector(features)

    def forward(
        self,
        input_ids: torch.Tensor,
        boxes: torch.Tensor | None = None,
        page_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(input_ids, boxes, page_ids)
