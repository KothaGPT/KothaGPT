"""Multimodal encoder modules for KothaGPT."""

from __future__ import annotations


class VisionEncoder:
    """Frozen ViT-style vision encoder projecting image patches to hidden space."""

    def __init__(self, hidden_size: int, mm_hidden: int = 512):
        self.hidden_size = hidden_size
        self.mm_hidden = mm_hidden

    def encode(self, images):
        """Encode images into hidden-space embeddings."""
        raise NotImplementedError


class AudioEncoder:
    """Frozen audio encoder projecting audio features to hidden space."""

    def __init__(self, hidden_size: int, mm_hidden: int = 512):
        self.hidden_size = hidden_size
        self.mm_hidden = mm_hidden

    def encode(self, audio):
        """Encode audio into hidden-space embeddings."""
        raise NotImplementedError


class DocumentEncoder:
    """Document encoder combining OCR text, layout features, and page structure."""

    def __init__(self, hidden_size: int, mm_hidden: int = 512):
        self.hidden_size = hidden_size
        self.mm_hidden = mm_hidden

    def encode(self, document):
        """Encode document into multimodal embeddings."""
        raise NotImplementedError
