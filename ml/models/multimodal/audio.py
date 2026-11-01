"""Waveform encoder and hidden-space projector."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AudioProjector(nn.Module):
    """Project audio-frame features into the decoder hidden dimension."""

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

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.shape[-1] != self.mm_hidden:
            raise ValueError(f"expected audio features with width {self.mm_hidden}")
        return self.projection(frame_features)


class AudioEncoder(nn.Module):
    """Encode mono waveforms into a sequence of projected log-spectral frames.

    Inputs are [samples] or [batch, samples] floating-point tensors. This
    self-contained spectral encoder does not require pretrained audio packages.
    """

    def __init__(
        self,
        hidden_size: int,
        mm_hidden: int = 512,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()
        if hidden_size <= 0 or mm_hidden <= 0 or n_fft < 4 or hop_length <= 0:
            raise ValueError("invalid audio encoder dimensions")
        self.hidden_size = hidden_size
        self.mm_hidden = mm_hidden
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectral_projection = nn.Linear(n_fft // 2 + 1, mm_hidden)
        self.projector = AudioProjector(mm_hidden, hidden_size)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Return [batch, frame_count, hidden_size] embeddings."""
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2:
            raise ValueError("audio must have shape [batch, samples]")
        if not audio.is_floating_point():
            raise TypeError("audio must be a floating-point tensor")
        if audio.shape[0] == 0:
            raise ValueError("audio batch must contain at least one waveform")
        if audio.shape[-1] == 0:
            raise ValueError("audio must contain at least one sample")
        if not torch.isfinite(audio).all():
            raise ValueError("audio must contain only finite samples")

        audio = audio.float()
        if audio.shape[-1] < self.n_fft:
            audio = F.pad(audio, (0, self.n_fft - audio.shape[-1]))
        window = torch.hann_window(self.n_fft, device=audio.device, dtype=audio.dtype)
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        log_magnitude = torch.log1p(spectrum.abs()).transpose(1, 2)
        features = F.gelu(
            self.spectral_projection(log_magnitude.to(dtype=self.spectral_projection.weight.dtype))
        )
        return self.projector(features)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio)
