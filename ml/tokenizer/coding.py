"""Coding-specific tokenizer extensions (WS-3).

Registers the code special tokens used by the coding stack:
  <code>, </code>, <function>, </function>, <bug>, </bug>, <explain>, </explain>
  <bangla_comment>, </bangla_comment>, <test>, </test>

These are additive: they extend an existing tokenizer's vocabulary without
breaking the frozen base vocab. The extensions are versioned so that a
tokenizer + coding-extension pair is content-addressable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

__all__ = [
    "CODING_SPECIAL_TOKENS",
    "CodingTokenExtensions",
    "apply_coding_extensions",
    "coding_extensions_digest",
]

# Ordered list of coding special tokens. New tokens must be appended, never
# inserted in the middle, to keep checkpoint compatibility.
CODING_SPECIAL_TOKENS: list[str] = [
    "<code>",
    "</code>",
    "<function>",
    "</function>",
    "<bug>",
    "</bug>",
    "<explain>",
    "</explain>",
    "<bangla_comment>",
    "</bangla_comment>",
    "<test>",
    "</test>",
]


@dataclass
class CodingTokenExtensions:
    """Coding special-token extension for a tokenizer."""

    tokens: list[str] = field(default_factory=lambda: list(CODING_SPECIAL_TOKENS))
    base_vocab_size: int = 0
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("tokens must be non-empty")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("coding tokens must be unique")
        if self.base_vocab_size < 0:
            raise ValueError("base_vocab_size must be non-negative")

    @property
    def offset(self) -> int:
        """First id assigned to a coding extension token."""
        return self.base_vocab_size

    def token_ids(self) -> dict[str, int]:
        """Map each coding token to its id in the extended vocabulary."""
        return {tok: self.offset + i for i, tok in enumerate(self.tokens)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "base_vocab_size": self.base_vocab_size,
            "tokens": list(self.tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodingTokenExtensions:
        if not isinstance(data, dict):
            raise TypeError("extensions data must be a mapping")
        return cls(
            tokens=list(data.get("tokens", CODING_SPECIAL_TOKENS)),
            base_vocab_size=int(data.get("base_vocab_size", 0)),
            version=str(data.get("version", "v1")),
        )

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def apply_coding_extensions(tokenizer, extensions: CodingTokenExtensions | None = None) -> dict[str, int]:
    """Return the coding token ids for a tokenizer, deriving base size from it.

    Does not mutate the tokenizer; callers that need a persistent extended
    vocab should save the returned mapping alongside the tokenizer.
    """
    base_vocab = getattr(tokenizer, "vocab", None)
    if not isinstance(base_vocab, dict):
        raise TypeError("tokenizer must expose a .vocab dict")
    base_size = len(base_vocab)
    ext = extensions or CodingTokenExtensions(base_vocab_size=base_size)
    return ext.token_ids()


def coding_extensions_digest(extensions: CodingTokenExtensions) -> str:
    return extensions.digest()


# Late import for typing only.
from typing import Any