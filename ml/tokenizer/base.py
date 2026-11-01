"""Base tokenizer interface shared by the BPE, Unigram, and WordPiece implementations.

Both algorithms operate word-based with a leading "▁" (U+2581) space marker,
GPT-2 style: every word is encoded as "▁"+word and the marker is restored to a
space on decode. This lets merges learn common word-stem tokens and keeps
spacing lossless for single-spaced text.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

UNK = "<unk>"
SPECIAL_TOKENS = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
_WORD_MARKER = "\u2581"

# Characters always present in the vocabulary so that unseen Bangla/ASCII text
# can still be tokenized instead of collapsing to <unk>. The full Bengali block
# is a small fixed overhead (~128 codepoints) and includes vowel signs and
# combining marks. The danda (। U+0964) is shared with Devanagari but is the
# standard Bangla sentence terminator, so it is included explicitly.
_BENGALI_BLOCK = [chr(cp) for cp in range(0x0980, 0x0A00)]
_DANDA = ["\u0964", "\u0965"]
_ASCII_PRINTABLE = [chr(cp) for cp in range(0x20, 0x7F)]
_COMMON_PUNCT = [
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2018",  # left single quote
    "\u2019",  # right single quote
    "\u201c",  # left double quote
    "\u201d",  # right double quote
    "\u2026",  # horizontal ellipsis
    "\u00ab",  # left guillemet
    "\u00bb",  # right guillemet
]
BOOTSTRAP_CHARS: frozenset[str] = frozenset(
    _BENGALI_BLOCK + _DANDA + _ASCII_PRINTABLE + _COMMON_PUNCT + [_WORD_MARKER]
)


class BaseTokenizer(ABC):
    """Common vocabulary, encode/decode and persistence behaviour."""

    type: str = "base"

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab: dict[str, int] = vocab
        self.id_to_token: dict[int, str] = {i: t for t, i in vocab.items()}
        self.unk_id: int = vocab[UNK]

    @abstractmethod
    def _encode_word(self, word: str) -> list[str]:
        """Return the subword tokens for a single ``▁``-prefixed word."""

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids (no special tokens)."""
        ids: list[int] = []
        for word in text.split():
            for token in self._encode_word(_WORD_MARKER + word):
                ids.append(self.vocab.get(token, self.unk_id))
        return ids

    def tokenize(self, text: str) -> list[str]:
        """Encode text into a list of token strings."""
        return [self.id_to_token.get(i, UNK) for i in self.encode(text)]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to text."""
        tokens = [self.id_to_token.get(i, UNK) for i in ids]
        return "".join(tokens).replace(_WORD_MARKER, " ").strip()

    def stats(self, text: str) -> dict[str, float]:
        """Token-efficiency metrics for a single text."""
        ids = self.encode(text)
        chars = len(text)
        words = max(len(text.split()), 1)
        return {
            "tokens": len(ids),
            "chars": chars,
            "words": words,
            "tokens_per_char": len(ids) / chars if chars else 0.0,
            "tokens_per_word": len(ids) / words,
            "unk": sum(1 for i in ids if i == self.unk_id),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "vocab": self.vocab}

    def save(self, out_dir: Path) -> Path:
        """Persist the tokenizer to ``out_dir/tokenizer.json``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "tokenizer.json"
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        path.write_text(payload + "\n", encoding="utf-8")
        return path


def load_tokenizer(path: str | Path) -> BaseTokenizer:
    """Load a tokenizer saved by ``BaseTokenizer.save`` (auto-detects type)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = data["type"]
    if kind == "bpe":
        from .bpe import BpeTokenizer

        return BpeTokenizer.from_dict(data)
    if kind == "unigram":
        from .unigram import UnigramTokenizer

        return UnigramTokenizer.from_dict(data)
    if kind == "wordpiece":
        from .wordpiece import WordPieceTokenizer

        return WordPieceTokenizer.from_dict(data)
    raise ValueError(f"unknown tokenizer type: {kind}")
