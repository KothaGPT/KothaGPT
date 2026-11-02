"""Bangla punctuation normalization (Phase 2 — WS-3).

Canonical forms:
- danda: ``॥`` collapses to the single danda ``।`` (U+0964); runs collapse,
  whitespace before a danda is removed, and a space is inserted after it when
  it is directly followed by a letter (so ``।।`` and ``word।`` normalise).
- quotes: curly quotes (``“ ” ‘ ’ „ ‚ « »``) map to straight ``"``/``'``.
- dashes: en/em dashes (``– —``) map to hyphen-minus ``-``.
- ellipsis: ``…`` maps to ``...``.
- digits: Bengali (০-৯) and Arabic-Indic (٠-٩) digits can be mapped to ASCII
  (or the reverse) via :func:`convert_digits`.
"""

from __future__ import annotations

import re

__all__ = [
    "convert_digits",
    "normalize_bangla_punctuation",
    "normalize_digits",
]

_DANDA = "\u0964"

_DANDA_DOUBLE_RE = re.compile("\u0965")  # ॥
_DANDA_RUN_RE = re.compile("\u0964+")
_DANDA_SPACE_RE = re.compile(r"\s+\u0964")
_DANDA_JOIN_RE = re.compile(r"\u0964(?=[^\s\u0964])")

_CURLY_QUOTE_TABLE = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
    }
)
_DASH_TABLE = str.maketrans({"\u2013": "-", "\u2014": "-"})
_ELLIPSIS_RE = re.compile("\u2026")

_BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ASCII_DIGITS = "0123456789"

_BENGALI_TO_ASCII = str.maketrans(_BENGALI_DIGITS, _ASCII_DIGITS)
_ASCII_TO_BENGALI = str.maketrans(_ASCII_DIGITS, _BENGALI_DIGITS)
_ARABIC_TO_ASCII = str.maketrans(_ARABIC_INDIC_DIGITS, _ASCII_DIGITS)


def convert_digits(text: str, target: str = "ascii") -> str:
    """Normalize digit forms.

    ``target`` is ``"ascii"`` (Bangla + Arabic-Indic -> ASCII) or ``"bengali"``
    (ASCII -> Bangla). Any other value returns ``text`` unchanged.
    """
    if target == "ascii":
        text = text.translate(_BENGALI_TO_ASCII)
        return text.translate(_ARABIC_TO_ASCII)
    if target == "bengali":
        return text.translate(_ASCII_TO_BENGALI)
    return text


def normalize_digits(text: str, style: str = "ascii") -> str:
    """Alias for :func:`convert_digits` using a style name."""
    return convert_digits(text, target=style)


def normalize_bangla_punctuation(text: str) -> str:
    """Apply the canonical Bangla punctuation rules."""
    text = _DANDA_DOUBLE_RE.sub(_DANDA, text)
    text = _DANDA_RUN_RE.sub(_DANDA, text)
    text = _DANDA_SPACE_RE.sub(_DANDA, text)
    text = _DANDA_JOIN_RE.sub(_DANDA + " ", text)
    text = text.translate(_CURLY_QUOTE_TABLE)
    text = text.translate(_DASH_TABLE)
    return _ELLIPSIS_RE.sub("...", text)
