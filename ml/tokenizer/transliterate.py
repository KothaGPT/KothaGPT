"""Rule-based Bangla <-> Latin transliteration (Phase 2 — WS-5).

Deterministic, longest-match Avro-style table. This is a *generic* phonetic
transliterator, not a full Avro implementation: ambiguous Bangla letters that
share a Latin spelling (ত/ট -> t, দ/ড -> d, ন/ণ -> n, শ/ষ/স -> sh/s) use a
documented default. See ``docs/bangla-transliteration.md`` for the variant
table and coverage notes.
"""

from __future__ import annotations

__all__ = [
    "BENGALI_CONSONANTS",
    "BENGALI_VOWELS",
    "VOWEL_MATRAS",
    "bangla_to_latin",
    "latin_to_bangla",
]

# --- Latin -> Bangla -------------------------------------------------------

# Standalone vowel forms (word-initial or after a non-consonant).
_BENGALI_VOWELS = {
    "a": "\u0986",  # আ
    "aa": "\u0986",
    "i": "\u0987",  # ই
    "ii": "\u0988",  # ঈ
    "u": "\u0989",  # উ
    "uu": "\u098a",  # ঊ
    "e": "\u098f",  # এ
    "oi": "\u0990",  # ঐ
    "o": "\u0985",  # অ
    "ou": "\u0994",  # ঔ
    "rri": "\u098b",  # ঋ
}

# Matra (vowel sign) forms, used after a consonant.
_VOWEL_MATRAS = {
    "a": "\u09be",  # া
    "aa": "\u09be",
    "i": "\u09bf",  # ি
    "ii": "\u09c0",  # ী
    "u": "\u09c1",  # ু
    "uu": "\u09c2",  # ূ
    "e": "\u09c7",  # ে
    "oi": "\u09c8",  # ৈ
    "o": "\u09cb",  # ো
    "ou": "\u09cc",  # ৌ
    "rri": "\u09c3",  # ৃ
}

# Consonants, longest match first. Order matters for digraphs.
# Lowercase maps to the dental/normal series (ত, দ, ন, থ ...); capitalised
# forms map to the retroflex series (ট, ড, ণ, ঠ ...) per ISO-15919-ish style.
_BENGALI_CONSONANTS = {
    "chh": "\u099b",  # ছ
    "kh": "\u0996",  # খ
    "gh": "\u0998",  # ঘ
    "ng": "\u0982",  # ং (anusvara; the common romanized "ng")
    "jh": "\u099d",  # ঝ
    "Th": "\u09a0",  # ঠ
    "th": "\u09a5",  # থ
    "Dh": "\u09a2",  # ঢ
    "dh": "\u09a7",  # ধ
    "ph": "\u09ab",  # ফ
    "bh": "\u09ad",  # ভ
    "sh": "\u09b6",  # শ
    "Rh": "\u09dd",  # ঢ়
    "ch": "\u099a",  # চ (before "c")
    "c": "\u099a",  # চ
    "k": "\u0995",  # ক
    "g": "\u0997",  # গ
    "j": "\u099c",  # জ
    "z": "\u09af",  # য (generic z -> য)
    "T": "\u099f",  # ট
    "t": "\u09a4",  # ত
    "D": "\u09a1",  # ড
    "d": "\u09a6",  # দ
    "N": "\u09a3",  # ণ
    "n": "\u09a8",  # ন
    "p": "\u09aa",  # প
    "f": "\u09ab",  # ফ
    "b": "\u09ac",  # ব
    "v": "\u09ad",  # ভ (generic v)
    "m": "\u09ae",  # ম
    "R": "\u09dc",  # ড়
    "r": "\u09b0",  # র
    "l": "\u09b2",  # ল
    "S": "\u09b7",  # ষ
    "s": "\u09b8",  # স
    "h": "\u09b9",  # হ
    "Y": "\u09df",  # য়
    "y": "\u09af",  # য
    "q": "\u0995",  # q -> ক (loanword assist)
    "x": "\u0995\u09cd\u09b7",  # x -> ক্ষ
}

_HASANTA = "\u09cd"  # ্
_ZWNJ = "\u200c"  # used in conjuncts to separate merged letters

# Signs that are matched via the consonant table but are not base consonants:
# they must not trigger a hasanta on the following letter.
_SIGN_GLYPHS = frozenset("\u0982\u0983\u0981")  # ং ঃ ঁ

_CONSONANT_KEYS = sorted(_BENGALI_CONSONANTS, key=len, reverse=True)
_VOWEL_KEYS = sorted(set(_BENGALI_VOWELS) | set(_VOWEL_MATRAS), key=len, reverse=True)


def _match_table(text: str, i: int, table: dict, keys: list[str]) -> tuple[str, str] | None:
    """Return (matched_latin, bangla) for the longest table entry at index i."""
    for key in keys:
        if text.startswith(key, i):
            return key, table[key]
    return None


def latin_to_bangla(text: str) -> str:
    """Transliterate romanized Bangla to Bangla script (Avro-style).

    Consonants take a following vowel as a matra; a consonant followed by
    another consonant or a boundary gets a hasanta (্).
    """
    out: list[str] = []
    i = 0
    n = len(text)
    after_consonant = False
    while i < n:
        ch = text[i]
        if ch.isspace():
            out.append(ch)
            i += 1
            after_consonant = False
            continue
        if ch.isdigit():
            out.append(ch)
            i += 1
            after_consonant = False
            continue
        if ch in ".,!?;:()\"'`":
            out.append(ch)
            i += 1
            after_consonant = False
            continue
        vowel = _match_table(text, i, _VOWEL_MATRAS, _VOWEL_KEYS)
        consonant = _match_table(text, i, _BENGALI_CONSONANTS, _CONSONANT_KEYS)
        if vowel is not None:
            key, sign = vowel
            if after_consonant and key in _VOWEL_MATRAS:
                out.append(sign)
            else:
                standalone = _BENGALI_VOWELS[key]
                out.append(standalone)
            i += len(key)
            after_consonant = False
            continue
        if consonant is not None:
            key, glyph = consonant
            if after_consonant:
                out.append(_HASANTA)
            out.append(glyph)
            i += len(key)
            after_consonant = glyph not in _SIGN_GLYPHS
            continue
        # Unknown Latin letter: pass through.
        out.append(ch)
        i += 1
        after_consonant = False
    return "".join(out)


# --- Bangla -> Latin -------------------------------------------------------

_LATIN_VOWELS = {
    "\u0985": "o",
    "\u0986": "a",
    "\u0987": "i",
    "\u0988": "i",
    "\u0989": "u",
    "\u098a": "u",
    "\u098f": "e",
    "\u0990": "oi",
    "\u0993": "o",
    "\u0994": "ou",
    "\u098b": "ri",
}
_LATIN_MATRAS = {
    "\u09be": "a",
    "\u09bf": "i",
    "\u09c0": "i",
    "\u09c1": "u",
    "\u09c2": "u",
    "\u09c7": "e",
    "\u09c8": "oi",
    "\u09cb": "o",
    "\u09cc": "ou",
    "\u09c3": "ri",
}
_LATIN_CONSONANTS = {
    "\u0995": "k",
    "\u0996": "kh",
    "\u0997": "g",
    "\u0998": "gh",
    "\u0999": "ng",
    "\u099a": "c",
    "\u099b": "chh",
    "\u099c": "j",
    "\u099d": "jh",
    "\u099e": "ng",
    "\u099f": "t",
    "\u09a0": "th",
    "\u09a1": "d",
    "\u09a2": "dh",
    "\u09a3": "n",
    "\u09a4": "t",
    "\u09a5": "th",
    "\u09a6": "d",
    "\u09a7": "dh",
    "\u09a8": "n",
    "\u09aa": "p",
    "\u09ab": "ph",
    "\u09ac": "b",
    "\u09ad": "bh",
    "\u09ae": "m",
    "\u09af": "j",
    "\u09b0": "r",
    "\u09b2": "l",
    "\u09b6": "sh",
    "\u09b7": "sh",
    "\u09b8": "s",
    "\u09b9": "h",
    "\u09dc": "r",
    "\u09dd": "rh",
    "\u09df": "y",
    "\u09ce": "t",
}
_LATIN_SIGNS = {
    "\u0982": "ng",
    "\u0983": "h",
    "\u0981": "n",  # ং ঃ ঁ
}


def bangla_to_latin(text: str) -> str:
    """Transliterate Bangla script to readable Latin (lossy, deterministic)."""
    out: list[str] = []
    for ch in text:
        if ch in _LATIN_CONSONANTS:
            out.append(_LATIN_CONSONANTS[ch])
        elif ch == _HASANTA:
            continue
        elif ch in _LATIN_MATRAS:
            out.append(_LATIN_MATRAS[ch])
        elif ch in _LATIN_VOWELS:
            out.append(_LATIN_VOWELS[ch])
        elif ch in _LATIN_SIGNS:
            out.append(_LATIN_SIGNS[ch])
        else:
            out.append(ch)
    return "".join(out)


VOWEL_MATRAS = dict(_VOWEL_MATRAS)
BENGALI_VOWELS = dict(_BENGALI_VOWELS)
BENGALI_CONSONANTS = dict(_BENGALI_CONSONANTS)
