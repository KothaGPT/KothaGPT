"""Tests for Phase 2 Wave A — Bangla language foundation preprocessing.

Covers WS-2 (Unicode normalization), WS-3 (punctuation normalization),
WS-4 (mixed Bangla/English text) and WS-5 (transliteration).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.pipeline import mixed, normalize, punctuation
from ml.tokenizer import transliterate

FIXTURES = Path(__file__).parent / "fixtures"
TRANSLIT_FIXTURE = FIXTURES / "transliteration.json"


# --- WS-2: Unicode normalization -------------------------------------------


def test_fix_bengali_yaphala_converts_zwj_to_hasanta():
    text = "ক\u200dয"  # ka + ZWJ + ya
    out = normalize.fix_bengali_yaphala(text)
    assert out == "ক\u09cd\u09af"  # ka + hasanta + ya


def test_fix_bengali_yaphala_leaves_plain_text_alone():
    text = "বাংলা ভাষা"
    assert normalize.fix_bengali_yaphala(text) == text


def test_unicode_normalize_nfc_keeps_conjuncts():
    text = "বাংলা\u200d\u200bলেখা"
    out = normalize.unicode_normalize(text)
    assert "\u200b" not in out  # zero-width space removed
    assert "\u200d" in out  # ZWJ kept


def test_nfkc_latin_only_normalizes_latin_but_not_bangla():
    text = "বাংলা ﬁnancial\u3000awful"  # ligature in the latin run
    out = normalize.nfkc_latin_only(text)
    assert "fi" in out  # ligature decomposed in the latin segment
    assert out.startswith("বাংলা")  # bangla text left untouched
    assert "বাংলা financial" in out


# --- WS-3: punctuation normalization ----------------------------------------


def test_danda_double_and_runs_collapse():
    assert punctuation.normalize_bangla_punctuation("কথা॥") == "কথা।"
    assert punctuation.normalize_bangla_punctuation("কথা।।।") == "কথা।"


def test_danda_spacing_rules():
    assert punctuation.normalize_bangla_punctuation("কথা।পর") == "কথা। পর"
    assert punctuation.normalize_bangla_punctuation("কথা ।") == "কথা।"


def test_curly_quotes_and_dashes_and_ellipsis():
    text = "তিনি বললেন \u201cহ্যালো\u201d \u2014 তারপর \u2026"
    out = punctuation.normalize_bangla_punctuation(text)
    assert "\u201c" not in out and "\u201d" not in out
    assert "\u2014" not in out and out.count('"') == 2
    assert "..." in out


def test_convert_digits_bengali_to_ascii_and_back():
    assert punctuation.convert_digits("১২৩") == "123"
    assert punctuation.convert_digits("123", target="bengali") == "১২৩"
    assert punctuation.convert_digits("٤٥", target="ascii") == "45"
    assert punctuation.convert_digits("12", target="keep") == "12"


def test_normalize_text_digit_style():
    out = normalize.normalize_text("আমার ১২৩ টাকা", digit_style="ascii")
    assert "123" in out


# --- WS-4: mixed Bangla/English text ---------------------------------------


def test_bengali_ratio_and_is_bangla():
    bn = "বাংলা ভাষা বাংলাদেশের মানুষের মাতৃভাষা।"
    assert mixed.bengali_ratio(bn) > 0.9
    assert mixed.is_bangla(bn)
    assert not mixed.is_bangla("This is an English sentence.")


def test_script_of():
    assert mixed.script_of("ক") == "bn"
    assert mixed.script_of("K") == "la"
    assert mixed.script_of("1") == "other"


def test_split_script_runs():
    runs = mixed.split_script_runs("Sundayসকালে 123")
    assert runs[0] == "Sunday"
    assert runs[1] == "সকালে 123"


def test_ensure_script_spacing():
    assert mixed.ensure_script_spacing("Sundayসকালে") == "Sunday সকালে"
    assert mixed.ensure_script_spacing("সকালেSunday") == "সকালে Sunday"
    assert mixed.ensure_script_spacing("plain") == "plain"


# --- WS-5: transliteration ---------------------------------------------------


def test_latin_to_bangla_known_words():
    assert transliterate.latin_to_bangla("ami") == "আমি"
    assert transliterate.latin_to_bangla("kotha") == "কোথা"
    assert transliterate.latin_to_bangla("bangla") == "বাংলা"
    assert transliterate.latin_to_bangla("bangladesh") == "বাংলাদেশ"


def test_latin_to_bangla_consonant_hasanta_joining():
    # kkha -> ক + ্ + খ + া
    assert transliterate.latin_to_bangla("kkha") == "ক\u09cdখা"


def test_latin_to_bangla_keeps_digits_and_punctuation():
    out = transliterate.latin_to_bangla("amar 5 ta")
    assert "5" in out
    out = transliterate.latin_to_bangla("ki, korcho?")
    assert "," in out and "?" in out


def test_bangla_to_latin_roundtrip_readable():
    assert transliterate.bangla_to_latin("বাংলা") == "bangla"
    assert transliterate.bangla_to_latin("কোথা") == "kotha"
    assert transliterate.bangla_to_latin("আমি") == "ami"


def test_bangla_to_latin_skips_hasanta():
    assert transliterate.bangla_to_latin("ক্কা") == "kka"


def test_transliterate_helpers_exposed():
    assert "া" in transliterate.VOWEL_MATRAS["a"]
    assert transliterate.BENGALI_CONSONANTS["k"] == "ক"
    assert transliterate.BENGALI_VOWELS["a"] == "আ"


@pytest.mark.parametrize("key", ["latin_to_bangla", "bangla_to_latin"])
def test_transliteration_golden_fixture(key: str):
    data = json.loads(TRANSLIT_FIXTURE.read_text(encoding="utf-8"))
    table = data[key]
    fn = getattr(transliterate, key)
    matches = sum(1 for src, expected in table.items() if fn(src) == expected)
    accuracy = matches / len(table)
    assert accuracy >= 0.95, f"{key} accuracy {accuracy:.2f} below 0.95"
