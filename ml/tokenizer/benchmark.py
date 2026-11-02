"""Token-efficiency benchmarks and the sample test texts.

The benchmark covers the Phase 1B test matrix plus the Phase 2 (WS-7) sets:
Bangla text, Bangla/English mixed text, punctuation-heavy text, romanized
(transliterated) Bangla, digits, proper nouns / English loanwords, emoji, and
social-media style code-switched text.

Metrics per set:
- ``tokens_per_char`` / ``tokens_per_word`` (efficiency)
- ``unk_rate`` (unknown-token rate; target < 0.5% on the dev sets)
- ``decode_fidelity`` (encode -> decode == original; target 100%)
- ``compression_vs_byte`` / ``compression_vs_char`` (tokens relative to a
  byte-level / char-level tokenizer baseline)
- ``paragraph_stability`` (consistency of tokens/char across paragraphs)
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

from .base import BaseTokenizer
from .transliterate import latin_to_bangla

__all__ = [
    "GATED_SETS",
    "GATE_THRESHOLDS",
    "PREPROCESS",
    "SAMPLE_TEXTS",
    "check_benchmark",
    "run_benchmark",
]

SAMPLE_TEXTS: dict[str, str] = {
    "bangla": (
        "বাংলা ভাষা বাংলাদেশের মানুষের মাতৃভাষা। এটি একটি প্রাচীন এবং সমৃদ্ধ ভাষা, "
        "যার ইতিহাস হাজার বছরের পুরনো। রবীন্দ্রনাথ ঠাকুর, নজরুল ইসলাম এবং "
        "জীবনানন্দ দাশের মতো কবিরা এই ভাষায় তাদের অমর সৃষ্টি রেখে গেছেন। "
        "বাংলা সাহিত্য বিশ্বসাহিত্যের এক অনন্য সম্পদ। বাংলাদেশের স্বাধীনতা যুদ্ধের "
        "ইতিহাসে বাংলা ভাষা আন্দোলন একটি গুরুত্বপূর্ণ অধ্যায়।"
    ),
    "mixed": (
        "আমার ছোট্ট সোনামণি এবং আমি, একটি সুন্দর Sunday সকালে ঘুরতে গিয়েছিলাম। "
        "We visited the National Museum এবং তারপর lunch করলাম। The weather was "
        "beautiful এবং সবকিছু মনে হচ্ছিল perfect।"
    ),
    "punctuation": (
        "বাংলা ভাষা! এটি কি সত্যিই সুন্দর? হ্যাঁ, অবশ্যই! কমা, সেমিকোলন; কোলন: "
        "এবং ব্র্যাকেট (নিয়মিত) ব্যবহার করা হয়। প্রশ্নবোধক? বিস্ময়বোধক! "
        'উদ্ধৃতি "চিহ্ন" এবং ড্যাশ — সবই এখানে আছে।'
    ),
    "emoji": (
        "আমি বাংলা ভালোবাসি ❤️ এই ভাষা আমার গর্ব 🥰 বাংলাদেশ 🇧🇩 এ জন্মেছি আমরা 🎉 শুভ জন্মদিন 🎂 ধন্যবাদ 🙏 হাসি 😂"
    ),
    "code": (
        "def add(a, b):\n"
        "    # বাংলা মন্তব্য: দুটি সংখ্যা যোগ করুন\n"
        "    result = a + b\n"
        '    print(f"সমষ্টি: {result}")\n'
        "    return result"
    ),
    # WS-7: romanized Bangla, converted to Bangla script before tokenizing.
    "translit": (
        "ami bangla bhalobashi. eta amader matribhasha. bangladesher manush "
        "pratidin bangla kotha bole. amra lekhi, porhi ebong swapno dekhi. "
        "bangla sahitya bishwasahityer ek onnopo sompod."
    ),
    "digits": (
        "১২৩৪৫৬৭৮৯০ — ২০২৫ সালে বাংলাদেশের জনসংখ্যা প্রায় ১৭ কোটি। "
        "এক কোটি = ১০,০০০,০০০। তিন দশমিক পাঁচ = ৩.৫। শতকরা ৯৯.৯% মানুষ কথা বলে।"
    ),
    "names": (
        "রবীন্দ্রনাথ ঠাকুর, কাজী নজরুল ইসলাম এবং জসীমউদ্দীন বাংলা সাহিত্যের পথিকৃৎ। "
        "ঢাকা, চট্টগ্রাম, সিলেট এবং খুলনা বাংলাদেশের প্রধান শহর। "
        "হুগলি নদী, পদ্মা সেতু, Cox's Bazar, Sundarban এবং ঢাকা বিশ্ববিদ্যালয়।"
    ),
    "social": (
        "আজকের দিনটা খুব সুন্দর ছিল! 😊 বন্ধুদের সাথে hangout করে dope day। "
        "সকালে coffee খেলাম ☕ তারপর gym এ গেলাম 💪 OMG, ভাই কি happen করলো 😅 "
        "লাইক 👍 শেয়ার 🔁 কমেন্ট 📝 #trending #bangla"
    ),
}

# Sets whose texts are fed through a preprocessor before tokenizing.
PREPROCESS: dict[str, Callable[[str], str]] = {
    "translit": latin_to_bangla,
}

# Sets used for the hard CI gate. They are Bangla/ASCII only (no emoji or
# supplementary planes), so a bootstrap-covered tokenizer must be lossless.
GATED_SETS: frozenset[str] = frozenset(
    {"bangla", "mixed", "punctuation", "translit", "digits", "names"}
)

GATE_THRESHOLDS: dict[str, float] = {
    # The frozen tokenizer must not regress on the headline efficiency metric.
    "max_avg_tokens_per_char": 2.0,
    # Unknown-token rate target from the plan (< 0.5% on the dev sets).
    "max_dev_unk_rate": 0.005,
    # decode round-trip must be lossless on the dev sets.
    "min_dev_decode_fidelity": 1.0,
    # The tokenizer should beat a per-character tokenizer on Bangla dev sets.
    "min_dev_compression_vs_char": 1.0,
}


def _paragraph_stability(text: str, tokenizer: BaseTokenizer) -> float:
    """Fractional tokens/char consistency across paragraphs (1.0 = stable)."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return 1.0
    tpc = [tokenizer.stats(p)["tokens_per_char"] for p in paragraphs]
    mean = statistics.mean(tpc)
    if mean <= 0:
        return 0.0
    std = statistics.pstdev(tpc)
    return max(0.0, 1.0 - std / mean)


def _set_metrics(
    tokenizer: BaseTokenizer, text: str, *, preprocess: Callable[[str], str] | None = None
) -> dict[str, float]:
    """Rich metrics for a single benchmark text."""
    if preprocess is not None:
        text = preprocess(text)
    ids = tokenizer.encode(text)
    chars = len(text)
    byte_len = len(text.encode("utf-8"))
    tokens = len(ids)
    words = max(len(text.split()), 1)
    unk = sum(1 for i in ids if i == tokenizer.unk_id)
    return {
        "tokens": tokens,
        "chars": chars,
        "bytes": byte_len,
        "words": words,
        "tokens_per_char": tokens / chars if chars else 0.0,
        "tokens_per_word": tokens / words,
        "unk": unk,
        "unk_rate": unk / tokens if tokens else 0.0,
        "decode_fidelity": 1.0 if tokenizer.decode(ids) == text else 0.0,
        "compression_vs_byte": byte_len / tokens if tokens else 0.0,
        "compression_vs_char": chars / tokens if tokens else 0.0,
        "paragraph_stability": _paragraph_stability(text, tokenizer),
    }


def run_benchmark(tokenizer: BaseTokenizer, texts: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the efficiency benchmark; returns per-set metrics and aggregates."""
    if texts is None:
        texts = SAMPLE_TEXTS
    per_set: dict[str, dict[str, float]] = {}
    for name, text in texts.items():
        per_set[name] = _set_metrics(tokenizer, text, preprocess=PREPROCESS.get(name))

    def _avg(key: str, names: list[str]) -> float:
        values = [per_set[n][key] for n in names if n in per_set]
        return statistics.mean(values) if values else 0.0

    def _min(key: str, names: list[str]) -> float:
        values = [per_set[n][key] for n in names if n in per_set]
        return min(values) if values else 1.0

    def _max(key: str, names: list[str]) -> float:
        values = [per_set[n][key] for n in names if n in per_set]
        return max(values) if values else 0.0

    all_names = list(per_set)
    gated = [n for n in GATED_SETS if n in per_set]
    return {
        "per_set": per_set,
        "num_sets": len(all_names),
        "avg_tokens_per_char": _avg("tokens_per_char", all_names),
        "avg_unk_rate": _avg("unk_rate", all_names),
        "min_decode_fidelity": _min("decode_fidelity", all_names),
        "avg_compression_vs_byte": _avg("compression_vs_byte", all_names),
        "avg_compression_vs_char": _avg("compression_vs_char", all_names),
        "avg_paragraph_stability": _avg("paragraph_stability", all_names),
        # Aggregates over the gated (Bangla dev) sets only.
        "dev_avg_tokens_per_char": _avg("tokens_per_char", gated),
        "dev_max_unk_rate": _max("unk_rate", gated),
        "dev_min_decode_fidelity": _min("decode_fidelity", gated),
        "dev_min_compression_vs_char": _min("compression_vs_char", gated),
    }


def check_benchmark(
    result: dict[str, Any], thresholds: dict[str, float] | None = None
) -> list[str]:
    """Return a list of threshold violations; empty list means the gate passes."""
    t = dict(GATE_THRESHOLDS if thresholds is None else thresholds)
    failures: list[str] = []
    checks = (
        (
            "avg_tokens_per_char",
            "max_avg_tokens_per_char",
            result.get("avg_tokens_per_char", 0.0),
            ">",
        ),
        ("dev_max_unk_rate", "max_dev_unk_rate", result.get("dev_max_unk_rate", 1.0), ">"),
        (
            "dev_min_decode_fidelity",
            "min_dev_decode_fidelity",
            result.get("dev_min_decode_fidelity", 0.0),
            "<",
        ),
        (
            "dev_min_compression_vs_char",
            "min_dev_compression_vs_char",
            result.get("dev_min_compression_vs_char", 0.0),
            "<",
        ),
    )
    for metric, key, value, op in checks:
        if key not in t:
            continue
        bound = t[key]
        bad = value > bound if op == ">" else value < bound
        if bad:
            failures.append(f"{metric}={value:.4f} violates {key}={bound}")
    return failures
