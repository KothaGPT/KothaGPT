"""WS-11: deterministic long-context benchmark generator.

Writes ``data/benchmarks/bangla/long``:

- ``needle.jsonl`` — needle-in-a-haystack recall records. A distinctive
  fact sentence is embedded inside a long Bangla haystack at a controlled
  depth (25/50/75%) and target context length (0.5k/1k/2k/4k/8k).
- ``long_ppl.jsonl`` — perplexity-vs-context probes: a coherent reference
  paragraph whose ppl the harness measures given a growing filler prefix,
  to detect ppl degradation when the context exceeds the trained length.

Every record carries a deterministic ``record_id``; the dev/test split is a
stable 80/20 hash of that id (same scheme as the v1 benchmark).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_OUT_DIR = Path(__file__).parent / "long"
_DEV_FRACTION = 8

_HAYSTACK_CHARS_PER_TOKEN = 6.0

_FILLER = (
    "বাংলা ভাষা একটি সমৃদ্ধ ইন্দো-আর্য ভাষা। এই ভাষায় বিশ্বজুড়ে "
    "প্রায় তিরিশ কোটি মানুষ কথা বলেন। বাংলা সাহিত্যের ইতিহাস বহু শতাব্দী "
    "প্রাচীন, এবং এর শিল্পীরা বিশ্বজুড়ে খ্যাতি অর্জন করেছেন। "
    "ভাষাটি বাংলাদেশ, পশ্চিমবঙ্গ, ত্রিপুরা এবং আসামের কিছু অঞ্চলে সরকারি "
    "ভাষা হিসেবে ব্যবহৃত হয়। প্রতিদিন লক্ষ লক্ষ মানুষ বাংলায় সংবাদপত্র, "
    "বই এবং অনলাইন নিবন্ধ পড়েন।"
)

# (needle, question, answer) — verbatim-span answers for recall checking.
_NEEDLES: list[tuple[str, str, str]] = [
    ("পদ্মা সেতু ২০২২ সালের ২৫ জুন উদ্বোধন করা হয়।", "পদ্মা সেতু কবে উদ্বোধন করা হয়?", "২০২২ সালের ২৫ জুন"),
    ("জাতীয় কবি কাজী নজরুল ইসলাম ১৮৯৯ সালে জন্মগ্রহণ করেন।", "কাজী নজরুল ইসলাম কত সালে জন্মগ্রহণ করেন?", "১৮৯৯"),
    ("ঢাকা শহরের পুরনাম ১৬০৮ সালে প্রতিষ্ঠিত হয়েছিল।", "ঢাকার পুরনাম কত সালে প্রতিষ্ঠিত হয়?", "১৬০৮"),
    (
        "ভাষা শহীদদের স্মরণে ১৯৫২ সালে প্রথম শহীদ মিনার নির্মিত হয়।",
        "প্রথম শহীদ মিনার কত সালে নির্মিত হয়?",
        "১৯৫২",
    ),
    ("বাংলা একাডেমি ১৯৫৫ সালে প্রতিষ্ঠিত হয়।", "বাংলা একাডেমি কত সালে প্রতিষ্ঠিত হয়?", "১৯৫৫"),
]


def _split(record_id: str) -> str:
    digest = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8], 16)
    return "dev" if digest % 10 < _DEV_FRACTION else "test"


def _filler_tokens(count: int) -> str:
    """Deterministic filler text of approximately ``count`` tokens."""
    sentence = _FILLER.split("। ")[0] + "। "
    chunks: list[str] = []
    total = 0
    i = 0
    while total < count:
        variant = f"এটি একটি সাধারণ উদাহরণ বাক্য, সংখ্যা {i}। {sentence}"
        chunks.append(variant)
        total += len(variant) / _HAYSTACK_CHARS_PER_TOKEN
        i += 1
    return " ".join(chunks)


def _build_needle_records() -> list[dict]:
    records: list[dict] = []
    for context_len in (512, 1024, 2048, 4096, 8192):
        for depth in (0.25, 0.5, 0.75):
            for needle, question, answer in _NEEDLES:
                filler_total = context_len - len(needle) / _HAYSTACK_CHARS_PER_TOKEN
                haystack = _filler_tokens(int(filler_total))
                marker = int(len(haystack) * depth)
                inserted = haystack[:marker] + " " + needle + " " + haystack[marker:]
                record_id = f"needle-{context_len}-{int(depth * 100)}-{answer[:6]}"
                records.append(
                    {
                        "record_id": record_id,
                        "task": "needle",
                        "split": _split(record_id),
                        "context_len": context_len,
                        "depth_pct": depth,
                        "haystack": inserted,
                        "question": question,
                        "answer": answer,
                    }
                )
    return records


def _build_ppl_records() -> list[dict]:
    records: list[dict] = []
    reference = (
        "বাংলা ভাষার বর্ণমালায় বায়ান্নটি অক্ষর রয়েছে এবং এটি মোটামুটি "
        "একটি ফোনেটিক ভাষা, অর্থাৎ যেভাবে লেখা হয় সেভাবেই উচ্চারণ করা হয়। "
        "বাংলা লিপি ব্রাহ্মী পরিবারের অন্তর্গত এবং এটি পঞ্চম সবচেয়ে বেশি ব্যবহৃত "
        "লেখার পদ্ধতি।"
    )
    for context_len in (512, 2048, 8192):
        record_id = f"ppl-probe-{context_len}"
        records.append(
            {
                "record_id": record_id,
                "task": "long_ppl",
                "split": _split(record_id),
                "context_len": context_len,
                "filler": _filler_tokens(context_len),
                "reference": reference,
            }
        )
    return records


def generate() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (_OUT_DIR / "needle.jsonl").open("w", encoding="utf-8") as fh:
        for record in _build_needle_records():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (_OUT_DIR / "long_ppl.jsonl").open("w", encoding="utf-8") as fh:
        for record in _build_ppl_records():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "name": "bangla",
        "version": "long",
        "tasks": {
            "needle": {"total": len(_build_needle_records())},
            "long_ppl": {"total": len(_build_ppl_records())},
        },
    }
    (_OUT_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate()
    print(f"wrote {_OUT_DIR}")
