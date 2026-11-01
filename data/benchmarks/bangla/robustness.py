"""Generate deterministic dialect and mixed-script Bangla robustness records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "v1"
_TASK = "bangla_robustness"

_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "family": "formal",
        "dialect": "standard",
        "script_pattern": "bn",
        "phenomena": ["formal_register"],
        "region": "bn-BD",
        "domain": "daily_life",
        "source_text": "আমি প্রতিদিন সকাল সাতটায় বিদ্যালয়ে যাই।",
        "question": "আমি কখন বিদ্যালয়ে যাই?",
        "reference": "সকাল সাতটায়।",
    },
    {
        "family": "formal",
        "dialect": "standard",
        "script_pattern": "bn",
        "phenomena": ["formal_register"],
        "region": "bn-BD",
        "domain": "shopping",
        "source_text": "বাজার থেকে তিন কেজি চাল এবং এক লিটার তেল কেনা হয়েছে।",
        "question": "বাজার থেকে কী কী কেনা হয়েছে?",
        "reference": "তিন কেজি চাল এবং এক লিটার তেল।",
    },
    {
        "family": "formal",
        "dialect": "standard",
        "script_pattern": "bn",
        "phenomena": ["formal_register"],
        "region": "bn-BD",
        "domain": "travel",
        "source_text": "ঢাকা থেকে রাজশাহীগামী ট্রেনটি দুপুর দুইটায় ছাড়বে।",
        "question": "ট্রেনটি কখন ছাড়বে?",
        "reference": "দুপুর দুইটায়।",
    },
    {
        "family": "formal",
        "dialect": "standard",
        "script_pattern": "bn",
        "phenomena": ["formal_register"],
        "region": "bn-BD",
        "domain": "weather",
        "source_text": "বৃষ্টির সম্ভাবনা থাকায় তিনি সঙ্গে একটি ছাতা নিয়েছেন।",
        "question": "তিনি সঙ্গে কী নিয়েছেন?",
        "reference": "একটি ছাতা।",
    },
    {
        "family": "colloquial",
        "dialect": "colloquial",
        "script_pattern": "bn",
        "phenomena": ["colloquial_register"],
        "region": "bn-BD",
        "domain": "shopping",
        "source_text": "বাজার থেইকা দুই কেজি চাল আনছি।",
        "question": "বাজার থেকে কী আনা হয়েছে?",
        "reference": "দুই কেজি চাল।",
    },
    {
        "family": "colloquial",
        "dialect": "colloquial",
        "script_pattern": "bn",
        "phenomena": ["colloquial_register"],
        "region": "bn-BD",
        "domain": "weather",
        "source_text": "আজকে বৃষ্টি হইলে ছাতা নিয়া বাইর হমু।",
        "question": "বৃষ্টি হলে সঙ্গে কী নেওয়া হবে?",
        "reference": "ছাতা নেওয়া হবে।",
    },
    {
        "family": "colloquial",
        "dialect": "colloquial",
        "script_pattern": "bn",
        "phenomena": ["colloquial_register"],
        "region": "bn-BD",
        "domain": "school",
        "source_text": "কালকে আমরা বিকেলে মাঠে খেলতে যামু।",
        "question": "কখন মাঠে খেলতে যাওয়া হবে?",
        "reference": "কালকে বিকেলে।",
    },
    {
        "family": "colloquial",
        "dialect": "colloquial",
        "script_pattern": "bn",
        "phenomena": ["colloquial_register"],
        "region": "bn-BD",
        "domain": "daily_life",
        "source_text": "নানুর বাসায় কাল রাইতে থাকমু।",
        "question": "কাল রাতে কোথায় থাকা হবে?",
        "reference": "নানুর বাসায়।",
    },
    {
        "family": "colloquial",
        "dialect": "colloquial",
        "script_pattern": "bn",
        "phenomena": ["colloquial_register"],
        "region": "bn-BD",
        "domain": "daily_life",
        "source_text": "আমার ফোনের চার্জ শেষ, একটু চার্জার দে তো।",
        "question": "কী চাওয়া হয়েছে?",
        "reference": "একটি চার্জার।",
    },
    {
        "family": "code_switched",
        "dialect": "colloquial",
        "script_pattern": "bn_en",
        "phenomena": ["code_switching"],
        "region": "bn-BD",
        "domain": "work",
        "source_text": "আমি আজ ৫টার meeting-এ join করব।",
        "question": "কখন meeting-এ যোগ দেওয়া হবে?",
        "reference": "আজ ৫টায়।",
    },
    {
        "family": "code_switched",
        "dialect": "colloquial",
        "script_pattern": "bn_en",
        "phenomena": ["code_switching"],
        "region": "bn-BD",
        "domain": "work",
        "source_text": "ফাইলটা email-এ পাঠিয়ে দাও, please।",
        "question": "কী পাঠাতে বলা হয়েছে?",
        "reference": "ফাইলটি।",
    },
    {
        "family": "code_switched",
        "dialect": "colloquial",
        "script_pattern": "bn_en",
        "phenomena": ["code_switching"],
        "region": "bn-BD",
        "domain": "school",
        "source_text": "আজকের class online হবে, teacher জানিয়েছেন।",
        "question": "আজকের class কীভাবে হবে?",
        "reference": "অনলাইনে হবে।",
    },
    {
        "family": "code_switched",
        "dialect": "colloquial",
        "script_pattern": "bn_en",
        "phenomena": ["code_switching"],
        "region": "bn-BD",
        "domain": "travel",
        "source_text": "বাসে traffic বেশি, তাই আমি late হব।",
        "question": "দেরি হওয়ার কারণ কী?",
        "reference": "বাসে traffic বেশি।",
    },
    {
        "family": "code_switched",
        "dialect": "colloquial",
        "script_pattern": "bn_en",
        "phenomena": ["code_switching"],
        "region": "bn-BD",
        "domain": "work",
        "source_text": "project-এর deadline কাল, আজ রাতেই submit করব।",
        "question": "কখন project জমা দেওয়া হবে?",
        "reference": "আজ রাতেই।",
    },
    {
        "family": "transliterated",
        "dialect": "colloquial",
        "script_pattern": "latin_bn",
        "phenomena": ["transliteration"],
        "region": "bn-BD",
        "domain": "school",
        "source_text": "ami ajke school theke boi niye asbo",
        "question": "স্কুল থেকে কী আনা হবে?",
        "reference": "স্কুল থেকে বই আনা হবে।",
    },
    {
        "family": "transliterated",
        "dialect": "colloquial",
        "script_pattern": "latin_bn",
        "phenomena": ["transliteration"],
        "region": "bn-BD",
        "domain": "travel",
        "source_text": "kal shokal 8tay train chharbe",
        "question": "ট্রেন কখন ছাড়বে?",
        "reference": "ট্রেন কাল সকাল ৮টায় ছাড়বে।",
    },
    {
        "family": "transliterated",
        "dialect": "colloquial",
        "script_pattern": "latin_bn",
        "phenomena": ["transliteration"],
        "region": "bn-BD",
        "domain": "shopping",
        "source_text": "amar bon bazar theke aam kineche",
        "question": "বাজার থেকে আম কে কিনেছে?",
        "reference": "আমার বোন বাজার থেকে আম কিনেছে।",
    },
    {
        "family": "transliterated",
        "dialect": "colloquial",
        "script_pattern": "latin_bn",
        "phenomena": ["transliteration"],
        "region": "bn-BD",
        "domain": "daily_life",
        "source_text": "bristi hole amra bari te thakbo",
        "question": "বৃষ্টি হলে কোথায় থাকা হবে?",
        "reference": "বৃষ্টি হলে আমরা বাড়িতে থাকব।",
    },
    {
        "family": "transliterated",
        "dialect": "colloquial",
        "script_pattern": "latin_bn",
        "phenomena": ["transliteration"],
        "region": "bn-BD",
        "domain": "daily_life",
        "source_text": "ami dupure bhat kheyechi",
        "question": "দুপুরে কী খাওয়া হয়েছে?",
        "reference": "দুপুরে ভাত খাওয়া হয়েছে।",
    },
    {
        "family": "noisy_mixed",
        "dialect": "colloquial",
        "script_pattern": "bn_noisy",
        "phenomena": ["code_switching", "repeated_punctuation"],
        "region": "bn-BD",
        "domain": "travel",
        "source_text": "বাসটা ৮:৩০-এ আসবে!!!",
        "question": "বাস কখন আসবে?",
        "reference": "বাসটি ৮টা ৩০ মিনিটে আসবে।",
    },
    {
        "family": "noisy_mixed",
        "dialect": "colloquial",
        "script_pattern": "bn_en_noisy",
        "phenomena": ["code_switching", "spacing_noise"],
        "region": "bn-BD",
        "domain": "work",
        "source_text": "আজ file টা ইমেইল কইরো প্লিজ",
        "question": "আজ কী করতে বলা হয়েছে?",
        "reference": "আজ file-টি ইমেইল করতে বলা হয়েছে।",
    },
    {
        "family": "noisy_mixed",
        "dialect": "colloquial",
        "script_pattern": "latin_bn_noisy",
        "phenomena": ["transliteration", "repeated_punctuation"],
        "region": "bn-BD",
        "domain": "weather",
        "source_text": "aj brishti hobe naki???",
        "question": "কীসের সম্ভাবনা জানতে চাওয়া হয়েছে?",
        "reference": "আজ বৃষ্টির সম্ভাবনা আছে কি না জানতে চাওয়া হয়েছে।",
    },
    {
        "family": "noisy_mixed",
        "dialect": "colloquial",
        "script_pattern": "bn_en_noisy",
        "phenomena": ["code_switching", "repeated_characters"],
        "region": "bn-BD",
        "domain": "school",
        "source_text": "কালকে class ta সকাল ১০টায় হবেee",
        "question": "কালকের class কখন হবে?",
        "reference": "কালকের class সকাল ১০টায় হবে।",
    },
    {
        "family": "noisy_mixed",
        "dialect": "colloquial",
        "script_pattern": "latin_bn_noisy",
        "phenomena": ["transliteration", "code_switching"],
        "region": "bn-BD",
        "domain": "daily_life",
        "source_text": "ami bus e kore office jabo",
        "question": "কীভাবে office-এ যাওয়া হবে?",
        "reference": "বাসে করে office-এ যাওয়া হবে।",
    },
)


def _split(record_id: str) -> str:
    digest = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8], 16)
    return "dev" if digest % 10 < 8 else "test"


def build_robustness() -> list[dict[str, Any]]:
    """Build the curated v1 robustness set with stable IDs and splits."""
    records = []
    for index, example in enumerate(_EXAMPLES):
        record_id = f"robustness:{index:03d}"
        record = {
            "task": _TASK,
            "record_id": record_id,
            "split": _split(record_id),
            **example,
            "prompt": (
                "উক্ত বক্তব্য থেকে প্রশ্নের উত্তর বাংলায় এক বাক্যে দাও।\n"
                f"বক্তব্য: {example['source_text']}\nপ্রশ্ন: {example['question']}"
            ),
        }
        records.append(record)
    return records


def generate(out_dir: Path | None = None) -> list[dict[str, Any]]:
    """Write robustness JSONL and register task counts in the benchmark manifest."""
    out_dir = Path(out_dir or _DATA_DIR)
    records = build_robustness()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{_TASK}.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_path = out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {"total": len(records)}
    for split in ("dev", "test"):
        counts[split] = sum(record["split"] == split for record in records)
    manifest.setdefault("tasks", {})[_TASK] = counts
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return records


if __name__ == "__main__":
    generate()
