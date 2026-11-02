"""Deterministic source/domain-stratified train/validation/test splits.

Assignments are reproducible for a fixed dataset and independent of input
order.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from urllib.parse import urlsplit

__all__ = [
    "assert_no_split_contamination",
    "split_record",
    "split_records",
    "split_set",
]

def split_key(text: str) -> int:
    """Deterministic 32-bit bucket derived from the text content."""
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:4], "big")


def split_set(text: str, validation_ratio: float = 0.02) -> str:
    """Return ``train`` or ``validation`` for ``text``."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")
    bucket = split_key(text)
    if bucket / 0xFFFFFFFF < validation_ratio:
        return "validation"
    return "train"


def split_record(record: dict, *, text_key: str = "text", validation_ratio: float = 0.02) -> dict:
    """Attach a ``split`` field to a record (does not mutate the input)."""
    copy = dict(record)
    copy["split"] = split_set(record[text_key], validation_ratio)
    return copy


def _stratum(record: dict, *, source_key: str, domain_key: str) -> str:
    domain = record.get(domain_key)
    if domain:
        normalized_domain = str(domain).strip().lower()
        if normalized_domain:
            return f"domain:{normalized_domain}"
    for value in (record.get("url"), record.get(source_key)):
        if value:
            host = urlsplit(str(value)).hostname
            if host:
                return f"domain:{host.lower()}"
    source = record.get(source_key)
    return f"source:{source}" if source else "source:unknown"


def _allocate_counts(size: int, ratios: list[tuple[str, float]]) -> dict[str, int]:
    expected = {name: size * ratio for name, ratio in ratios}
    counts = {name: int(value) for name, value in expected.items()}
    remainder = size - sum(counts.values())
    ranked = sorted(
        ratios,
        key=lambda item: (-(expected[item[0]] - counts[item[0]]), item[0]),
    )
    for name, _ in ranked[:remainder]:
        counts[name] += 1

    active = [name for name, ratio in ratios if ratio > 0]
    if size >= len(active):
        for name in active:
            if counts[name] == 0:
                donor = max(
                    (candidate for candidate in active if counts[candidate] > 1),
                    key=lambda candidate: (counts[candidate] - expected[candidate], candidate),
                )
                counts[donor] -= 1
                counts[name] = 1
    return counts


def split_records(
    records: list[dict],
    *,
    validation_ratio: float = 0.02,
    test_ratio: float = 0.01,
    text_key: str = "text",
    source_key: str = "source",
    domain_key: str = "domain",
) -> list[dict]:
    """Assign deterministic, source/domain-stratified train/validation/test splits.

    Explicit ``domain`` wins, followed by a URL host from ``url`` or ``source``,
    then the source value. Exact duplicate text is assigned together so it
    cannot leak across splits.
    """
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in [0, 1), got {validation_ratio}")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1")

    ratios = [("train", 1.0 - validation_ratio - test_ratio)]
    if validation_ratio:
        ratios.append(("validation", validation_ratio))
    if test_ratio:
        ratios.append(("test", test_ratio))

    duplicate_strata: dict[str, set[str]] = defaultdict(set)
    for record in records:
        duplicate_strata[record[text_key]].add(
            _stratum(record, source_key=source_key, domain_key=domain_key)
        )

    grouped_texts: dict[str, list[str]] = defaultdict(list)
    for text, strata in duplicate_strata.items():
        grouped_texts[min(strata)].append(text)

    assignments: dict[str, str] = {}
    for stratum, texts in grouped_texts.items():
        ordered = sorted(
            texts,
            key=lambda text: (split_key(f"{stratum}\0{text}"), text),
        )
        counts = _allocate_counts(len(ordered), ratios)
        offset = 0
        for name, _ in ratios:
            for text in ordered[offset : offset + counts[name]]:
                assignments[text] = name
            offset += counts[name]

    unique_count = len(assignments)
    split_counts = {name: sum(value == name for value in assignments.values()) for name, _ in ratios}
    minimum_sizes = {"validation": 2, "test": len(ratios)}
    for name, minimum_size in minimum_sizes.items():
        if name not in split_counts or unique_count < minimum_size or split_counts[name]:
            continue
        donors = [candidate for candidate, count in split_counts.items() if count > 1]
        if not donors:
            continue
        donor = max(donors, key=lambda candidate: (split_counts[candidate], candidate))
        promoted = min(
            (text for text, assigned in assignments.items() if assigned == donor),
            key=lambda text: (split_key(text), text),
        )
        assignments[promoted] = name
        split_counts[donor] -= 1
        split_counts[name] += 1

    return [dict(record, split=assignments[record[text_key]]) for record in records]


def assert_no_split_contamination(records: list[dict], *, text_key: str = "text") -> None:
    """Raise when identical text occurs in more than one split."""
    split_by_digest: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if "split" not in record:
            raise ValueError("every record must have a split before contamination checking")
        digest = sha256(record[text_key].encode("utf-8")).hexdigest()
        split_by_digest[digest].add(record["split"])

    conflicts = [splits for splits in split_by_digest.values() if len(splits) > 1]
    if conflicts:
        raise ValueError(
            f"cross-split exact-text contamination detected for {len(conflicts)} text hash(es)"
        )


def ensure_nonempty_validation(records: list[dict], *, text_key: str = "text") -> list[dict]:
    """Guarantee at least one validation record for corpora large enough to split.

    Content hashing is per-record, so small corpora can round to an empty
    validation split (e.g. 12 docs at ratio 0.02), which breaks downstream
    tokenization/training. Promotes the record whose bucket is closest to the
    validation threshold, keeping the split deterministic and stable.
    """
    if len(records) < 2 or any(r["split"] == "validation" for r in records):
        return records
    idx = min(range(len(records)), key=lambda i: split_key(records[i][text_key]))
    promoted = dict(records[idx])
    promoted["split"] = "validation"
    return records[:idx] + [promoted] + records[idx + 1 :]
