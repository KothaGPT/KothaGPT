"""Quality-gated instruction dataset ingestion and versioned shard output."""

from __future__ import annotations

import json
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any

from data.manifest import DatasetRegistry, get_registry
from data.pipeline import (
    copyright,
    dedup,
    io,
    normalize,
    quality,
    scoring,
    spam,
    split,
    stats,
    toxic,
    version,
)
from data.pipeline.config import enforce_source_provenance
from data.pipeline.schema import normalize_instruction_record

__all__ = ["run_instruction_pipeline"]


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            try:
                yield normalize_instruction_record(record)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid instruction record: {exc}") from exc


def _record_text(record: dict[str, Any]) -> str:
    return "\n".join(message["content"] for message in record["messages"])


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_record(record: dict[str, Any], unicode_form: str) -> dict[str, Any]:
    normalized = dict(record)
    normalized["messages"] = [
        {
            "role": message["role"],
            "content": normalize.clean_whitespace(
                normalize.unicode_normalize(message["content"], form=unicode_form)
            ),
        }
        for message in record["messages"]
    ]
    if any(not message["content"] for message in normalized["messages"]):
        raise ValueError("message content must remain non-empty after normalization")
    normalized["text"] = _record_text(normalized)
    digest = sha256(
        json.dumps(
            normalized["messages"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata = dict(normalized.get("metadata", {}))
    metadata.setdefault("record_id", digest[:16])
    normalized["metadata"] = metadata
    return normalized


def run_instruction_pipeline(
    input_path: str | Path,
    out_root: str | Path,
    *,
    min_score: float = 0.5,
    validation_ratio: float = 0.02,
    test_ratio: float = 0.01,
    min_chars: int = 1,
    max_chars: int = 1_000_000,
    min_words: int = 1,
    near_dedup: bool = False,
    check_spam: bool = True,
    check_toxic: bool = True,
    pii_mode: str = "drop",
    strict: bool = True,
    version_label: str | None = None,
    copyrighted_titles_path: str | Path | None = None,
    registry: DatasetRegistry | None = None,
) -> dict[str, Any]:
    """Validate, safety-filter, score, split, and version instruction JSONL."""
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0, 1], got {min_score}")
    if pii_mode not in {"drop", "mask"}:
        raise ValueError(f"pii_mode must be 'drop' or 'mask', got {pii_mode!r}")
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"instruction input not found: {input_file}")
    records = [_normalize_record(record, "NFC") for record in _iter_jsonl(input_file)]
    if not records:
        raise ValueError(f"no instruction records found in {input_file}")

    registry = registry or get_registry()
    accepted, provenance_rejected = enforce_source_provenance(records, registry, strict=strict)
    title_blocklist = copyright.load_title_blocklist(copyrighted_titles_path)
    license_filtered = []
    for record in accepted:
        keep, _ = copyright.copyright_gate(
            record, require_license=True, title_blocklist=title_blocklist
        )
        if keep:
            license_filtered.append(record)

    filtered = []
    for record in license_filtered:
        messages = []
        pii_found = False
        for message in record["messages"]:
            content = message["content"]
            if pii_mode == "mask":
                content, _ = quality.redact_pii(content)
            elif quality.contains_pii(content):
                pii_found = True
            messages.append({"role": message["role"], "content": content})
        if pii_found:
            continue
        record = dict(record, messages=messages)
        record["text"] = _record_text(record)
        keep, _ = quality.length_filter(
            record["text"], min_chars=min_chars, max_chars=max_chars, min_words=min_words
        )
        if not keep:
            continue
        if check_spam and not spam.spam_gate(record["text"])[0]:
            continue
        if check_toxic and not toxic.toxic_gate(record["text"])[0]:
            continue
        filtered.append(record)

    filtered.sort(
        key=lambda record: (
            record["text"],
            record["source"],
            record["license"],
            record["metadata"]["record_id"],
            json.dumps(record, ensure_ascii=False, sort_keys=True),
        )
    )
    deduped, dedup_counts = dedup.deduplicate_with_stats(
        filtered, exact=True, near=near_dedup, near_sharded=True
    )
    scored = []
    for record in deduped:
        result = scoring.score_instruction_record(record)
        if result["quality_score"] < min_score:
            continue
        scored.append({**record, **result})
    split_records = split.split_records(
        scored, validation_ratio=validation_ratio, test_ratio=test_ratio
    )
    split.assert_no_split_contamination(split_records)
    rejected_counts = {
        "provenance": len(provenance_rejected),
        "license": len(accepted) - len(license_filtered),
        "safety": len(license_filtered) - len(filtered),
        "exact_or_near_duplicate": dedup_counts["removed_exact"]
        + dedup_counts["removed_near"],
        "quality": len(deduped) - len(scored),
    }
    if not split_records:
        reasons = ", ".join(f"{gate}={count}" for gate, count in rejected_counts.items())
        raise ValueError(f"no instruction records passed validation and gates ({reasons})")

    score_stats = scoring.score_distribution(split_records)
    summary: dict[str, Any] = {
        "raw": len(records),
        "after_provenance": len(accepted),
        "provenance_rejected": provenance_rejected,
        "rejected": rejected_counts,
        "after_license": len(license_filtered),
        "after_safety": len(filtered),
        "after_dedup": len(deduped),
        "quality_rejected": len(deduped) - len(scored),
        "dedup": dedup_counts,
        "train": sum(record["split"] == "train" for record in split_records),
        "validation": sum(record["split"] == "validation" for record in split_records),
        "test": sum(record["split"] == "test" for record in split_records),
        "quality_scores": score_stats,
    }
    stats_records = (dict(record, text=record["text"]) for record in split_records)
    stats_data = stats.compute_stats(stats_records)
    stats_data["quality_scores"] = score_stats
    summary["stats"] = stats_data
    summary["after_filter"] = len(scored)

    out_root_path = Path(out_root)
    cfg = {
        "input_path": str(input_file),
        "input_sha256": _file_sha256(input_file),
        "min_score": min_score,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "min_words": min_words,
        "near_dedup": near_dedup,
        "check_spam": check_spam,
        "check_toxic": check_toxic,
        "pii_mode": pii_mode,
        "strict": strict,
        "copyrighted_titles_path": (
            str(copyrighted_titles_path) if copyrighted_titles_path is not None else None
        ),
    }
    dataset_version = version.DatasetVersion.create(
        config=cfg, counts=summary, shards=[], files={}, version_label=version_label
    )
    version_dir = out_root_path / dataset_version.version_id
    report_dir = version_dir / "report"
    stats.write_report(stats_data, report_dir)
    instruction_dir = version_dir / "instruction"
    shards: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        split_dir = instruction_dir / split_name
        output_records = []
        for record in split_records:
            if record["split"] != split_name:
                continue
            output = {key: value for key, value in record.items() if key != "text"}
            output["metadata"] = {**output.get("metadata", {}), "split": split_name}
            output_records.append(output)
        written = io.write_shards(
            iter(output_records),
            split_dir,
            gzip_output=True,
        )
        shards.extend({**shard, "split": split_name} for shard in written)
    dataset_version.shards = shards
    dataset_version.files = {
        "instruction_dir": str(instruction_dir.relative_to(out_root_path)),
        "report_dir": str(report_dir.relative_to(out_root_path)),
    }
    manifest_path = version.write_manifest(dataset_version, version_dir)
    summary.update(
        version_id=dataset_version.version_id,
        manifest=str(manifest_path),
        shards=shards,
    )
    return summary
