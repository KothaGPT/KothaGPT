"""Pipeline configuration and the full-pipeline driver."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from data.manifest import DatasetRegistry, get_registry
from data.pipeline import (
    copyright,
    dedup,
    io,
    normalize,
    quality,
    spam,
    split,
    stats,
    toxic,
    version,
)

__all__ = ["PipelineConfig", "enforce_source_provenance", "run_pipeline"]


@dataclass
class PipelineConfig:
    """Knobs for the Phase 1A data pipeline.

    Defaults favour aggressive but safe filtering for a small first corpus.
    """

    # Inputs / outputs
    raw_dir: str = "data/raw"
    out_root: str = "data/processed"
    version_label: str | None = None

    # Normalization
    unicode_form: str = "NFC"
    remove_html: bool = True
    remove_markup: bool = True

    # Quality filter
    min_chars: int = 100
    max_chars: int = 1_000_000
    min_words: int = 20
    require_bangla: bool = True
    min_bangla_ratio: float = 0.5
    allow_pii: bool = False
    pii_mode: str = "drop"

    # Spam filter
    check_spam: bool = True
    spam_threshold: float = 0.6

    # Toxic-content filter
    check_toxic: bool = True
    toxic_classifier: str | None = None
    toxic_classifier_threshold: float = 0.8

    # Copyright / licensing gate
    require_license: bool = False
    license_map_path: str | None = None
    license_allowlist: tuple[str, ...] | None = None
    copyrighted_titles_path: str | None = None

    # Deduplication
    dedup_exact: bool = True
    dedup_near: bool = False
    dedup_threshold: float = 0.8

    # Exact-dedup scale options: persistent cross-source state file, or a
    # fixed-memory Bloom filter.
    dedup_state_path: str | None = None
    dedup_bloom: bool = False
    dedup_bloom_capacity: int = 1_000_000
    dedup_bloom_fp_rate: float = 0.01

    # Split
    validation_ratio: float = 0.02
    test_ratio: float = 0.01

    # Shards
    shard_size: int = 100_000
    gzip_output: bool = True

    # Provenance / source registry
    strict: bool = True
    """Production mode (default): fail closed when the source registry is empty,
    a record's source is unregistered, or a license is missing/disallowed.
    Set ``strict=False`` to opt into the permissive development bypass."""

    # Reporting
    sample_note: str | None = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def enforce_source_provenance(
    records: list[dict],
    registry: DatasetRegistry | None = None,
    *,
    strict: bool = True,
) -> tuple[list[dict], list[str]]:
    """Filter records to those whose ``source`` is registered and allowed.

    Returns ``(kept, rejected_reasons)``.

    Production mode (``strict=True``, the default) fails closed:
      * an empty registry rejects every record (provenance required);
      * a record without a ``source`` field is rejected;
      * an unregistered or disallowed source is rejected.

    Development bypass (``strict=False``) is an explicit opt-in: every record is
    allowed through with no provenance check, regardless of registry state.
    """
    registry = registry or get_registry()
    if not strict:
        # Explicit development bypass: allow all records, no provenance check.
        return records, []
    if not registry.list():
        rejected = [
            f"{r.get('id', '?')}: no sources registered (provenance required in production mode)"
            for r in records
        ]
        return [], rejected
    kept: list[dict] = []
    rejected: list[str] = []
    for record in records:
        source = record.get("source")
        if not source:
            rejected.append(
                f"{record.get('id', '?')}: record missing 'source' field (no provenance)"
            )
            continue
        ok, reason = registry.is_allowed(source)
        if not ok:
            rejected.append(f"{source}: {reason}")
            continue
        kept.append(record)
    return kept, rejected


def run_pipeline(cfg: PipelineConfig, registry: DatasetRegistry | None = None) -> dict:
    """Execute raw -> normalize -> filter -> dedup -> split -> version.

    Returns a summary dict with per-step counts and the emitted version.
    """
    cfg_dict = cfg.as_dict()
    source = Path(cfg.raw_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"raw directory not found: {source}")

    summary: dict = {}

    # Copyright/licensing gate inputs.
    license_map = copyright.load_license_map(cfg.license_map_path) if cfg.license_map_path else {}
    title_blocklist = copyright.load_title_blocklist(cfg.copyrighted_titles_path)
    license_allowlist = (
        frozenset(cfg.license_allowlist) if cfg.license_allowlist else copyright.DEFAULT_ALLOWLIST
    )

    # 0. Provenance gate (KGPT-003): reject records whose source is not
    # registered and allowed. Production mode (strict=True, the default) fails
    # closed when the registry is empty or a source is unregistered/disallowed;
    # strict=False is an explicit development bypass.
    registry = registry or get_registry()
    provenance_rejected: list[str] = []

    # 1. Read raw records.
    records = list(io.iter_records(source))
    summary["raw"] = len(records)
    records, provenance_rejected = enforce_source_provenance(
        records, registry, strict=cfg.strict
    )
    summary["after_provenance"] = len(records)
    summary["provenance_rejected"] = provenance_rejected

    # 2. Normalize.
    normalized = []
    for record in records:
        text = normalize.normalize_text(
            record["text"],
            unicode_form=cfg.unicode_form,
            remove_html=cfg.remove_html,
            remove_markup=cfg.remove_markup,
        )
        if not text:
            continue
        record = dict(record)
        record["text"] = text
        normalized.append(record)
    summary["normalized"] = len(normalized)

    # 3. Copyright / licensing gate.
    licensed = []
    for record in normalized:
        keep, _ = copyright.copyright_gate(
            record,
            license_map=license_map,
            allowlist=license_allowlist,
            require_license=cfg.require_license,
            title_blocklist=title_blocklist,
        )
        if not keep:
            continue
        licensed.append(record)
    summary["after_license"] = len(licensed)

    # 4. Quality filter.
    filtered = []
    for record in licensed:
        text = record["text"]
        pii_allowed = cfg.allow_pii
        if cfg.pii_mode == "mask":
            text, _pii_counts = quality.redact_pii(text)
            if text != record["text"]:
                record = dict(record)
                record["text"] = text
            pii_allowed = True  # PII already handled by masking
        keep, _ = quality.quality_filter(
            text,
            min_chars=cfg.min_chars,
            max_chars=cfg.max_chars,
            min_words=cfg.min_words,
            require_bangla=cfg.require_bangla,
            min_bangla_ratio=cfg.min_bangla_ratio,
            allow_pii=pii_allowed,
        )
        if keep:
            filtered.append(record)
    summary["after_filter"] = len(filtered)

    # 5. Spam filter.
    clean = []
    for record in filtered:
        if cfg.check_spam:
            keep, _spam_reasons = spam.spam_gate(record["text"], threshold=cfg.spam_threshold)
            if not keep:
                continue
        clean.append(record)
    summary["after_spam"] = len(clean)

    # 6. Toxic-content filter.
    safe = []
    toxic_classifier = toxic.load_classifier(cfg.toxic_classifier) if cfg.toxic_classifier else None
    for record in clean:
        if cfg.check_toxic:
            keep, _toxic_reasons = toxic.toxic_gate(
                record["text"],
                classifier=toxic_classifier,
                classifier_threshold=cfg.toxic_classifier_threshold,
            )
            if not keep:
                continue
        safe.append(record)
    summary["after_toxic"] = len(safe)

    # 7. Deduplicate.
    if cfg.dedup_state_path:
        exact_store: set | dedup.BloomFilter | dedup.ExactDedupState = dedup.ExactDedupState(
            cfg.dedup_state_path
        )
    elif cfg.dedup_bloom:
        exact_store = dedup.BloomFilter(cfg.dedup_bloom_capacity, fp_rate=cfg.dedup_bloom_fp_rate)
    else:
        exact_store = set()
    deduped, dedup_counts = dedup.deduplicate_with_stats(
        safe,
        exact=cfg.dedup_exact,
        near=cfg.dedup_near,
        threshold=cfg.dedup_threshold,
        exact_store=exact_store,
        near_sharded=True,
    )
    if isinstance(exact_store, dedup.ExactDedupState):
        exact_store.save()
    summary["after_dedup"] = len(deduped)
    summary["dedup"] = dedup_counts

    # 8. Deterministic source/domain-stratified split.
    split_records = split.split_records(
        deduped,
        validation_ratio=cfg.validation_ratio,
        test_ratio=cfg.test_ratio,
    )
    split.assert_no_split_contamination(split_records)
    summary["train"] = sum(1 for r in split_records if r["split"] == "train")
    summary["validation"] = sum(1 for r in split_records if r["split"] == "validation")
    summary["test"] = sum(1 for r in split_records if r["split"] == "test")

    # 9. Statistics report.
    stats_data = stats.compute_stats(iter(split_records))
    if "dedup" in summary:
        dedup_counts = dict(summary["dedup"])
        removed = dedup_counts["removed_exact"] + dedup_counts["removed_near"]
        dedup_counts["rate"] = removed / dedup_counts["input"] if dedup_counts["input"] else 0.0
        summary["dedup"] = dedup_counts
        stats_data["dedup"] = dedup_counts
    summary["stats"] = stats_data

    # 10. Versioned output.
    out_root = Path(cfg.out_root)
    version_obj = version.DatasetVersion.create(
        config=cfg_dict,
        counts=summary,
        shards=[],
        files={},
        version_label=cfg.version_label,
    )
    version_dir = out_root / version_obj.version_id
    stats.write_report(stats_data, version_dir / "report")

    train_dir = version_dir / "train"
    val_dir = version_dir / "validation"
    train_shards = io.write_shards(
        (r for r in split_records if r["split"] == "train"),
        train_dir,
        shard_size=cfg.shard_size,
        gzip_output=cfg.gzip_output,
    )
    val_shards = io.write_shards(
        (r for r in split_records if r["split"] == "validation"),
        val_dir,
        shard_size=cfg.shard_size,
        gzip_output=cfg.gzip_output,
    )
    test_shards: list[dict] = []
    version_obj.shards = train_shards + val_shards
    version_obj.files = {
        "train_dir": str(train_dir.relative_to(out_root)),
        "validation_dir": str(val_dir.relative_to(out_root)),
        "report_dir": str((version_dir / "report").relative_to(out_root)),
    }
    if cfg.test_ratio > 0:
        test_dir = version_dir / "test"
        test_shards = io.write_shards(
            (r for r in split_records if r["split"] == "test"),
            test_dir,
            shard_size=cfg.shard_size,
            gzip_output=cfg.gzip_output,
        )
        version_obj.shards += test_shards
        version_obj.files["test_dir"] = str(test_dir.relative_to(out_root))
    version_obj.config = cfg_dict

    manifest_path = version.write_manifest(version_obj, version_dir)
    summary["version_id"] = version_obj.version_id
    summary["manifest"] = str(manifest_path)
    summary["train_shards"] = train_shards
    summary["validation_shards"] = val_shards
    summary["test_shards"] = test_shards
    return summary
