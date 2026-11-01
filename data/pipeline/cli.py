"""CLI entry point for the Phase 1A Bangla data pipeline.

Usage:
    python -m data.pipeline.cli run [options]
    python -m data.pipeline.cli normalize FILE
    python -m data.pipeline.cli check TEXT
    python -m data.pipeline.cli version
    python -m data.pipeline.cli sources [list|check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.manifest import (
    DatasetRegistry,
    DatasetSource,
    get_registry,
    load_registry,
    save_registry,
)
from data.pipeline import normalize as norm
from data.pipeline import quality
from data.pipeline.config import PipelineConfig, run_pipeline

_MAIN = (
    "raw -> normalize -> quality filter -> deduplicate -> train/validation "
    "split -> stats -> versioned shards"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kothagpt-data",
        description="Phase 1A — Bangla dataset pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline (" + _MAIN + ").")
    run_p.add_argument("--raw-dir", default="data/raw")
    run_p.add_argument("--out-root", default="data/processed")
    run_p.add_argument("--version-label", default=None)
    run_p.add_argument("--unicode-form", default="NFC")
    run_p.add_argument("--keep-html", action="store_true", help="skip HTML stripping")
    run_p.add_argument("--keep-markup", action="store_true", help="skip markup stripping")
    run_p.add_argument("--min-chars", type=int, default=100)
    run_p.add_argument("--max-chars", type=int, default=1_000_000)
    run_p.add_argument("--min-words", type=int, default=20)
    run_p.add_argument("--no-bangla-check", action="store_true")
    run_p.add_argument("--min-bangla-ratio", type=float, default=0.5)
    run_p.add_argument("--allow-pii", action="store_true")
    run_p.add_argument(
        "--pii-mode",
        choices=("drop", "mask"),
        default="drop",
        help="drop documents containing PII, or mask PII spans and keep them",
    )
    run_p.add_argument(
        "--no-spam-check",
        action="store_true",
        help="skip the spam filter",
    )
    run_p.add_argument(
        "--spam-threshold",
        type=float,
        default=0.6,
        help="spam score threshold in [0,1] (default 0.6)",
    )
    run_p.add_argument(
        "--no-toxic-check",
        action="store_true",
        help="skip the toxic-content filter",
    )
    run_p.add_argument(
        "--toxic-classifier",
        default=None,
        help="optional 'module:callable' returning a toxicity score in [0,1]",
    )
    run_p.add_argument(
        "--toxic-threshold",
        type=float,
        default=0.8,
        help="classifier threshold in [0,1] (default 0.8)",
    )
    run_p.add_argument(
        "--require-license",
        action="store_true",
        help="drop records without an allowed license",
    )
    run_p.add_argument(
        "--license-map",
        default=None,
        help="JSON file mapping source path/pattern -> license (e.g. data/raw/.licenses.json)",
    )
    run_p.add_argument(
        "--license-allowlist",
        default=None,
        help="comma-separated allowed license ids (default: standard open allowlist)",
    )
    run_p.add_argument(
        "--copyrighted-titles",
        default=None,
        help="file of known-copyrighted titles to block (one per line)",
    )
    run_p.add_argument("--no-exact-dedup", action="store_true")
    run_p.add_argument("--near-dedup", action="store_true", help="enable MinHash LSH dedup")
    run_p.add_argument("--dedup-threshold", type=float, default=0.8)
    run_p.add_argument(
        "--dedup-state",
        metavar="PATH",
        default=None,
        help="persistent exact-dedup state file for cross-source dedup",
    )
    run_p.add_argument(
        "--bloom-filter",
        action="store_true",
        help="use a fixed-memory Bloom filter for exact dedup",
    )
    run_p.add_argument(
        "--bloom-capacity",
        type=int,
        default=1_000_000,
        help="expected document count for the Bloom filter",
    )
    run_p.add_argument(
        "--bloom-fp-rate",
        type=float,
        default=0.01,
        help="Bloom filter false-positive rate in (0, 1)",
    )
    run_p.add_argument("--validation-ratio", type=float, default=0.02)
    run_p.add_argument("--shard-size", type=int, default=100_000)
    run_p.add_argument("--no-gzip", action="store_true")

    norm_p = sub.add_parser("normalize", help="Normalize a text file or stdin.")
    norm_p.add_argument("file", nargs="?", default="-")
    norm_p.add_argument("--unicode-form", default="NFC")
    norm_p.add_argument("--no-html", action="store_true")
    norm_p.add_argument("--no-markup", action="store_true")

    check_p = sub.add_parser(
        "check", help="Run language/PII/length checks on a text file or stdin."
    )
    check_p.add_argument("file", nargs="?", default="-")

    version_p = sub.add_parser("version", help="Show the CURRENT dataset version.")
    version_p.add_argument("--out-root", default="data/processed")

    sources_p = sub.add_parser(
        "sources",
        help="Manage the dataset provenance registry (KGPT-003).",
    )
    sources_sub = sources_p.add_subparsers(dest="sources_command", required=True)
    sources_list = sources_sub.add_parser("list", help="List registered sources.")
    sources_list.add_argument("--registry", default=None)
    sources_check = sources_sub.add_parser("check", help="Check whether sources are allowed.")
    sources_check.add_argument("--registry", default=None)
    sources_check.add_argument(
        "sources", nargs="*", help="source names to check"
    )
    sources_add = sources_sub.add_parser("add", help="Register a new source.")
    sources_add.add_argument("--registry", default="data/processed/registry.json")
    sources_add.add_argument("--name", required=True)
    sources_add.add_argument("--url", required=True)
    sources_add.add_argument("--license", required=True)
    sources_add.add_argument("--provenance", default="")
    sources_add.add_argument("--notes", default="")
    sources_add.add_argument("--no-training", action="store_true")
    sources_add.add_argument("--no-distribution", action="store_true")

    return parser


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _cmd_normalize(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    result = norm.normalize_text(
        text,
        unicode_form=args.unicode_form,
        remove_html=not args.no_html,
        remove_markup=not args.no_markup,
    )
    sys.stdout.write(result + "\n")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    lang = quality.detect_language(text)
    ratio = quality.bengali_ratio(text)
    pii = quality.contains_pii(text)
    kept, _ = quality.length_filter(text, min_chars=1, max_chars=10**12, min_words=0)
    report = {
        "language": lang,
        "bengali_ratio": round(ratio, 3),
        "chars": len(text),
        "words": len(text.split()),
        "pii": pii,
        "length_ok": kept,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    current = Path(args.out_root) / "CURRENT"
    if not current.exists():
        print("No dataset version yet. Run 'kothagpt-data run'.", file=sys.stderr)
        return 1
    version_id = current.read_text(encoding="utf-8").strip()
    manifest = Path(args.out_root) / version_id / "MANIFEST.json"
    print(f"version_id: {version_id}")
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        counts = data.get("counts", {})
        print(f"docs (after dedup): {counts.get('after_dedup', 'n/a')}")
        print(f"train: {counts.get('train', 'n/a')}")
        print(f"validation: {counts.get('validation', 'n/a')}")
        print(f"manifest: {manifest}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(
        raw_dir=args.raw_dir,
        out_root=args.out_root,
        version_label=args.version_label,
        unicode_form=args.unicode_form,
        remove_html=not args.keep_html,
        remove_markup=not args.keep_markup,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_words=args.min_words,
        require_bangla=not args.no_bangla_check,
        min_bangla_ratio=args.min_bangla_ratio,
        allow_pii=args.allow_pii,
        pii_mode=args.pii_mode,
        check_spam=not args.no_spam_check,
        spam_threshold=args.spam_threshold,
        check_toxic=not args.no_toxic_check,
        toxic_classifier=args.toxic_classifier,
        toxic_classifier_threshold=args.toxic_threshold,
        require_license=args.require_license,
        license_map_path=args.license_map,
        license_allowlist=tuple(a.strip() for a in args.license_allowlist.split(",") if a.strip())
        if args.license_allowlist
        else None,
        copyrighted_titles_path=args.copyrighted_titles,
        dedup_exact=not args.no_exact_dedup,
        dedup_near=args.near_dedup,
        dedup_threshold=args.dedup_threshold,
        dedup_state_path=args.dedup_state,
        dedup_bloom=args.bloom_filter,
        dedup_bloom_capacity=args.bloom_capacity,
        dedup_bloom_fp_rate=args.bloom_fp_rate,
        validation_ratio=args.validation_ratio,
        shard_size=args.shard_size,
        gzip_output=not args.no_gzip,
    )
    summary = run_pipeline(cfg)
    print(f"Pipeline complete: {_MAIN}")
    print(f"  raw docs:        {summary['raw']:,}")
    print(f"  normalized:      {summary['normalized']:,}")
    print(f"  after license:   {summary.get('after_license', summary['normalized']):,}")
    print(f"  after filter:    {summary['after_filter']:,}")
    print(f"  after spam:      {summary.get('after_spam', summary['after_filter']):,}")
    print(f"  after toxic:     {summary.get('after_toxic', summary['after_spam']):,}")
    print(f"  after dedup:     {summary['after_dedup']:,}")
    dedup_counts = summary.get("dedup")
    if dedup_counts:
        print(
            f"  dedup rate:      {dedup_counts['rate']:.2%} "
            f"(exact {dedup_counts['removed_exact']:,}, "
            f"near {dedup_counts['removed_near']:,})"
        )
    print(f"  train / val:     {summary['train']:,} / {summary['validation']:,}")
    print(f"  version id:      {summary['version_id']}")
    print(f"  manifest:        {summary['manifest']}")
    return 0


def _cmd_sources(args: argparse.Namespace) -> int:
    if args.sources_command == "list":
        reg = load_registry(args.registry) if args.registry else get_registry()
        sources = reg.list()
        if not sources:
            print("No sources registered.")
            return 0
        print(f"{len(sources)} source(s):")
        for src in sources:
            ok, reason = reg.is_allowed(src.name)
            status = "ALLOWED" if ok else f"BLOCKED ({reason})"
            print(
                f"  {src.name:<24} license={src.license:<16} "
                f"training={src.allow_training} {status}"
            )
        return 0

    if args.sources_command == "check":
        reg = load_registry(args.registry) if args.registry else get_registry()
        if not args.sources:
            print("no sources specified", file=sys.stderr)
            return 1
        failed = False
        for name in args.sources:
            ok, reason = reg.is_allowed(name)
            print(f"{name}: {'OK' if ok else 'BLOCKED'} — {reason}")
            if not ok:
                failed = True
        return 1 if failed else 0

    if args.sources_command == "add":
        reg = load_registry(args.registry) if args.registry else DatasetRegistry()
        try:
            src = DatasetSource(
                name=args.name,
                url=args.url,
                license=args.license,
                provenance=args.provenance,
                notes=args.notes,
                allow_training=not args.no_training,
                allow_distribution=not args.no_distribution,
            )
            reg.register(src)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        out = save_registry(reg, args.registry)
        print(f"registered {args.name} -> {out}")
        return 0

    raise AssertionError(f"unhandled sources command {args.sources_command}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "normalize":
        return _cmd_normalize(args)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "version":
        return _cmd_version(args)
    if args.command == "sources":
        return _cmd_sources(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
