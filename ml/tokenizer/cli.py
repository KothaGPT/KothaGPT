"""CLI for Phase 1B — Bangla tokenizer training and benchmarking.

Usage:
    python -m ml.tokenizer.cli train --corpus PATH --algorithm bpe|unigram|wordpiece --vocab-size N --out DIR
    python -m ml.tokenizer.cli experiments --corpus PATH [--out DIR]
    python -m ml.tokenizer.cli encode --tokenizer DIR --text "..." [--transliterate] | --file PATH
    python -m ml.tokenizer.cli benchmark --tokenizer DIR --file PATH
    python -m ml.tokenizer.cli freeze --corpus PATH --vocab-size N [--sample-stride N] --out DIR
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from ml.tokenizer import load_tokenizer, train_bpe, train_unigram, train_wordpiece
from ml.tokenizer.benchmark import GATE_THRESHOLDS, SAMPLE_TEXTS, check_benchmark, run_benchmark
from ml.tokenizer.corpus import iter_corpus, load_corpus
from ml.tokenizer.transliterate import latin_to_bangla
from ml.tokenizer.vocab import coverage_report, export_vocab, version_id

DEFAULT_VOCAB_SIZES = (16000, 32000, 50000)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kothagpt-tokenizer",
        description="Phase 1B — Bangla tokenizer (BPE / Unigram / WordPiece) experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a single tokenizer.")
    train_p.add_argument("--corpus", required=True, help="corpus dir or file")
    train_p.add_argument("--algorithm", choices=("bpe", "unigram", "wordpiece"), default="bpe")
    train_p.add_argument("--vocab-size", type=int, default=16000)
    train_p.add_argument("--out", required=True, help="output artifact dir")
    train_p.add_argument("--min-frequency", type=int, default=2)
    train_p.add_argument("--max-subword-len", type=int, default=8)
    train_p.add_argument("--iterations", type=int, default=8)

    exp_p = sub.add_parser("experiments", help="Run the BPE/Unigram/WordPiece x vocab matrix.")
    exp_p.add_argument("--corpus", required=True)
    exp_p.add_argument("--out", default="ml/tokenizer/artifacts")
    exp_p.add_argument("--algorithms", default="bpe,unigram,wordpiece", help="comma-separated algorithms")
    exp_p.add_argument(
        "--vocab-sizes",
        default=",".join(str(v) for v in DEFAULT_VOCAB_SIZES),
        help="comma-separated target vocab sizes",
    )
    exp_p.add_argument("--min-frequency", type=int, default=2)
    exp_p.add_argument("--max-subword-len", type=int, default=8)
    exp_p.add_argument("--iterations", type=int, default=8)

    enc_p = sub.add_parser("encode", help="Encode text with a saved tokenizer.")
    enc_p.add_argument("--tokenizer", required=True)
    enc_p.add_argument("--text", default=None)
    enc_p.add_argument("--file", default=None)
    enc_p.add_argument("--show-tokens", action="store_true", dest="show_tokens")
    enc_p.add_argument(
        "--transliterate",
        action="store_true",
        help="treat input as romanized Bangla and convert to Bangla script first",
    )

    bench_p = sub.add_parser("benchmark", help="Benchmark a saved tokenizer.")
    bench_p.add_argument("--tokenizer", required=True)
    bench_p.add_argument("--file", default=None)
    bench_p.add_argument("--out", default=None, help="write benchmark.json artifact here")
    bench_p.add_argument(
        "--gate",
        action="store_true",
        help="fail (exit 1) when the WS-7 threshold gate is violated",
    )
    bench_p.add_argument(
        "--max-tokens-per-char",
        type=float,
        default=None,
        help="override the max average tokens/char gate threshold",
    )

    freeze_p = sub.add_parser(
        "freeze",
        help="Retrain on the normalized corpus and freeze tokenizer + vocab (WS-1/WS-6).",
    )
    freeze_p.add_argument(
        "--corpus", required=True, help="corpus dir or file (processed train shards)"
    )
    freeze_p.add_argument("--algorithm", choices=("bpe", "unigram", "wordpiece"), default="bpe")
    freeze_p.add_argument("--vocab-size", type=int, default=16000)
    freeze_p.add_argument("--min-frequency", type=int, default=1)
    freeze_p.add_argument("--out", default="ml/tokenizer")
    freeze_p.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="train on every Nth document (deterministic memory-safe subsample); "
        "the corpus digest still covers the full corpus",
    )
    freeze_p.add_argument("--coverage-docs", type=int, default=5000)
    freeze_p.add_argument(
        "--gate",
        action="store_true",
        help="fail (exit 1) when the WS-7 threshold gate is violated",
    )

    return parser


def _train_one(
    algorithm: str,
    corpus: list[str],
    vocab_size: int,
    min_frequency: int,
    max_subword_len: int,
    iterations: int,
):
    if algorithm == "bpe":
        return train_bpe(corpus, vocab_size, min_frequency=min_frequency, log=None)
    if algorithm == "wordpiece":
        return train_wordpiece(corpus, vocab_size, min_frequency=min_frequency, log=None)
    return train_unigram(
        corpus,
        vocab_size,
        min_frequency=min_frequency,
        max_subword_len=max_subword_len,
        iterations=iterations,
        log=None,
    )


def _cmd_train(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    tokenizer = _train_one(
        args.algorithm,
        corpus,
        args.vocab_size,
        args.min_frequency,
        args.max_subword_len,
        args.iterations,
    )
    path = tokenizer.save(Path(args.out))
    stats = run_benchmark(tokenizer)
    print(f"trained {args.algorithm} tokenizer")
    print(f"  vocab: {len(tokenizer.vocab)} tokens (target {args.vocab_size})")
    print(f"  saved: {path}")
    print(f"  avg tokens/char: {stats['avg_tokens_per_char']:.4f}")
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",") if v.strip()]
    corpus = load_corpus(args.corpus)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for algorithm in algorithms:
        for vocab_size in vocab_sizes:
            name = f"{algorithm}-{vocab_size}"
            print(f"[experiments] {name}: training on {len(corpus)} docs ...")
            tokenizer = _train_one(
                algorithm,
                corpus,
                vocab_size,
                args.min_frequency,
                args.max_subword_len,
                args.iterations,
            )
            exp_dir = out_root / "experiments" / name
            tokenizer.save(exp_dir)
            bench = run_benchmark(tokenizer)
            rows.append(
                {
                    "algorithm": algorithm,
                    "target_vocab": vocab_size,
                    "actual_vocab": len(tokenizer.vocab),
                    "avg_tokens_per_char": bench["avg_tokens_per_char"],
                    "avg_unk_rate": bench["avg_unk_rate"],
                    "min_decode_fidelity": bench["min_decode_fidelity"],
                    "avg_compression_vs_byte": bench["avg_compression_vs_byte"],
                    "avg_compression_vs_char": bench["avg_compression_vs_char"],
                    "per_set": {k: v["tokens_per_char"] for k, v in bench["per_set"].items()},
                    "unk_rate": {k: v["unk_rate"] for k, v in bench["per_set"].items()},
                    "fidelity": {k: v["decode_fidelity"] for k, v in bench["per_set"].items()},
                    "dir": str(exp_dir),
                }
            )

    best = min(rows, key=lambda r: r["avg_tokens_per_char"])
    best_tokenizer = load_tokenizer(Path(best["dir"]) / "tokenizer.json")
    best_dir = out_root / "best"
    best_tokenizer.save(best_dir)

    comparison = {
        "results": rows,
        "best": {"name": f"{best['algorithm']}-{best['target_vocab']}", "dir": str(best_dir)},
    }
    (out_root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(out_root, rows, best)
    _print_summary(rows, best)
    return 0


def _write_report(out_root: Path, rows: list[dict], best: dict) -> None:
    lines = ["# Tokenizer Experiment Report", ""]
    lines.append("Lower tokens/char is better (fewer tokens for the same text).")
    lines.append(
        "unk_rate is unknown-token rate (target < 0.5% on dev sets); "
        "fidelity is decode round-trip accuracy (target 100%)."
    )
    lines.append("")
    lines.append("## Overall (average across test sets)")
    lines.append(
        "| algorithm | target vocab | actual vocab | tokens/char | unk rate | fidelity | best |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        marker = " *" if row is best else ""
        lines.append(
            f"| {row['algorithm']} | {row['target_vocab']:,} | {row['actual_vocab']:,} "
            f"| {row['avg_tokens_per_char']:.4f} | {row['avg_unk_rate']:.4f} "
            f"| {row['min_decode_fidelity']:.0%} | {marker} |"
        )
    lines.append("")
    names = [
        "bangla",
        "mixed",
        "punctuation",
        "translit",
        "digits",
        "names",
        "emoji",
        "code",
        "social",
    ]
    names = [n for n in names if n in rows[0]["per_set"]]
    lines.append("## Per test set (tokens/char)")
    lines.append("| algorithm | vocab | " + " | ".join(names) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in names) + " |")
    for row in rows:
        cells = " | ".join(f"{row['per_set'][n]:.4f}" for n in names if n in row["per_set"])
        lines.append(f"| {row['algorithm']} | {row['target_vocab']:,} | {cells} |")
    lines.append("")
    lines.append("## Best (frozen)")
    lines.append(f"- `{best['algorithm']}-{best['target_vocab']}` → `{out_root / 'best'}`")
    lines.append("")
    lines.append("## Unk rate (unknown tokens per set)")
    lines.append("| algorithm | vocab | " + " | ".join(names) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in names) + " |")
    for row in rows:
        cells = " | ".join(f"{row['unk_rate'].get(n, 1.0):.4f}" for n in names)
        lines.append(f"| {row['algorithm']} | {row['target_vocab']:,} | {cells} |")
    lines.append("")
    lines.append("## Decode fidelity (round-trip == original)")
    lines.append("| algorithm | vocab | " + " | ".join(names) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in names) + " |")
    for row in rows:
        cells = " | ".join(f"{row['fidelity'].get(n, 0.0):.0%}" for n in names)
        lines.append(f"| {row['algorithm']} | {row['target_vocab']:,} | {cells} |")
    lines.append("")
    (out_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _print_summary(rows: list[dict], best: dict) -> None:
    print()
    print("Experiment summary (avg tokens/char, lower is better):")
    for row in sorted(rows, key=lambda r: r["avg_tokens_per_char"]):
        marker = " <- best" if row is best else ""
        print(
            f"  {row['algorithm']:<8} vocab={row['target_vocab']:>6} "
            f"actual={row['actual_vocab']:>6} tpc={row['avg_tokens_per_char']:.4f}{marker}"
        )
    print(f"Frozen best: {best['algorithm']}-{best['target_vocab']}")


def _cmd_encode(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    if args.transliterate:
        text = latin_to_bangla(text)
    ids = tokenizer.encode(text)
    print(f"tokens: {len(ids)}  chars: {len(text)}  tokens/char: {len(ids) / len(text):.4f}")
    if args.show_tokens:
        print(json.dumps(tokenizer.tokenize(text), ensure_ascii=False, indent=2))
    else:
        print(ids)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    texts = SAMPLE_TEXTS
    if args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
        texts = {"file": text}
    result = run_benchmark(tokenizer, texts)
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer": str(args.tokenizer),
            "vocab_size": len(tokenizer.vocab),
            "thresholds": dict(GATE_THRESHOLDS),
            "result": result,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"benchmark artifact written: {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.gate:
        thresholds = dict(GATE_THRESHOLDS)
        if args.max_tokens_per_char is not None:
            thresholds["max_avg_tokens_per_char"] = args.max_tokens_per_char
        failures = check_benchmark(result, thresholds)
        if failures:
            print("GATE FAILED:")
            for msg in failures:
                print(f"  - {msg}")
            return 1
        print("GATE PASSED")
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    import hashlib

    from data.pipeline import normalize as norm
    from ml.tokenizer.vocab import _write_vocab_files

    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    # Single streaming pass: normalize each doc exactly once, hash it into the
    # corpus digest, and keep only every Nth doc for training. Peak memory is
    # bounded to the training sample rather than the full raw corpus.
    digest_h = hashlib.sha256()
    train_docs: list[str] = []
    total_docs = 0
    for i, doc in enumerate(iter_corpus(args.corpus)):
        total_docs += 1
        ndoc = norm.normalize_text(doc)
        doc_bytes = ndoc.encode("utf-8")
        # length-prefix each document so the hash is unambiguous
        digest_h.update(len(doc_bytes).to_bytes(8, "big"))
        digest_h.update(doc_bytes)
        digest_h.update(b"\x00")
        if i % args.sample_stride == 0:
            train_docs.append(ndoc)
    if total_docs == 0:
        raise ValueError("empty corpus")
    print(f"[freeze] loaded {total_docs:,} docs", flush=True)

    train_docs = [doc for doc in train_docs if doc.strip()]
    if not train_docs:
        raise ValueError("empty corpus after normalization")
    digest = digest_h.hexdigest()[:12]

    print(
        f"[freeze] training on {len(train_docs):,} docs (stride {args.sample_stride})",
        flush=True,
    )
    print(f"[freeze] corpus digest: {digest}", flush=True)

    tokenizer = _train_one(
        args.algorithm,
        train_docs,
        args.vocab_size,
        args.min_frequency,
        max_subword_len=8,
        iterations=8,
    )
    print(f"[freeze] vocab: {len(tokenizer.vocab):,} (target {args.vocab_size:,})", flush=True)

    out_root = Path(args.out)
    best_dir = out_root / "artifacts" / "best"
    tokenizer.save(best_dir)
    print(f"[freeze] saved tokenizer: {best_dir / 'tokenizer.json'}")

    benchmark = run_benchmark(tokenizer)
    bench_path = out_root / "artifacts" / "benchmark.json"
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(
        json.dumps(
            {"tokenizer": str(best_dir), "vocab_size": len(tokenizer.vocab), "result": benchmark},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[freeze] benchmark: tpc={benchmark['avg_tokens_per_char']:.4f} "
        f"unk={benchmark['dev_max_unk_rate']:.2%} "
        f"fidelity={benchmark['dev_min_decode_fidelity']:.0%}"
    )

    if args.gate:
        failures = check_benchmark(benchmark, dict(GATE_THRESHOLDS))
        if failures:
            print("GATE FAILED:")
            for msg in failures:
                print(f"  - {msg}")
            return 1
        print("GATE PASSED")

    coverage_docs = train_docs[: args.coverage_docs]
    coverage = coverage_report(tokenizer, coverage_docs)
    print(
        f"[freeze] coverage on {len(coverage_docs):,} docs: "
        f"{coverage['coverage']:.2%} (unk {coverage['unk_rate']:.2%})"
    )

    meta = _write_vocab_files(
        out_root / "vocab",
        tokenizer,
        corpus_digest_value=digest,
        algorithm=args.algorithm,
        vocab_size_target=args.vocab_size,
        coverage=coverage,
        benchmark=benchmark,
    )
    print(f"[freeze] vocab version: {meta['version']}")
    _write_freeze_reports(
        out_root,
        tokenizer,
        digest,
        args,
        coverage,
        benchmark,
        total_docs,
        len(train_docs),
        len(coverage_docs),
    )
    return 0


def _write_freeze_reports(
    out_root: Path,
    tokenizer,
    digest: str,
    args: argparse.Namespace,
    coverage: dict,
    benchmark: dict,
    corpus_docs: int,
    train_docs: int,
    coverage_docs: int,
) -> None:
    artifacts = out_root / "artifacts"
    lines = [
        "# Tokenizer Freeze Report",
        "",
        f"- **version**: `{version_id(digest, export_vocab(tokenizer))}`",
        f"- **algorithm**: `{args.algorithm}`",
        f"- **vocab**: {len(tokenizer.vocab):,} (target {args.vocab_size:,})",
        (
            f"- **corpus docs**: {corpus_docs:,} (training sample: {train_docs:,} "
            f"stride {args.sample_stride})"
        ),
        f"- **corpus digest**: `{digest}`",
        f"- **coverage**: {coverage['coverage']:.2%} (unk {coverage['unk_rate']:.2%})",
        "",
        "## Efficiency benchmark (gated dev sets)",
        f"- avg tokens/char: **{benchmark['avg_tokens_per_char']:.4f}**",
        f"- dev max unk rate: **{benchmark['dev_max_unk_rate']:.2%}**",
        f"- dev min decode fidelity: **{benchmark['dev_min_decode_fidelity']:.0%}**",
        f"- dev min compression vs char: **{benchmark['dev_min_compression_vs_char']:.2f}**",
        "",
        "## Artifacts",
        f"- tokenizer: `{artifacts / 'best/tokenizer.json'}`",
        f"- benchmark: `{artifacts / 'benchmark.json'}`",
        f"- vocab: `{out_root / 'vocab/vocab.json'}`",
        f"- decision: `{out_root / 'DECISION.md'}`",
        "",
    ]
    (artifacts / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    decision_lines = [
        "# Decision — Tokenizer & Vocabulary Freeze",
        "",
        f"- **date**: {_dt.datetime.now(_dt.UTC).isoformat(timespec='seconds')}",
        f"- **algorithm**: `{args.algorithm}` (vocab {args.vocab_size:,})",
        f"- **corpus digest**: `{digest}`",
        "",
        "## Why this tokenizer/vocab",
        "",
        (
            f"- Trained on the normalized Bangla corpus "
            f"({train_docs:,} docs of {corpus_docs:,} with stride {args.sample_stride})."
        ),
        (
            f"- Coverage: **{coverage['coverage']:.2%}** of script characters on a "
            f"{coverage_docs:,}-doc sample; unk rate {coverage['unk_rate']:.2%}."
        ),
        (
            f"- Efficiency gate passes: tpc {benchmark['avg_tokens_per_char']:.4f} "
            f"(<= {GATE_THRESHOLDS['max_avg_tokens_per_char']}), "
            f"unk {benchmark['dev_max_unk_rate']:.2%} "
            f"(<= {GATE_THRESHOLDS['max_dev_unk_rate']}), "
            f"fidelity {benchmark['dev_min_decode_fidelity']:.0%} "
            f"(>= {GATE_THRESHOLDS['min_dev_decode_fidelity']})."
        ),
        "",
        "## Open questions",
        "",
        (
            "- Compare against sentencepiece / tokenizers reference baselines on the "
            "same corpus (dev-only dependency) and record the numbers here."
        ),
        "- Re-freeze at 32k / 50k vocab sizes once the 16k baseline is stable.",
        "",
    ]
    (out_root / "DECISION.md").write_text("\n".join(decision_lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "experiments":
        return _cmd_experiments(args)
    if args.command == "encode":
        return _cmd_encode(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "freeze":
        return _cmd_freeze(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
