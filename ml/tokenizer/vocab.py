"""Vocabulary export, coverage and versioning helpers (WS-1 / WS-6).

A *frozen* vocabulary is a plain JSON object mapping token -> integer id,
plus a metadata file that pins the algorithm, corpus digest and vocabulary
digest. ``version_id`` follows the content-addressed scheme ``MAJOR.MINOR+PATCH``
(``1.0.0+<corpus>.<vocab>``) so any retrain with a different corpus or vocab
yields a distinct, comparable version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base import _WORD_MARKER, UNK, BaseTokenizer

__all__ = ["VOCAB_VERSION", "corpus_digest", "coverage_report", "export_vocab", "version_id"]

VOCAB_VERSION = "1.0.0"

_DIGEST_LEN = 8
_CORPUS_DIGEST_LEN = 12


def corpus_digest(texts: list[str]) -> str:
    """Content-addressed digest of the normalized corpus (first 12 hex chars)."""
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:_CORPUS_DIGEST_LEN]


def export_vocab(tokenizer: BaseTokenizer) -> dict[str, int]:
    """Return the tokenizer's ``{token: id}`` mapping for serialization."""
    return dict(tokenizer.vocab)


def coverage_report(tokenizer: BaseTokenizer, texts: list[str]) -> dict[str, Any]:
    """Measure how much of ``texts`` is covered by the tokenizer's vocabulary.

    Reports:
    - ``coverage``: fraction of characters covered by known (non-``<unk>``)
      tokens over all texts;
    - ``unk_rate``: fraction of emitted tokens that are ``<unk>``;
    - ``num_tokens`` / ``tokens_per_char``: compression efficiency;
    - ``num_known``: number of distinct non-special vocabulary entries.
    """
    chars = 0
    known_chars = 0
    unk = 0
    tokens = 0
    for text in texts:
        ids = tokenizer.encode(text)
        tokens += len(ids)
        unk += sum(1 for i in ids if i == tokenizer.unk_id)
        # Whitespace is encoded as word-marker tokens (restored on decode), so
        # coverage measures the script characters proper. Exclude *all*
        # whitespace (spaces, newlines, tabs) from the denominator, not just
        # U+0020.
        chars += sum(1 for c in text if not c.isspace())
        token_strings = tokenizer.tokenize(text)
        # Every word is prefixed with one word-marker char (often embedded in a
        # merged token), so subtract one marker char per word from the "known"
        # char count.
        markers = sum(tok.count(_WORD_MARKER) for tok in token_strings if tok != UNK)
        known_chars += sum(len(tok) for tok in token_strings if tok != UNK) - markers
    coverage = known_chars / chars if chars else 0.0
    return {
        "coverage": coverage,
        "unk_rate": unk / tokens if tokens else 0.0,
        "num_tokens": tokens,
        "tokens_per_char": tokens / chars if chars else 0.0,
        "num_known": len(tokenizer.vocab) - len({"<unk>", "<pad>", "<s>", "</s>"}),
    }


def version_id(corpus: str, vocab: dict[str, int]) -> str:
    """Return ``MAJOR.MINOR.PATCH+<corpus>.<vocab>`` for a frozen vocab."""
    digest = hashlib.sha256(json.dumps(vocab, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return f"{VOCAB_VERSION}+{corpus}.{digest.hexdigest()[:_DIGEST_LEN]}"


def _write_vocab_files(
    out_dir: Path,
    tokenizer: BaseTokenizer,
    *,
    corpus_digest_value: str,
    algorithm: str,
    vocab_size_target: int,
    coverage: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    """Write ``vocab.json``, ``vocab.md``, ``vocab-meta.json`` and ``REPORT.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = export_vocab(tokenizer)
    version = version_id(corpus_digest_value, vocab)

    meta = {
        "version": version,
        "algorithm": algorithm,
        "vocab_version": VOCAB_VERSION,
        "target_vocab_size": vocab_size_target,
        "actual_vocab_size": len(vocab),
        "corpus_digest": corpus_digest_value,
        "coverage": coverage,
        "benchmark": benchmark,
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "vocab-meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "vocab.md").write_text(_render_vocab_md(meta, vocab), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(_render_vocab_report(meta), encoding="utf-8")
    return meta


def _render_vocab_md(meta: dict[str, Any], vocab: dict[str, int]) -> str:
    lines = ["# Vocabulary", ""]
    lines.append(f"- version: `{meta['version']}`")
    lines.append(f"- algorithm: `{meta['algorithm']}`")
    lines.append(
        f"- vocab size: {meta['actual_vocab_size']:,} (target {meta['target_vocab_size']:,})"
    )
    lines.append(f"- corpus digest: `{meta['corpus_digest']}`")
    lines.append(f"- coverage: {meta['coverage']['coverage']:.2%}")
    lines.append("")
    lines.append("## Top 200 tokens")
    lines.append("| id | token |")
    lines.append("| --- | --- |")
    for token, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        lines.append(f"| {tid} | `{token}` |")
        if tid >= 199:
            break
    return "\n".join(lines) + "\n"


def _render_vocab_report(meta: dict[str, Any]) -> str:
    c = meta["coverage"]
    b = meta["benchmark"]
    return "\n".join(
        [
            "# Vocabulary Freeze Report",
            "",
            f"- **version**: `{meta['version']}`",
            f"- **algorithm**: `{meta['algorithm']}`",
            f"- **vocab size**: {meta['actual_vocab_size']:,} (target {meta['target_vocab_size']:,})",
            f"- **corpus digest**: `{meta['corpus_digest']}`",
            "",
            "## Coverage",
            f"- coverage: **{c['coverage']:.2%}**",
            f"- unk rate: **{c['unk_rate']:.2%}**",
            f"- tokens/char: **{c['tokens_per_char']:.4f}**",
            "",
            "## Efficiency benchmark",
            f"- avg tokens/char: **{b.get('avg_tokens_per_char', 0):.4f}**",
            f"- dev max unk rate: **{b.get('dev_max_unk_rate', 0):.2%}**",
            f"- dev min decode fidelity: **{b.get('dev_min_decode_fidelity', 0):.0%}**",
            f"- dev min compression vs char: **{b.get('dev_min_compression_vs_char', 0):.2f}**",
            "",
            "## Metrics",
            "| metric | value | target | status |",
            "| --- | --- | --- | --- |",
        ]
        + [
            "| {} | {:.4f} | {} | {} |".format(
                name,
                value := meta["benchmark"].get(key, 0),
                bound,
                "PASS" if (value >= bound if op == "min" else value <= bound) else "FAIL",
            )
            for name, key, bound, op in (
                ("avg tokens/char", "avg_tokens_per_char", 2.0, "max"),
                ("dev max unk rate", "dev_max_unk_rate", 0.005, "max"),
                ("dev min decode fidelity", "dev_min_decode_fidelity", 1.0, "min"),
                ("dev min compression vs char", "dev_min_compression_vs_char", 1.0, "min"),
            )
        ]
        + ["", "Benchmark report: `ml/tokenizer/artifacts/benchmark.json`"]
    )
