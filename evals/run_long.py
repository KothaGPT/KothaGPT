"""WS-11: long-context evaluation harness.

Builds a ``KothaGPT`` from ``--config`` (optionally loading ``--checkpoint``),
optionally extends its positional cache via ``--extend`` with RoPE theta
scaling, then runs the ``data/benchmarks/bangla/long`` probes:

- ``needle`` — needle-in-a-haystack recall at increasing context/depth.
- ``long_ppl`` — reference perplexity as the filler prefix grows past the
  native training context (the WS-11 metric: <5% ppl degradation on
  extension via theta scaling).

Writes ``evals/results/<name>-<date>.json`` and ``.REPORT.md``.

Usage:
    python -m evals.run_long --config ml/configs/long.yaml
    python -m evals.run_long --config ml/configs/small.yaml \\
        --checkpoint runs/pretrain/checkpoints/step-0001000.pt \\
        --extend 16384 --theta 500000 --scaling ntk --factor 4.0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import torch

from ml.models import KothaGPT, load_config
from ml.trainer.checkpoint import load_checkpoint

_EVALS_DIR = Path(__file__).parent
_RESULTS_DIR = _EVALS_DIR / "results"
_DATA_DIR = Path(__file__).parent.parent / "data/benchmarks/bangla/long"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_model(config_path: str | Path, checkpoint: str | Path | None, device: str) -> KothaGPT:
    config = load_config(config_path)
    model = KothaGPT(config.model)
    if checkpoint is not None:
        payload = load_checkpoint(checkpoint)
        model.load_state_dict(payload["model_state"])
    return model.to(device)


@torch.no_grad()
def reference_ppl(
    model: KothaGPT,
    tokenizer: Any,
    prefix_text: str,
    reference_text: str,
    device: str,
) -> float:
    """Perplexity of ``reference_text`` conditioned on ``prefix_text`` (loss on ref only)."""
    prefix_ids = tokenizer.encode(prefix_text)
    ref_ids = tokenizer.encode(reference_text)
    if not ref_ids:
        return math.nan
    if len(prefix_ids) + len(ref_ids) > model.config.max_position_embeddings:
        return math.nan
    input_ids = torch.tensor([prefix_ids + ref_ids], dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prefix_ids) - 1 : len(prefix_ids) - 1 + len(ref_ids)] = torch.tensor(
        ref_ids, dtype=torch.long
    )
    loss = model(input_ids=input_ids, labels=labels)["loss"].item()
    return math.exp(min(loss, 20))


@torch.no_grad()
def needle_recall(
    model: KothaGPT,
    tokenizer: Any,
    record: dict[str, Any],
    device: str,
) -> tuple[str, bool]:
    prompt = record["haystack"] + " প্রশ্ন: " + record["question"] + "\nউত্তর: "
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(prompt_ids, max_new_tokens=24, temperature=1e-6)
    prediction = tokenizer.decode(out[0, prompt_ids.shape[1] :].tolist())
    return prediction, record["answer"] in prediction


def run(
    config_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    extend: int | None = None,
    theta: float | None = None,
    scaling: str | None = None,
    factor: float | None = None,
    data_dir: Path = _DATA_DIR,
    limit: int | None = None,
) -> dict[str, Any]:
    from ml.tokenizer import load_tokenizer

    model = build_model(config_path, checkpoint, device)
    cfg = load_config(config_path)
    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    extend_info: dict[str, Any] | None = None
    if extend is not None:
        model.extend_context_length(extend, theta, scaling, factor)
        extend_info = {
            "new_max_position_embeddings": extend,
            "rope_theta": theta or model.config.rope_theta,
            "rope_scaling": scaling or model.config.rope_scaling,
            "rope_scaling_factor": factor or model.config.rope_scaling_factor,
        }

    needle_records = _load_jsonl(data_dir / "needle.jsonl")
    if limit is not None:
        needle_records = needle_records[:limit]
    per_depth: dict[float, list[bool]] = {}
    per_context: dict[int, list[bool]] = {}
    for record in needle_records:
        _, hit = needle_recall(model, tokenizer, record, device)
        per_depth.setdefault(record["depth_pct"], []).append(hit)
        per_context.setdefault(record["context_len"], []).append(hit)

    needle_summary = {
        "instances": len(needle_records),
        "recall": sum(v for hits in per_depth.values() for v in hits) / max(len(needle_records), 1),
        "by_depth": {
            f"{int(d * 100)}": sum(hits) / len(hits) for d, hits in sorted(per_depth.items())
        },
        "by_context": {str(k): sum(hits) / len(hits) for k, hits in sorted(per_context.items())},
    }

    ppl_records = _load_jsonl(data_dir / "long_ppl.jsonl")
    if limit is not None:
        ppl_records = ppl_records[:limit]
    ppl_rows: list[dict[str, Any]] = []
    for record in ppl_records:
        filler = record["filler"]
        ref = record["reference"]
        for frac in (0.25, 0.5, 0.75, 1.0):
            cut = int(len(filler) * frac)
            prefix = filler[:cut]
            ppl = reference_ppl(model, tokenizer, prefix, ref, device)
            within = (
                frac * record["context_len"] + len(tokenizer.encode(ref))
                <= cfg.model.max_position_embeddings
            )
            ppl_rows.append(
                {
                    "record_id": record["record_id"],
                    "context_len": record["context_len"],
                    "prefix_tokens": len(tokenizer.encode(prefix)),
                    "ppl": ppl,
                    "within_native_ctx": within,
                }
            )

    within_rows = [r for r in ppl_rows if r["within_native_ctx"] and not math.isnan(r["ppl"])]
    beyond_rows = [r for r in ppl_rows if not r["within_native_ctx"] and not math.isnan(r["ppl"])]
    degradation_pct = None
    if within_rows and beyond_rows:
        baseline = sum(r["ppl"] for r in within_rows) / len(within_rows)
        beyond = sum(r["ppl"] for r in beyond_rows) / len(beyond_rows)
        degradation_pct = (beyond - baseline) / baseline * 100.0

    results = {
        "config": str(config_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "extend": extend_info,
        "needle": needle_summary,
        "long_ppl": {
            "rows": ppl_rows,
            "baseline_ppl": _mean_ppl(within_rows),
            "beyond_ppl": _mean_ppl(beyond_rows),
            "degradation_pct": degradation_pct,
        },
        "date": _dt.datetime.now(_dt.UTC).date().isoformat(),
    }
    return results


def _mean_ppl(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(r["ppl"] for r in rows) / len(rows)


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Long-Context Evaluation Report",
        "",
        f"- **config**: {results['config']}",
        f"- **checkpoint**: {results['checkpoint'] or 'random-init'}",
        f"- **extend**: {results['extend'] or 'native'}",
        f"- **date**: {results['date']}",
        "",
        "## Needle-in-a-haystack recall",
        "",
        f"- **instances**: {results['needle']['instances']}",
        f"- **overall recall**: {results['needle']['recall']:.4f}",
        "",
        "| context_len | recall |",
        "| --- | --- |",
    ]
    for k, v in results["needle"]["by_context"].items():
        lines.append(f"| {k} | {v:.4f} |")
    lines += [
        "",
        "## Reference ppl vs context growth",
        "",
        "| record_id | prefix_tokens | ppl | within native ctx |",
        "| --- | --- | --- | --- |",
    ]
    for row in results["long_ppl"]["rows"]:
        lines.append(
            f"| {row['record_id']} | {row['prefix_tokens']} | {row['ppl']:.4f} | {row['within_native_ctx']} |"
        )
    p = results["long_ppl"]["degradation_pct"]
    lines.append(
        f"\n**ppl degradation on extension**: {p:.2f}%"
        if p is not None
        else "\n**ppl degradation on extension**: n/a"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extend", type=int, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--scaling", choices=("none", "linear", "ntk"), default=None)
    parser.add_argument("--factor", type=float, default=None)
    parser.add_argument("--data-dir", default=str(_DATA_DIR))
    parser.add_argument("--out-dir", default=str(_RESULTS_DIR))
    parser.add_argument("--name", default="long")
    parser.add_argument(
        "--limit", type=int, default=None, help="max instances per task (smoke runs)"
    )
    args = parser.parse_args(argv)

    results = run(
        args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        extend=args.extend,
        theta=args.theta,
        scaling=args.scaling,
        factor=args.factor,
        data_dir=Path(args.data_dir),
        limit=args.limit,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.name}-{results['date']}.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{args.name}-{results['date']}.REPORT.md").write_text(
        _render_report(results), encoding="utf-8"
    )
    print(f"wrote {out_json}")
    print(f"needle recall: {results['needle']['recall']:.4f}")
    print(f"ppl degradation on extension: {results['long_ppl']['degradation_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
