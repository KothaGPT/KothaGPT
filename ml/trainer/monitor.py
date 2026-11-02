"""Training monitoring: console progress + JSONL history file.

Always writes ``<out>/history.jsonl`` (one JSON object per macro-step) and
prints a compact progress line to stdout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Monitor:
    def __init__(self, out_dir: str | Path, rank: int = 0) -> None:
        self.out_dir = Path(out_dir)
        self.rank = rank
        self.start_time = time.monotonic()

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        record = {
            "step": step,
            "elapsed_s": round(time.monotonic() - self.start_time, 2),
            **metrics,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        parts = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
        )
        print(f"step {step}: {parts}", flush=True)
