"""WS-4 Batch inference — continuous batching stub."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class BatchItem:
    prompt: str
    max_tokens: int = 32

class Batcher:
    def __init__(self, engine, max_batch: int = 4):
        self.engine = engine
        self.max_batch = max_batch

    def generate_many(self, items: list[BatchItem]) -> list[list[str]]:
        # Simple: sequential but batched interface; real impl would pack KV cache
        results: list[list[str]] = []
        for it in items:
            results.append(list(self.engine.generate(it.prompt, max_new_tokens=it.max_tokens)))
        return results

    def stream_many(self, items: list[BatchItem]) -> Iterator[tuple[int, str]]:
        # Interleaved streaming stub
        gens = [self.engine.generate(it.prompt, max_new_tokens=it.max_tokens) for it in items]
        active = {i: g for i, g in enumerate(gens)}
        while active:
            for idx, g in list(active.items()):
                try:
                    tok = next(g)
                    yield idx, tok
                except StopIteration:
                    del active[idx]
