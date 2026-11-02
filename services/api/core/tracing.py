"""WS-15 Tracing stubs — dependency-free OpenTelemetry-style tracing.

Provides a no-op tracer by default so instrumented code never crashes when
OpenTelemetry is not installed. When OTEL is available, a real tracer can be
swapped in behind the same interface.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    start_time: float = field(default_factory=__import__("time").time)
    end_time: float | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({"name": name, "attributes": attributes or {}})

    def end(self, status: str = "ok") -> None:
        self.status = status
        self.end_time = __import__("time").time()


@dataclass
class Tracer:
    """No-op tracer. Replace with an OpenTelemetry tracer in production."""

    service_name: str = "kothagpt"

    def start_span(
        self, name: str, *, parent: Span | None = None, attributes: dict[str, Any] | None = None
    ) -> Span:
        import uuid

        return Span(
            name=name,
            trace_id=(parent.trace_id if parent else uuid.uuid4().hex),
            span_id=uuid.uuid4().hex,
            parent_id=(parent.span_id if parent else None),
            attributes=attributes or {},
        )


tracer = Tracer()


@contextmanager
def trace_span(name: str, *, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
    """Context manager convenience for starting and ending a span."""
    span = tracer.start_span(name, attributes=attributes)
    try:
        yield span
    except Exception as exc:
        span.set_attribute("error", str(exc))
        span.end(status="error")
        raise
    else:
        span.end(status="ok")


__all__ = ["Span", "Tracer", "trace_span", "tracer"]
