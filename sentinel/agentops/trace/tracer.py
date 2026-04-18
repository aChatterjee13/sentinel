"""Span-based agent tracer with OpenTelemetry-compatible semantics."""

from __future__ import annotations

import contextvars
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from sentinel.config.schema import AgentOpsConfig, TracingConfig
from sentinel.core.types import AgentTrace, Span, SpanStatus

log = structlog.get_logger(__name__)

MAX_SPANS_PER_TRACE = 1000


_current_trace: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_current_trace", default=None
)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_current_span", default=None
)


class AgentTracer:
    """Captures hierarchical spans for a single agent run.

    Spans are stored in memory by default and can be exported through
    pluggable exporters. The tracer is thread-safe and uses
    :class:`contextvars` so spans propagate across async boundaries.

    Context propagation for multi-agent scenarios
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    When an agent delegates work to another agent (A → B), trace context
    must propagate so the full delegation chain appears as a single
    distributed trace.

    * **Same process**: both agents share the :data:`_current_trace` and
      :data:`_current_span` context-vars. Open a nested :meth:`span`
      inside the parent trace and the child spans will automatically
      attach under it via ``parent_id``.

    * **Cross-process / cross-service**: serialise the ``trace_id`` and
      current ``span_id`` into the delegation payload (e.g. as HTTP
      headers ``X-Sentinel-Trace-Id`` / ``X-Sentinel-Parent-Span-Id``
      following W3C Trace-Context conventions). The receiving
      ``AgentTracer`` should call :meth:`trace` with ``trace_id`` and
      ``parent_span_id`` metadata so exporters can stitch the spans
      together.

    * **OpenTelemetry export**: When the OTLP exporter is configured, the
      ``trace_id`` maps to an OTel trace and each :class:`Span` maps to
      an OTel span, preserving the parent-child hierarchy across agent
      boundaries.
    """

    def __init__(self, config: TracingConfig | None = None, exporters: list[Any] | None = None):
        self.config = config or TracingConfig()
        self.exporters = exporters or []
        self._lock = threading.Lock()
        self._traces: deque[AgentTrace] = deque(maxlen=1000)

    @classmethod
    def from_config(cls, config: AgentOpsConfig | str | Any) -> AgentTracer:
        if isinstance(config, str):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            return cls(cfg.agentops.tracing)
        if isinstance(config, AgentOpsConfig):
            return cls(config.tracing)
        return cls(config.agentops.tracing)  # type: ignore[union-attr]

    # ── Trace lifecycle ───────────────────────────────────────────

    @contextmanager
    def trace(self, agent_name: str, **metadata: Any) -> Iterator[dict[str, Any]]:
        """Open a new trace context for an agent run."""
        trace_state = {
            "trace_id": uuid4().hex[:16],
            "agent_name": agent_name,
            "started_at": datetime.now(timezone.utc),
            "spans": [],
            "metadata": metadata,
            "total_tokens": 0,
            "total_cost": 0.0,
            "tool_call_count": 0,
        }
        token = _current_trace.set(trace_state)
        try:
            yield trace_state
            trace_state.setdefault("status", SpanStatus.OK)
        except Exception as e:
            trace_state["status"] = SpanStatus.ERROR
            trace_state.setdefault("metadata", {})["error"] = str(e)
            raise
        finally:
            trace_state["ended_at"] = datetime.now(timezone.utc)
            self._finalise(trace_state)
            _current_trace.reset(token)

    def trace_agent(self, agent_name: str, **metadata: Any):
        """Decorator form of :meth:`trace`."""

        def decorator(fn):
            from functools import wraps

            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                with self.trace(agent_name, **metadata):
                    return fn(*args, **kwargs)

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with self.trace(agent_name, **metadata):
                    return await fn(*args, **kwargs)

            import asyncio

            return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper

        return decorator

    # ── Span lifecycle ────────────────────────────────────────────

    @contextmanager
    def span(self, name: str, kind: str = "step", **attributes: Any) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(
            parent_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            start_time=datetime.now(timezone.utc),
            attributes=attributes,
        )
        token = _current_span.set(span)
        try:
            yield span
            self._record(span, status=SpanStatus.OK)
        except Exception as e:
            self._record(span, status=SpanStatus.ERROR, error=str(e))
            raise
        finally:
            _current_span.reset(token)

    def _record(self, span: Span, status: SpanStatus, error: str | None = None) -> None:
        end_time = datetime.now(timezone.utc)
        attrs = dict(span.attributes)
        if error:
            attrs["error"] = error
        finalised = span.model_copy(
            update={"end_time": end_time, "status": status, "attributes": attrs}
        )
        trace_state = _current_trace.get()
        if trace_state is not None:
            if len(trace_state["spans"]) >= MAX_SPANS_PER_TRACE:
                log.warning(
                    "trace.max_spans_exceeded",
                    trace_id=trace_state["trace_id"],
                    max=MAX_SPANS_PER_TRACE,
                )
                return
            trace_state["spans"].append(finalised)
            if span.kind == "tool_call":
                trace_state["tool_call_count"] += 1
            tokens = attrs.get("tokens", 0)
            cost = attrs.get("cost", 0.0)
            if isinstance(tokens, (int, float)):
                trace_state["total_tokens"] += int(tokens)
            if isinstance(cost, (int, float)):
                trace_state["total_cost"] += float(cost)

    # ── Telemetry helpers ─────────────────────────────────────────

    def add_event(self, name: str, **attributes: Any) -> None:
        span = _current_span.get()
        if span is None:
            return
        span.events.append(
            {"name": name, "timestamp": datetime.now(timezone.utc).isoformat(), **attributes}
        )  # type: ignore[arg-type]

    def add_attributes(self, **attributes: Any) -> None:
        span = _current_span.get()
        if span is None:
            return
        span.attributes.update(attributes)  # type: ignore[arg-type]

    def update_trace(self, **fields: Any) -> None:
        trace_state = _current_trace.get()
        if trace_state is None:
            return
        for k, v in fields.items():
            if k in {"total_tokens", "total_cost", "tool_call_count"} and isinstance(
                v, (int, float)
            ):
                trace_state[k] += v
            else:
                trace_state.setdefault("metadata", {})[k] = v

    # ── Finalisation ──────────────────────────────────────────────

    def _finalise(self, state: dict[str, Any]) -> None:
        trace = AgentTrace(
            trace_id=state["trace_id"],
            agent_name=state["agent_name"],
            started_at=state["started_at"],
            ended_at=state.get("ended_at"),
            spans=list(state.get("spans", [])),
            total_tokens=int(state.get("total_tokens", 0)),
            total_cost=float(state.get("total_cost", 0.0)),
            tool_call_count=int(state.get("tool_call_count", 0)),
            status=state.get("status", SpanStatus.OK),
            metadata=state.get("metadata", {}),
        )
        with self._lock:
            self._traces.append(trace)
        failed_count = 0
        for exporter in self.exporters:
            try:
                exporter.export(trace)
            except Exception as e:
                failed_count += 1
                log.warning(
                    "trace.export_failed",
                    exporter=type(exporter).__name__,
                    error=str(e),
                )
        if failed_count == len(self.exporters) and self.exporters:
            log.error("trace.all_exporters_failed", trace_id=trace.trace_id)

    # ── Read API ──────────────────────────────────────────────────

    def get_last_trace(self) -> AgentTrace | None:
        with self._lock:
            return self._traces[-1] if self._traces else None

    def get_recent(self, n: int = 10) -> list[AgentTrace]:
        with self._lock:
            return list(self._traces)[-n:]

    def all_traces(self) -> list[AgentTrace]:
        with self._lock:
            return list(self._traces)
