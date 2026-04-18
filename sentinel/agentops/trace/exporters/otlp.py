"""OpenTelemetry OTLP exporter — lazy import."""

from __future__ import annotations

from typing import Any

import structlog

from sentinel.agentops.trace.exporters.base import BaseExporter
from sentinel.core.exceptions import TraceError
from sentinel.core.types import AgentTrace, Span

log = structlog.get_logger(__name__)


class OTLPExporter(BaseExporter):
    """Forward traces to an OTLP collector (Jaeger, Tempo, Honeycomb, ...).

    Requires the `agentops` extra: ``pip install sentinel-mlops[agentops]``.
    """

    name = "otlp"

    def __init__(self, endpoint: str, service_name: str = "sentinel-agent"):
        try:
            from opentelemetry import trace as ot  # type: ignore[import-not-found]
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,  # type: ignore[import-not-found]
            )
        except ImportError as e:
            raise TraceError(
                "agentops extra not installed — `pip install sentinel-mlops[agentops]`"
            ) from e
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        ot.set_tracer_provider(provider)
        self._tracer = ot.get_tracer("sentinel.agentops")

    def export(self, trace: AgentTrace) -> None:
        for span in trace.spans:
            self._export_span(trace, span)

    def _export_span(self, trace: AgentTrace, span: Span) -> None:
        try:
            with self._tracer.start_as_current_span(span.name) as ot_span:  # type: ignore[union-attr]
                ot_span.set_attribute("agent.name", trace.agent_name)
                ot_span.set_attribute("agent.trace_id", trace.trace_id)
                ot_span.set_attribute("span.kind", span.kind)
                for k, v in span.attributes.items():
                    if isinstance(v, (str, int, float, bool)):
                        ot_span.set_attribute(k, v)
                if span.status.value == "error":
                    ot_span.set_status(self._error_status())
        except Exception as e:
            log.warning("otlp.export_failed", error=str(e))

    @staticmethod
    def _error_status() -> Any:
        try:
            from opentelemetry.trace import Status, StatusCode  # type: ignore[import-not-found]

            return Status(StatusCode.ERROR)
        except Exception:
            return None
