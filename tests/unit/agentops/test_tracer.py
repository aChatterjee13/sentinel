"""Unit tests for sentinel.agentops.trace.tracer — AgentTracer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sentinel.agentops.trace.tracer import AgentTracer
from sentinel.config.schema import TracingConfig
from sentinel.core.types import SpanStatus


@pytest.fixture
def tracer() -> AgentTracer:
    return AgentTracer(config=TracingConfig())


class TestTraceLifecycle:
    """Opening and closing a trace context."""

    def test_trace_creates_agent_trace(self, tracer: AgentTracer) -> None:
        with tracer.trace("test_agent") as state:
            assert state["agent_name"] == "test_agent"
            assert state["trace_id"]

        result = tracer.get_last_trace()
        assert result is not None
        assert result.agent_name == "test_agent"
        assert result.status == SpanStatus.OK
        assert result.ended_at is not None

    def test_trace_captures_error_status(self, tracer: AgentTracer) -> None:
        with pytest.raises(ValueError, match="boom"), tracer.trace("fail_agent"):
            raise ValueError("boom")

        result = tracer.get_last_trace()
        assert result is not None
        assert result.status == SpanStatus.ERROR
        assert result.metadata.get("error") == "boom"

    def test_trace_metadata_passthrough(self, tracer: AgentTracer) -> None:
        with tracer.trace("meta_agent", env="staging", run_id="abc"):
            pass

        result = tracer.get_last_trace()
        assert result is not None
        assert result.metadata["env"] == "staging"
        assert result.metadata["run_id"] == "abc"


class TestSpan:
    """Span creation and recording inside a trace."""

    def test_span_recorded_in_trace(self, tracer: AgentTracer) -> None:
        with tracer.trace("agent"):
            with tracer.span("plan", kind="step"):
                pass
            with tracer.span("execute", kind="tool_call"):
                pass

        result = tracer.get_last_trace()
        assert result is not None
        assert len(result.spans) == 2
        assert result.spans[0].name == "plan"
        assert result.spans[1].name == "execute"

    def test_tool_call_span_increments_count(self, tracer: AgentTracer) -> None:
        with tracer.trace("agent"):
            with tracer.span("search", kind="tool_call"):
                pass
            with tracer.span("fetch", kind="tool_call"):
                pass
            with tracer.span("think", kind="step"):
                pass

        result = tracer.get_last_trace()
        assert result is not None
        assert result.tool_call_count == 2

    def test_span_captures_error(self, tracer: AgentTracer) -> None:
        with pytest.raises(RuntimeError), tracer.trace("agent"), tracer.span("bad_step"):
            raise RuntimeError("span error")

        result = tracer.get_last_trace()
        assert result is not None
        assert result.spans[0].status == SpanStatus.ERROR
        assert result.spans[0].attributes.get("error") == "span error"

    def test_span_accumulates_tokens_and_cost(self, tracer: AgentTracer) -> None:
        with tracer.trace("agent"):
            with tracer.span("llm_call", kind="llm_call", tokens=500, cost=0.02):
                pass
            with tracer.span("llm_call_2", kind="llm_call", tokens=300, cost=0.01):
                pass

        result = tracer.get_last_trace()
        assert result is not None
        assert result.total_tokens == 800
        assert result.total_cost == pytest.approx(0.03)

    def test_span_parent_id_linked(self, tracer: AgentTracer) -> None:
        with tracer.trace("agent"), tracer.span("parent") as parent_span:
            with tracer.span("child") as child_span:
                captured_parent_id = child_span.parent_id

        assert captured_parent_id == parent_span.span_id


class TestReadAPI:
    """get_last_trace, get_recent, all_traces."""

    def test_get_last_trace_empty(self, tracer: AgentTracer) -> None:
        assert tracer.get_last_trace() is None

    def test_get_recent_returns_latest(self, tracer: AgentTracer) -> None:
        for i in range(5):
            with tracer.trace(f"agent_{i}"):
                pass

        recent = tracer.get_recent(n=3)
        assert len(recent) == 3
        assert recent[-1].agent_name == "agent_4"

    def test_all_traces(self, tracer: AgentTracer) -> None:
        for i in range(3):
            with tracer.trace(f"a_{i}"):
                pass
        assert len(tracer.all_traces()) == 3


class TestTraceDecorator:
    """@trace_agent decorator for sync functions."""

    def test_decorator_wraps_sync(self, tracer: AgentTracer) -> None:
        @tracer.trace_agent("decorated_agent")
        def my_task():
            return 42

        result = my_task()
        assert result == 42
        trace = tracer.get_last_trace()
        assert trace is not None
        assert trace.agent_name == "decorated_agent"
        assert trace.status == SpanStatus.OK


class TestTelemetryHelpers:
    """add_event, add_attributes, update_trace."""

    def test_update_trace_accumulates_tokens(self, tracer: AgentTracer) -> None:
        with tracer.trace("agent"):
            tracer.update_trace(total_tokens=100)
            tracer.update_trace(total_tokens=200)

        result = tracer.get_last_trace()
        assert result is not None
        assert result.total_tokens == 300

    def test_update_trace_stores_metadata(self, tracer: AgentTracer) -> None:
        with tracer.trace("agent"):
            tracer.update_trace(custom_key="hello")

        result = tracer.get_last_trace()
        assert result is not None
        assert result.metadata["custom_key"] == "hello"

    def test_update_trace_noop_outside_context(self, tracer: AgentTracer) -> None:
        tracer.update_trace(total_tokens=999)
        assert tracer.get_last_trace() is None


class TestExporter:
    """Exporter integration."""

    def test_exporter_called_on_trace_end(self, tracer: AgentTracer) -> None:
        exporter = MagicMock()
        tracer.exporters.append(exporter)

        with tracer.trace("agent"):
            pass

        exporter.export.assert_called_once()
        exported_trace = exporter.export.call_args[0][0]
        assert exported_trace.agent_name == "agent"

    def test_exporter_failure_does_not_crash(self, tracer: AgentTracer) -> None:
        bad_exporter = MagicMock()
        bad_exporter.export.side_effect = RuntimeError("export failed")
        tracer.exporters.append(bad_exporter)

        with tracer.trace("agent"):
            pass

        result = tracer.get_last_trace()
        assert result is not None
