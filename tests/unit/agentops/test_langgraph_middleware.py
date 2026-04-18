"""Tests for Gap-C: LangGraph auto-instrumentation middleware."""

from __future__ import annotations

from typing import Any

import pytest

from sentinel.agentops.integrations import LangGraphMiddleware, MonitoredGraph
from sentinel.agentops.trace.tracer import AgentTracer


class FakeGraph:
    """Fake LangGraph compiled graph for testing."""

    name = "test_graph"

    def __init__(self, nodes: list[tuple[str, dict[str, Any]]]) -> None:
        self._nodes = nodes

    def stream(self, input: Any, config: Any = None, **kwargs: Any):
        for node_name, output in self._nodes:
            yield {node_name: output}

    async def astream(self, input: Any, config: Any = None, **kwargs: Any):
        for node_name, output in self._nodes:
            yield {node_name: output}


class TestLangGraphMiddleware:
    def test_wrap_returns_monitored_graph(self) -> None:
        tracer = AgentTracer()
        middleware = LangGraphMiddleware(tracer)
        graph = FakeGraph([("node_a", {"key": "val"})])
        monitored = middleware.wrap(graph)
        assert isinstance(monitored, MonitoredGraph)

    def test_custom_agent_name(self) -> None:
        tracer = AgentTracer()
        middleware = LangGraphMiddleware(tracer)
        graph = FakeGraph([])
        monitored = middleware.wrap(graph, agent_name="custom_agent")
        assert monitored._agent_name == "custom_agent"

    def test_default_agent_name_from_graph(self) -> None:
        tracer = AgentTracer()
        middleware = LangGraphMiddleware(tracer)
        graph = FakeGraph([])
        monitored = middleware.wrap(graph)
        assert monitored._agent_name == "test_graph"


class TestMonitoredGraphInvoke:
    def test_invoke_traces_nodes(self) -> None:
        tracer = AgentTracer()
        middleware = LangGraphMiddleware(tracer)
        graph = FakeGraph(
            [
                ("plan", {"thought": "I need to search"}),
                ("search", {"results": [1, 2, 3]}),
                ("synthesise", {"output": "done"}),
            ]
        )
        monitored = middleware.wrap(graph, agent_name="test_agent")
        result = monitored.invoke({"input": "test"})

        # Result should be the last node's output
        assert result == {"output": "done"}

        # Should have one trace
        trace = tracer.get_last_trace()
        assert trace is not None
        assert trace.agent_name == "test_agent"
        assert len(trace.spans) == 3
        assert trace.spans[0].name == "plan"
        assert trace.spans[1].name == "search"
        assert trace.spans[2].name == "synthesise"

    def test_invoke_empty_graph(self) -> None:
        tracer = AgentTracer()
        graph = FakeGraph([])
        monitored = LangGraphMiddleware(tracer).wrap(graph)
        result = monitored.invoke({"input": "test"})
        assert result is None
        trace = tracer.get_last_trace()
        assert trace is not None
        assert len(trace.spans) == 0

    def test_invoke_records_output_keys(self) -> None:
        tracer = AgentTracer()
        graph = FakeGraph(
            [
                ("step1", {"key_a": 1, "key_b": 2}),
            ]
        )
        monitored = LangGraphMiddleware(tracer).wrap(graph)
        monitored.invoke({"input": "x"})
        trace = tracer.get_last_trace()
        assert trace is not None
        span = trace.spans[0]
        assert set(span.attributes.get("output_keys", [])) == {"key_a", "key_b"}


class TestMonitoredGraphAinvoke:
    @pytest.mark.asyncio
    async def test_ainvoke_traces_nodes(self) -> None:
        tracer = AgentTracer()
        graph = FakeGraph(
            [
                ("plan", {"thought": "planning"}),
                ("act", {"action": "doing"}),
            ]
        )
        monitored = LangGraphMiddleware(tracer).wrap(graph, agent_name="async_agent")
        result = await monitored.ainvoke({"input": "test"})

        assert result == {"action": "doing"}
        trace = tracer.get_last_trace()
        assert trace is not None
        assert trace.agent_name == "async_agent"
        assert len(trace.spans) == 2


class TestMonitoredGraphPassthrough:
    def test_getattr_forwards_to_graph(self) -> None:
        tracer = AgentTracer()
        graph = FakeGraph([])
        graph.custom_attr = "hello"  # type: ignore[attr-defined]
        monitored = LangGraphMiddleware(tracer).wrap(graph)
        assert monitored.custom_attr == "hello"  # type: ignore[attr-defined]

    def test_getattr_name(self) -> None:
        tracer = AgentTracer()
        graph = FakeGraph([])
        monitored = LangGraphMiddleware(tracer).wrap(graph)
        assert monitored.name == "test_graph"
