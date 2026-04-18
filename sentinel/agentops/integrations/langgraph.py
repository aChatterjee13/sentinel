"""LangGraph auto-instrumentation middleware.

Wraps a compiled LangGraph ``StateGraph`` so that every node execution
is captured as a span in the :class:`AgentTracer`. No LangGraph import
at module level — the middleware duck-types against ``stream()`` /
``astream()``.

Example:
    >>> from sentinel.agentops import AgentTracer
    >>> from sentinel.agentops.integrations import LangGraphMiddleware
    >>> tracer = AgentTracer()
    >>> middleware = LangGraphMiddleware(tracer)
    >>> monitored = middleware.wrap(compiled_graph)
    >>> result = monitored.invoke({"input": "..."})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentinel.agentops.trace.tracer import AgentTracer


class LangGraphMiddleware:
    """Zero-code instrumentation for LangGraph compiled graphs.

    Args:
        tracer: An :class:`AgentTracer` instance for span recording.

    Example:
        >>> middleware = LangGraphMiddleware(tracer)
        >>> monitored = middleware.wrap(compiled_graph)
        >>> result = monitored.invoke({"input": "..."})
    """

    def __init__(self, tracer: AgentTracer) -> None:
        self._tracer = tracer

    def wrap(self, graph: Any, *, agent_name: str | None = None) -> MonitoredGraph:
        """Wrap a compiled LangGraph graph with tracing.

        Args:
            graph: A compiled LangGraph ``StateGraph`` (or any object
                exposing ``.stream()`` / ``.astream()``).
            agent_name: Optional override for the agent name in traces.
                Defaults to ``graph.name`` or ``"langgraph_agent"``.

        Returns:
            A :class:`MonitoredGraph` proxy.
        """
        return MonitoredGraph(graph, self._tracer, agent_name=agent_name)


class MonitoredGraph:
    """Proxy around a LangGraph ``CompiledStateGraph``.

    Intercepts ``invoke()`` and ``ainvoke()`` to record per-node spans
    through the tracer. All other attribute access is forwarded to the
    wrapped graph for compatibility.
    """

    def __init__(
        self,
        graph: Any,
        tracer: AgentTracer,
        *,
        agent_name: str | None = None,
    ) -> None:
        self._graph = graph
        self._tracer = tracer
        self._agent_name = agent_name or getattr(graph, "name", "langgraph_agent")

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Synchronous invocation with tracing.

        Uses ``graph.stream()`` under the hood to intercept per-node
        events. Each node produces a span in the trace.

        Args:
            input: The input state for the graph.
            config: Optional LangGraph config dict.
            **kwargs: Forwarded to ``graph.stream()``.

        Returns:
            The final output state from the graph.
        """
        with self._tracer.trace(self._agent_name):
            final_state = None
            for event in self._graph.stream(input, config=config, **kwargs):
                if isinstance(event, dict):
                    for node_name, node_output in event.items():
                        with self._tracer.span(node_name, kind="step"):
                            self._tracer.add_attributes(
                                output_keys=list(node_output.keys())
                                if isinstance(node_output, dict)
                                else [],
                            )
                        final_state = node_output
            return final_state

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Asynchronous invocation with tracing.

        Uses ``graph.astream()`` under the hood to intercept per-node
        events. Each node produces a span in the trace.

        Args:
            input: The input state for the graph.
            config: Optional LangGraph config dict.
            **kwargs: Forwarded to ``graph.astream()``.

        Returns:
            The final output state from the graph.
        """
        with self._tracer.trace(self._agent_name):
            final_state = None
            async for event in self._graph.astream(input, config=config, **kwargs):
                if isinstance(event, dict):
                    for node_name, node_output in event.items():
                        with self._tracer.span(node_name, kind="step"):
                            self._tracer.add_attributes(
                                output_keys=list(node_output.keys())
                                if isinstance(node_output, dict)
                                else [],
                            )
                        final_state = node_output
            return final_state

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped graph."""
        return getattr(self._graph, name)
