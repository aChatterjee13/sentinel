"""Render traces as text timelines + decision trees."""

from __future__ import annotations

from sentinel.core.types import AgentTrace, Span


class TraceVisualiser:
    """Convert :class:`AgentTrace` objects into human-readable views.

    The default implementations are pure text — no GUI dependencies.
    Plug in Plotly/Dash separately for interactive dashboards.
    """

    def timeline(self, trace: AgentTrace) -> str:
        """Render a flat ASCII timeline of all spans."""
        lines = [f"trace {trace.trace_id} — agent={trace.agent_name}"]
        for span in trace.spans:
            duration = span.duration_ms
            duration_str = f"{duration:7.1f}ms" if duration is not None else "  ----  "
            lines.append(f"  {duration_str}  [{span.kind:12s}] {span.name}  ({span.status.value})")
        return "\n".join(lines)

    def tree(self, trace: AgentTrace) -> str:
        """Render the parent/child span hierarchy."""
        children: dict[str | None, list[Span]] = {}
        for span in trace.spans:
            children.setdefault(span.parent_id, []).append(span)

        lines: list[str] = [f"trace {trace.trace_id}"]

        def _walk(parent_id: str | None, depth: int) -> None:
            for span in children.get(parent_id, []):
                lines.append(
                    f"{'  ' * (depth + 1)}- {span.name} [{span.kind}] ({span.status.value})"
                )
                _walk(span.span_id, depth + 1)

        _walk(None, 0)
        return "\n".join(lines)

    def summary(self, trace: AgentTrace) -> dict[str, object]:
        """Return a small dict suitable for dashboards."""
        return {
            "trace_id": trace.trace_id,
            "agent": trace.agent_name,
            "status": trace.status.value,
            "duration_ms": trace.duration_ms,
            "step_count": trace.step_count,
            "tool_calls": trace.tool_call_count,
            "tokens": trace.total_tokens,
            "cost": trace.total_cost,
        }
