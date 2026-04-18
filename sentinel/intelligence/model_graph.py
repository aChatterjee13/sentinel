"""Multi-model dependency DAG with cascade alerting."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sentinel.config.schema import ModelGraphConfig


@dataclass(frozen=True)
class ModelNode:
    """A node in the model dependency graph."""

    name: str
    metadata: dict[str, Any]


class ModelGraph:
    """Tracks upstream/downstream model relationships.

    When an upstream model drifts, all downstream models are notified so the
    notification engine can fire cascade alerts.  All public methods are
    thread-safe.  ``add_edge`` performs eager cycle detection and rolls back
    the edge if it would introduce a cycle.

    Example:
        >>> graph = ModelGraph(config.model_graph)
        >>> graph.add_edge("feature_pipeline", "fraud_classifier")
        >>> graph.get_descendants("feature_pipeline")
        ['fraud_classifier']
    """

    def __init__(self, config: ModelGraphConfig | None = None):
        self._lock = threading.RLock()
        self._nodes: dict[str, ModelNode] = {}
        self._out: dict[str, set[str]] = defaultdict(set)
        self._in: dict[str, set[str]] = defaultdict(set)
        self.cascade_alerts = config.cascade_alerts if config else True
        if config is not None:
            for edge in config.dependencies:
                self.add_edge(edge.upstream, edge.downstream)

    def add_node(self, name: str, **metadata: Any) -> ModelNode:
        """Register a node in the graph.

        Args:
            name: Unique node identifier.
            **metadata: Arbitrary key-value pairs attached to the node.

        Returns:
            The newly created ``ModelNode``.
        """
        with self._lock:
            node = ModelNode(name=name, metadata=metadata)
            self._nodes[name] = node
            return node

    def add_edge(self, upstream: str, downstream: str) -> None:
        """Add a directed edge, rejecting it if a cycle would result.

        Args:
            upstream: Source node name.
            downstream: Target node name.

        Raises:
            ValueError: If the edge would create a cycle in the DAG.
        """
        with self._lock:
            if upstream not in self._nodes:
                self.add_node(upstream)
            if downstream not in self._nodes:
                self.add_node(downstream)
            self._out[upstream].add(downstream)
            self._in[downstream].add(upstream)
            # Eager cycle detection — verify no path from downstream back to upstream
            if self._has_path(downstream, upstream):
                # Rollback
                self._out[upstream].discard(downstream)
                self._in[downstream].discard(upstream)
                raise ValueError(
                    f"adding edge {upstream} → {downstream} creates a cycle"
                )

    # ── private helpers (must be called under self._lock) ─────

    def _has_path(self, src: str, dst: str) -> bool:
        """BFS reachability check from *src* to *dst*.

        Must be called while ``self._lock`` is held.
        """
        visited: set[str] = set()
        queue = [src]
        while queue:
            current = queue.pop(0)
            if current == dst:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._out.get(current, set()))
        return False

    # ── public queries ────────────────────────────────────────

    def get_descendants(self, name: str) -> list[str]:
        """Return all transitive downstream models."""
        with self._lock:
            seen: set[str] = set()
            stack = list(self._out.get(name, set()))
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(self._out.get(current, set()))
            return sorted(seen)

    def get_ancestors(self, name: str) -> list[str]:
        """Return all transitive upstream models."""
        with self._lock:
            seen: set[str] = set()
            stack = list(self._in.get(name, set()))
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(self._in.get(current, set()))
            return sorted(seen)

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm — useful for retrain ordering."""
        with self._lock:
            in_degree = {n: len(self._in.get(n, set())) for n in self._nodes}
            queue = [n for n, d in in_degree.items() if d == 0]
            order: list[str] = []
            while queue:
                current = queue.pop(0)
                order.append(current)
                for child in self._out.get(current, set()):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            if len(order) != len(self._nodes):
                raise ValueError("graph contains a cycle")
            return order

    def cascade_impact(self, source_model: str) -> dict[str, list[str]]:
        """Return the cascade chain for a source model drift event."""
        with self._lock:
            if not self.cascade_alerts:
                return {}
            descendants = self.get_descendants(source_model)
            return {
                "source": [source_model],
                "downstream_affected": descendants,
                "depth": len(descendants),
            }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph to a plain dict."""
        with self._lock:
            return {
                "nodes": [
                    {"name": n.name, "metadata": n.metadata} for n in self._nodes.values()
                ],
                "edges": [
                    {"upstream": u, "downstream": d}
                    for u, ds in self._out.items()
                    for d in ds
                ],
            }
