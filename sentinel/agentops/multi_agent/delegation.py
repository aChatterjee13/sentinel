"""Track agent-to-agent delegations across a multi-agent workflow."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from sentinel.core.exceptions import AgentError

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DelegationLink:
    """A single delegation edge between two agents."""

    run_id: str
    source: str
    target: str
    task: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class DelegationTracker:
    """In-memory delegation graph for the active runs.

    The tracker records every ``source → target`` delegation, exposes the
    chain for each run, and answers simple structural queries (depth,
    cycles, fan-out) used by the multi-agent monitor.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._links: deque[DelegationLink] = deque(maxlen=max_history)
        self._by_run: dict[str, list[DelegationLink]] = defaultdict(list)
        self._lock = threading.RLock()

    def record(
        self,
        run_id: str,
        source: str,
        target: str,
        task: str,
        **metadata: Any,
    ) -> DelegationLink:
        if source == target:
            raise AgentError(f"self-delegation not allowed: {source}")
        link = DelegationLink(
            run_id=run_id,
            source=source,
            target=target,
            task=task,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
        )
        with self._lock:
            self._links.append(link)
            self._by_run[run_id].append(link)
        log.debug("delegation.recorded", run_id=run_id, source=source, target=target)
        return link

    def chain(self, run_id: str) -> list[DelegationLink]:
        with self._lock:
            return list(self._by_run.get(run_id, []))

    def depth(self, run_id: str) -> int:
        with self._lock:
            return len(self._by_run.get(run_id, []))

    def has_cycle(self, run_id: str) -> bool:
        """Detect cycles in the delegation graph using DFS.

        Detects any cycle including A→B→C→A (transitive) and A↔B
        (bidirectional).
        """
        with self._lock:
            links = self._by_run.get(run_id, [])
            if not links:
                return False
            # Build adjacency list
            graph: dict[str, list[str]] = {}
            for link in links:
                graph.setdefault(link.source, []).append(link.target)
            # DFS for back-edges
            WHITE, GRAY, BLACK = 0, 1, 2
            color: dict[str, int] = dict.fromkeys(graph, WHITE)
            # Also add targets that may not be sources
            for link in links:
                color.setdefault(link.target, WHITE)

            def dfs(node: str) -> bool:
                color[node] = GRAY
                for neighbor in graph.get(node, []):
                    if color.get(neighbor, WHITE) == GRAY:
                        return True  # Back edge = cycle
                    if color.get(neighbor, WHITE) == WHITE and dfs(neighbor):
                        return True
                color[node] = BLACK
                return False

            return any(
                dfs(node) for node, c in list(color.items()) if c == WHITE
            )

    def fan_out(self, run_id: str, source: str) -> list[str]:
        with self._lock:
            return [link.target for link in self._by_run.get(run_id, []) if link.source == source]

    def end_run(self, run_id: str) -> list[DelegationLink]:
        with self._lock:
            return self._by_run.pop(run_id, [])
