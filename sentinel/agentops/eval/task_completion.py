"""Track agent task completion rates."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from sentinel.config.schema import AgentEvaluationConfig

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TaskCompletionResult:
    """Result of a single task completion event."""

    agent: str
    task_type: str
    success: bool
    score: float
    duration_ms: float | None
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskCompletionTracker:
    """Sliding-window tracker of task success rates per agent and task type.

    Each completion is recorded with a binary success flag and an
    optional graded score in ``[0, 1]``. The tracker exposes per-agent
    and per-task success rates and emits a warning when the rolling
    success rate drops below the configured minimum.
    """

    def __init__(self, config: AgentEvaluationConfig | None = None, window: int = 500):
        self.config = config or AgentEvaluationConfig()
        self.window = window
        self._results: dict[tuple[str, ...], deque[TaskCompletionResult]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        task_cfg = self.config.task_completion or {}
        self.min_success_rate = float(task_cfg.get("min_success_rate", 0.85))
        self.track_by: list[str] = task_cfg.get("track_by", ["agent", "task_type"])
        self._lock = threading.Lock()

    def _make_key(self, agent: str, task_type: str, **metadata: Any) -> tuple[str, ...]:
        """Build a tracking key from the configured dimensions."""
        parts: list[str] = []
        for dim in self.track_by:
            if dim == "agent":
                parts.append(agent)
            elif dim == "task_type":
                parts.append(task_type)
            else:
                parts.append(str(metadata.get(dim, "unknown")))
        return tuple(parts) if parts else (agent, task_type)

    def record(
        self,
        agent: str,
        task_type: str,
        success: bool,
        *,
        score: float | None = None,
        duration_ms: float | None = None,
        **metadata: Any,
    ) -> TaskCompletionResult:
        result = TaskCompletionResult(
            agent=agent,
            task_type=task_type,
            success=bool(success),
            score=float(score) if score is not None else (1.0 if success else 0.0),
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
        )
        with self._lock:
            key = self._make_key(agent, task_type, **metadata)
            self._results[key].append(result)
            rate = self._success_rate_unlocked(agent=agent, task_type=task_type)
        if rate is not None and rate < self.min_success_rate:
            log.warning(
                "task_completion.below_threshold",
                agent=agent,
                task_type=task_type,
                rate=rate,
                threshold=self.min_success_rate,
            )
        return result

    def success_rate(
        self, *, agent: str | None = None, task_type: str | None = None
    ) -> float | None:
        with self._lock:
            return self._success_rate_unlocked(agent=agent, task_type=task_type)

    def _success_rate_unlocked(
        self, *, agent: str | None = None, task_type: str | None = None
    ) -> float | None:
        """Compute success rate without acquiring lock (caller must hold it)."""
        items = list(self._iter(agent=agent, task_type=task_type))
        if not items:
            return None
        return sum(1 for r in items if r.success) / len(items)

    def average_score(
        self, *, agent: str | None = None, task_type: str | None = None
    ) -> float | None:
        with self._lock:
            items = list(self._iter(agent=agent, task_type=task_type))
        if not items:
            return None
        return sum(r.score for r in items) / len(items)

    def average_duration_ms(
        self, *, agent: str | None = None, task_type: str | None = None
    ) -> float | None:
        with self._lock:
            items = [r for r in self._iter(agent=agent, task_type=task_type) if r.duration_ms]
        if not items:
            return None
        return sum(r.duration_ms or 0.0 for r in items) / len(items)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {}
            for key, results in self._results.items():
                label = "::".join(key)
                out[label] = {
                    "n": len(results),
                    "success_rate": (
                        sum(1 for r in results if r.success) / len(results)
                        if results
                        else 0.0
                    ),
                    "avg_score": (
                        sum(r.score for r in results) / len(results) if results else 0.0
                    ),
                }
            return out

    def _iter(self, *, agent: str | None, task_type: str | None) -> Iterable[TaskCompletionResult]:
        for _key, results in self._results.items():
            for r in results:
                if agent is not None and r.agent != agent:
                    continue
                if task_type is not None and r.task_type != task_type:
                    continue
                yield r
