"""Detect infinite loops, circular delegations, and thrashing in agents."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import structlog

from sentinel.config.schema import LoopDetectionConfig
from sentinel.core.exceptions import LoopDetectedError

log = structlog.get_logger(__name__)


@dataclass
class _RunState:
    iterations: int = 0
    delegation_depth: int = 0
    delegation_chain: list[str] = field(default_factory=list)
    recent_actions: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=20))
    repeat_counts: dict[tuple[str, str], int] = field(default_factory=dict)


class LoopDetector:
    """Stateful per-run loop detector.

    The agent runtime calls :meth:`step` before every reasoning turn and
    :meth:`record_tool_call` after every tool invocation. The detector
    maintains a small ring buffer of recent actions and raises
    :class:`LoopDetectedError` (which the runtime should catch and use
    to terminate the run gracefully) when patterns appear.
    """

    def __init__(self, config: LoopDetectionConfig | None = None):
        self.config = config or LoopDetectionConfig()
        self._runs: dict[str, _RunState] = {}
        self._lock = threading.Lock()

    def begin_run(self, run_id: str) -> None:
        with self._lock:
            self._runs[run_id] = _RunState()

    def end_run(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)

    def step(self, run_id: str) -> None:
        with self._lock:
            state = self._runs.setdefault(run_id, _RunState())
            state.iterations += 1
            if state.iterations > self.config.max_iterations:
                raise LoopDetectedError(
                    f"max iterations exceeded: {state.iterations} > {self.config.max_iterations}"
                )

    def record_tool_call(self, run_id: str, tool: str, inputs: dict[str, Any]) -> None:
        with self._lock:
            state = self._runs.setdefault(run_id, _RunState())
            signature = (tool, _hash_inputs(inputs))
            state.recent_actions.append(signature)
            state.repeat_counts[signature] = state.repeat_counts.get(signature, 0) + 1
            if state.repeat_counts[signature] > self.config.max_repeated_tool_calls:
                raise LoopDetectedError(
                    f"tool '{tool}' called {state.repeat_counts[signature]} times with same input"
                )
            self._check_thrashing(state)

    def record_delegation(self, run_id: str, target_agent: str) -> None:
        with self._lock:
            state = self._runs.setdefault(run_id, _RunState())
            if target_agent in state.delegation_chain:
                raise LoopDetectedError(f"circular delegation involving '{target_agent}'")
            state.delegation_chain.append(target_agent)
            state.delegation_depth = max(state.delegation_depth, len(state.delegation_chain))
            if len(state.delegation_chain) > self.config.max_delegation_depth:
                raise LoopDetectedError(
                    f"delegation depth {len(state.delegation_chain)} exceeds max {self.config.max_delegation_depth}"
                )

    def record_delegation_end(self, run_id: str, target_agent: str) -> None:
        """Remove a completed delegation from the chain."""
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return
            if target_agent in state.delegation_chain:
                state.delegation_chain.remove(target_agent)

    def _check_thrashing(self, state: _RunState) -> None:
        window = list(state.recent_actions)[-self.config.thrash_window :]
        if len(window) < self.config.thrash_window:
            return
        unique = set(window)
        if len(unique) <= 2 and len(window) >= self.config.thrash_window:
            raise LoopDetectedError(
                f"thrashing detected: alternating between {len(unique)} states for {len(window)} steps"
            )


def _hash_inputs(inputs: dict[str, Any]) -> str:
    import json

    return json.dumps(inputs, sort_keys=True, default=str)
