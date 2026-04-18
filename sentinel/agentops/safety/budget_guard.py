"""Token, cost, time, and action budget enforcement for agent runs."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from sentinel.config.schema import BudgetConfig
from sentinel.core.exceptions import BudgetExceededError

log = structlog.get_logger(__name__)

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_DURATION_FACTORS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _seconds(spec: str) -> float:
    match = _DURATION.fullmatch(spec.strip())
    if not match:
        return 0.0
    return float(match.group(1)) * _DURATION_FACTORS[match.group(2)]


@dataclass
class _RunBudget:
    started_at: float = field(default_factory=time.time)
    tokens_used: int = 0
    cost_used: float = 0.0
    tool_calls: int = 0


class BudgetGuard:
    """Per-run resource budget enforcement.

    The runtime calls :meth:`add_tokens`, :meth:`add_cost`, and
    :meth:`record_tool_call` after every measurable action. The guard
    checks the configured limits and either:

    - ``graceful_stop``: signals via :class:`BudgetExceededError`
      (callers should catch and finalise partial output)
    - ``escalate``: same exception, but tagged for escalation
    - ``hard_kill``: same exception with no recovery hint
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        escalation_callback: Any | None = None,
    ):
        self.config = config or BudgetConfig()
        self._max_time = _seconds(self.config.max_time_per_run)
        self._runs: dict[str, _RunBudget] = {}
        self._escalation_callback = escalation_callback
        self._lock = threading.Lock()

    def begin_run(self, run_id: str) -> None:
        with self._lock:
            self._runs[run_id] = _RunBudget()

    def end_run(self, run_id: str) -> _RunBudget | None:
        with self._lock:
            return self._runs.pop(run_id, None)

    def add_tokens(self, run_id: str, tokens: int) -> None:
        with self._lock:
            self._check_time_unlocked(run_id)
            state = self._runs.setdefault(run_id, _RunBudget())
            state.tokens_used += tokens
            if state.tokens_used > self.config.max_tokens_per_run:
                self._raise(run_id, "token", state.tokens_used, self.config.max_tokens_per_run)

    def add_cost(self, run_id: str, cost: float) -> None:
        with self._lock:
            self._check_time_unlocked(run_id)
            state = self._runs.setdefault(run_id, _RunBudget())
            state.cost_used += cost
            if state.cost_used > self.config.max_cost_per_run:
                self._raise(run_id, "cost", state.cost_used, self.config.max_cost_per_run)

    def record_tool_call(self, run_id: str) -> None:
        with self._lock:
            self._check_time_unlocked(run_id)
            state = self._runs.setdefault(run_id, _RunBudget())
            state.tool_calls += 1
            if state.tool_calls > self.config.max_tool_calls_per_run:
                self._raise(run_id, "tool_calls", state.tool_calls, self.config.max_tool_calls_per_run)

    def check_time(self, run_id: str) -> None:
        with self._lock:
            self._check_time_unlocked(run_id)

    def _check_time_unlocked(self, run_id: str) -> None:
        """Check time budget without acquiring lock (caller must hold it)."""
        state = self._runs.get(run_id)
        if state is None or self._max_time <= 0:
            return
        elapsed = time.time() - state.started_at
        if elapsed > self._max_time:
            self._raise(run_id, "time", elapsed, self._max_time)

    def remaining(self, run_id: str) -> dict[str, float]:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return {}
            return {
                "tokens": max(0, self.config.max_tokens_per_run - state.tokens_used),
                "cost": max(0.0, self.config.max_cost_per_run - state.cost_used),
                "tool_calls": max(0, self.config.max_tool_calls_per_run - state.tool_calls),
                "time_seconds": max(0.0, self._max_time - (time.time() - state.started_at)),
            }

    def _raise(self, run_id: str, kind: str, used: float, limit: float) -> None:
        log.warning("budget.exceeded", run_id=run_id, kind=kind, used=used, limit=limit)
        message = f"{kind} budget exceeded: {used} > {limit}"
        if self.config.on_exceeded == "escalate":
            if self._escalation_callback is not None:
                self._escalation_callback(run_id=run_id, kind=kind, used=used, limit=limit)
            raise BudgetExceededError(f"{message} (escalated)")
        if self.config.on_exceeded == "hard_kill":
            raise BudgetExceededError(f"{message} (hard_kill)")
        raise BudgetExceededError(f"{message} (graceful_stop)")

    @property
    def on_exceeded(self) -> Literal["graceful_stop", "escalate", "hard_kill"]:
        return self.config.on_exceeded
