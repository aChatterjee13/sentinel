"""Tool call monitoring — success/failure/latency, rate limiting, audit."""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from sentinel.agentops.tool_audit.permissions import PermissionMatrix
from sentinel.agentops.tool_audit.replay import ToolReplayStore
from sentinel.config.schema import AgentOpsConfig, ToolAuditConfig
from sentinel.core.exceptions import AgentError

log = structlog.get_logger(__name__)


_RATE_PATTERN = re.compile(r"^(\d+)/(sec|second|min|minute|hour|h)$")


@dataclass
class ToolCallRecord:
    """A single tool invocation."""

    agent: str
    tool: str
    inputs: dict[str, Any]
    output: Any = None
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ToolAuditMonitor:
    """Track every tool call: latency, errors, rate limits, replay storage."""

    def __init__(
        self,
        config: ToolAuditConfig | None = None,
        permissions: PermissionMatrix | None = None,
        replay_store: ToolReplayStore | None = None,
        audit_trail: Any = None,
    ):
        self.config = config or ToolAuditConfig()
        self.permissions = permissions or PermissionMatrix(self.config)
        self.replay = replay_store
        self.audit = audit_trail
        self._calls: dict[str, deque[ToolCallRecord]] = defaultdict(lambda: deque(maxlen=2000))
        self._timestamps: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls, config: AgentOpsConfig | str | Any, audit_trail: Any = None
    ) -> ToolAuditMonitor:
        if isinstance(config, str):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            return cls(cfg.agentops.tool_audit, audit_trail=audit_trail)
        if isinstance(config, AgentOpsConfig):
            return cls(config.tool_audit, audit_trail=audit_trail)
        return cls(config.agentops.tool_audit, audit_trail=audit_trail)  # type: ignore[union-attr]

    # ── Pre-call ──────────────────────────────────────────────────

    def authorise(self, agent: str, tool: str, inputs: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.permissions.enforce(agent, tool)
            self._enforce_rate_limit(agent, tool)
            if self.config.parameter_validation and inputs is not None:
                self._validate_inputs(tool, inputs)

    def _enforce_rate_limit(self, agent: str, tool: str) -> None:
        rate = self.config.rate_limits.get(tool) or self.config.rate_limits.get("default")
        if not rate:
            return
        match = _RATE_PATTERN.match(rate)
        if not match:
            return
        n = int(match.group(1))
        unit = match.group(2)
        seconds = {"sec": 1, "second": 1, "min": 60, "minute": 60, "hour": 3600, "h": 3600}[unit]
        now = time.time()
        bucket = self._timestamps[(agent, tool)]
        cutoff = now - seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= n:
            raise AgentError(f"rate limit exceeded for {agent}/{tool}: {rate}")
        bucket.append(now)

    def _validate_inputs(self, tool: str, inputs: dict[str, Any]) -> None:
        # Basic validation hook — production deployments should plug in
        # JSON Schema or pydantic models per tool. We at least reject
        # obvious shell-injection patterns in string parameters.
        for key, value in inputs.items():
            if isinstance(value, str) and any(
                token in value for token in (";", "&&", "|", "`", "$(")
            ):
                log.warning("tool_audit.suspicious_input", tool=tool, key=key)

    # ── Recording ─────────────────────────────────────────────────

    def record(
        self,
        agent: str,
        tool: str,
        inputs: dict[str, Any],
        output: Any = None,
        success: bool = True,
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> ToolCallRecord:
        record = ToolCallRecord(
            agent=agent,
            tool=tool,
            inputs=inputs,
            output=output,
            success=success,
            error=error,
            latency_ms=latency_ms,
        )
        with self._lock:
            self._calls[tool].append(record)
        if self.replay is not None:
            try:
                self.replay.save(record)
            except Exception:
                log.warning("tool_audit.replay_save_failed", tool=tool)
        if self.audit is not None:
            self.audit.log(
                event_type="agent.tool_call",
                actor=agent,
                tool=tool,
                success=success,
                error=error,
                latency_ms=latency_ms,
            )
        return record

    # ── Telemetry ─────────────────────────────────────────────────

    def stats(self, tool: str) -> dict[str, float]:
        with self._lock:
            records = list(self._calls.get(tool, []))
        if not records:
            return {}
        successes = sum(1 for r in records if r.success)
        latencies = sorted(r.latency_ms for r in records)
        return {
            "calls": float(len(records)),
            "success_rate": successes / len(records),
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
        }

    def all_stats(self) -> dict[str, dict[str, float]]:
        with self._lock:
            tools = list(self._calls.keys())
        return {tool: self.stats(tool) for tool in tools}
