"""Unit tests for sentinel.agentops.tool_audit.monitor — ToolAuditMonitor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sentinel.agentops.tool_audit.monitor import ToolAuditMonitor, ToolCallRecord
from sentinel.config.schema import ToolAuditConfig
from sentinel.core.exceptions import AgentError, ToolPermissionError


@pytest.fixture
def config() -> ToolAuditConfig:
    return ToolAuditConfig(
        permissions={
            "claims_agent": {
                "allowed": ["policy_search", "llm_extraction"],
                "blocked": ["payment_execute"],
            }
        },
        parameter_validation=True,
        rate_limits={"default": "100/min", "payment_execute": "5/min"},
    )


@pytest.fixture
def monitor(config: ToolAuditConfig) -> ToolAuditMonitor:
    return ToolAuditMonitor(config=config)


class TestAuthorise:
    """Pre-call permission and rate-limit enforcement."""

    def test_allowed_tool_passes(self, monitor: ToolAuditMonitor) -> None:
        monitor.authorise("claims_agent", "policy_search")

    def test_blocked_tool_raises(self, monitor: ToolAuditMonitor) -> None:
        with pytest.raises(ToolPermissionError, match="payment_execute"):
            monitor.authorise("claims_agent", "payment_execute")

    def test_unknown_agent_permissive(self, monitor: ToolAuditMonitor) -> None:
        monitor.authorise("unknown_agent", "any_tool")

    def test_unlisted_tool_denied_when_allowlist_exists(self, monitor: ToolAuditMonitor) -> None:
        with pytest.raises(ToolPermissionError, match="database_write"):
            monitor.authorise("claims_agent", "database_write")

    def test_rate_limit_exceeded(self, config: ToolAuditConfig) -> None:
        cfg = ToolAuditConfig(
            permissions={},
            rate_limits={"default": "3/min"},
        )
        mon = ToolAuditMonitor(config=cfg)
        mon.authorise("a", "tool_x")
        mon.authorise("a", "tool_x")
        mon.authorise("a", "tool_x")
        with pytest.raises(AgentError, match="rate limit exceeded"):
            mon.authorise("a", "tool_x")

    def test_no_rate_limit_when_not_configured(self) -> None:
        mon = ToolAuditMonitor(config=ToolAuditConfig(rate_limits={}))
        for _ in range(200):
            mon.authorise("agent", "tool")


class TestRecord:
    """Recording tool call results."""

    def test_record_success(self, monitor: ToolAuditMonitor) -> None:
        rec = monitor.record(
            agent="claims_agent",
            tool="policy_search",
            inputs={"query": "policy 123"},
            output={"results": [1, 2]},
            success=True,
            latency_ms=42.0,
        )
        assert isinstance(rec, ToolCallRecord)
        assert rec.agent == "claims_agent"
        assert rec.success is True

    def test_record_failure(self, monitor: ToolAuditMonitor) -> None:
        rec = monitor.record(
            agent="claims_agent",
            tool="policy_search",
            inputs={},
            success=False,
            error="timeout",
            latency_ms=5000.0,
        )
        assert rec.success is False
        assert rec.error == "timeout"

    def test_record_with_audit_trail(self) -> None:
        audit = MagicMock()
        mon = ToolAuditMonitor(audit_trail=audit)
        mon.record(agent="a", tool="t", inputs={}, success=True, latency_ms=10.0)
        audit.log.assert_called_once()
        call_kwargs = audit.log.call_args[1]
        assert call_kwargs["event_type"] == "agent.tool_call"
        assert call_kwargs["success"] is True

    def test_record_with_replay_store(self) -> None:
        replay = MagicMock()
        mon = ToolAuditMonitor(replay_store=replay)
        mon.record(agent="a", tool="t", inputs={"k": "v"}, output="result")
        replay.save.assert_called_once()

    def test_record_replay_failure_swallowed(self) -> None:
        replay = MagicMock()
        replay.save.side_effect = OSError("disk full")
        mon = ToolAuditMonitor(replay_store=replay)
        rec = mon.record(agent="a", tool="t", inputs={})
        assert rec is not None


class TestStats:
    """Telemetry statistics for tool calls."""

    def test_stats_empty_tool(self, monitor: ToolAuditMonitor) -> None:
        assert monitor.stats("nonexistent") == {}

    def test_stats_single_call(self, monitor: ToolAuditMonitor) -> None:
        monitor.record(agent="a", tool="search", inputs={}, latency_ms=50.0)
        s = monitor.stats("search")
        assert s["calls"] == 1.0
        assert s["success_rate"] == 1.0
        assert s["p50_latency_ms"] == 50.0

    def test_stats_mixed_success_failure(self, monitor: ToolAuditMonitor) -> None:
        for i in range(10):
            monitor.record(
                agent="a",
                tool="api",
                inputs={},
                success=(i < 7),
                latency_ms=float(i * 10),
            )
        s = monitor.stats("api")
        assert s["calls"] == 10.0
        assert s["success_rate"] == pytest.approx(0.7)

    def test_all_stats(self, monitor: ToolAuditMonitor) -> None:
        monitor.record(agent="a", tool="t1", inputs={})
        monitor.record(agent="a", tool="t2", inputs={})
        result = monitor.all_stats()
        assert "t1" in result
        assert "t2" in result


class TestInputValidation:
    """Parameter validation catches suspicious inputs."""

    def test_suspicious_input_does_not_raise(self, monitor: ToolAuditMonitor) -> None:
        # Validation logs a warning but doesn't block
        monitor.authorise("claims_agent", "policy_search", inputs={"q": "hello; rm -rf /"})

    def test_clean_input_passes(self, monitor: ToolAuditMonitor) -> None:
        monitor.authorise("claims_agent", "policy_search", inputs={"q": "normal query"})

    def test_validation_skipped_when_disabled(self) -> None:
        cfg = ToolAuditConfig(parameter_validation=False)
        mon = ToolAuditMonitor(config=cfg)
        mon.authorise("agent", "tool", inputs={"q": "$(malicious)"})
