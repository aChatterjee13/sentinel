"""Unit tests for AgentOps safety: loop detection, budgets, escalation, sandbox."""

from __future__ import annotations

import pytest

from sentinel.agentops.safety.budget_guard import BudgetGuard
from sentinel.agentops.safety.escalation import EscalationManager
from sentinel.agentops.safety.loop_detector import LoopDetector
from sentinel.agentops.safety.sandbox import ActionSandbox
from sentinel.config.schema import (
    BudgetConfig,
    EscalationConfig,
    EscalationTrigger,
    LoopDetectionConfig,
    SandboxConfig,
)
from sentinel.core.exceptions import (
    AgentError,
    BudgetExceededError,
    EscalationRequired,
    LoopDetectedError,
)


class TestLoopDetector:
    def test_step_within_limit_passes(self) -> None:
        d = LoopDetector(LoopDetectionConfig(max_iterations=5))
        d.begin_run("r1")
        for _ in range(5):
            d.step("r1")

    def test_step_exceeds_max_iterations(self) -> None:
        d = LoopDetector(LoopDetectionConfig(max_iterations=3))
        d.begin_run("r1")
        for _ in range(3):
            d.step("r1")
        with pytest.raises(LoopDetectedError):
            d.step("r1")

    def test_repeated_tool_calls_trigger(self) -> None:
        d = LoopDetector(LoopDetectionConfig(max_repeated_tool_calls=2, thrash_window=20))
        d.begin_run("r1")
        d.record_tool_call("r1", "search", {"q": "x"})
        d.record_tool_call("r1", "search", {"q": "x"})
        with pytest.raises(LoopDetectedError):
            d.record_tool_call("r1", "search", {"q": "x"})

    def test_circular_delegation(self) -> None:
        d = LoopDetector(LoopDetectionConfig(max_delegation_depth=10))
        d.begin_run("r1")
        d.record_delegation("r1", "agent_b")
        with pytest.raises(LoopDetectedError):
            d.record_delegation("r1", "agent_b")  # repeat = cycle

    def test_delegation_depth_exceeded(self) -> None:
        d = LoopDetector(LoopDetectionConfig(max_delegation_depth=2))
        d.begin_run("r1")
        d.record_delegation("r1", "a")
        d.record_delegation("r1", "b")
        with pytest.raises(LoopDetectedError):
            d.record_delegation("r1", "c")

    def test_thrashing_detected(self) -> None:
        d = LoopDetector(
            LoopDetectionConfig(
                max_repeated_tool_calls=100,
                thrash_window=4,
            )
        )
        d.begin_run("r1")
        # Alternate between two states
        for _ in range(4):
            d.record_tool_call("r1", "a", {"k": 1})
            try:
                d.record_tool_call("r1", "b", {"k": 2})
            except LoopDetectedError:
                return
        pytest.fail("expected thrashing detection")

    def test_end_run_clears_state(self) -> None:
        d = LoopDetector(LoopDetectionConfig(max_iterations=3))
        d.begin_run("r1")
        d.step("r1")
        d.end_run("r1")
        # Re-running should not be affected by previous state
        d.begin_run("r1")
        d.step("r1")


class TestBudgetGuard:
    def test_tokens_under_budget(self) -> None:
        g = BudgetGuard(BudgetConfig(max_tokens_per_run=1000))
        g.begin_run("r1")
        g.add_tokens("r1", 500)
        g.add_tokens("r1", 400)

    def test_tokens_exceeded(self) -> None:
        g = BudgetGuard(BudgetConfig(max_tokens_per_run=100))
        g.begin_run("r1")
        with pytest.raises(BudgetExceededError):
            g.add_tokens("r1", 200)

    def test_cost_exceeded(self) -> None:
        g = BudgetGuard(BudgetConfig(max_cost_per_run=1.00))
        g.begin_run("r1")
        with pytest.raises(BudgetExceededError):
            g.add_cost("r1", 2.00)

    def test_tool_call_budget(self) -> None:
        g = BudgetGuard(BudgetConfig(max_tool_calls_per_run=2))
        g.begin_run("r1")
        g.record_tool_call("r1")
        g.record_tool_call("r1")
        with pytest.raises(BudgetExceededError):
            g.record_tool_call("r1")

    def test_remaining_budget(self) -> None:
        g = BudgetGuard(BudgetConfig(max_tokens_per_run=1000, max_cost_per_run=10.0))
        g.begin_run("r1")
        g.add_tokens("r1", 250)
        remaining = g.remaining("r1")
        assert remaining["tokens"] == 750
        assert remaining["cost"] == 10.0

    def test_on_exceeded_escalate_tag(self) -> None:
        g = BudgetGuard(BudgetConfig(max_tokens_per_run=10, on_exceeded="escalate"))
        g.begin_run("r1")
        with pytest.raises(BudgetExceededError, match="escalate"):
            g.add_tokens("r1", 100)

    def test_on_exceeded_hard_kill_tag(self) -> None:
        g = BudgetGuard(BudgetConfig(max_tokens_per_run=10, on_exceeded="hard_kill"))
        g.begin_run("r1")
        with pytest.raises(BudgetExceededError, match="hard_kill"):
            g.add_tokens("r1", 100)


class TestEscalationManager:
    def test_no_triggers_no_escalation(self) -> None:
        m = EscalationManager(EscalationConfig(triggers=[]))
        decision = m.check("r1", confidence=0.99)
        assert not decision.triggered

    def test_low_confidence_triggers(self) -> None:
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="confidence_below",
                        threshold=0.5,
                        action="human_handoff",
                    )
                ]
            )
        )
        decision = m.check("r1", confidence=0.2)
        assert decision.triggered
        assert decision.action == "human_handoff"

    def test_consecutive_failures_trigger(self) -> None:
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="consecutive_tool_failures",
                        threshold=3,
                        action="human_handoff",
                    )
                ]
            )
        )
        for _ in range(3):
            m.check("r1", success=False)
        decision = m.check("r1", success=False)
        assert decision.triggered

    def test_success_resets_failure_counter(self) -> None:
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="consecutive_tool_failures",
                        threshold=3,
                        action="human_handoff",
                    )
                ]
            )
        )
        m.check("r1", success=False)
        m.check("r1", success=False)
        m.check("r1", success=True)  # reset
        decision = m.check("r1", success=False)
        assert not decision.triggered

    def test_sensitive_data_trigger(self) -> None:
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="sensitive_data_detected",
                        action="human_approval",
                    )
                ]
            )
        )
        decision = m.check("r1", action_text="customer SSN: 123-45-6789")
        assert decision.triggered

    def test_regulatory_context_trigger(self) -> None:
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="regulatory_context",
                        patterns=["financial advice", "medical diagnosis"],
                        action="human_approval",
                    )
                ]
            )
        )
        decision = m.check("r1", action_text="Provide financial advice on whether to invest")
        assert decision.triggered

    def test_raise_if_required(self) -> None:
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="confidence_below",
                        threshold=0.5,
                        action="human_handoff",
                    )
                ]
            )
        )
        decision = m.check("r1", confidence=0.1)
        with pytest.raises(EscalationRequired):
            m.raise_if_required(decision)


class TestActionSandbox:
    def test_non_destructive_passes(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete", "drop"]))
        decision = s.evaluate("read user profile")
        assert decision.allowed
        assert not decision.requires_approval

    def test_destructive_approve_first(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        decision = s.evaluate("delete all user records")
        assert not decision.allowed
        assert decision.requires_approval

    def test_destructive_dry_run(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["execute"], mode="dry_run"))
        decision = s.evaluate("execute payment")
        assert decision.allowed
        assert decision.dry_run

    def test_execute_dry_run_returns_none(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="dry_run"))
        called = []
        result = s.execute("delete files", lambda: called.append(1) or "done")
        assert result is None
        assert called == []  # operation not actually invoked

    def test_execute_with_approver(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        result = s.execute("delete x", lambda: "ok", approver=lambda a: True)
        assert result == "ok"

    def test_execute_denied_by_approver(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        with pytest.raises(AgentError):
            s.execute("delete x", lambda: "ok", approver=lambda a: False)

    def test_execute_no_approver_blocks(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        with pytest.raises(AgentError):
            s.execute("delete x", lambda: "ok")
