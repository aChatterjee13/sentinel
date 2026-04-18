"""Tests for AgentOps logic bug fixes.

Covers:
1. DelegationTracker DFS-based cycle detection
2. ConsensusEvaluator escalation raises EscalationRequired
3. ActionSandbox word-boundary matching
4. ActionSandbox audit trail integration
5. EscalationManager notification callback
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sentinel.agentops.multi_agent.consensus import ConsensusEvaluator
from sentinel.agentops.multi_agent.delegation import DelegationTracker
from sentinel.agentops.safety.escalation import EscalationManager
from sentinel.agentops.safety.sandbox import ActionSandbox
from sentinel.config.schema import (
    EscalationConfig,
    EscalationTrigger,
    MultiAgentConfig,
    SandboxConfig,
)
from sentinel.core.exceptions import AgentError, EscalationRequired

# ── Fix 1: DelegationTracker cycle detection ────────────────────────


class TestDelegationCycleDFS:
    def test_transitive_cycle_detected(self) -> None:
        """A→B→C→A must be detected as a cycle."""
        t = DelegationTracker()
        t.record("r1", "a", "b", "task1")
        t.record("r1", "b", "c", "task2")
        t.record("r1", "c", "a", "task3")
        assert t.has_cycle("r1")

    def test_linear_chain_no_cycle(self) -> None:
        """A→B→C is not a cycle."""
        t = DelegationTracker()
        t.record("r1", "a", "b", "task1")
        t.record("r1", "b", "c", "task2")
        assert not t.has_cycle("r1")

    def test_bidirectional_cycle_detected(self) -> None:
        """A↔B (bidirectional edge) is a cycle."""
        t = DelegationTracker()
        t.record("r1", "a", "b", "task1")
        t.record("r1", "b", "a", "task2")
        assert t.has_cycle("r1")

    def test_self_delegation_raises(self) -> None:
        """Self-delegation (A→A) must raise AgentError."""
        t = DelegationTracker()
        with pytest.raises(AgentError, match="self-delegation not allowed"):
            t.record("r1", "a", "a", "task")

    def test_empty_run_no_cycle(self) -> None:
        t = DelegationTracker()
        assert not t.has_cycle("nonexistent")

    def test_longer_transitive_cycle(self) -> None:
        """A→B→C→D→B should detect the B→C→D→B cycle."""
        t = DelegationTracker()
        t.record("r1", "a", "b", "t1")
        t.record("r1", "b", "c", "t2")
        t.record("r1", "c", "d", "t3")
        t.record("r1", "d", "b", "t4")
        assert t.has_cycle("r1")


# ── Fix 2: Consensus escalation raises EscalationRequired ──────────


class TestConsensusEscalate:
    def test_escalate_raises_on_disagreement(self) -> None:
        """conflict_action='escalate' must raise on conflict."""
        e = ConsensusEvaluator(
            MultiAgentConfig(
                consensus={"min_agreement": 0.8, "conflict_action": "escalate"}
            )
        )
        with pytest.raises(EscalationRequired, match="consensus conflict"):
            e.evaluate({"a1": "approve", "a2": "approve", "a3": "reject"})

    def test_escalate_no_raise_on_consensus(self) -> None:
        """No exception when agreement is met, even with escalate action."""
        e = ConsensusEvaluator(
            MultiAgentConfig(
                consensus={"min_agreement": 0.5, "conflict_action": "escalate"}
            )
        )
        result = e.evaluate({"a1": "approve", "a2": "approve", "a3": "reject"})
        assert result.has_consensus

    def test_weighted_escalate_raises(self) -> None:
        """Weighted vote with escalate also raises on conflict."""
        e = ConsensusEvaluator(
            MultiAgentConfig(
                consensus={"min_agreement": 0.9, "conflict_action": "weighted_vote"}
            )
        )
        # Switch to escalate for this test
        e.conflict_action = "escalate"
        votes = {"a1": "yes", "a2": "no", "a3": "maybe"}
        with pytest.raises(EscalationRequired):
            e.evaluate(votes, weights={"a1": 1.0, "a2": 1.0, "a3": 1.0})

    def test_majority_vote_no_raise(self) -> None:
        """conflict_action='majority_vote' returns result, no exception."""
        e = ConsensusEvaluator(
            MultiAgentConfig(
                consensus={"min_agreement": 0.8, "conflict_action": "majority_vote"}
            )
        )
        result = e.evaluate({"a1": "approve", "a2": "approve", "a3": "reject"})
        assert not result.has_consensus
        assert result.conflict


# ── Fix 3: Sandbox word-boundary matching ───────────────────────────


class TestSandboxWordBoundary:
    def test_delete_matches_delete_file(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        decision = s.evaluate("delete file")
        assert decision.requires_approval

    def test_delete_matches_file_dot_delete(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        decision = s.evaluate("file.delete()")
        assert decision.requires_approval

    def test_delete_does_not_match_undelete(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        decision = s.evaluate("undelete record")
        assert decision.allowed

    def test_delete_does_not_match_delete_backup_as_compound(self) -> None:
        """'delete' should not match 'predelete' (substring at end)."""
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        decision = s.evaluate("predelete check")
        assert decision.allowed

    def test_execute_matches_execute_payment(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["execute"], mode="dry_run"))
        decision = s.evaluate("execute payment")
        assert decision.dry_run

    def test_write_does_not_match_rewrite(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["write"], mode="approve_first"))
        decision = s.evaluate("rewrite the summary")
        assert decision.allowed

    def test_tool_parameter_word_boundary(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        decision = s.evaluate("process record", tool="undelete_tool")
        assert decision.allowed

    def test_tool_parameter_matches_exact(self) -> None:
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="approve_first"))
        # "delete" as a standalone word in the tool name triggers the sandbox
        decision = s.evaluate("process record", tool="delete records")
        assert decision.requires_approval


# ── Fix 4: Sandbox audit trail integration ──────────────────────────


class TestSandboxAudit:
    def test_execute_logs_evaluate_event(self) -> None:
        audit = MagicMock()
        s = ActionSandbox(
            SandboxConfig(destructive_ops=["delete"], mode="dry_run"),
            audit_trail=audit,
        )
        s.execute("delete file", lambda: "done")
        audit.log.assert_any_call(
            event_type="sandbox.evaluated",
            action="delete file",
            tool=None,
            allowed=True,
            requires_approval=False,
            dry_run=True,
            mode="dry_run",
        )

    def test_execute_logs_approval_decision(self) -> None:
        audit = MagicMock()
        s = ActionSandbox(
            SandboxConfig(destructive_ops=["delete"], mode="approve_first"),
            audit_trail=audit,
        )
        s.execute("delete file", lambda: "ok", approver=lambda a: True)
        audit.log.assert_any_call(
            event_type="sandbox.approval_decision",
            action="delete file",
            approved=True,
        )

    def test_execute_logs_denial(self) -> None:
        audit = MagicMock()
        s = ActionSandbox(
            SandboxConfig(destructive_ops=["delete"], mode="approve_first"),
            audit_trail=audit,
        )
        with pytest.raises(AgentError):
            s.execute("delete file", lambda: "ok", approver=lambda a: False)
        audit.log.assert_any_call(
            event_type="sandbox.approval_decision",
            action="delete file",
            approved=False,
        )

    def test_no_audit_trail_does_not_fail(self) -> None:
        """No audit trail => execute still works (no crash)."""
        s = ActionSandbox(SandboxConfig(destructive_ops=["delete"], mode="dry_run"))
        result = s.execute("delete file", lambda: "done")
        assert result is None


# ── Fix 5: EscalationManager notification callback ─────────────────


class TestEscalationNotification:
    def test_callback_called_on_trigger(self) -> None:
        cb = MagicMock()
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="confidence_below",
                        threshold=0.5,
                        action="human_handoff",
                    )
                ]
            ),
            notification_callback=cb,
        )
        m.check("r1", confidence=0.2)
        cb.assert_called_once_with(
            run_id="r1",
            trigger="confidence_below",
            action="human_handoff",
            reason="confidence 0.20 < 0.5",
        )

    def test_callback_not_called_when_no_trigger(self) -> None:
        cb = MagicMock()
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="confidence_below",
                        threshold=0.5,
                        action="human_handoff",
                    )
                ]
            ),
            notification_callback=cb,
        )
        m.check("r1", confidence=0.9)
        cb.assert_not_called()

    def test_callback_error_does_not_prevent_escalation(self) -> None:
        cb = MagicMock(side_effect=RuntimeError("notification failed"))
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="confidence_below",
                        threshold=0.5,
                        action="human_handoff",
                    )
                ]
            ),
            notification_callback=cb,
        )
        decision = m.check("r1", confidence=0.2)
        assert decision.triggered
        cb.assert_called_once()

    def test_no_callback_configured_works(self) -> None:
        m = EscalationManager(
            EscalationConfig(
                triggers=[
                    EscalationTrigger(
                        condition="confidence_below",
                        threshold=0.5,
                        action="human_handoff",
                    )
                ]
            ),
        )
        decision = m.check("r1", confidence=0.2)
        assert decision.triggered
