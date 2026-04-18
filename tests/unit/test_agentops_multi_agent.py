"""Unit tests for AgentOps multi-agent monitoring."""

from __future__ import annotations

from sentinel.agentops.multi_agent.consensus import ConsensusEvaluator
from sentinel.agentops.multi_agent.delegation import DelegationTracker
from sentinel.agentops.multi_agent.orchestration import MultiAgentMonitor
from sentinel.config.schema import MultiAgentConfig


class TestDelegationTracker:
    def test_record_creates_link(self) -> None:
        t = DelegationTracker()
        link = t.record("r1", "orchestrator", "specialist", "summarise")
        assert link.source == "orchestrator"
        assert link.target == "specialist"

    def test_chain_returns_history(self) -> None:
        t = DelegationTracker()
        t.record("r1", "a", "b", "task1")
        t.record("r1", "b", "c", "task2")
        chain = t.chain("r1")
        assert len(chain) == 2
        assert [link.target for link in chain] == ["b", "c"]

    def test_depth_counts_links(self) -> None:
        t = DelegationTracker()
        t.record("r1", "a", "b", "task")
        t.record("r1", "b", "c", "task")
        t.record("r1", "c", "d", "task")
        assert t.depth("r1") == 3

    def test_has_cycle_detects_back_edge(self) -> None:
        t = DelegationTracker()
        t.record("r1", "a", "b", "x")
        t.record("r1", "b", "a", "y")  # cycle
        assert t.has_cycle("r1")

    def test_has_no_cycle_for_linear_chain(self) -> None:
        t = DelegationTracker()
        t.record("r1", "a", "b", "x")
        t.record("r1", "b", "c", "y")
        assert not t.has_cycle("r1")

    def test_fan_out_lists_targets(self) -> None:
        t = DelegationTracker()
        t.record("r1", "orchestrator", "agent_a", "x")
        t.record("r1", "orchestrator", "agent_b", "y")
        t.record("r1", "agent_a", "agent_c", "z")
        targets = t.fan_out("r1", "orchestrator")
        assert set(targets) == {"agent_a", "agent_b"}

    def test_end_run_clears_state(self) -> None:
        t = DelegationTracker()
        t.record("r1", "a", "b", "x")
        cleared = t.end_run("r1")
        assert len(cleared) == 1
        assert t.depth("r1") == 0


class TestConsensusEvaluator:
    def test_full_agreement(self) -> None:
        e = ConsensusEvaluator()
        result = e.evaluate({"a1": "approve", "a2": "approve", "a3": "approve"})
        assert result.has_consensus
        assert result.agreement == 1.0
        assert result.decision == "approve"

    def test_majority_consensus(self) -> None:
        e = ConsensusEvaluator(MultiAgentConfig(consensus={"min_agreement": 0.6}))
        result = e.evaluate({"a1": "approve", "a2": "approve", "a3": "reject"})
        assert result.has_consensus
        assert abs(result.agreement - 2 / 3) < 1e-6

    def test_no_consensus_below_threshold(self) -> None:
        e = ConsensusEvaluator(MultiAgentConfig(consensus={"min_agreement": 0.8, "conflict_action": "majority_vote"}))
        result = e.evaluate({"a1": "approve", "a2": "approve", "a3": "reject"})
        assert not result.has_consensus
        assert result.conflict

    def test_empty_votes(self) -> None:
        e = ConsensusEvaluator()
        result = e.evaluate({})
        assert result.conflict
        assert result.n_voters == 0

    def test_canonicalisation(self) -> None:
        e = ConsensusEvaluator()
        # Different cases should be treated as same
        result = e.evaluate({"a1": "APPROVE", "a2": "approve", "a3": "Approve"})
        assert result.has_consensus

    def test_weighted_vote(self) -> None:
        e = ConsensusEvaluator(
            MultiAgentConfig(consensus={"conflict_action": "weighted_vote", "min_agreement": 0.6})
        )
        votes = {"a1": "approve", "a2": "reject", "a3": "reject"}
        weights = {"a1": 5.0, "a2": 1.0, "a3": 1.0}  # a1 outweighs others
        result = e.evaluate(votes, weights=weights)
        assert result.decision == "approve"

    def test_disagreement_rate(self) -> None:
        e = ConsensusEvaluator(MultiAgentConfig(consensus={"min_agreement": 0.8, "conflict_action": "majority_vote"}))
        history = [
            e.evaluate({"a": "x", "b": "x"}),  # consensus
            e.evaluate({"a": "x", "b": "y"}),  # conflict
            e.evaluate({"a": "x", "b": "y"}),  # conflict
        ]
        rate = e.disagreement_rate(history)
        assert abs(rate - 2 / 3) < 1e-6


class TestMultiAgentMonitor:
    def test_records_delegation(self) -> None:
        m = MultiAgentMonitor()
        link = m.on_delegation("r1", "a", "b", "task")
        assert link.source == "a"
        assert link.target == "b"

    def test_evaluate_consensus_records_history(self) -> None:
        m = MultiAgentMonitor(MultiAgentConfig(consensus={"conflict_action": "majority_vote"}))
        m.evaluate_consensus({"a1": "yes", "a2": "yes", "a3": "no"})
        m.evaluate_consensus({"a1": "yes", "a2": "yes"})
        rate = m.consensus_disagreement_rate()
        assert 0.0 <= rate <= 1.0

    def test_bottleneck_detection(self) -> None:
        m = MultiAgentMonitor(
            MultiAgentConfig(
                bottleneck_detection={"latency_percentile": "p95", "threshold_ms": 100.0}
            )
        )
        for _ in range(20):
            m.on_agent_complete("slow_agent", 200.0)
            m.on_agent_complete("fast_agent", 10.0)
        bottlenecks = m.bottlenecks()
        names = [b["agent"] for b in bottlenecks]
        assert "slow_agent" in names
        assert "fast_agent" not in names

    def test_stats_includes_disagreement(self) -> None:
        m = MultiAgentMonitor()
        m.on_agent_complete("a", 100.0)
        stats = m.stats()
        assert "tracked_agents" in stats
        assert "a" in stats["tracked_agents"]
