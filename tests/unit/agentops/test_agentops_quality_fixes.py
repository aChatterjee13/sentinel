"""Tests for AgentOps quality/functional fixes.

Covers: time budget, golden dataset error classification, trajectory
configurable threshold, task-completion track_by, and A2A capability index.
"""

from __future__ import annotations

import time

import pytest

from sentinel.agentops.agent_registry import AgentRegistry, AgentSpec
from sentinel.agentops.eval.golden_datasets import (
    GoldenDataset,
    GoldenExample,
    GoldenSuiteRunner,
)
from sentinel.agentops.eval.task_completion import TaskCompletionTracker
from sentinel.agentops.eval.trajectory import TrajectoryEvaluator, TrajectoryScore
from sentinel.agentops.safety.budget_guard import BudgetGuard, _seconds
from sentinel.config.schema import (
    AgentEvaluationConfig,
    AgentRegistryConfig,
    BudgetConfig,
)
from sentinel.core.exceptions import BudgetExceededError

# ── Time budget tests ────────────────────────────────────────────────


class TestTimeBudget:
    """BudgetGuard time-limit enforcement (previously untested)."""

    def test_time_budget_exceeded(self) -> None:
        guard = BudgetGuard(BudgetConfig(max_time_per_run="100ms"))
        guard.begin_run("r1")
        time.sleep(0.15)
        with pytest.raises(BudgetExceededError, match="time"):
            guard.check_time("r1")

    def test_time_budget_not_exceeded(self) -> None:
        guard = BudgetGuard(BudgetConfig(max_time_per_run="10s"))
        guard.begin_run("r1")
        guard.check_time("r1")  # should not raise

    def test_time_remaining(self) -> None:
        guard = BudgetGuard(BudgetConfig(max_time_per_run="5s"))
        guard.begin_run("r1")
        remaining = guard.remaining("r1")
        assert 4.0 < remaining["time_seconds"] <= 5.0

    def test_duration_parsing(self) -> None:
        assert _seconds("300s") == pytest.approx(300.0)
        assert _seconds("5m") == pytest.approx(300.0)
        assert _seconds("1h") == pytest.approx(3600.0)
        assert _seconds("500ms") == pytest.approx(0.5)


# ── Golden dataset error classification ──────────────────────────────


class TestGoldenErrorClassification:
    """GoldenSuiteRunner classifies timeout vs budget vs generic errors."""

    @staticmethod
    def _dataset(n: int = 1) -> GoldenDataset:
        return GoldenDataset(
            name="test",
            version="1.0",
            examples=[
                GoldenExample(
                    example_id=f"ex{i}",
                    input={"x": i},
                    expected_output="ok",
                )
                for i in range(n)
            ],
        )

    def test_golden_timeout(self) -> None:
        def slow_runner(_input: dict) -> dict:
            time.sleep(2)
            return {"output": "ok"}

        runner = GoldenSuiteRunner()
        result = runner.run(self._dataset(), slow_runner, timeout_seconds=0.1)
        assert result.failed == 1
        rec = result.examples[0]
        assert rec["failure_reason"] == "timeout"

    def test_golden_budget_exceeded(self) -> None:
        def budget_runner(_input: dict) -> dict:
            raise BudgetExceededError("token budget exceeded")

        runner = GoldenSuiteRunner()
        result = runner.run(self._dataset(), budget_runner)
        assert result.failed == 1
        rec = result.examples[0]
        assert rec["failure_reason"].startswith("budget_exceeded")

    def test_golden_error_classification(self) -> None:
        def bad_runner(_input: dict) -> dict:
            raise ValueError("bad value")

        runner = GoldenSuiteRunner()
        result = runner.run(self._dataset(), bad_runner)
        rec = result.examples[0]
        assert rec["failure_reason"] == "error: ValueError: bad value"

    def test_golden_success_has_no_failure_reason(self) -> None:
        runner = GoldenSuiteRunner()
        result = runner.run(self._dataset(), lambda _: {"output": "ok"})
        assert result.passed == 1
        assert result.examples[0]["failure_reason"] is None


# ── Trajectory configurable threshold ────────────────────────────────


class TestTrajectoryThreshold:
    """TrajectoryScore.passed honours the configured pass_threshold."""

    def test_trajectory_custom_threshold(self) -> None:
        cfg = AgentEvaluationConfig(trajectory={"pass_threshold": 0.5})
        evaluator = TrajectoryEvaluator(cfg)
        score = evaluator.score(
            actual_steps=["a", "b", "c", "d"],
            optimal_steps=["a", "b", "c"],
        )
        # Score should be above 0.5 but below 0.7
        assert score.passed is True
        assert score.score >= 0.5

    def test_trajectory_default_threshold(self) -> None:
        evaluator = TrajectoryEvaluator()
        assert evaluator.pass_threshold == pytest.approx(0.7)
        # Perfect match should pass
        score = evaluator.score(["a", "b"], ["a", "b"])
        assert score.passed is True
        assert score.score == pytest.approx(1.0)

    def test_trajectory_score_below_threshold_fails(self) -> None:
        cfg = AgentEvaluationConfig(trajectory={"pass_threshold": 0.9})
        evaluator = TrajectoryEvaluator(cfg)
        # Many extra steps → large penalty → low score
        score = evaluator.score(
            actual_steps=["a", "x", "y", "z", "w", "v"],
            optimal_steps=["a", "b"],
        )
        assert score.passed is False

    def test_trajectory_score_is_field_not_property(self) -> None:
        ts = TrajectoryScore(
            score=0.5,
            optimal_steps=3,
            actual_steps=3,
            extra_steps=0,
            passed=True,
        )
        assert ts.passed is True


# ── TaskCompletion track_by ──────────────────────────────────────────


class TestTaskCompletionTrackBy:
    """TaskCompletionTracker uses track_by config for keying."""

    def test_default_track_by(self) -> None:
        tracker = TaskCompletionTracker()
        tracker.record("agent_a", "classify", True)
        tracker.record("agent_a", "classify", False)
        assert tracker.success_rate(agent="agent_a", task_type="classify") == pytest.approx(0.5)

    def test_track_by_custom_dimensions(self) -> None:
        cfg = AgentEvaluationConfig(
            task_completion={
                "track_by": ["agent", "task_type", "region"],
                "min_success_rate": 0.0,
            }
        )
        tracker = TaskCompletionTracker(config=cfg)
        tracker.record("a", "classify", True, region="us")
        tracker.record("a", "classify", False, region="eu")

        # The keys are ("a", "classify", "us") and ("a", "classify", "eu")
        keys = list(tracker._results.keys())
        assert len(keys) == 2
        assert ("a", "classify", "us") in keys
        assert ("a", "classify", "eu") in keys

    def test_track_by_single_dimension(self) -> None:
        cfg = AgentEvaluationConfig(
            task_completion={"track_by": ["agent"], "min_success_rate": 0.0}
        )
        tracker = TaskCompletionTracker(config=cfg)
        tracker.record("a", "classify", True)
        tracker.record("a", "summarise", False)
        # Both keyed by ("a",) only
        keys = list(tracker._results.keys())
        assert keys == [("a",)]


# ── A2A capability index ─────────────────────────────────────────────


class TestCapabilityIndex:
    """AgentRegistry.find_by_capability uses a reverse index."""

    @pytest.fixture
    def registry(self, tmp_path) -> AgentRegistry:
        return AgentRegistry(
            config=AgentRegistryConfig(capability_manifest=True),
            root=tmp_path / "agents",
        )

    def test_find_by_capability_indexed(self, registry: AgentRegistry) -> None:
        registry.register(
            AgentSpec(name="a", version="1.0", capabilities=["search", "extract"])
        )
        registry.register(
            AgentSpec(name="b", version="1.0", capabilities=["search"])
        )
        registry.register(
            AgentSpec(name="c", version="1.0", capabilities=["payment"])
        )
        found = registry.find_by_capability("search")
        names = {s.name for s in found}
        assert names == {"a", "b"}

    def test_capability_index_after_register(self, registry: AgentRegistry) -> None:
        registry.register(
            AgentSpec(name="d", version="1.0", capabilities=["claims"])
        )
        assert "d" in registry._capability_index["claims"]
        # Register another with the same capability
        registry.register(
            AgentSpec(name="e", version="1.0", capabilities=["claims", "underwriting"])
        )
        assert "e" in registry._capability_index["claims"]
        assert "e" in registry._capability_index["underwriting"]

    def test_capability_index_rebuilt_on_load(self, tmp_path) -> None:
        root = tmp_path / "agents"
        reg1 = AgentRegistry(
            config=AgentRegistryConfig(capability_manifest=True), root=root
        )
        reg1.register(
            AgentSpec(name="f", version="1.0", capabilities=["ocr"])
        )
        # New instance reloads from disk and rebuilds index
        reg2 = AgentRegistry(
            config=AgentRegistryConfig(capability_manifest=True), root=root
        )
        assert "f" in reg2._capability_index["ocr"]
        found = reg2.find_by_capability("ocr")
        assert len(found) == 1 and found[0].name == "f"

    def test_find_by_capability_no_match(self, registry: AgentRegistry) -> None:
        registry.register(
            AgentSpec(name="g", version="1.0", capabilities=["x"])
        )
        assert registry.find_by_capability("nonexistent") == []
