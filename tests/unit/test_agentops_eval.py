"""Unit tests for AgentOps evaluation: task completion, trajectory, golden suite."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.agentops.eval.golden_datasets import (
    GoldenDataset,
    GoldenExample,
    GoldenSuiteRunner,
)
from sentinel.agentops.eval.task_completion import TaskCompletionTracker
from sentinel.agentops.eval.trajectory import TrajectoryEvaluator
from sentinel.config.schema import AgentEvaluationConfig


class TestTaskCompletionTracker:
    def test_record_and_success_rate(self) -> None:
        t = TaskCompletionTracker()
        t.record("agent_a", "summarise", success=True)
        t.record("agent_a", "summarise", success=True)
        t.record("agent_a", "summarise", success=False)
        rate = t.success_rate(agent="agent_a", task_type="summarise")
        assert rate is not None
        assert abs(rate - 2 / 3) < 1e-6

    def test_global_success_rate(self) -> None:
        t = TaskCompletionTracker()
        t.record("a", "x", success=True)
        t.record("b", "y", success=False)
        assert t.success_rate() == 0.5

    def test_average_score(self) -> None:
        t = TaskCompletionTracker()
        t.record("a", "x", success=True, score=0.8)
        t.record("a", "x", success=True, score=0.6)
        avg = t.average_score(agent="a", task_type="x")
        assert avg is not None
        assert abs(avg - 0.7) < 1e-6

    def test_average_duration_ms(self) -> None:
        t = TaskCompletionTracker()
        t.record("a", "x", success=True, duration_ms=100.0)
        t.record("a", "x", success=True, duration_ms=200.0)
        assert t.average_duration_ms(agent="a") == 150.0

    def test_stats_returns_per_agent_breakdown(self) -> None:
        t = TaskCompletionTracker()
        t.record("a", "x", success=True)
        t.record("b", "y", success=False)
        stats = t.stats()
        assert "a::x" in stats
        assert "b::y" in stats
        assert stats["a::x"]["success_rate"] == 1.0
        assert stats["b::y"]["success_rate"] == 0.0

    def test_min_success_rate_from_config(self) -> None:
        cfg = AgentEvaluationConfig(task_completion={"min_success_rate": 0.9})
        t = TaskCompletionTracker(cfg)
        assert t.min_success_rate == 0.9

    def test_no_data_returns_none(self) -> None:
        t = TaskCompletionTracker()
        assert t.success_rate(agent="ghost") is None


class TestTrajectoryEvaluator:
    def test_perfect_match_scores_one(self) -> None:
        e = TrajectoryEvaluator()
        score = e.score(["plan", "search", "synthesise"], ["plan", "search", "synthesise"])
        assert score.score == 1.0
        assert score.passed
        assert score.extra_steps == 0

    def test_extra_steps_penalty(self) -> None:
        e = TrajectoryEvaluator(AgentEvaluationConfig(trajectory={"penalty_per_extra_step": 0.1}))
        score = e.score(
            ["plan", "extra", "search", "extra2", "synthesise"],
            ["plan", "search", "synthesise"],
        )
        assert score.extra_steps == 2
        assert score.score < 1.0

    def test_missing_steps_reported(self) -> None:
        e = TrajectoryEvaluator()
        score = e.score(["plan"], ["plan", "search", "synthesise"])
        assert "search" in score.missing_steps
        assert "synthesise" in score.missing_steps
        assert score.score < 0.7

    def test_completely_wrong_trajectory_scores_zero(self) -> None:
        e = TrajectoryEvaluator()
        score = e.score(["foo", "bar"], ["plan", "search", "synthesise"])
        assert score.score == 0.0

    def test_empty_optimal_with_actual(self) -> None:
        e = TrajectoryEvaluator()
        score = e.score(["plan"], [])
        assert score.score == 0.0

    def test_empty_both_returns_one(self) -> None:
        e = TrajectoryEvaluator()
        score = e.score([], [])
        assert score.score == 1.0

    def test_out_of_order_detection(self) -> None:
        e = TrajectoryEvaluator()
        score = e.score(["search", "plan", "synthesise"], ["plan", "search", "synthesise"])
        # All steps present but ordering wrong
        assert score.out_of_order


class TestGoldenDataset:
    def test_from_dict(self) -> None:
        ex = GoldenExample.from_dict(
            {
                "id": "ex1",
                "input": {"q": "hi"},
                "expected_output": "hello",
                "expected_steps": ["plan", "respond"],
            }
        )
        assert ex.example_id == "ex1"
        assert ex.expected_steps == ["plan", "respond"]

    def test_from_file(self, tmp_path: Path) -> None:
        data = {
            "name": "smoke",
            "version": "1.0",
            "examples": [
                {"id": "e1", "input": {"q": "x"}, "expected_output": "y"},
                {"id": "e2", "input": {"q": "a"}, "expected_output": "b"},
            ],
        }
        path = tmp_path / "smoke.json"
        path.write_text(json.dumps(data))
        ds = GoldenDataset.from_file(path)
        assert len(ds) == 2
        assert ds.name == "smoke"


class TestGoldenSuiteRunner:
    def _dataset(self) -> GoldenDataset:
        return GoldenDataset(
            name="t",
            version="1.0",
            examples=[
                GoldenExample(
                    example_id="e1",
                    input={"q": "hi"},
                    expected_output="hello",
                    expected_steps=["plan", "respond"],
                ),
                GoldenExample(
                    example_id="e2",
                    input={"q": "bye"},
                    expected_output="goodbye",
                    expected_steps=["plan", "respond"],
                ),
            ],
        )

    def test_runs_passing_examples(self) -> None:
        runner = GoldenSuiteRunner()
        ds = self._dataset()

        def fake(inp: dict) -> dict:
            mapping = {"hi": "hello", "bye": "goodbye"}
            return {"output": mapping[inp["q"]], "steps": ["plan", "respond"]}

        result = runner.run(ds, fake)
        assert result.total == 2
        assert result.passed == 2
        assert result.pass_rate == 1.0

    def test_records_failure_when_output_mismatches(self) -> None:
        runner = GoldenSuiteRunner()
        ds = self._dataset()

        def fake(inp: dict) -> dict:
            return {"output": "wrong", "steps": ["plan", "respond"]}

        result = runner.run(ds, fake)
        assert result.failed == 2
        assert result.pass_rate == 0.0

    def test_handles_runner_exceptions(self) -> None:
        runner = GoldenSuiteRunner()
        ds = self._dataset()

        def boom(inp: dict) -> dict:
            raise RuntimeError("agent crashed")

        result = runner.run(ds, boom)
        assert result.failed == 2
        assert all("error" in r for r in result.examples)
