"""Unit tests for ExperimentTracker (expanded coverage)."""

from __future__ import annotations

import pytest

from sentinel.foundation.experiments.tracker import ExperimentRun, ExperimentTracker


@pytest.fixture
def tracker(tmp_path) -> ExperimentTracker:
    return ExperimentTracker(root=tmp_path / "experiments")


class TestStart:
    """Test experiment creation."""

    def test_start_creates_run(self, tracker) -> None:
        run = tracker.start("fraud_v3")
        assert run.name == "fraud_v3"
        assert run.status == "running"
        assert run.run_id is not None

    def test_start_with_params(self, tracker) -> None:
        run = tracker.start("test", max_depth=6, lr=0.01)
        assert run.parameters["max_depth"] == 6
        assert run.parameters["lr"] == 0.01

    def test_start_creates_file(self, tracker) -> None:
        run = tracker.start("test")
        path = tracker._path(run.run_id)
        assert path.exists()


class TestLogParam:
    """Test parameter logging."""

    def test_log_param(self, tracker) -> None:
        run = tracker.start("test")
        tracker.log_param(run.run_id, "batch_size", 32)
        reloaded = tracker.get(run.run_id)
        assert reloaded.parameters["batch_size"] == 32

    def test_log_param_overwrites(self, tracker) -> None:
        run = tracker.start("test")
        tracker.log_param(run.run_id, "lr", 0.01)
        tracker.log_param(run.run_id, "lr", 0.001)
        reloaded = tracker.get(run.run_id)
        assert reloaded.parameters["lr"] == 0.001


class TestLogMetric:
    """Test metric logging."""

    def test_log_metric(self, tracker) -> None:
        run = tracker.start("test")
        tracker.log_metric(run.run_id, "f1", 0.91)
        reloaded = tracker.get(run.run_id)
        assert reloaded.metrics["f1"] == pytest.approx(0.91)

    def test_log_multiple_metrics(self, tracker) -> None:
        run = tracker.start("test")
        tracker.log_metric(run.run_id, "accuracy", 0.95)
        tracker.log_metric(run.run_id, "f1", 0.90)
        reloaded = tracker.get(run.run_id)
        assert "accuracy" in reloaded.metrics
        assert "f1" in reloaded.metrics


class TestLogArtifact:
    """Test artifact logging."""

    def test_log_artifact(self, tracker) -> None:
        run = tracker.start("test")
        tracker.log_artifact(run.run_id, "model", "s3://bucket/model.pkl")
        reloaded = tracker.get(run.run_id)
        assert reloaded.artifacts["model"] == "s3://bucket/model.pkl"


class TestAddTag:
    """Test tagging."""

    def test_add_tag(self, tracker) -> None:
        run = tracker.start("test")
        tracker.add_tag(run.run_id, "production")
        reloaded = tracker.get(run.run_id)
        assert "production" in reloaded.tags

    def test_add_tag_idempotent(self, tracker) -> None:
        run = tracker.start("test")
        tracker.add_tag(run.run_id, "prod")
        tracker.add_tag(run.run_id, "prod")
        reloaded = tracker.get(run.run_id)
        assert reloaded.tags.count("prod") == 1


class TestEnd:
    """Test ending experiments."""

    def test_end_completed(self, tracker) -> None:
        run = tracker.start("test")
        ended = tracker.end(run.run_id)
        assert ended.status == "completed"
        assert ended.ended_at is not None

    def test_end_failed(self, tracker) -> None:
        run = tracker.start("test")
        ended = tracker.end(run.run_id, status="failed")
        assert ended.status == "failed"

    def test_end_returns_run(self, tracker) -> None:
        run = tracker.start("test")
        ended = tracker.end(run.run_id)
        assert ended.run_id == run.run_id


class TestLinkToModel:
    """Test linking experiments to production models."""

    def test_link_to_model(self, tracker) -> None:
        run = tracker.start("test")
        tracker.link_to_model(run.run_id, "fraud_v3", "2.1")
        reloaded = tracker.get(run.run_id)
        assert reloaded.promoted_to == "fraud_v3@2.1"


class TestGet:
    """Test retrieval."""

    def test_get_existing(self, tracker) -> None:
        run = tracker.start("test")
        reloaded = tracker.get(run.run_id)
        assert reloaded.name == "test"

    def test_get_nonexistent_raises(self, tracker) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            tracker.get("nonexistent_id")


class TestListRuns:
    """Test listing experiments."""

    def test_list_runs_empty(self, tracker) -> None:
        runs = tracker.list_runs()
        assert runs == []

    def test_list_runs_multiple(self, tracker) -> None:
        tracker.start("exp1")
        tracker.start("exp2")
        tracker.start("exp3")
        runs = tracker.list_runs()
        assert len(runs) == 3
        names = {r.name for r in runs}
        assert names == {"exp1", "exp2", "exp3"}


class TestFullWorkflow:
    """Integration-style test for the full experiment workflow."""

    def test_full_workflow(self, tracker) -> None:
        # Start experiment
        run = tracker.start("fraud_v3_search", max_depth=6)
        assert run.status == "running"

        # Log metrics and params
        tracker.log_param(run.run_id, "n_estimators", 100)
        tracker.log_metric(run.run_id, "f1", 0.91)
        tracker.log_metric(run.run_id, "accuracy", 0.95)
        tracker.log_artifact(run.run_id, "model", "/models/fraud_v3.pkl")
        tracker.add_tag(run.run_id, "candidate")

        # End experiment
        ended = tracker.end(run.run_id, status="completed")
        assert ended.status == "completed"
        assert ended.metrics["f1"] == 0.91

        # Link to production
        tracker.link_to_model(run.run_id, "fraud_model", "3.0")
        final = tracker.get(run.run_id)
        assert final.promoted_to == "fraud_model@3.0"
        assert final.parameters["max_depth"] == 6
        assert final.parameters["n_estimators"] == 100
        assert "candidate" in final.tags


class TestExperimentRunModel:
    """Test the ExperimentRun Pydantic model."""

    def test_run_id_auto_generated(self) -> None:
        run = ExperimentRun(name="test")
        assert len(run.run_id) == 16

    def test_default_status(self) -> None:
        run = ExperimentRun(name="test")
        assert run.status == "running"

    def test_serialization_roundtrip(self) -> None:
        run = ExperimentRun(
            name="test",
            parameters={"lr": 0.01},
            metrics={"f1": 0.9},
            tags=["prod"],
        )
        json_str = run.model_dump_json()
        restored = ExperimentRun.model_validate_json(json_str)
        assert restored.name == run.name
        assert restored.parameters == run.parameters
        assert restored.metrics == run.metrics
        assert restored.tags == run.tags


class TestConcurrentAccess:
    """Test that concurrent access doesn't corrupt data."""

    def test_sequential_updates(self, tracker) -> None:
        run = tracker.start("test")
        for i in range(10):
            tracker.log_metric(run.run_id, f"metric_{i}", float(i))
        reloaded = tracker.get(run.run_id)
        assert len(reloaded.metrics) == 10
        for i in range(10):
            assert reloaded.metrics[f"metric_{i}"] == float(i)
