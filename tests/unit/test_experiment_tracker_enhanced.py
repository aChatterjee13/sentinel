"""Unit tests for the enhanced ExperimentTracker.

Covers experiment management, run lifecycle, metric history, search/compare,
dataset and model linking, backward-compat aliases, storage layout, and
thread safety.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from sentinel.foundation.experiments.tracker import (
    Experiment,
    ExperimentRun,
    ExperimentTracker,
    MetricEntry,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def tracker(tmp_path: Any) -> ExperimentTracker:
    """Fresh tracker with a temporary storage path."""
    return ExperimentTracker(storage_path=tmp_path / "store")


@pytest.fixture
def seeded_tracker(tracker: ExperimentTracker) -> ExperimentTracker:
    """Tracker with a few experiments and runs pre-seeded."""
    tracker.create_experiment("search", description="Hyper-param search")
    r1 = tracker.start_run("search", name="run-a", params={"lr": 0.01})
    tracker.log_metric(r1.run_id, "f1", 0.80, step=1)
    tracker.log_metric(r1.run_id, "f1", 0.85, step=2)
    tracker.log_metric(r1.run_id, "f1", 0.90, step=3)
    tracker.log_metric(r1.run_id, "accuracy", 0.92)
    tracker.end_run(r1.run_id)

    r2 = tracker.start_run("search", name="run-b", params={"lr": 0.001})
    tracker.log_metric(r2.run_id, "f1", 0.70)
    tracker.log_metric(r2.run_id, "accuracy", 0.88)
    tracker.end_run(r2.run_id, status="failed")

    tracker.create_experiment("deploy", tags=["prod"])
    r3 = tracker.start_run("deploy", name="run-c", params={"lr": 0.01})
    tracker.log_metric(r3.run_id, "f1", 0.95)
    tracker.end_run(r3.run_id)

    return tracker


# ── Experiment management ─────────────────────────────────────────


class TestCreateExperiment:
    """Tests for experiment creation."""

    def test_creates_and_returns(self, tracker: ExperimentTracker) -> None:
        exp = tracker.create_experiment("alpha")
        assert isinstance(exp, Experiment)
        assert exp.name == "alpha"
        assert exp.created_at is not None

    def test_idempotent_same_name(self, tracker: ExperimentTracker) -> None:
        e1 = tracker.create_experiment("beta", description="first")
        e2 = tracker.create_experiment("beta", description="second")
        assert e1.name == e2.name
        assert e1.created_at == e2.created_at
        # Description stays from first creation (idempotent)
        assert e2.description == "first"

    def test_with_tags_and_metadata(self, tracker: ExperimentTracker) -> None:
        exp = tracker.create_experiment(
            "gamma",
            tags=["prod", "v2"],
            metadata={"team": "ml"},
        )
        assert exp.tags == ["prod", "v2"]
        assert exp.metadata == {"team": "ml"}


class TestGetExperiment:
    """Tests for experiment retrieval."""

    def test_get_existing(self, tracker: ExperimentTracker) -> None:
        tracker.create_experiment("delta")
        exp = tracker.get_experiment("delta")
        assert exp.name == "delta"

    def test_get_nonexistent_raises(self, tracker: ExperimentTracker) -> None:
        with pytest.raises(KeyError, match="not found"):
            tracker.get_experiment("nope")


class TestListExperiments:
    """Tests for listing experiments."""

    def test_returns_all(self, tracker: ExperimentTracker) -> None:
        tracker.create_experiment("a")
        tracker.create_experiment("b")
        tracker.create_experiment("c")
        exps = tracker.list_experiments()
        assert len(exps) == 3
        names = {e.name for e in exps}
        assert names == {"a", "b", "c"}

    def test_empty(self, tracker: ExperimentTracker) -> None:
        assert tracker.list_experiments() == []


# ── Run management ────────────────────────────────────────────────


class TestStartRun:
    """Tests for starting runs."""

    def test_auto_creates_experiment(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("new_exp", name="run-1")
        assert run.experiment_name == "new_exp"
        # Experiment should exist now
        exp = tracker.get_experiment("new_exp")
        assert exp.name == "new_exp"

    def test_with_params(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp", params={"lr": 0.01, "batch": 32})
        assert run.parameters["lr"] == 0.01
        assert run.parameters["batch"] == 32

    def test_nested_parent_run_id(self, tracker: ExperimentTracker) -> None:
        parent = tracker.start_run("exp", name="parent")
        child = tracker.start_run("exp", name="child", parent_run_id=parent.run_id)
        assert child.parent_run_id == parent.run_id

    def test_with_tags(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp", tags=["candidate", "v2"])
        assert run.tags == ["candidate", "v2"]

    def test_status_is_running(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        assert run.status == "running"

    def test_creates_file(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        assert tracker._path(run.run_id).exists()


class TestEndRun:
    """Tests for ending runs."""

    def test_sets_status_and_timestamp(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        ended = tracker.end_run(run.run_id)
        assert ended.status == "completed"
        assert ended.ended_at is not None

    def test_failed_status(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        ended = tracker.end_run(run.run_id, status="failed")
        assert ended.status == "failed"

    def test_returns_updated_run(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.log_metric(run.run_id, "f1", 0.9)
        ended = tracker.end_run(run.run_id)
        assert ended.metrics["f1"] == pytest.approx(0.9)


class TestGetRun:
    """Tests for run retrieval."""

    def test_get_existing(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp", name="hello")
        loaded = tracker.get_run(run.run_id)
        assert loaded.name == "hello"

    def test_get_nonexistent_raises(self, tracker: ExperimentTracker) -> None:
        with pytest.raises(KeyError, match="not found"):
            tracker.get_run("does_not_exist")


# ── Logging ───────────────────────────────────────────────────────


class TestLogMetric:
    """Tests for metric logging."""

    def test_appends_to_history(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.log_metric(run.run_id, "loss", 0.5, step=1)
        tracker.log_metric(run.run_id, "loss", 0.3, step=2)
        history = tracker.get_metric_history(run.run_id, "loss")
        assert len(history) == 2
        assert history[0].value == pytest.approx(0.5)
        assert history[1].value == pytest.approx(0.3)

    def test_updates_latest_snapshot(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.log_metric(run.run_id, "acc", 0.8)
        tracker.log_metric(run.run_id, "acc", 0.9)
        loaded = tracker.get_run(run.run_id)
        assert loaded.metrics["acc"] == pytest.approx(0.9)

    def test_multiple_steps_create_time_series(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        for i in range(5):
            tracker.log_metric(run.run_id, "loss", 1.0 / (i + 1), step=i)
        history = tracker.get_metric_history(run.run_id, "loss")
        assert len(history) == 5
        assert [e.step for e in history] == [0, 1, 2, 3, 4]


class TestLogParams:
    """Tests for parameter logging."""

    def test_stores_all_params(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.log_params(run.run_id, {"lr": 0.01, "epochs": 10, "opt": "adam"})
        loaded = tracker.get_run(run.run_id)
        assert loaded.parameters == {"lr": 0.01, "epochs": 10, "opt": "adam"}


class TestLogArtifact:
    """Tests for artifact logging."""

    def test_stores_uri(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.log_artifact(run.run_id, "model", "s3://bucket/model.pkl")
        loaded = tracker.get_run(run.run_id)
        assert loaded.artifacts["model"] == "s3://bucket/model.pkl"


class TestLogDataset:
    """Tests for dataset linking."""

    def test_stores_ref(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.log_dataset(run.run_id, "fraud_training@1.0")
        loaded = tracker.get_run(run.run_id)
        assert "fraud_training@1.0" in loaded.dataset_refs

    def test_idempotent(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.log_dataset(run.run_id, "ds@1")
        tracker.log_dataset(run.run_id, "ds@1")
        loaded = tracker.get_run(run.run_id)
        assert loaded.dataset_refs.count("ds@1") == 1


class TestLinkToModel:
    """Tests for model version linking."""

    def test_stores_model_ref(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.link_to_model(run.run_id, "fraud_model", "3.0")
        loaded = tracker.get_run(run.run_id)
        assert loaded.promoted_to == "fraud_model@3.0"


class TestAddTag:
    """Tests for tagging."""

    def test_adds_tag(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.add_tag(run.run_id, "candidate")
        loaded = tracker.get_run(run.run_id)
        assert "candidate" in loaded.tags

    def test_idempotent(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.add_tag(run.run_id, "x")
        tracker.add_tag(run.run_id, "x")
        loaded = tracker.get_run(run.run_id)
        assert loaded.tags.count("x") == 1


# ── Search & compare ─────────────────────────────────────────────


class TestListRuns:
    """Tests for listing runs."""

    def test_filtered_by_experiment(self, seeded_tracker: ExperimentTracker) -> None:
        runs = seeded_tracker.list_runs(experiment_name="search")
        assert len(runs) == 2

    def test_all_runs(self, seeded_tracker: ExperimentTracker) -> None:
        runs = seeded_tracker.list_runs()
        assert len(runs) == 3


class TestSearchRuns:
    """Tests for search_runs with filter expressions."""

    def test_metrics_filter_gt(self, seeded_tracker: ExperimentTracker) -> None:
        results = seeded_tracker.search_runs(filter_expr="metrics.f1 > 0.85")
        assert all(r.metrics["f1"] > 0.85 for r in results)
        assert len(results) >= 2  # run-a (0.90) and run-c (0.95)

    def test_combined_filter(self, seeded_tracker: ExperimentTracker) -> None:
        results = seeded_tracker.search_runs(filter_expr="params.lr < 0.01 AND metrics.f1 > 0.5")
        # Only run-b has lr=0.001 and f1=0.70
        assert len(results) == 1
        assert results[0].parameters["lr"] == 0.001

    def test_order_by_desc(self, seeded_tracker: ExperimentTracker) -> None:
        results = seeded_tracker.search_runs(order_by="metrics.f1 DESC")
        f1s = [r.metrics["f1"] for r in results]
        assert f1s == sorted(f1s, reverse=True)

    def test_order_by_asc(self, seeded_tracker: ExperimentTracker) -> None:
        results = seeded_tracker.search_runs(order_by="metrics.f1 ASC")
        f1s = [r.metrics["f1"] for r in results]
        assert f1s == sorted(f1s)

    def test_scoped_to_experiment(self, seeded_tracker: ExperimentTracker) -> None:
        results = seeded_tracker.search_runs(
            experiment_name="search",
            filter_expr="metrics.f1 > 0.85",
        )
        assert all(r.experiment_name == "search" for r in results)

    def test_max_results(self, seeded_tracker: ExperimentTracker) -> None:
        results = seeded_tracker.search_runs(max_results=1)
        assert len(results) == 1


class TestCompareRuns:
    """Tests for side-by-side run comparison."""

    def test_returns_side_by_side(self, seeded_tracker: ExperimentTracker) -> None:
        runs = seeded_tracker.list_runs(experiment_name="search")
        ids = [r.run_id for r in runs]
        comparison = seeded_tracker.compare_runs(ids)
        assert "params_diff" in comparison
        assert "metrics_latest" in comparison
        assert "status" in comparison
        assert "duration_s" in comparison
        assert "artifacts" in comparison
        # Both runs should appear in metrics_latest
        for metric_dict in comparison["metrics_latest"].values():
            assert set(metric_dict.keys()) == set(ids)


class TestGetMetricHistory:
    """Tests for metric history retrieval."""

    def test_returns_full_history(self, seeded_tracker: ExperimentTracker) -> None:
        runs = seeded_tracker.list_runs(experiment_name="search")
        run_a = next(r for r in runs if r.name == "run-a")
        history = seeded_tracker.get_metric_history(run_a.run_id, "f1")
        assert len(history) == 3
        assert [e.value for e in history] == [
            pytest.approx(0.80),
            pytest.approx(0.85),
            pytest.approx(0.90),
        ]

    def test_missing_metric_returns_empty(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        assert tracker.get_metric_history(run.run_id, "nope") == []


# ── Backward compatibility ────────────────────────────────────────


class TestBackwardCompat:
    """The old start/end/get/list_runs API must keep working."""

    def test_start(self, tracker: ExperimentTracker) -> None:
        run = tracker.start("my_test", lr=0.01)
        assert run.name == "my_test"
        assert run.parameters["lr"] == 0.01
        assert run.status == "running"

    def test_end(self, tracker: ExperimentTracker) -> None:
        run = tracker.start("test")
        ended = tracker.end(run.run_id)
        assert ended.status == "completed"
        assert ended.ended_at is not None

    def test_get(self, tracker: ExperimentTracker) -> None:
        run = tracker.start("test")
        loaded = tracker.get(run.run_id)
        assert loaded.name == "test"

    def test_log_param_and_read(self, tracker: ExperimentTracker) -> None:
        run = tracker.start("test")
        tracker.log_param(run.run_id, "x", 42)
        assert tracker.get(run.run_id).parameters["x"] == 42

    def test_log_metric_and_read_as_float(self, tracker: ExperimentTracker) -> None:
        run = tracker.start("test")
        tracker.log_metric(run.run_id, "f1", 0.91)
        loaded = tracker.get(run.run_id)
        assert loaded.metrics["f1"] == pytest.approx(0.91)

    def test_list_runs_no_filter(self, tracker: ExperimentTracker) -> None:
        tracker.start("a")
        tracker.start("b")
        runs = tracker.list_runs()
        assert len(runs) == 2

    def test_link_to_model_compat(self, tracker: ExperimentTracker) -> None:
        run = tracker.start("test")
        tracker.link_to_model(run.run_id, "fraud", "2.1")
        assert tracker.get(run.run_id).promoted_to == "fraud@2.1"


# ── Storage layout ────────────────────────────────────────────────


class TestStorageLayout:
    """Verify the on-disk file structure."""

    def test_run_files_in_runs_dir(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        path = tracker._path(run.run_id)
        assert path.exists()
        assert path.parent.name == "runs"

    def test_experiment_file_created(self, tracker: ExperimentTracker) -> None:
        tracker.create_experiment("my_exp")
        assert (tracker._root / "experiments" / "my_exp.json").exists()


# ── Thread safety ─────────────────────────────────────────────────


class TestThreadSafety:
    """Concurrent writes must not corrupt data."""

    def test_concurrent_metric_writes(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        errors: list[Exception] = []

        def writer(metric_name: str) -> None:
            try:
                for i in range(20):
                    tracker.log_metric(run.run_id, metric_name, float(i), step=i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"m{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        loaded = tracker.get_run(run.run_id)
        # All four metric keys should be present
        assert len(loaded.metrics) == 4
        for i in range(4):
            assert f"m{i}" in loaded.metrics


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_experiment_no_runs(self, tracker: ExperimentTracker) -> None:
        tracker.create_experiment("empty")
        runs = tracker.list_runs(experiment_name="empty")
        assert runs == []

    def test_run_with_no_metrics(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        ended = tracker.end_run(run.run_id)
        assert ended.metrics == {}
        assert ended.metric_history == {}

    def test_search_no_results(self, tracker: ExperimentTracker) -> None:
        tracker.start_run("exp")
        results = tracker.search_runs(filter_expr="metrics.f1 > 100")
        assert results == []

    def test_log_metrics_batch(self, tracker: ExperimentTracker) -> None:
        run = tracker.start_run("exp")
        tracker.log_metrics(run.run_id, {"a": 1.0, "b": 2.0}, step=0)
        loaded = tracker.get_run(run.run_id)
        assert loaded.metrics["a"] == pytest.approx(1.0)
        assert loaded.metrics["b"] == pytest.approx(2.0)
        assert len(loaded.metric_history["a"]) == 1
        assert len(loaded.metric_history["b"]) == 1


# ── Data models ───────────────────────────────────────────────────


class TestModels:
    """Tests for the Pydantic data models."""

    def test_experiment_run_frozen(self) -> None:
        run = ExperimentRun(name="test")
        with pytest.raises(Exception):
            run.status = "done"  # type: ignore[misc]

    def test_metric_entry_frozen(self) -> None:
        entry = MetricEntry(value=0.9, timestamp="2025-01-01T00:00:00Z")
        with pytest.raises(Exception):
            entry.value = 0.5  # type: ignore[misc]

    def test_experiment_frozen(self) -> None:
        exp = Experiment(name="x", created_at="2025-01-01T00:00:00Z")
        with pytest.raises(Exception):
            exp.name = "y"  # type: ignore[misc]

    def test_run_serialization_roundtrip(self) -> None:
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

    def test_run_id_auto_generated(self) -> None:
        run = ExperimentRun(name="test")
        assert len(run.run_id) == 16
