"""Tests for Fix 1-6: Intelligence and Foundation layer improvements."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from sentinel.config.schema import (
    AuditConfig,
    BusinessKPIConfig,
    KPIMapping,
)
from sentinel.foundation.audit.compliance import ComplianceReporter
from sentinel.foundation.audit.lineage import LineageTracker
from sentinel.foundation.audit.trail import AuditTrail
from sentinel.foundation.experiments.tracker import ExperimentTracker
from sentinel.intelligence.kpi_linker import KPILinker

# ── Helpers ───────────────────────────────────────────────────────


def _make_kpi_config(mappings: list[dict] | None = None) -> BusinessKPIConfig:
    """Build a minimal BusinessKPIConfig."""
    if mappings is None:
        mappings = [
            {
                "model_metric": "precision",
                "business_kpi": "fraud_catch_rate",
                "data_source": "warehouse://fraud",
            }
        ]
    return BusinessKPIConfig(
        mappings=[KPIMapping(**m) for m in mappings],
    )


def _make_trail(tmp_path: Path) -> AuditTrail:
    cfg = AuditConfig(path=str(tmp_path / "audit"))
    return AuditTrail(cfg)


# ── Fix 1: KPI auto-refresh ──────────────────────────────────────


class TestKPIAutoRefresh:
    def test_auto_refresh_calls_refresh_repeatedly(self) -> None:
        """Start auto-refresh with short interval, verify multiple calls."""
        config = _make_kpi_config()
        linker = KPILinker(config)
        call_count = 0
        lock = threading.Lock()

        def counting_fetcher(src: str) -> float:
            nonlocal call_count
            with lock:
                call_count += 1
            return 0.95

        linker.set_fetcher(counting_fetcher)
        linker.start_auto_refresh(interval_seconds=0.1)
        try:
            time.sleep(0.55)
        finally:
            linker.stop_auto_refresh()
        with lock:
            assert call_count >= 2, f"Expected >=2 refresh calls, got {call_count}"

    def test_stop_auto_refresh_actually_stops(self) -> None:
        """Verify stop_auto_refresh() halts further calls."""
        config = _make_kpi_config()
        linker = KPILinker(config)
        call_count = 0
        lock = threading.Lock()

        def counting_fetcher(src: str) -> float:
            nonlocal call_count
            with lock:
                call_count += 1
            return 0.95

        linker.set_fetcher(counting_fetcher)
        linker.start_auto_refresh(interval_seconds=0.05)
        time.sleep(0.2)
        linker.stop_auto_refresh()
        with lock:
            snapshot = call_count
        time.sleep(0.25)
        with lock:
            assert call_count == snapshot, "Timer kept running after stop"


# ── Fix 2: Compliance configurable risk ──────────────────────────


class TestComplianceRiskLevel:
    def test_custom_risk_level_in_eu_ai_act(self, tmp_path: Path) -> None:
        trail = _make_trail(tmp_path)
        trail.log("deployment_started", model_name="m1", model_version="1.0")
        reporter = ComplianceReporter(trail, risk_level="limited")
        report = reporter.generate("eu_ai_act", "m1")
        assert report["risk_classification"] == "limited"

    def test_default_risk_level_is_high(self, tmp_path: Path) -> None:
        trail = _make_trail(tmp_path)
        trail.log("deployment_started", model_name="m1", model_version="1.0")
        reporter = ComplianceReporter(trail)
        report = reporter.generate("eu_ai_act", "m1")
        assert report["risk_classification"] == "high"


# ── Fix 3: Compliance fairness metrics ───────────────────────────


class TestComplianceFairness:
    def test_fca_report_includes_fairness_issues(self, tmp_path: Path) -> None:
        trail = _make_trail(tmp_path)
        trail.log("drift_detected", model_name="m1")
        trail.log(
            "cohort_analysis",
            model_name="m1",
            disparity_flagged=True,
            cohort="age_group",
            metric="accuracy",
            disparity=0.12,
        )
        trail.log(
            "cohort_analysis",
            model_name="m1",
            disparity_flagged=False,
            cohort="gender",
            metric="accuracy",
            disparity=0.01,
        )
        reporter = ComplianceReporter(trail)
        report = reporter.generate("fca_consumer_duty", "m1")

        fm = report["fairness_monitoring"]
        assert fm["status"] == "active"
        assert fm["cohort_analyses"] == 2
        assert fm["issue_count"] == 1
        assert fm["fairness_issues"][0]["cohort"] == "age_group"

    def test_fca_enriched_fields(self, tmp_path: Path) -> None:
        trail = _make_trail(tmp_path)
        trail.log("alert_fired", model_name="m1")
        trail.log("deployment_started", model_name="m1", model_version="2.0")
        trail.log("retrain_triggered", model_name="m1", trigger="drift")
        trail.log("approval_decision", model_name="m1", approved=True)
        trail.log("prediction_logged", model_name="m1")

        reporter = ComplianceReporter(trail)
        report = reporter.generate("fca_consumer_duty", "m1")

        assert report["summary"]["alerts_fired"] == 1
        assert report["summary"]["deployments"] == 1
        assert "2.0" in report["model_governance"]["versions_deployed"]
        assert "drift" in report["model_governance"]["retrain_triggers"]
        assert report["human_oversight"]["approval_decisions"] == 1
        assert report["outcome_tracking"] == "enabled"

    def test_fca_no_signals(self, tmp_path: Path) -> None:
        trail = _make_trail(tmp_path)
        # Only log an unrelated event
        trail.log("model_registered", model_name="m1")
        reporter = ComplianceReporter(trail)
        report = reporter.generate("fca_consumer_duty", "m1")
        assert report["fairness_monitoring"]["status"] == "no_signals"
        assert report["outcome_tracking"] == "not_configured"


# ── Fix 4: Lineage persistence ───────────────────────────────────


class TestLineagePersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        tracker = LineageTracker(path=tmp_path / "lineage.json")
        tracker.add_node("ds1", "dataset", source="s3://bucket/data")
        tracker.add_node("exp1", "experiment")
        tracker.add_node("mv1", "model_version", version="1.0")
        tracker.add_edge("ds1", "exp1", "trained_on")
        tracker.add_edge("exp1", "mv1", "produced_by")

        tracker.save()

        loaded = LineageTracker.load(tmp_path / "lineage.json")
        original = tracker.to_dict()
        reloaded = loaded.to_dict()

        assert len(reloaded["nodes"]) == len(original["nodes"])
        assert len(reloaded["edges"]) == len(original["edges"])

        orig_node_ids = {n["id"] for n in original["nodes"]}
        loaded_node_ids = {n["id"] for n in reloaded["nodes"]}
        assert orig_node_ids == loaded_node_ids

        orig_edge_set = {(e["src"], e["dst"], e["relation"]) for e in original["edges"]}
        loaded_edge_set = {(e["src"], e["dst"], e["relation"]) for e in reloaded["edges"]}
        assert orig_edge_set == loaded_edge_set

    def test_save_explicit_path(self, tmp_path: Path) -> None:
        tracker = LineageTracker()
        tracker.add_node("n1", "dataset")
        out = tmp_path / "sub" / "graph.json"
        tracker.save(out)
        assert out.exists()
        loaded = LineageTracker.load(out)
        assert len(loaded.to_dict()["nodes"]) == 1

    def test_save_no_path_raises(self) -> None:
        tracker = LineageTracker()
        with pytest.raises(ValueError, match="no path"):
            tracker.save()


# ── Fix 5: Audit file-level index ────────────────────────────────


class TestAuditFileIndex:
    def test_index_skips_files_without_event_type(self, tmp_path: Path) -> None:
        trail = _make_trail(tmp_path)
        trail.log("drift_detected", model_name="m1")
        trail.log("alert_fired", model_name="m1")
        trail.log("model_registered", model_name="m2")

        # Index should know the current file has all three types
        results = list(trail.query(event_type="drift_detected"))
        assert len(results) == 1
        assert results[0].event_type == "drift_detected"

    def test_index_skips_files_without_model_name(self, tmp_path: Path) -> None:
        trail = _make_trail(tmp_path)
        trail.log("drift_detected", model_name="model_a")
        trail.log("drift_detected", model_name="model_b")

        results = list(trail.query(model_name="model_a"))
        assert len(results) == 1
        assert results[0].model_name == "model_a"

    def test_index_populated_on_write(self, tmp_path: Path) -> None:
        trail = _make_trail(tmp_path)
        trail.log("drift_detected", model_name="m1")
        trail.log("alert_fired", model_name="m2")

        # Verify the internal indexes exist and are populated
        assert len(trail._file_type_index) > 0
        types_in_files = set()
        for types in trail._file_type_index.values():
            types_in_files |= types
        assert "drift_detected" in types_in_files
        assert "alert_fired" in types_in_files


# ── Fix 6: Experiment filter OR ──────────────────────────────────


class TestExperimentFilterOR:
    def test_or_filter_matches_either_condition(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(storage_path=tmp_path / "experiments")
        tracker.create_experiment("exp1")

        r1 = tracker.start_run("exp1", params={"lr": 0.01})
        tracker.log_metric(r1.run_id, "f1", 0.85)
        tracker.log_metric(r1.run_id, "accuracy", 0.80)
        tracker.end_run(r1.run_id)

        r2 = tracker.start_run("exp1", params={"lr": 0.1})
        tracker.log_metric(r2.run_id, "f1", 0.70)
        tracker.log_metric(r2.run_id, "accuracy", 0.95)
        tracker.end_run(r2.run_id)

        r3 = tracker.start_run("exp1", params={"lr": 0.5})
        tracker.log_metric(r3.run_id, "f1", 0.60)
        tracker.log_metric(r3.run_id, "accuracy", 0.70)
        tracker.end_run(r3.run_id)

        results = tracker.search_runs(
            "exp1",
            filter_expr="metrics.f1 > 0.8 OR metrics.accuracy > 0.9",
        )
        result_ids = {r.run_id for r in results}
        assert r1.run_id in result_ids, "r1 (f1=0.85) should match"
        assert r2.run_id in result_ids, "r2 (accuracy=0.95) should match"
        assert r3.run_id not in result_ids, "r3 should not match"

    def test_and_still_works(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(storage_path=tmp_path / "experiments")
        tracker.create_experiment("exp1")

        r1 = tracker.start_run("exp1")
        tracker.log_metric(r1.run_id, "f1", 0.9)
        tracker.log_metric(r1.run_id, "accuracy", 0.95)
        tracker.end_run(r1.run_id)

        r2 = tracker.start_run("exp1")
        tracker.log_metric(r2.run_id, "f1", 0.9)
        tracker.log_metric(r2.run_id, "accuracy", 0.80)
        tracker.end_run(r2.run_id)

        results = tracker.search_runs(
            "exp1",
            filter_expr="metrics.f1 > 0.8 AND metrics.accuracy > 0.9",
        )
        assert len(results) == 1
        assert results[0].run_id == r1.run_id

    def test_or_with_and_groups(self, tmp_path: Path) -> None:
        """Test mixed: 'A AND B OR C AND D'."""
        tracker = ExperimentTracker(storage_path=tmp_path / "experiments")
        tracker.create_experiment("exp1")

        r1 = tracker.start_run("exp1")
        tracker.log_metric(r1.run_id, "f1", 0.9)
        tracker.log_metric(r1.run_id, "accuracy", 0.6)
        tracker.log_metric(r1.run_id, "precision", 0.95)
        tracker.end_run(r1.run_id)

        r2 = tracker.start_run("exp1")
        tracker.log_metric(r2.run_id, "f1", 0.5)
        tracker.log_metric(r2.run_id, "accuracy", 0.95)
        tracker.log_metric(r2.run_id, "precision", 0.95)
        tracker.end_run(r2.run_id)

        # r1 matches first group (f1>0.8 AND accuracy>0.5)
        # r2 matches second group (accuracy>0.9 AND precision>0.9)
        results = tracker.search_runs(
            "exp1",
            filter_expr="metrics.f1 > 0.8 AND metrics.accuracy > 0.5 OR metrics.accuracy > 0.9 AND metrics.precision > 0.9",
        )
        result_ids = {r.run_id for r in results}
        assert r1.run_id in result_ids
        assert r2.run_id in result_ids
