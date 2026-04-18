"""Thread-safety, eager cycle detection, assert-to-raise, and input validation tests.

Covers fixes 1-7 from the v4 hardening pass:
  1. ModelGraph thread safety + eager cycle detection
  2. KPILinker thread safety
  3. LineageTracker thread safety
  4. CalendarDriftDetector assert → raise
  5. BaseDriftDetector lock provision
  6. Domain adapter negative-threshold validation
  7. Domain adapter invalid-task / invalid-method validation
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    BusinessKPIConfig,
    DataDriftConfig,
    DomainConfig,
    DriftConfig,
    KPIMapping,
    ModelConfig,
    SentinelConfig,
)
from sentinel.domains.graph.adapter import GraphAdapter
from sentinel.domains.nlp.adapter import NLPAdapter
from sentinel.domains.recommendation.adapter import RecommendationAdapter
from sentinel.domains.timeseries.adapter import TimeSeriesAdapter
from sentinel.domains.timeseries.drift import (
    CalendarDriftDetector,
    StationarityDriftDetector,
)
from sentinel.foundation.audit.lineage import LineageTracker
from sentinel.intelligence.kpi_linker import KPILinker
from sentinel.intelligence.model_graph import ModelGraph

# ── helpers ───────────────────────────────────────────────────────


def _domain_config(domain: str, options: dict[str, Any] | None = None) -> SentinelConfig:
    """Build a minimal SentinelConfig with the given domain options."""
    domains = DomainConfig()
    if options is not None:
        setattr(domains, domain, options)
    return SentinelConfig(
        model=ModelConfig(name="test_model", domain=domain),
        drift=DriftConfig(data=DataDriftConfig(method="psi", threshold=0.2, window="7d")),
        alerts=AlertsConfig(),
        audit=AuditConfig(storage="local"),
        domains=domains,
    )


# ═══════════════════════════════════════════════════════════════════
# Fix 1 — ModelGraph thread safety + eager cycle detection
# ═══════════════════════════════════════════════════════════════════


class TestModelGraphThreadSafety:
    """Concurrent edge additions must not corrupt the internal state."""

    def test_concurrent_add_edges(self) -> None:
        graph = ModelGraph()
        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                for j in range(20):
                    src = f"node_{idx}_{j}"
                    dst = f"sink_{idx}"
                    graph.add_edge(src, dst)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"threads raised: {errors}"
        # Each worker adds 20 edges to a unique sink → 200 total edges
        d = graph.to_dict()
        assert len(d["edges"]) == 200

    def test_eager_cycle_detection_two_node(self) -> None:
        graph = ModelGraph()
        graph.add_edge("A", "B")
        with pytest.raises(ValueError, match="creates a cycle"):
            graph.add_edge("B", "A")
        # First edge should survive
        assert "B" in graph.get_descendants("A")
        assert "A" not in graph.get_descendants("B")

    def test_eager_cycle_detection_three_node(self) -> None:
        graph = ModelGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        with pytest.raises(ValueError, match="creates a cycle"):
            graph.add_edge("C", "A")
        # Original chain intact
        assert graph.get_descendants("A") == ["B", "C"]

    def test_self_loop_rejected(self) -> None:
        graph = ModelGraph()
        with pytest.raises(ValueError, match="creates a cycle"):
            graph.add_edge("X", "X")

    def test_valid_dag_still_sorts(self) -> None:
        graph = ModelGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        order = graph.topological_sort()
        assert order.index("A") < order.index("B") < order.index("C")


# ═══════════════════════════════════════════════════════════════════
# Fix 2 — KPILinker thread safety
# ═══════════════════════════════════════════════════════════════════


class TestKPILinkerThreadSafety:
    """Concurrent refresh() + report() must not corrupt cached KPIs."""

    @staticmethod
    def _kpi_config() -> BusinessKPIConfig:
        return BusinessKPIConfig(
            mappings=[
                KPIMapping(
                    model_metric="accuracy",
                    business_kpi="fraud_catch_rate",
                    data_source="warehouse://test",
                ),
            ]
        )

    def test_concurrent_refresh_and_report(self) -> None:
        linker = KPILinker(self._kpi_config())
        counter = {"calls": 0}

        def fetcher(source: str) -> float:
            counter["calls"] += 1
            return 0.95

        linker.set_fetcher(fetcher)
        errors: list[Exception] = []
        barrier = threading.Barrier(6)

        def refresh_worker() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    linker.refresh()
            except Exception as exc:
                errors.append(exc)

        def report_worker() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    linker.report({"accuracy": 0.9})
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=refresh_worker) for _ in range(3)]
            + [threading.Thread(target=report_worker) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"threads raised: {errors}"


# ═══════════════════════════════════════════════════════════════════
# Fix 3 — LineageTracker thread safety
# ═══════════════════════════════════════════════════════════════════


class TestLineageTrackerThreadSafety:
    """Concurrent add_node/add_edge/get_ancestors must not corrupt state."""

    def test_concurrent_operations(self) -> None:
        tracker = LineageTracker()
        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def writer(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                for j in range(20):
                    nid = f"n_{idx}_{j}"
                    tracker.add_node(nid, kind="dataset")
                    if j > 0:
                        prev = f"n_{idx}_{j - 1}"
                        tracker.add_edge(prev, nid, relation="derived_from")
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    tracker.to_dict()
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=writer, args=(i,)) for i in range(8)]
            + [threading.Thread(target=reader) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"threads raised: {errors}"
        d = tracker.to_dict()
        # 8 writers x 20 nodes = 160
        assert len(d["nodes"]) == 160


# ═══════════════════════════════════════════════════════════════════
# Fix 4 — CalendarDriftDetector assert → raise
# ═══════════════════════════════════════════════════════════════════


class TestCalendarDriftDetectorRaisesRuntime:
    """detect() before fit() must raise RuntimeError, not AssertionError."""

    def test_detect_before_fit_raises_runtime(self) -> None:
        det = CalendarDriftDetector(model_name="m", threshold=0.2, seasonality=7)
        with pytest.raises(RuntimeError, match="not fitted"):
            det.detect([1, 2, 3])

    def test_fit_then_detect_succeeds(self) -> None:
        det = CalendarDriftDetector(model_name="m", threshold=0.2, seasonality=7)
        det.fit(np.arange(28, dtype=float))
        report = det.detect(np.arange(28, dtype=float))
        assert not report.is_drifted

    def test_stationarity_detect_before_fit_raises_runtime(self) -> None:
        det = StationarityDriftDetector(model_name="m", threshold=0.2)
        with pytest.raises(RuntimeError, match="not fitted"):
            det.detect([1, 2, 3])


# ═══════════════════════════════════════════════════════════════════
# Fix 5 — BaseDriftDetector lock provision
# ═══════════════════════════════════════════════════════════════════


class TestBaseDriftDetectorLock:
    """Verify that the lock is present on every drift detector instance."""

    def test_lock_exists(self) -> None:
        det = CalendarDriftDetector(model_name="m", threshold=0.1)
        assert hasattr(det, "_lock")
        assert isinstance(det._lock, type(threading.Lock()))


# ═══════════════════════════════════════════════════════════════════
# Fix 6 — Domain adapter negative-threshold validation
# ═══════════════════════════════════════════════════════════════════


class TestDomainThresholdValidation:
    """Negative thresholds must be rejected at construction time."""

    def test_timeseries_negative_threshold(self) -> None:
        cfg = _domain_config("timeseries", {"drift": {"threshold": -0.5}})
        with pytest.raises(ValueError, match="non-negative"):
            TimeSeriesAdapter(cfg)

    def test_nlp_negative_oov_threshold(self) -> None:
        cfg = _domain_config("nlp", {"drift": {"vocabulary": {"oov_rate_threshold": -1.0}}})
        with pytest.raises(ValueError, match="non-negative"):
            NLPAdapter(cfg)

    def test_recommendation_negative_item_threshold(self) -> None:
        cfg = _domain_config(
            "recommendation", {"drift": {"item_distribution": {"threshold": -0.1}}}
        )
        with pytest.raises(ValueError, match="non-negative"):
            RecommendationAdapter(cfg)

    def test_graph_negative_topology_threshold(self) -> None:
        cfg = _domain_config("graph", {"drift": {"topology": {"threshold": -0.3}}})
        with pytest.raises(ValueError, match="non-negative"):
            GraphAdapter(cfg)

    def test_base_drift_detector_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CalendarDriftDetector(model_name="m", threshold=-0.1)


# ═══════════════════════════════════════════════════════════════════
# Fix 7 — Domain adapter invalid task / method validation
# ═══════════════════════════════════════════════════════════════════


class TestDomainTaskValidation:
    """Invalid task / graph_type names must be rejected."""

    def test_nlp_invalid_task(self) -> None:
        cfg = _domain_config("nlp", {"task": "astrology"})
        with pytest.raises(ValueError, match="invalid NLP task"):
            NLPAdapter(cfg)

    def test_nlp_valid_tasks_accepted(self) -> None:
        for task in ("ner", "classification", "sentiment", "topic_modelling"):
            cfg = _domain_config("nlp", {"task": task})
            adapter = NLPAdapter(cfg)
            assert adapter.task == task

    def test_graph_invalid_task(self) -> None:
        cfg = _domain_config("graph", {"task": "astrology"})
        with pytest.raises(ValueError, match="invalid graph task"):
            GraphAdapter(cfg)

    def test_graph_invalid_graph_type(self) -> None:
        cfg = _domain_config("graph", {"graph_type": "fantasy"})
        with pytest.raises(ValueError, match="invalid graph_type"):
            GraphAdapter(cfg)

    def test_graph_valid_tasks_accepted(self) -> None:
        for task in ("link_prediction", "node_classification", "kg_completion"):
            cfg = _domain_config("graph", {"task": task})
            adapter = GraphAdapter(cfg)
            assert adapter.task == task
