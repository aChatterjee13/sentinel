"""Tests for the enhanced CostMonitor functionality."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import pytest

from sentinel.config.schema import CostMonitorConfig
from sentinel.core.types import CostMetrics
from sentinel.observability.cost_monitor import CostMonitor

# ── Helpers ────────────────────────────────────────────────────────


def _make_monitor(
    thresholds: dict[str, float] | None = None,
    window_size: int = 1000,
    cost_per_prediction: float = 0.0,
) -> CostMonitor:
    config = CostMonitorConfig(alert_thresholds=thresholds or {})
    return CostMonitor(
        config,
        model_name="test_model",
        window_size=window_size,
        cost_per_prediction=cost_per_prediction,
    )


# ── Empty state defaults ──────────────────────────────────────────


class TestEmptyState:
    def test_empty_snapshot_has_nan_latencies(self) -> None:
        mon = _make_monitor()
        snap = mon.snapshot()
        assert math.isnan(snap.latency_ms_p50)
        assert math.isnan(snap.latency_ms_p95)
        assert math.isnan(snap.latency_ms_p99)

    def test_empty_snapshot_has_zero_error_rate(self) -> None:
        mon = _make_monitor()
        snap = mon.snapshot()
        assert snap.error_rate == 0.0
        assert snap.error_count == 0
        assert snap.success_count == 0

    def test_empty_snapshot_has_zero_throughput(self) -> None:
        mon = _make_monitor()
        snap = mon.snapshot()
        assert snap.throughput_rps == 0.0
        assert snap.sample_count == 0


# ── Throughput threshold ──────────────────────────────────────────


class TestThroughputThreshold:
    def test_low_throughput_triggers_breach(self) -> None:
        mon = _make_monitor(thresholds={"throughput_rps": 100.0})
        # Record two samples close together → low throughput
        mon.record(10.0)
        time.sleep(0.05)
        mon.record(10.0)
        breaches = mon.check_thresholds()
        assert any("throughput" in b for b in breaches)

    def test_no_breach_when_above_threshold(self) -> None:
        mon = _make_monitor(thresholds={"throughput_rps": 0.1})
        mon.record(10.0)
        time.sleep(0.01)
        mon.record(10.0)
        breaches = mon.check_thresholds()
        assert not any("throughput" in b for b in breaches)


# ── Error rate tracking ───────────────────────────────────────────


class TestErrorRate:
    def test_record_error_increments_count(self) -> None:
        mon = _make_monitor()
        mon.record_error()
        mon.record_error()
        snap = mon.snapshot()
        assert snap.error_count == 2
        assert snap.success_count == 0

    def test_error_rate_computation(self) -> None:
        mon = _make_monitor()
        # 3 successes, 1 error → 25% error rate
        for _ in range(3):
            mon.record(10.0)
        mon.record_error(latency_ms=50.0)
        snap = mon.snapshot()
        assert snap.error_rate == pytest.approx(0.25)
        assert snap.error_count == 1
        assert snap.success_count == 3

    def test_error_with_latency_appears_in_window(self) -> None:
        mon = _make_monitor()
        mon.record_error(latency_ms=500.0)
        snap = mon.snapshot()
        assert snap.sample_count == 1
        assert snap.latency_ms_p50 == pytest.approx(500.0)

    def test_error_without_latency_not_in_window(self) -> None:
        mon = _make_monitor()
        mon.record_error()
        snap = mon.snapshot()
        # No latency samples → nan
        assert snap.sample_count == 0
        assert math.isnan(snap.latency_ms_p50)
        # But error count is tracked
        assert snap.error_count == 1

    def test_error_rate_threshold_breach(self) -> None:
        mon = _make_monitor(thresholds={"error_rate": 0.1})
        # 1 success, 1 error → 50% error rate → breach
        mon.record(10.0)
        mon.record_error()
        breaches = mon.check_thresholds()
        assert any("error rate" in b for b in breaches)

    def test_no_error_rate_breach_when_below(self) -> None:
        mon = _make_monitor(thresholds={"error_rate": 0.5})
        # 9 successes, 1 error → 10% error rate → no breach
        for _ in range(9):
            mon.record(10.0)
        mon.record_error()
        breaches = mon.check_thresholds()
        assert not any("error rate" in b for b in breaches)


# ── Auto-record from log_prediction ──────────────────────────────


class TestAutoRecordFromLogPrediction:
    def test_latency_ms_forwarded_to_cost_monitor(self) -> None:
        from sentinel.config.schema import ModelConfig, SentinelConfig
        from sentinel.core.client import SentinelClient

        cfg = SentinelConfig(model=ModelConfig(name="auto_test"))
        client = SentinelClient(cfg)
        client.log_prediction(
            features={"x": 1.0},
            prediction=0,
            latency_ms=42.5,
        )
        snap = client.cost_monitor.snapshot()
        assert snap.sample_count == 1
        assert snap.latency_ms_p50 == pytest.approx(42.5)
        assert snap.success_count == 1

    def test_no_latency_no_record(self) -> None:
        from sentinel.config.schema import ModelConfig, SentinelConfig
        from sentinel.core.client import SentinelClient

        cfg = SentinelConfig(model=ModelConfig(name="auto_test2"))
        client = SentinelClient(cfg)
        client.log_prediction(features={"x": 1.0}, prediction=0)
        snap = client.cost_monitor.snapshot()
        assert snap.sample_count == 0


# ── Metric persistence ────────────────────────────────────────────


class TestFlushAndLoadMetrics:
    def test_flush_creates_valid_jsonl(self, tmp_path: Path) -> None:
        mon = _make_monitor()
        mon.record(10.0)
        mon.record(20.0)
        fpath = tmp_path / "metrics.jsonl"
        count = mon.flush_metrics(fpath)
        assert count == 1
        lines = fpath.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "timestamp" in data
        assert "latency_ms_p50" in data
        assert "error_rate" in data
        assert "sample_count" in data
        assert data["sample_count"] == 2

    def test_flush_appends_multiple(self, tmp_path: Path) -> None:
        mon = _make_monitor()
        fpath = tmp_path / "metrics.jsonl"
        mon.record(10.0)
        mon.flush_metrics(fpath)
        mon.record(20.0)
        count = mon.flush_metrics(fpath)
        assert count == 2

    def test_load_metrics_reads_back(self, tmp_path: Path) -> None:
        mon = _make_monitor()
        fpath = tmp_path / "metrics.jsonl"
        for i in range(5):
            mon.record(float(i * 10))
            mon.flush_metrics(fpath)
        loaded = CostMonitor.load_metrics(fpath, n=3)
        assert len(loaded) == 3
        # All should parse
        for rec in loaded:
            assert "timestamp" in rec
            assert "latency_ms_p50" in rec

    def test_load_metrics_from_nonexistent_file(self, tmp_path: Path) -> None:
        loaded = CostMonitor.load_metrics(tmp_path / "nope.jsonl")
        assert loaded == []

    def test_load_metrics_n_larger_than_file(self, tmp_path: Path) -> None:
        mon = _make_monitor()
        fpath = tmp_path / "metrics.jsonl"
        mon.record(10.0)
        mon.flush_metrics(fpath)
        loaded = CostMonitor.load_metrics(fpath, n=1000)
        assert len(loaded) == 1

    def test_flush_creates_parent_dirs(self, tmp_path: Path) -> None:
        mon = _make_monitor()
        mon.record(5.0)
        fpath = tmp_path / "sub" / "dir" / "metrics.jsonl"
        count = mon.flush_metrics(fpath)
        assert count == 1
        assert fpath.exists()


# ── Thread safety ─────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_record_and_snapshot(self) -> None:
        mon = _make_monitor(window_size=5000)
        errors: list[Exception] = []
        stop = threading.Event()

        def writer() -> None:
            for i in range(200):
                try:
                    mon.record(float(i))
                except Exception as exc:
                    errors.append(exc)

        def reader() -> None:
            while not stop.is_set():
                try:
                    mon.snapshot()
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        reader_thread.join()

        assert errors == [], f"Thread safety errors: {errors}"
        snap = mon.snapshot()
        assert snap.sample_count > 0
        assert snap.success_count == 800  # 4 writers x 200

    def test_concurrent_record_and_record_error(self) -> None:
        mon = _make_monitor(window_size=5000)
        errors: list[Exception] = []

        def do_records() -> None:
            for i in range(100):
                try:
                    mon.record(float(i))
                except Exception as exc:
                    errors.append(exc)

        def do_errors() -> None:
            for _ in range(50):
                try:
                    mon.record_error(latency_ms=999.0)
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=do_records)
        t2 = threading.Thread(target=do_errors)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []
        snap = mon.snapshot()
        assert snap.success_count == 100
        assert snap.error_count == 50
        assert snap.error_rate == pytest.approx(50 / 150)


# ── Existing API backward compat ──────────────────────────────────


class TestBackwardCompat:
    def test_time_request_context_manager(self) -> None:
        mon = _make_monitor()
        with mon.time_request():
            time.sleep(0.01)
        snap = mon.snapshot()
        assert snap.sample_count == 1
        assert snap.latency_ms_p50 >= 10.0  # at least 10ms
        assert snap.success_count == 1

    def test_snapshot_returns_cost_metrics(self) -> None:
        mon = _make_monitor(cost_per_prediction=0.005)
        mon.record(10.0)
        snap = mon.snapshot()
        assert isinstance(snap, CostMetrics)
        assert snap.cost_per_prediction == 0.005

    def test_latency_threshold_still_works(self) -> None:
        mon = _make_monitor(thresholds={"latency_p99_ms": 50.0})
        mon.record(100.0)
        breaches = mon.check_thresholds()
        assert any("p99 latency" in b for b in breaches)
