"""Tests for drift detection windowing, NaN handling, and report persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sentinel import SentinelClient
from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)
from sentinel.core.types import AlertSeverity, DriftReport, PredictionRecord
from sentinel.observability.drift.data_drift import (
    JSDivergenceDriftDetector,
    KSDriftDetector,
    PSIDriftDetector,
    WassersteinDriftDetector,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _make_config(
    *,
    window: str = "7d",
    reference: str = "baseline",
    audit_path: str = "./audit/",
) -> SentinelConfig:
    return SentinelConfig(
        model=ModelConfig(name="test_model", domain="tabular"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window=window, reference=reference),
        ),
        alerts=AlertsConfig(),
        audit=AuditConfig(storage="local", path=audit_path),
    )


# ── _parse_window ─────────────────────────────────────────────────


class TestParseWindow:
    def test_days(self) -> None:
        result = SentinelClient._parse_window("7d")
        assert result == timedelta(days=7)

    def test_hours(self) -> None:
        result = SentinelClient._parse_window("24h")
        assert result == timedelta(hours=24)

    def test_minutes(self) -> None:
        result = SentinelClient._parse_window("30m")
        assert result == timedelta(minutes=30)

    def test_count_only(self) -> None:
        result = SentinelClient._parse_window("1000")
        assert result == 1000

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot parse"):
            SentinelClient._parse_window("7x")


# ── Time-based windowing ──────────────────────────────────────────


class TestTimeBasedWindowing:
    def test_filters_old_records(self, tmp_path: str) -> None:
        """Records older than the window period should be excluded."""
        config = _make_config(window="7d", audit_path=str(tmp_path / "audit"))
        client = SentinelClient(config)
        rng = np.random.default_rng(0)
        client.fit_baseline(rng.normal(size=(200, 1)))

        now = datetime.now(timezone.utc)

        # Add old records (10 days ago) — should be filtered out
        for v in rng.normal(size=50):
            rec = PredictionRecord(
                model_name="test_model",
                features={"x": float(v)},
                prediction=0,
                timestamp=now - timedelta(days=10),
            )
            with client._lock:
                client._prediction_buffer.append(rec)

        # Add recent records (1 day ago) — should be kept
        for v in rng.normal(size=100):
            rec = PredictionRecord(
                model_name="test_model",
                features={"x": float(v)},
                prediction=0,
                timestamp=now - timedelta(days=1),
            )
            with client._lock:
                client._prediction_buffer.append(rec)

        # _extract_window should only return the recent 100
        with client._lock:
            windowed = client._extract_window()
        assert len(windowed) == 100

    def test_all_records_within_window(self, tmp_path: str) -> None:
        """If all records are within the window, all should be included."""
        config = _make_config(window="7d", audit_path=str(tmp_path / "audit"))
        client = SentinelClient(config)

        now = datetime.now(timezone.utc)
        for i in range(50):
            rec = PredictionRecord(
                model_name="test_model",
                features={"x": float(i)},
                prediction=0,
                timestamp=now - timedelta(hours=i),
            )
            with client._lock:
                client._prediction_buffer.append(rec)

        with client._lock:
            windowed = client._extract_window()
        assert len(windowed) == 50


# ── Count-based windowing ─────────────────────────────────────────


class TestCountBasedWindowing:
    def test_takes_last_n_records(self, tmp_path: str) -> None:
        config = _make_config(window="50", audit_path=str(tmp_path / "audit"))
        client = SentinelClient(config)

        for i in range(200):
            rec = PredictionRecord(
                model_name="test_model",
                features={"x": float(i)},
                prediction=0,
            )
            with client._lock:
                client._prediction_buffer.append(rec)

        with client._lock:
            windowed = client._extract_window()
        assert len(windowed) == 50
        # Should be the last 50 records
        assert float(windowed[0].features["x"]) == 150.0

    def test_fewer_than_n_returns_all(self, tmp_path: str) -> None:
        config = _make_config(window="1000", audit_path=str(tmp_path / "audit"))
        client = SentinelClient(config)

        for i in range(30):
            rec = PredictionRecord(
                model_name="test_model",
                features={"x": float(i)},
                prediction=0,
            )
            with client._lock:
                client._prediction_buffer.append(rec)

        with client._lock:
            windowed = client._extract_window()
        assert len(windowed) == 30


# ── Previous-window reference mode ────────────────────────────────


class TestPreviousWindowReference:
    def test_switches_baseline(self, tmp_path: str) -> None:
        """In previous_window mode, the baseline should change each check."""
        config = _make_config(
            window="1000",
            reference="previous_window",
            audit_path=str(tmp_path / "audit"),
        )
        client = SentinelClient(config)

        rng = np.random.default_rng(42)

        # First batch — will become baseline (auto-fit since not fitted)
        for v in rng.normal(loc=0.0, size=200):
            client.log_prediction(features={"x": float(v)}, prediction=0)
        report1 = client.check_drift()
        assert not report1.is_drifted

        # Clear buffer and add shifted data
        client.clear_buffer()
        for v in rng.normal(loc=5.0, scale=2.0, size=200):
            client.log_prediction(features={"x": float(v)}, prediction=0)

        # Second check should use first batch as reference and detect drift
        report2 = client.check_drift()
        assert report2.is_drifted

    def test_previous_window_data_updates(self, tmp_path: str) -> None:
        config = _make_config(
            window="1000",
            reference="previous_window",
            audit_path=str(tmp_path / "audit"),
        )
        client = SentinelClient(config)
        assert client._previous_window_data is None

        rng = np.random.default_rng(0)
        for v in rng.normal(size=100):
            client.log_prediction(features={"x": float(v)}, prediction=0)
        client.check_drift()

        # After check, previous_window_data should be populated
        assert client._previous_window_data is not None
        assert client._previous_window_data.shape[0] == 100


# ── NaN handling ──────────────────────────────────────────────────


class TestNaNHandling:
    def test_nan_rows_dropped_psi(self, rng: np.random.Generator) -> None:
        ref = rng.normal(size=(200, 2))
        cur = rng.normal(size=(200, 2))
        # Inject NaN in some rows
        cur[0, 0] = np.nan
        cur[10, 1] = np.nan
        cur[50, :] = np.nan

        d = PSIDriftDetector(model_name="m", threshold=0.2)
        d.fit(ref)
        report = d.detect(cur)
        # Should complete without error; NaN rows dropped
        assert isinstance(report, DriftReport)

    def test_nan_rows_dropped_ks(self, rng: np.random.Generator) -> None:
        ref = rng.normal(size=(200, 2))
        cur = rng.normal(size=(200, 2))
        cur[5, 0] = np.nan
        cur[15, 1] = np.nan

        d = KSDriftDetector(model_name="m", threshold=0.05)
        d.fit(ref)
        report = d.detect(cur)
        assert isinstance(report, DriftReport)

    def test_nan_rows_dropped_js(self, rng: np.random.Generator) -> None:
        ref = rng.normal(size=(200, 2))
        cur = rng.normal(size=(200, 2))
        cur[:3, :] = np.nan

        d = JSDivergenceDriftDetector(model_name="m", threshold=0.1)
        d.fit(ref)
        report = d.detect(cur)
        assert isinstance(report, DriftReport)

    def test_nan_rows_dropped_wasserstein(self, rng: np.random.Generator) -> None:
        ref = rng.normal(size=(200, 2))
        cur = rng.normal(size=(200, 2))
        cur[0, :] = np.nan

        d = WassersteinDriftDetector(model_name="m", threshold=0.5)
        d.fit(ref)
        report = d.detect(cur)
        assert isinstance(report, DriftReport)

    def test_all_nan_returns_safe_report(self) -> None:
        """When all rows are NaN the detector should return a non-drifted report."""
        ref = np.ones((100, 2))
        cur = np.full((50, 2), np.nan)

        d = PSIDriftDetector(model_name="m", threshold=0.2)
        d.fit(ref)
        report = d.detect(cur)
        assert not report.is_drifted
        assert report.severity == AlertSeverity.INFO
        assert "warning" in report.metadata

    def test_nan_in_reference_dropped(self, rng: np.random.Generator) -> None:
        """NaN in reference data should be dropped during fit."""
        ref = rng.normal(size=(200, 2))
        ref[0, :] = np.nan
        ref[10, 0] = np.nan
        cur = rng.normal(size=(200, 2))

        d = PSIDriftDetector(model_name="m", threshold=0.2)
        d.fit(ref)
        # Reference should have NaN rows removed
        assert d._reference.shape[0] == 198
        report = d.detect(cur)
        assert isinstance(report, DriftReport)


# ── DriftReport.window field ──────────────────────────────────────


class TestDriftReportWindow:
    def test_window_populated_on_buffer_check(self, tmp_path: str) -> None:
        config = _make_config(window="7d", audit_path=str(tmp_path / "audit"))
        client = SentinelClient(config)
        rng = np.random.default_rng(0)
        client.fit_baseline(rng.normal(size=(200, 1)))
        for v in rng.normal(size=200):
            client.log_prediction(features={"x": float(v)}, prediction=0)
        report = client.check_drift()
        assert report.window == "7d"

    def test_window_populated_on_explicit_data(self, tmp_path: str) -> None:
        config = _make_config(window="24h", audit_path=str(tmp_path / "audit"))
        client = SentinelClient(config)
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(200, 2))
        cur = rng.normal(size=(200, 2))
        client.fit_baseline(ref)
        report = client.check_drift(cur)
        assert report.window == "24h"


# ── Drift history persistence ─────────────────────────────────────


class TestDriftHistoryPersistence:
    def test_persist_and_read_back(self, tmp_path: str) -> None:
        audit_dir = str(tmp_path / "audit")
        config = _make_config(audit_path=audit_dir)
        client = SentinelClient(config)

        rng = np.random.default_rng(0)
        client.fit_baseline(rng.normal(size=(200, 1)))
        for v in rng.normal(size=200):
            client.log_prediction(features={"x": float(v)}, prediction=0)

        report = client.check_drift()
        history = client.get_drift_history(n=10)
        assert len(history) == 1
        assert history[0].method == report.method
        assert history[0].model_name == "test_model"

    def test_multiple_reports_persist(self, tmp_path: str) -> None:
        audit_dir = str(tmp_path / "audit")
        config = _make_config(audit_path=audit_dir)
        client = SentinelClient(config)

        rng = np.random.default_rng(0)
        ref = rng.normal(size=(200, 1))
        client.fit_baseline(ref)

        for _ in range(3):
            client.clear_buffer()
            for v in rng.normal(size=100):
                client.log_prediction(features={"x": float(v)}, prediction=0)
            client.check_drift()

        history = client.get_drift_history(n=10)
        assert len(history) == 3

    def test_get_drift_history_limits_count(self, tmp_path: str) -> None:
        audit_dir = str(tmp_path / "audit")
        config = _make_config(audit_path=audit_dir)
        client = SentinelClient(config)

        rng = np.random.default_rng(0)
        ref = rng.normal(size=(200, 1))
        client.fit_baseline(ref)

        for _ in range(5):
            client.clear_buffer()
            for v in rng.normal(size=100):
                client.log_prediction(features={"x": float(v)}, prediction=0)
            client.check_drift()

        history = client.get_drift_history(n=2)
        assert len(history) == 2

    def test_get_drift_history_empty(self, tmp_path: str) -> None:
        audit_dir = str(tmp_path / "audit")
        config = _make_config(audit_path=audit_dir)
        client = SentinelClient(config)
        history = client.get_drift_history()
        assert history == []

    def test_history_returns_newest_first(self, tmp_path: str) -> None:
        audit_dir = str(tmp_path / "audit")
        config = _make_config(audit_path=audit_dir)
        client = SentinelClient(config)

        rng = np.random.default_rng(0)
        ref = rng.normal(size=(200, 1))
        client.fit_baseline(ref)

        reports = []
        for _ in range(3):
            client.clear_buffer()
            for v in rng.normal(size=100):
                client.log_prediction(features={"x": float(v)}, prediction=0)
            reports.append(client.check_drift())

        history = client.get_drift_history(n=3)
        # Newest first — last check should be first in history
        assert history[0].report_id == reports[-1].report_id

    def test_drift_history_file_created(self, tmp_path: str) -> None:
        audit_dir = str(tmp_path / "audit")
        config = _make_config(audit_path=audit_dir)
        client = SentinelClient(config)

        rng = np.random.default_rng(0)
        client.fit_baseline(rng.normal(size=(200, 1)))
        for v in rng.normal(size=100):
            client.log_prediction(features={"x": float(v)}, prediction=0)
        client.check_drift()

        history_path = tmp_path / "audit" / "drift_history" / "test_model.jsonl"
        assert history_path.exists()
        lines = history_path.read_text().strip().splitlines()
        assert len(lines) == 1
