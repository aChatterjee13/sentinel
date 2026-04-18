"""Tests for FeatureHealthMonitor."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from sentinel.config.schema import FeatureHealthConfig
from sentinel.core.types import AlertSeverity, FeatureHealthReport
from sentinel.observability.drift.base import BaseDriftDetector
from sentinel.observability.feature_health import FeatureHealthMonitor

# ── Helpers ────────────────────────────────────────────────────────


class _StubDetector(BaseDriftDetector):
    """Detector that returns preset drift results."""

    def __init__(
        self,
        feature_scores: dict[str, float] | None = None,
        drifted: list[str] | None = None,
    ) -> None:
        super().__init__(model_name="test", threshold=0.2)
        self._scores = feature_scores or {}
        self._drifted = drifted or []
        self._fitted = True

    def fit(self, reference_data: Any) -> None:
        self._fitted = True

    def detect(self, current_data: Any) -> Any:
        from sentinel.core.types import DriftReport

        return DriftReport(
            model_name="test",
            method="stub",
            is_drifted=bool(self._drifted),
            severity=AlertSeverity.HIGH if self._drifted else AlertSeverity.INFO,
            test_statistic=max(self._scores.values()) if self._scores else 0.0,
            feature_scores=self._scores,
            drifted_features=self._drifted,
        )

    def reset(self) -> None:
        pass


@pytest.fixture()
def config() -> FeatureHealthConfig:
    return FeatureHealthConfig(
        importance_method="builtin",
        alert_on_top_n_drift=2,
        recalculate_importance="weekly",
    )


# ── evaluate() ─────────────────────────────────────────────────────


class TestEvaluate:
    def test_no_drift_all_info(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        detector = _StubDetector(
            feature_scores={"a": 0.05, "b": 0.03},
            drifted=[],
        )
        report = monitor.evaluate(detector, np.zeros((10, 2)))

        assert isinstance(report, FeatureHealthReport)
        assert report.model_name == "m"
        assert len(report.features) == 2
        assert all(f.severity == AlertSeverity.INFO for f in report.features)
        assert report.top_n_drifted == []

    def test_drift_with_importances(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(
            config,
            model_name="m",
            importances={"a": 0.8, "b": 0.1},
        )
        detector = _StubDetector(
            feature_scores={"a": 0.25, "b": 0.22},
            drifted=["a", "b"],
        )
        report = monitor.evaluate(detector, np.zeros((10, 2)))

        assert len(report.features) == 2
        # 'a' has importance 0.8 → CRITICAL
        a_feat = next(f for f in report.features if f.name == "a")
        assert a_feat.severity == AlertSeverity.CRITICAL
        assert a_feat.is_drifted is True
        # 'b' has importance 0.1 → WARNING (drifted but low importance)
        b_feat = next(f for f in report.features if f.name == "b")
        assert b_feat.severity == AlertSeverity.WARNING

    def test_top_n_respects_config(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(
            config,
            model_name="m",
            importances={"a": 0.6, "b": 0.3, "c": 0.1},
        )
        detector = _StubDetector(
            feature_scores={"a": 0.3, "b": 0.25, "c": 0.21},
            drifted=["a", "b", "c"],
        )
        report = monitor.evaluate(detector, np.zeros((10, 3)))

        # alert_on_top_n_drift=2, so only top 2 by weighted score
        assert len(report.top_n_drifted) == 2
        assert "a" in report.top_n_drifted

    def test_null_rates_populated(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        detector = _StubDetector(feature_scores={"x": 0.1}, drifted=[])
        null_rates = {"x": 0.15}
        report = monitor.evaluate(detector, np.zeros((10, 1)), null_rates=null_rates)

        assert report.features[0].null_rate == 0.15

    def test_missing_importances_default_zero(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        detector = _StubDetector(
            feature_scores={"a": 0.25},
            drifted=["a"],
        )
        report = monitor.evaluate(detector, np.zeros((10, 1)))

        assert report.features[0].importance == 0.0
        assert report.features[0].severity == AlertSeverity.WARNING

    def test_summary_property(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        detector = _StubDetector(
            feature_scores={"a": 0.3, "b": 0.05},
            drifted=["a"],
        )
        report = monitor.evaluate(detector, np.zeros((10, 2)))
        assert "1/2 features drifted" in report.summary


# ── set_importances() ──────────────────────────────────────────────


class TestSetImportances:
    def test_update_importances(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        monitor.set_importances({"x": 0.7, "y": 0.3})
        assert monitor._importances == {"x": 0.7, "y": 0.3}

    def test_overwrite_previous(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m", importances={"x": 0.5})
        monitor.set_importances({"x": 0.9})
        assert monitor._importances["x"] == 0.9


# ── severity mapping ──────────────────────────────────────────────


class TestSeverity:
    def test_not_drifted_is_info(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        assert monitor._severity(False, 0.99) == AlertSeverity.INFO

    def test_high_importance_critical(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        assert monitor._severity(True, 0.7) == AlertSeverity.CRITICAL

    def test_medium_importance_high(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        assert monitor._severity(True, 0.3) == AlertSeverity.HIGH

    def test_low_importance_warning(self, config: FeatureHealthConfig) -> None:
        monitor = FeatureHealthMonitor(config, model_name="m")
        assert monitor._severity(True, 0.1) == AlertSeverity.WARNING


# ── static helpers ─────────────────────────────────────────────────


class TestComputeNullRates:
    def test_empty_returns_empty(self) -> None:
        assert FeatureHealthMonitor.compute_null_rates([]) == {}

    def test_no_nulls(self) -> None:
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        rates = FeatureHealthMonitor.compute_null_rates(rows)
        assert rates["a"] == 0.0
        assert rates["b"] == 0.0

    def test_some_nulls(self) -> None:
        rows = [{"a": None, "b": 2}, {"a": 3, "b": None}]
        rates = FeatureHealthMonitor.compute_null_rates(rows)
        assert rates["a"] == 0.5
        assert rates["b"] == 0.5


class TestComputeBuiltinImportance:
    def test_feature_importances(self) -> None:
        model = MagicMock()
        model.feature_importances_ = np.array([0.6, 0.4])
        result = FeatureHealthMonitor.compute_builtin_importance(model, ["a", "b"])
        assert abs(result["a"] - 0.6) < 1e-6
        assert abs(result["b"] - 0.4) < 1e-6

    def test_coef_attribute(self) -> None:
        model = MagicMock(spec=[])
        model.coef_ = np.array([[0.3, -0.7]])
        result = FeatureHealthMonitor.compute_builtin_importance(model, ["a", "b"])
        assert result["a"] > 0
        assert result["b"] > 0

    def test_no_importances_returns_empty(self) -> None:
        model = MagicMock(spec=[])
        result = FeatureHealthMonitor.compute_builtin_importance(model, ["a"])
        assert result == {}
