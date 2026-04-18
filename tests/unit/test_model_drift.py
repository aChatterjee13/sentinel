"""Unit tests for ModelPerformanceDriftDetector."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.exceptions import DriftDetectionError
from sentinel.core.types import AlertSeverity
from sentinel.observability.drift.model_drift import (
    METRIC_REGISTRY,
    ModelPerformanceDriftDetector,
    _accuracy,
    _f1,
    _mae,
    _precision,
    _recall,
    _rmse,
)


class TestMetricFunctions:
    """Unit tests for the standalone metric helper functions."""

    def test_accuracy(self) -> None:
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        assert _accuracy(y_true, y_pred) == pytest.approx(0.8)

    def test_precision(self) -> None:
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        # TP=2, FP=1 → precision = 2/3
        assert _precision(y_true, y_pred) == pytest.approx(2 / 3)

    def test_precision_no_positives(self) -> None:
        y_true = np.array([0, 0])
        y_pred = np.array([0, 0])
        assert _precision(y_true, y_pred) == 0.0

    def test_recall(self) -> None:
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        # TP=2, FN=1 → recall = 2/3
        assert _recall(y_true, y_pred) == pytest.approx(2 / 3)

    def test_recall_no_actual_positives(self) -> None:
        y_true = np.array([0, 0])
        y_pred = np.array([1, 0])
        assert _recall(y_true, y_pred) == 0.0

    def test_f1(self) -> None:
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0])
        assert _f1(y_true, y_pred) == pytest.approx(1.0)

    def test_f1_zero(self) -> None:
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])
        assert _f1(y_true, y_pred) == 0.0

    def test_mae(self) -> None:
        y_true = np.array([3.0, 5.0, 2.0])
        y_pred = np.array([2.5, 5.0, 3.0])
        assert _mae(y_true, y_pred) == pytest.approx(0.5)

    def test_rmse(self) -> None:
        y_true = np.array([3.0, 5.0])
        y_pred = np.array([3.0, 5.0])
        assert _rmse(y_true, y_pred) == pytest.approx(0.0)


class TestModelPerformanceDriftDetector:
    """Tests for ModelPerformanceDriftDetector."""

    def _make_detector(self, **kwargs):
        defaults = {"model_name": "test_model", "metrics": ["accuracy"]}
        defaults.update(kwargs)
        return ModelPerformanceDriftDetector(**defaults)

    # ── fit and detect ────────────────────────────────────────────

    def test_fit_from_arrays(self) -> None:
        det = self._make_detector()
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0])
        det.fit((y_true, y_pred))
        assert det.is_fitted()
        assert "accuracy" in det._baseline

    def test_fit_from_dict(self) -> None:
        det = self._make_detector()
        det.fit({"accuracy": 0.95})
        assert det.is_fitted()
        assert det._baseline["accuracy"] == 0.95

    def test_detect_drift_when_accuracy_drops(self) -> None:
        det = self._make_detector(threshold=0.05)
        det.fit({"accuracy": 0.90})
        # accuracy = 7/10 = 0.7, baseline = 0.9, delta = 0.2 > threshold 0.05
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 0])
        report = det.detect((y_true, y_pred))
        assert report.is_drifted

    def test_detect_stable_performance(self) -> None:
        det = self._make_detector(threshold=0.1)
        det.fit({"accuracy": 0.80})
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0])
        report = det.detect((y_true, y_pred))
        # accuracy = 1.0, baseline = 0.8 → delta = -0.2 (improvement)
        assert not report.is_drifted
        assert report.severity == AlertSeverity.INFO

    # ── performance decline ───────────────────────────────────────

    def test_detect_performance_decline(self) -> None:
        det = self._make_detector(threshold=0.05, metrics=["accuracy"])
        det.fit({"accuracy": 0.95})
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        report = det.detect((y_true, y_pred))
        assert report.is_drifted
        assert "accuracy" in report.drifted_features

    def test_severity_scales_with_delta(self) -> None:
        det = self._make_detector(threshold=0.05)
        det.fit({"accuracy": 0.95})
        # accuracy = 0.0 → delta = 0.95, which is >3x threshold → CRITICAL
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        report = det.detect((y_true, y_pred))
        assert report.severity == AlertSeverity.CRITICAL

    # ── multiple metrics ──────────────────────────────────────────

    def test_multiple_metrics(self) -> None:
        det = self._make_detector(metrics=["accuracy", "f1"], threshold=0.05)
        det.fit({"accuracy": 0.95, "f1": 0.90})
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0])
        report = det.detect((y_true, y_pred))
        assert "accuracy" in report.feature_scores
        assert "f1" in report.feature_scores

    def test_per_metric_thresholds(self) -> None:
        det = self._make_detector(
            metrics=["accuracy", "f1"],
            threshold=0.01,
            per_metric_thresholds={"f1": 0.5},
        )
        det.fit({"accuracy": 0.90, "f1": 0.90})
        # accuracy = 0.8 → delta = 0.1 > 0.01 → drifted
        # f1 might be fine with threshold 0.5
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        report = det.detect((y_true, y_pred))
        assert "accuracy" in report.drifted_features

    # ── threshold configuration ───────────────────────────────────

    def test_loose_threshold_no_drift(self) -> None:
        det = self._make_detector(threshold=0.5)
        det.fit({"accuracy": 0.90})
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        report = det.detect((y_true, y_pred))
        # accuracy = 0.8, delta = 0.1 < threshold 0.5
        assert not report.is_drifted

    # ── error handling ────────────────────────────────────────────

    def test_detect_before_fit_raises(self) -> None:
        det = self._make_detector()
        with pytest.raises(DriftDetectionError, match="fit"):
            det.detect((np.array([1]), np.array([1])))

    def test_length_mismatch_raises(self) -> None:
        det = self._make_detector()
        det.fit({"accuracy": 0.9})
        with pytest.raises(DriftDetectionError, match="length mismatch"):
            det.detect((np.array([1, 0]), np.array([1])))

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(DriftDetectionError, match="unknown metric"):
            self._make_detector(metrics=["nonexistent_metric"])

    # ── report metadata ───────────────────────────────────────────

    def test_report_contains_metadata(self) -> None:
        det = self._make_detector()
        det.fit({"accuracy": 0.90})
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 0])
        report = det.detect((y_true, y_pred))
        assert "baselines" in report.metadata
        assert "deltas" in report.metadata
        assert report.metadata["n_samples"] == 3
        assert report.method == "model_performance"
        assert report.model_name == "test_model"

    def test_metric_registry_contains_all_expected(self) -> None:
        expected = {"accuracy", "precision", "recall", "f1", "mae", "rmse"}
        assert set(METRIC_REGISTRY.keys()) == expected

    # ── reset ─────────────────────────────────────────────────────

    def test_reset_clears_fitted(self) -> None:
        det = self._make_detector()
        det.fit({"accuracy": 0.9})
        assert det.is_fitted()
        det.reset()
        assert not det.is_fitted()
