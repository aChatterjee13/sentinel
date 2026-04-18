"""Model performance drift — track metric decay against the baseline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from sentinel.core.exceptions import DriftDetectionError
from sentinel.core.types import AlertSeverity, DriftReport
from sentinel.observability.drift.base import BaseDriftDetector


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = _precision(y_true, y_pred)
    r = _recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


METRIC_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": _accuracy,
    "precision": _precision,
    "recall": _recall,
    "f1": _f1,
    "mae": _mae,
    "rmse": _rmse,
}


class ModelPerformanceDriftDetector(BaseDriftDetector):
    """Detects performance decay against registered baselines.

    Tracks one or more metrics and compares the running value against the
    baseline captured at registration time. Drift is signalled when any
    metric drops by more than its configured threshold.
    """

    method_name = "model_performance"
    requires_actuals = True

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.05,
        metrics: list[str] | None = None,
        baseline: dict[str, float] | None = None,
        per_metric_thresholds: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold)
        self.metrics = metrics or ["accuracy"]
        self._baseline: dict[str, float] = baseline or {}
        self._per_metric_thresholds = per_metric_thresholds or {}
        for m in self.metrics:
            if m not in METRIC_REGISTRY:
                raise DriftDetectionError(
                    f"unknown metric '{m}'. Available: {list(METRIC_REGISTRY)}"
                )

    def fit(self, reference: tuple[np.ndarray, np.ndarray] | dict[str, float]) -> None:  # type: ignore[override]
        """Either compute baselines from (y_true, y_pred) or accept them directly."""
        if isinstance(reference, dict):
            self._baseline = dict(reference)
        else:
            y_true = np.asarray(reference[0])
            y_pred = np.asarray(reference[1])
            self._baseline = {m: METRIC_REGISTRY[m](y_true, y_pred) for m in self.metrics}
        self._fitted = True

    def detect(  # type: ignore[override]
        self,
        current: tuple[np.ndarray, np.ndarray],
        **_: Any,
    ) -> DriftReport:
        if not self._fitted:
            raise DriftDetectionError("model performance detector must be fit() first")
        y_true = np.asarray(current[0])
        y_pred = np.asarray(current[1])
        if len(y_true) != len(y_pred):
            raise DriftDetectionError("y_true and y_pred length mismatch")

        scores: dict[str, float] = {}
        deltas: dict[str, float] = {}
        drifted: list[str] = []
        max_delta = 0.0
        for m in self.metrics:
            current_value = METRIC_REGISTRY[m](y_true, y_pred)
            baseline_value = self._baseline.get(m, current_value)
            delta = baseline_value - current_value  # positive = degradation
            threshold = self._per_metric_thresholds.get(m, self.threshold)
            scores[m] = current_value
            deltas[m] = delta
            if delta > threshold:
                drifted.append(m)
            max_delta = max(max_delta, delta)

        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=bool(drifted),
            severity=self._severity_from_score(max_delta),
            test_statistic=max_delta,
            feature_scores=scores,
            drifted_features=drifted,
            metadata={"baselines": self._baseline, "deltas": deltas, "n_samples": len(y_true)},
        )

    def _severity_from_score(self, score: float) -> AlertSeverity:
        # Override: bigger drops = more severe
        if score < self.threshold:
            return AlertSeverity.INFO
        if score < self.threshold * 2:
            return AlertSeverity.WARNING
        if score < self.threshold * 3:
            return AlertSeverity.HIGH
        return AlertSeverity.CRITICAL
