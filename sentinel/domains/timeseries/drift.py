"""Time series drift detectors — calendar tests, ACF shifts, stationarity."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from sentinel.core.types import DriftReport
from sentinel.observability.drift.base import BaseDriftDetector

log = structlog.get_logger(__name__)


def _autocorr(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    a = values[:-lag] - values[:-lag].mean()
    b = values[lag:] - values[lag:].mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


class CalendarDriftDetector(BaseDriftDetector):
    """Compare current observations against the same calendar period in the reference."""

    method_name = "calendar_test"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.2,
        seasonality: int = 7,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self.seasonality = max(1, int(seasonality))
        self._reference_means: np.ndarray | None = None
        self._reference_stds: np.ndarray | None = None

    def fit(self, reference: Any) -> None:
        arr = np.asarray(reference, dtype=float).flatten()
        if arr.size == 0:
            raise ValueError("reference is empty")
        bucket_means = np.zeros(self.seasonality)
        bucket_stds = np.zeros(self.seasonality)
        for s in range(self.seasonality):
            slice_ = arr[s :: self.seasonality]
            bucket_means[s] = slice_.mean() if slice_.size else 0.0
            bucket_stds[s] = slice_.std() if slice_.size else 1.0
        self._reference_means = bucket_means
        self._reference_stds = np.where(bucket_stds == 0, 1.0, bucket_stds)
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted:
            raise RuntimeError("CalendarDriftDetector not fitted")
        arr = np.asarray(current, dtype=float).flatten()
        per_bucket: dict[str, float] = {}
        worst = 0.0
        drifted: list[str] = []
        if self._reference_means is None or self._reference_stds is None:
            raise RuntimeError(
                "CalendarDriftDetector not fitted — reference data is missing"
            )
        for s in range(self.seasonality):
            slice_ = arr[s :: self.seasonality]
            if slice_.size == 0:
                continue
            z = abs(slice_.mean() - self._reference_means[s]) / self._reference_stds[s]
            per_bucket[f"phase_{s}"] = float(z)
            if z > worst:
                worst = float(z)
            if z >= self.threshold:
                drifted.append(f"phase_{s}")
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=worst >= self.threshold,
            severity=self._severity_from_score(worst),
            test_statistic=worst,
            feature_scores=per_bucket,
            drifted_features=drifted,
            metadata={"seasonality": self.seasonality},
        )


class ACFShiftDetector(BaseDriftDetector):
    """Detect autocorrelation structure shifts in residuals or values."""

    method_name = "acf_shift"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.2,
        lags: tuple[int, ...] = (1, 7, 30),
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self.lags = tuple(int(lag) for lag in lags)
        self._reference_acf: dict[int, float] | None = None

    def fit(self, reference: Any) -> None:
        arr = np.asarray(reference, dtype=float).flatten()
        if arr.size == 0:
            raise ValueError("reference is empty")
        self._reference_acf = {lag: _autocorr(arr, lag) for lag in self.lags}
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted or self._reference_acf is None:
            raise RuntimeError("ACFShiftDetector not fitted")
        arr = np.asarray(current, dtype=float).flatten()
        per_lag: dict[str, float] = {}
        worst = 0.0
        drifted: list[str] = []
        for lag, ref_value in self._reference_acf.items():
            if arr.size <= lag:
                continue
            current_acf = _autocorr(arr, lag)
            shift = abs(current_acf - ref_value)
            per_lag[f"lag_{lag}"] = float(shift)
            if shift > worst:
                worst = float(shift)
            if shift >= self.threshold:
                drifted.append(f"lag_{lag}")
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=worst >= self.threshold,
            severity=self._severity_from_score(worst),
            test_statistic=worst,
            feature_scores=per_lag,
            drifted_features=drifted,
            metadata={"lags": list(self.lags)},
        )


class StationarityDriftDetector(BaseDriftDetector):
    """Track mean and variance stability across sliding windows."""

    method_name = "stationarity"

    def __init__(self, model_name: str, threshold: float = 0.2, **kwargs: Any):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self._ref_mean: float | None = None
        self._ref_std: float | None = None

    def fit(self, reference: Any) -> None:
        arr = np.asarray(reference, dtype=float).flatten()
        if arr.size == 0:
            raise ValueError("reference is empty")
        self._ref_mean = float(arr.mean())
        self._ref_std = float(arr.std()) or 1.0
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted:
            raise RuntimeError("StationarityDriftDetector not fitted")
        arr = np.asarray(current, dtype=float).flatten()
        if self._ref_mean is None or self._ref_std is None:
            raise RuntimeError(
                "StationarityDriftDetector not fitted — reference data is missing"
            )
        mean_shift = abs(arr.mean() - self._ref_mean) / self._ref_std
        var_ratio = float(arr.std() / self._ref_std) if self._ref_std else 1.0
        var_shift = abs(np.log(var_ratio + 1e-9))
        worst = max(mean_shift, var_shift)
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=worst >= self.threshold,
            severity=self._severity_from_score(worst),
            test_statistic=worst,
            feature_scores={"mean_shift": mean_shift, "variance_shift": var_shift},
            drifted_features=[
                k
                for k, v in {"mean_shift": mean_shift, "variance_shift": var_shift}.items()
                if v >= self.threshold
            ],
            metadata={"ref_mean": self._ref_mean, "ref_std": self._ref_std},
        )
