"""Lightweight STL-style decomposition monitoring without statsmodels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Decomposition:
    """Trend / seasonal / residual components of a time series."""

    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray

    @property
    def trend_slope(self) -> float:
        n = len(self.trend)
        if n < 2:
            return 0.0
        x = np.arange(n)
        return float(np.polyfit(x, self.trend, 1)[0])

    @property
    def seasonal_amplitude(self) -> float:
        if self.seasonal.size == 0:
            return 0.0
        return float(self.seasonal.max() - self.seasonal.min())

    @property
    def residual_variance(self) -> float:
        if self.residual.size == 0:
            return 0.0
        return float(self.residual.var())


def decompose(values: Sequence[float], period: int) -> Decomposition:
    """Naive additive decomposition into trend, seasonal, residual.

    Uses a centred moving average for trend extraction. This avoids the
    ``statsmodels`` dependency while remaining accurate enough for
    monitoring purposes. Customers who want full STL can plug in their
    own implementation behind the ``DecompositionMonitor`` interface.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    period = max(1, int(period))
    if n < period * 2:
        return Decomposition(
            trend=np.full_like(arr, arr.mean()),
            seasonal=np.zeros_like(arr),
            residual=arr - arr.mean(),
        )
    window = period if period % 2 == 1 else period + 1
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    trend = np.array(
        [padded[i : i + window].mean() for i in range(n)],
        dtype=float,
    )
    detrended = arr - trend
    seasonal = np.zeros_like(arr)
    for s in range(period):
        idx = np.arange(s, n, period)
        seasonal[idx] = detrended[idx].mean() if idx.size else 0.0
    seasonal -= seasonal.mean()
    residual = arr - trend - seasonal
    return Decomposition(trend=trend, seasonal=seasonal, residual=residual)


class DecompositionMonitor:
    """Track trend slope, seasonal amplitude, and residual variance shifts."""

    def __init__(
        self,
        period: int,
        trend_slope_change_threshold: float = 0.1,
        seasonal_amplitude_change_pct: float = 20.0,
        residual_variance_increase_pct: float = 30.0,
    ):
        self.period = period
        self.trend_slope_change_threshold = trend_slope_change_threshold
        self.seasonal_amplitude_change_pct = seasonal_amplitude_change_pct
        self.residual_variance_increase_pct = residual_variance_increase_pct
        self._reference: Decomposition | None = None

    def fit(self, reference: Sequence[float]) -> Decomposition:
        self._reference = decompose(reference, self.period)
        return self._reference

    def evaluate(self, current: Sequence[float]) -> dict[str, float | bool]:
        if self._reference is None:
            raise RuntimeError("DecompositionMonitor not fitted")
        cur = decompose(current, self.period)
        slope_delta = abs(cur.trend_slope - self._reference.trend_slope)
        ref_amp = self._reference.seasonal_amplitude or 1.0
        amp_pct = abs(cur.seasonal_amplitude - self._reference.seasonal_amplitude) / ref_amp * 100
        ref_var = self._reference.residual_variance or 1.0
        var_pct = (cur.residual_variance - self._reference.residual_variance) / ref_var * 100
        return {
            "trend_slope_change": slope_delta,
            "seasonal_amplitude_change_pct": amp_pct,
            "residual_variance_change_pct": var_pct,
            "trend_alert": slope_delta > self.trend_slope_change_threshold,
            "seasonal_alert": amp_pct > self.seasonal_amplitude_change_pct,
            "residual_alert": var_pct > self.residual_variance_increase_pct,
        }
