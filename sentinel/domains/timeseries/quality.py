"""Forecast quality metrics — MASE, MAPE, coverage, directional accuracy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastQualityResult:
    """Aggregated forecast quality metrics."""

    mase: float | None
    mape: float | None
    smape: float | None
    rmse: float
    coverage: float | None
    interval_width: float | None
    winkler: float | None
    directional_accuracy: float | None


def mase(y_true: Sequence[float], y_pred: Sequence[float], season: int = 1) -> float:
    """Mean absolute scaled error against a seasonal naive baseline."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.size <= season:
        return float("nan")
    naive_errors = np.abs(y_true_arr[season:] - y_true_arr[:-season])
    scale = naive_errors.mean()
    if scale == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)) / scale)


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute percentage error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = y_true_arr != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])))


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Symmetric mean absolute percentage error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true_arr[mask] - y_pred_arr[mask]) / denom[mask]))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def coverage(
    y_true: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    """Fraction of true values that fall inside the prediction interval."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.size == 0:
        return 0.0
    return float(((y >= lo) & (y <= hi)).mean())


def interval_width(lower: Sequence[float], upper: Sequence[float]) -> float:
    """Average width of the prediction intervals."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.size == 0:
        return 0.0
    return float(np.mean(hi - lo))


def winkler_score(
    y_true: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    alpha: float = 0.05,
) -> float:
    """Joint coverage and width quality. Lower is better."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    penalty = np.where(y < lo, (2 / alpha) * (lo - y), 0.0) + np.where(
        y > hi, (2 / alpha) * (y - hi), 0.0
    )
    return float(np.mean(width + penalty))


def directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Fraction of predictions that match the actual direction of change."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.size < 2:
        return float("nan")
    actual_diff = np.sign(np.diff(y))
    pred_diff = np.sign(np.diff(p))
    return float((actual_diff == pred_diff).mean())


def evaluate_forecast(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    season: int = 1,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    alpha: float = 0.05,
) -> ForecastQualityResult:
    """Compute the full forecast quality bundle in one shot."""
    return ForecastQualityResult(
        mase=mase(y_true, y_pred, season=season),
        mape=mape(y_true, y_pred),
        smape=smape(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        coverage=coverage(y_true, lower, upper)
        if lower is not None and upper is not None
        else None,
        interval_width=(
            interval_width(lower, upper) if lower is not None and upper is not None else None
        ),
        winkler=(
            winkler_score(y_true, lower, upper, alpha=alpha)
            if lower is not None and upper is not None
            else None
        ),
        directional_accuracy=directional_accuracy(y_true, y_pred),
    )
