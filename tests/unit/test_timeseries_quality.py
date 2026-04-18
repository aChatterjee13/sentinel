"""Unit tests for time series forecast quality metrics."""

from __future__ import annotations

import math

import numpy as np

from sentinel.domains.timeseries.decomposition import (
    DecompositionMonitor,
    decompose,
)
from sentinel.domains.timeseries.quality import (
    coverage,
    directional_accuracy,
    evaluate_forecast,
    interval_width,
    mape,
    mase,
    rmse,
    smape,
    winkler_score,
)


class TestForecastMetrics:
    def test_perfect_forecast_zero_error(self) -> None:
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert rmse(y, y) == 0.0
        assert mape(y, y) == 0.0
        assert smape(y, y) == 0.0

    def test_mase_against_naive_baseline(self) -> None:
        # Linear series with naive lag-1 forecast (first prediction matches t=0).
        # MAE = 5/6 ≈ 0.833, in-sample naive scale = 1.0 → MASE ≈ 0.833.
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y_naive = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        score = mase(y_true, y_naive, season=1)
        assert math.isclose(score, 5 / 6, abs_tol=0.01)

    def test_mase_returns_nan_for_short_series(self) -> None:
        score = mase([1.0], [1.0], season=1)
        assert math.isnan(score)

    def test_coverage_full(self) -> None:
        y = [1.0, 2.0, 3.0]
        lo = [0.0, 0.0, 0.0]
        hi = [10.0, 10.0, 10.0]
        assert coverage(y, lo, hi) == 1.0

    def test_coverage_partial(self) -> None:
        y = [1.0, 5.0, 12.0]
        lo = [0.0, 0.0, 0.0]
        hi = [10.0, 10.0, 10.0]
        assert math.isclose(coverage(y, lo, hi), 2 / 3)

    def test_interval_width(self) -> None:
        assert interval_width([0.0, 0.0], [10.0, 20.0]) == 15.0

    def test_winkler_score_no_misses(self) -> None:
        y = [5.0]
        lo = [0.0]
        hi = [10.0]
        score = winkler_score(y, lo, hi, alpha=0.05)
        assert score == 10.0  # just the width

    def test_winkler_score_with_miss(self) -> None:
        y = [15.0]
        lo = [0.0]
        hi = [10.0]
        score = winkler_score(y, lo, hi, alpha=0.05)
        assert score > 10.0  # penalty applied for being above the upper bound

    def test_directional_accuracy(self) -> None:
        y = [1.0, 2.0, 3.0, 2.0, 4.0]
        # Same direction at every step
        p = [1.0, 1.5, 2.5, 1.0, 5.0]
        score = directional_accuracy(y, p)
        assert score == 1.0

    def test_evaluate_forecast_bundle(self) -> None:
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = [1.1, 1.9, 3.2, 3.8, 5.1]
        result = evaluate_forecast(y, p, season=1)
        assert result.rmse < 0.5
        assert result.mase is not None
        assert result.directional_accuracy == 1.0


class TestDecomposition:
    def test_decompose_recovers_trend(self) -> None:
        # Linear trend + small noise
        x = np.arange(60.0)
        series = list(0.5 * x + np.sin(2 * np.pi * x / 7))
        d = decompose(series, period=7)
        # Trend should be roughly monotonically increasing
        assert d.trend[-1] > d.trend[5]

    def test_decomposition_monitor_detects_trend_shift(self) -> None:
        rng = np.random.default_rng(0)
        baseline = (0.1 * np.arange(50) + rng.normal(scale=0.05, size=50)).tolist()
        steeper = (1.0 * np.arange(50) + rng.normal(scale=0.05, size=50)).tolist()
        monitor = DecompositionMonitor(
            period=7,
            trend_slope_change_threshold=0.1,
            seasonal_amplitude_change_pct=20.0,
            residual_variance_increase_pct=30.0,
        )
        monitor.fit(baseline)
        report = monitor.evaluate(steeper)
        assert report["trend_alert"] is True
