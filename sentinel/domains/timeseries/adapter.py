"""Time series domain adapter."""

from __future__ import annotations

from typing import Any

from sentinel.config.schema import SentinelConfig
from sentinel.domains.base import BaseDomainAdapter
from sentinel.domains.timeseries.decomposition import DecompositionMonitor
from sentinel.domains.timeseries.drift import (
    ACFShiftDetector,
    CalendarDriftDetector,
    StationarityDriftDetector,
)
from sentinel.domains.timeseries.quality import evaluate_forecast
from sentinel.observability.drift.base import BaseDriftDetector


class _ForecastQuality:
    """Named wrapper around the time series quality evaluator."""

    name = "forecast_quality"

    def __init__(self, season: int = 1):
        self.season = season

    def __call__(self, y_true, y_pred, **kwargs: Any):
        return evaluate_forecast(y_true, y_pred, season=self.season, **kwargs)


class TimeSeriesAdapter(BaseDomainAdapter):
    """Adapter for forecasting models — calendar drift, decomposition, MASE."""

    domain = "timeseries"

    def __init__(self, config: SentinelConfig):
        super().__init__(config)
        self.frequency = self.options.get("frequency", "daily")
        self.seasonality_periods = self.options.get("seasonality_periods", [7])
        self.primary_period = int(self.seasonality_periods[0]) if self.seasonality_periods else 7
        drift_cfg = self.options.get("drift", {})
        self.drift_method = drift_cfg.get("method", "calendar_test")
        self.threshold = float(drift_cfg.get("threshold", config.drift.data.threshold))
        if self.threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")
        decomp_cfg = self.options.get("decomposition_monitoring", {})
        self.decomposition = DecompositionMonitor(
            period=self.primary_period,
            trend_slope_change_threshold=float(decomp_cfg.get("trend_slope_change_threshold", 0.1)),
            seasonal_amplitude_change_pct=float(
                decomp_cfg.get("seasonal_amplitude_change_pct", 20.0)
            ),
            residual_variance_increase_pct=float(
                decomp_cfg.get("residual_variance_increase_pct", 30.0)
            ),
        )

    def get_drift_detectors(self) -> list[BaseDriftDetector]:
        detectors: list[BaseDriftDetector] = []
        if self.drift_method in {"calendar_test", "all"}:
            detectors.append(
                CalendarDriftDetector(
                    model_name=self.model_name,
                    threshold=self.threshold,
                    seasonality=self.primary_period,
                )
            )
        if self.drift_method in {"acf_shift", "temporal_covariate", "all"}:
            detectors.append(
                ACFShiftDetector(
                    model_name=self.model_name,
                    threshold=self.threshold,
                    lags=tuple(int(p) for p in self.seasonality_periods),
                )
            )
        if self.drift_method in {"stationarity", "all"}:
            detectors.append(
                StationarityDriftDetector(
                    model_name=self.model_name,
                    threshold=self.threshold,
                )
            )
        if not detectors:
            detectors.append(
                CalendarDriftDetector(
                    model_name=self.model_name,
                    threshold=self.threshold,
                    seasonality=self.primary_period,
                )
            )
        return detectors

    def get_quality_metrics(self) -> list[_ForecastQuality]:
        return [_ForecastQuality(season=self.primary_period)]

    def get_schema_validator(self) -> Any:
        # Time series uses the standard tabular validator on a windowed feature row.
        from sentinel.observability.data_quality import DataQualityChecker

        return DataQualityChecker(self.config.data_quality, model_name=self.model_name)

    def get_decomposition_monitor(self) -> DecompositionMonitor:
        return self.decomposition
