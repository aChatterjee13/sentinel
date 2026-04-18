"""Tabular adapter — wraps the existing core drift + quality logic."""

from __future__ import annotations

from typing import Any

from sentinel.config.schema import SentinelConfig
from sentinel.domains.base import BaseDomainAdapter
from sentinel.observability.data_quality import DataQualityChecker
from sentinel.observability.drift import create_drift_detector
from sentinel.observability.drift.base import BaseDriftDetector


class TabularQualityMetric:
    """Lightweight named wrapper around standard tabular metrics."""

    def __init__(self, name: str, func):
        self.name = name
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return self.func(*args, **kwargs)


def _accuracy(y_true, y_pred) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred, strict=False)) / len(y_true)


def _mae(y_true, y_pred) -> float:
    if not y_true:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred, strict=False)) / len(y_true)


class TabularAdapter(BaseDomainAdapter):
    """Default adapter for tabular ML models.

    Wires up the core drift detectors and a simple quality metric set
    so the existing :class:`SentinelClient` behaviour is preserved when
    no domain is configured.
    """

    domain = "tabular"

    def __init__(self, config: SentinelConfig):
        super().__init__(config)

    def get_drift_detectors(self) -> list[BaseDriftDetector]:
        cfg = self.config.drift.data
        return [
            create_drift_detector(
                method=cfg.method,
                model_name=self.model_name,
                threshold=cfg.threshold,
            )
        ]

    def get_quality_metrics(self) -> list[TabularQualityMetric]:
        if self.config.model.type == "regression":
            return [TabularQualityMetric("mae", _mae)]
        return [TabularQualityMetric("accuracy", _accuracy)]

    def get_schema_validator(self) -> DataQualityChecker:
        return DataQualityChecker(self.config.data_quality, model_name=self.model_name)
