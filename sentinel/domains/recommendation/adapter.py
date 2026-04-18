"""Recommendation systems domain adapter."""

from __future__ import annotations

from typing import Any

from sentinel.config.schema import SentinelConfig
from sentinel.domains.base import BaseDomainAdapter
from sentinel.domains.recommendation.bias import group_fairness
from sentinel.domains.recommendation.drift import (
    ItemDistributionDriftDetector,
    UserSegmentDriftDetector,
)
from sentinel.domains.recommendation.quality import evaluate_recommendations
from sentinel.observability.drift.base import BaseDriftDetector


class _RecQuality:
    name = "recommendation_quality"

    def __call__(self, *args: Any, **kwargs: Any):
        return evaluate_recommendations(*args, **kwargs)


class _Fairness:
    name = "group_fairness"

    def __call__(self, *args: Any, **kwargs: Any):
        return group_fairness(*args, **kwargs)


class RecommendationAdapter(BaseDomainAdapter):
    """Adapter for recommendation models — collaborative, content, hybrid, neural."""

    domain = "recommendation"

    def __init__(self, config: SentinelConfig):
        super().__init__(config)
        drift_cfg = self.options.get("drift", {})
        item_cfg = drift_cfg.get("item_distribution", {})
        user_cfg = drift_cfg.get("user_distribution", {})
        self.long_tail_threshold = float(item_cfg.get("long_tail_ratio_threshold", 0.3))
        self.item_threshold = float(item_cfg.get("threshold", 0.1))
        if self.item_threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {self.item_threshold}"
            )
        self.user_threshold = float(user_cfg.get("threshold", 0.1))
        if self.user_threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {self.user_threshold}"
            )
        quality_cfg = self.options.get("quality", {})
        self.k = int(quality_cfg.get("k", 10))

    def get_drift_detectors(self) -> list[BaseDriftDetector]:
        return [
            ItemDistributionDriftDetector(
                model_name=self.model_name,
                threshold=self.item_threshold,
                long_tail_ratio_threshold=self.long_tail_threshold,
            ),
            UserSegmentDriftDetector(
                model_name=self.model_name,
                threshold=self.user_threshold,
            ),
        ]

    def get_quality_metrics(self) -> list[Any]:
        return [_RecQuality(), _Fairness()]

    def get_schema_validator(self) -> Any:
        from sentinel.observability.data_quality import DataQualityChecker

        return DataQualityChecker(self.config.data_quality, model_name=self.model_name)
