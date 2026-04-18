"""Recommendation system drift detectors — item / user distribution + cold start."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from sentinel.core.types import DriftReport
from sentinel.domains.recommendation.quality import gini_coefficient
from sentinel.observability.drift.base import BaseDriftDetector


def _normalised(counter: Counter) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {str(k): v / total for k, v in counter.items()}


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    import math

    keys = set(p) | set(q)
    js = 0.0
    for k in keys:
        pv = p.get(k, 1e-12)
        qv = q.get(k, 1e-12)
        m = 0.5 * (pv + qv)
        if pv > 0:
            js += 0.5 * pv * math.log2(pv / m)
        if qv > 0:
            js += 0.5 * qv * math.log2(qv / m)
    return float(max(0.0, min(1.0, js)))


class ItemDistributionDriftDetector(BaseDriftDetector):
    """Detect shifts in the item distribution and rising long-tail concentration."""

    method_name = "item_distribution"

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.1,
        long_tail_ratio_threshold: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self.long_tail_ratio_threshold = long_tail_ratio_threshold
        self._reference_dist: dict[str, float] | None = None
        self._reference_gini: float | None = None

    def fit(self, reference: Any) -> None:
        items = _flatten(reference)
        counter = Counter(items)
        self._reference_dist = _normalised(counter)
        self._reference_gini = gini_coefficient(items)
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted or self._reference_dist is None:
            raise RuntimeError("ItemDistributionDriftDetector not fitted")
        items = _flatten(current)
        cur_counter = Counter(items)
        cur_dist = _normalised(cur_counter)
        js = _js_divergence(self._reference_dist, cur_dist)
        cur_gini = gini_coefficient(items)
        gini_shift = abs(cur_gini - (self._reference_gini or 0.0))
        worst = max(js, gini_shift)
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=worst >= self.threshold,
            severity=self._severity_from_score(worst),
            test_statistic=worst,
            feature_scores={
                "js_divergence": js,
                "gini_shift": gini_shift,
                "current_gini": cur_gini,
            },
            drifted_features=["item_distribution"] if worst >= self.threshold else [],
            metadata={"reference_gini": self._reference_gini},
        )


class UserSegmentDriftDetector(BaseDriftDetector):
    """Detect shifts in the user segment composition."""

    method_name = "user_segment"

    def __init__(self, model_name: str, threshold: float = 0.1, **kwargs: Any):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self._reference: dict[str, float] | None = None

    def fit(self, reference: Iterable[str]) -> None:
        counter = Counter(str(x) for x in reference)
        self._reference = _normalised(counter)
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted or self._reference is None:
            raise RuntimeError("UserSegmentDriftDetector not fitted")
        cur = _normalised(Counter(str(x) for x in current))
        js = _js_divergence(self._reference, cur)
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=js >= self.threshold,
            severity=self._severity_from_score(js),
            test_statistic=js,
            feature_scores=cur,
            drifted_features=sorted(cur)[:10] if js >= self.threshold else [],
            metadata={"reference": self._reference},
        )


def _flatten(value: Any) -> list[str]:
    """Flatten nested recommendation lists into a single list of item ids."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return [str(item) for sub in value for item in sub]
    return [str(item) for item in value]
