"""Graph topology drift detectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sentinel.core.types import DriftReport
from sentinel.domains.graph.structure import (
    Edge,
    TopologyStats,
    degree_distribution,
    topology_stats,
)
from sentinel.observability.drift.base import BaseDriftDetector


def _ks_distribution(a: dict[int, int], b: dict[int, int]) -> float:
    """Kolmogorov-Smirnov distance between two integer-valued distributions."""
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    total_a = sum(a.values()) or 1
    total_b = sum(b.values()) or 1
    cum_a = 0.0
    cum_b = 0.0
    max_diff = 0.0
    for k in keys:
        cum_a += a.get(k, 0) / total_a
        cum_b += b.get(k, 0) / total_b
        diff = abs(cum_a - cum_b)
        if diff > max_diff:
            max_diff = diff
    return float(max_diff)


class TopologyDriftDetector(BaseDriftDetector):
    """Detect topology drift via degree distribution + density + clustering."""

    method_name = "topology_drift"

    def __init__(self, model_name: str, threshold: float = 0.1, **kwargs: Any):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self._reference_stats: TopologyStats | None = None
        self._reference_degrees: dict[int, int] | None = None

    def fit(self, reference: Sequence[Edge]) -> None:
        edges = list(reference)
        self._reference_stats = topology_stats(edges)
        self._reference_degrees = dict(degree_distribution(edges))
        self._fitted = True

    def detect(self, current: Sequence[Edge], **kwargs: Any) -> DriftReport:
        if not self._fitted or self._reference_stats is None or self._reference_degrees is None:
            raise RuntimeError("TopologyDriftDetector not fitted")
        edges = list(current)
        cur_stats = topology_stats(edges)
        cur_degrees = dict(degree_distribution(edges))

        ks = _ks_distribution(self._reference_degrees, cur_degrees)
        density_shift = abs(cur_stats.density - self._reference_stats.density)
        clustering_shift = abs(
            cur_stats.clustering_coefficient - self._reference_stats.clustering_coefficient
        )
        component_shift = abs(cur_stats.n_components - self._reference_stats.n_components) / max(
            1, self._reference_stats.n_components
        )
        worst = max(ks, density_shift, clustering_shift, component_shift)

        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=worst >= self.threshold,
            severity=self._severity_from_score(worst),
            test_statistic=worst,
            feature_scores={
                "degree_ks": ks,
                "density_shift": density_shift,
                "clustering_shift": clustering_shift,
                "component_shift": component_shift,
            },
            drifted_features=[
                k
                for k, v in {
                    "degree_ks": ks,
                    "density_shift": density_shift,
                    "clustering_shift": clustering_shift,
                    "component_shift": component_shift,
                }.items()
                if v >= self.threshold
            ],
            metadata={
                "reference": self._reference_stats.__dict__,
                "current": cur_stats.__dict__,
            },
        )


class EntityVocabularyDriftDetector(BaseDriftDetector):
    """Track the rate of unseen entities in queries (KG-specific)."""

    method_name = "entity_vocabulary"

    def __init__(self, model_name: str, threshold: float = 0.1, **kwargs: Any):
        super().__init__(model_name=model_name, threshold=threshold, **kwargs)
        self._reference: set[str] = set()

    def fit(self, reference: Any) -> None:
        self._reference = {str(x) for x in reference}
        self._fitted = True

    def detect(self, current: Any, **kwargs: Any) -> DriftReport:
        if not self._fitted:
            raise RuntimeError("EntityVocabularyDriftDetector not fitted")
        items = [str(x) for x in current]
        if not items:
            return DriftReport(
                model_name=self.model_name,
                method=self.method_name,
                is_drifted=False,
                severity=self._severity_from_score(0.0),
                test_statistic=0.0,
            )
        unseen = [i for i in items if i not in self._reference]
        oov = len(unseen) / len(items)
        return DriftReport(
            model_name=self.model_name,
            method=self.method_name,
            is_drifted=oov >= self.threshold,
            severity=self._severity_from_score(oov),
            test_statistic=oov,
            feature_scores={"oov_rate": oov},
            drifted_features=unseen[:10] if oov >= self.threshold else [],
            metadata={"reference_size": len(self._reference)},
        )
