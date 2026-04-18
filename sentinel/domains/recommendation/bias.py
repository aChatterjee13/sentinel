"""Recommendation fairness and bias monitoring."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from sentinel.domains.recommendation.quality import ndcg_at_k


@dataclass
class FairnessReport:
    """Per-group quality with maximum disparity across protected groups."""

    metric: str
    per_group: dict[str, float]
    max_disparity: float
    passed: bool


def group_fairness(
    predictions: Sequence[Sequence[str]],
    ground_truth: Sequence[set[str]],
    groups: Sequence[str],
    *,
    metric: str = "ndcg_at_10",
    k: int = 10,
    max_disparity: float = 0.1,
) -> FairnessReport:
    """Compute per-group quality and the maximum disparity across groups."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        grouped[str(group)].append(idx)

    scores: dict[str, float] = {}
    for group, idxs in grouped.items():
        sub_preds = [predictions[i] for i in idxs]
        sub_truth = [ground_truth[i] for i in idxs]
        scores[group] = ndcg_at_k(sub_preds, sub_truth, k=k)

    disparity = max(scores.values()) - min(scores.values()) if scores else 0.0

    return FairnessReport(
        metric=metric,
        per_group=scores,
        max_disparity=disparity,
        passed=disparity <= max_disparity,
    )


def position_bias(predictions: Sequence[Sequence[str]], k: int = 10) -> dict[int, int]:
    """Frequency of items appearing at each position in the recommendation list."""
    counts: dict[int, int] = dict.fromkeys(range(k), 0)
    for preds in predictions:
        for i, _ in enumerate(preds[:k]):
            counts[i] = counts.get(i, 0) + 1
    return counts
