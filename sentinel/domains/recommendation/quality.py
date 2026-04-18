"""Recommendation quality metrics — NDCG, MAP, coverage, diversity, novelty."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass
class RankingQualityResult:
    """Aggregated ranking + beyond-accuracy metrics."""

    ndcg: float
    map_score: float
    coverage: float
    diversity: float
    novelty: float
    popularity_bias: float


def ndcg_at_k(
    predictions: Sequence[Sequence[str]], ground_truth: Sequence[set[str]], k: int = 10
) -> float:
    """Mean NDCG@k across users."""
    if not predictions:
        return 0.0
    scores = []
    for preds, truth in zip(predictions, ground_truth, strict=False):
        dcg = 0.0
        for i, item in enumerate(preds[:k]):
            if item in truth:
                dcg += 1.0 / math.log2(i + 2)
        ideal_hits = min(len(truth), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(sum(scores) / len(scores))


def map_at_k(
    predictions: Sequence[Sequence[str]],
    ground_truth: Sequence[set[str]],
    k: int = 10,
) -> float:
    """Mean Average Precision @ k."""
    if not predictions:
        return 0.0
    aps = []
    for preds, truth in zip(predictions, ground_truth, strict=False):
        if not truth:
            continue
        hits = 0
        precisions = []
        for i, item in enumerate(preds[:k]):
            if item in truth:
                hits += 1
                precisions.append(hits / (i + 1))
        ap = sum(precisions) / min(len(truth), k) if precisions else 0.0
        aps.append(ap)
    return float(sum(aps) / len(aps)) if aps else 0.0


def catalogue_coverage(
    predictions: Iterable[Sequence[str]],
    catalogue: set[str],
) -> float:
    """Fraction of the catalogue that appears in any recommendation list."""
    if not catalogue:
        return 0.0
    seen: set[str] = set()
    for preds in predictions:
        seen.update(preds)
    return len(seen & catalogue) / len(catalogue)


def diversity_intra_list(
    predictions: Sequence[Sequence[str]],
    similarity: dict[tuple[str, str], float] | None = None,
) -> float:
    """Mean intra-list diversity (1 - average pairwise similarity)."""
    if not predictions:
        return 0.0
    sim = similarity or {}
    scores = []
    for preds in predictions:
        if len(preds) < 2:
            scores.append(1.0)
            continue
        pairs = [
            sim.get((a, b), sim.get((b, a), 0.0))
            for i, a in enumerate(preds)
            for b in preds[i + 1 :]
        ]
        avg = sum(pairs) / len(pairs) if pairs else 0.0
        scores.append(1.0 - avg)
    return float(sum(scores) / len(scores))


def novelty_inverse_popularity(
    predictions: Sequence[Sequence[str]],
    popularity: dict[str, float],
) -> float:
    """Average self-information of recommended items (higher = more novel)."""
    if not predictions or not popularity:
        return 0.0
    total = sum(popularity.values()) or 1.0
    scores = []
    for preds in predictions:
        if not preds:
            scores.append(0.0)
            continue
        info = []
        for item in preds:
            p = popularity.get(item, 0.0) / total
            if p > 0:
                info.append(-math.log2(p))
        scores.append(sum(info) / len(preds) if preds else 0.0)
    return float(sum(scores) / len(scores))


def gini_coefficient(items: Iterable[str]) -> float:
    """Gini coefficient of recommendation frequency — high means popularity bias."""
    counts = sorted(Counter(items).values())
    n = len(counts)
    if n == 0:
        return 0.0
    cum = 0
    for i, c in enumerate(counts, 1):
        cum += i * c
    total = sum(counts)
    if total == 0:
        return 0.0
    return (2 * cum) / (n * total) - (n + 1) / n


def evaluate_recommendations(
    predictions: Sequence[Sequence[str]],
    ground_truth: Sequence[set[str]],
    catalogue: set[str],
    popularity: dict[str, float] | None = None,
    similarity: dict[tuple[str, str], float] | None = None,
    k: int = 10,
) -> RankingQualityResult:
    """Compute the ranking + beyond-accuracy metric bundle."""
    flat_items = [item for sub in predictions for item in sub]
    return RankingQualityResult(
        ndcg=ndcg_at_k(predictions, ground_truth, k=k),
        map_score=map_at_k(predictions, ground_truth, k=k),
        coverage=catalogue_coverage(predictions, catalogue),
        diversity=diversity_intra_list(predictions, similarity=similarity),
        novelty=novelty_inverse_popularity(predictions, popularity or {}),
        popularity_bias=gini_coefficient(flat_items),
    )
