"""Unit tests for recommendation system quality and bias metrics."""

from __future__ import annotations

import math

from sentinel.domains.recommendation.bias import group_fairness, position_bias
from sentinel.domains.recommendation.quality import (
    catalogue_coverage,
    diversity_intra_list,
    evaluate_recommendations,
    gini_coefficient,
    map_at_k,
    ndcg_at_k,
    novelty_inverse_popularity,
)


class TestRankingMetrics:
    def test_ndcg_perfect_ranking(self) -> None:
        preds = [["a", "b", "c"]]
        truth = [{"a", "b", "c"}]
        score = ndcg_at_k(preds, truth, k=3)
        assert math.isclose(score, 1.0)

    def test_ndcg_no_hits(self) -> None:
        preds = [["x", "y"]]
        truth = [{"a"}]
        score = ndcg_at_k(preds, truth, k=2)
        assert score == 0.0

    def test_map_at_k(self) -> None:
        preds = [["a", "x", "b"]]
        truth = [{"a", "b"}]
        score = map_at_k(preds, truth, k=3)
        # Position 1 hit (1/1 = 1.0), position 3 hit (2/3 ≈ 0.667)
        # Average / min(2, 3) = (1.0 + 0.667) / 2 ≈ 0.833
        assert score > 0.8

    def test_catalogue_coverage(self) -> None:
        preds = [["a", "b"], ["c"]]
        catalogue = {"a", "b", "c", "d"}
        cov = catalogue_coverage(preds, catalogue)
        assert math.isclose(cov, 0.75)

    def test_diversity_no_similarity_returns_one(self) -> None:
        preds = [["a", "b", "c"]]
        score = diversity_intra_list(preds)
        assert score == 1.0  # no similarity provided → assume orthogonal

    def test_novelty_unknown_pop_zero(self) -> None:
        preds = [["a", "b"]]
        score = novelty_inverse_popularity(preds, popularity={"a": 1.0, "b": 1.0})
        # Equal popularity → both items contribute -log2(0.5) = 1
        assert math.isclose(score, 1.0)


class TestGini:
    def test_uniform_zero_gini(self) -> None:
        items = ["a", "b", "c", "d"]
        assert math.isclose(gini_coefficient(items), 0.0, abs_tol=0.05)

    def test_concentrated_high_gini(self) -> None:
        items = ["a"] * 95 + ["b"] * 5
        score = gini_coefficient(items)
        assert score > 0.4

    def test_empty_returns_zero(self) -> None:
        assert gini_coefficient([]) == 0.0


class TestEvaluateBundle:
    def test_full_evaluation(self) -> None:
        preds = [["a", "b", "c"], ["b", "d"]]
        truth = [{"a", "b"}, {"d"}]
        catalogue = {"a", "b", "c", "d", "e"}
        popularity = {"a": 100, "b": 50, "c": 5, "d": 1, "e": 0}
        result = evaluate_recommendations(preds, truth, catalogue, popularity=popularity)
        assert 0.0 <= result.ndcg <= 1.0
        assert 0.0 <= result.map_score <= 1.0
        assert 0.0 <= result.coverage <= 1.0


class TestFairness:
    def test_group_fairness_no_disparity(self) -> None:
        preds = [["a", "b"], ["a", "b"]]
        truth = [{"a"}, {"a"}]
        groups = ["g1", "g2"]
        report = group_fairness(preds, truth, groups, k=2)
        assert report.max_disparity == 0.0
        assert report.passed


class TestPositionBias:
    def test_position_bias_uniform(self) -> None:
        preds = [["a", "b"], ["c", "d"], ["e", "f"]]
        bias = position_bias(preds, k=2)
        # Two positions, each filled three times
        assert bias[0] == bias[1] == 3
