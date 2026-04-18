"""Unit tests for sentinel.action.deployment.promotion.PromotionPolicy."""

from __future__ import annotations

import pytest

from sentinel.action.deployment.promotion import PromotionPolicy

# ── Tests ──────────────────────────────────────────────────────────


class TestSingleMetricPromotion:
    """Default mode: promote based on one metric's improvement %."""

    def test_promote_when_improvement_exceeds_threshold(self) -> None:
        policy = PromotionPolicy(metric="f1", improvement_pct=2.0)
        assert policy.should_promote(
            champion={"f1": 0.80},
            challenger={"f1": 0.85},  # +6.25%
        )

    def test_no_promote_when_improvement_below_threshold(self) -> None:
        policy = PromotionPolicy(metric="f1", improvement_pct=10.0)
        assert not policy.should_promote(
            champion={"f1": 0.80},
            challenger={"f1": 0.81},  # +1.25%
        )

    def test_promote_with_zero_improvement_threshold(self) -> None:
        """improvement_pct=0 means any equal-or-better challenger wins."""
        policy = PromotionPolicy(metric="accuracy", improvement_pct=0.0)
        assert policy.should_promote(
            champion={"accuracy": 0.90},
            challenger={"accuracy": 0.90},
        )

    def test_champion_zero_promotes_any_positive_challenger(self) -> None:
        """When champion metric is 0, any positive challenger should be promoted."""
        policy = PromotionPolicy(metric="f1", improvement_pct=5.0)
        assert policy.should_promote(
            champion={"f1": 0.0},
            challenger={"f1": 0.01},
        )

    def test_champion_zero_does_not_promote_zero_challenger(self) -> None:
        """Both zero: not promoted (no improvement)."""
        policy = PromotionPolicy(metric="f1", improvement_pct=1.0)
        assert not policy.should_promote(
            champion={"f1": 0.0},
            challenger={"f1": 0.0},
        )


class TestMinMetricsFloor:
    """Floor checks: challenger must meet absolute minimums."""

    def test_below_floor_blocks_promotion(self) -> None:
        policy = PromotionPolicy(
            metric="f1",
            improvement_pct=0.0,
            min_metrics={"accuracy": 0.85},
        )
        assert not policy.should_promote(
            champion={"f1": 0.70, "accuracy": 0.80},
            challenger={"f1": 0.90, "accuracy": 0.80},  # accuracy < 0.85
        )

    def test_meets_floor_allows_promotion(self) -> None:
        policy = PromotionPolicy(
            metric="f1",
            improvement_pct=0.0,
            min_metrics={"accuracy": 0.85},
        )
        assert policy.should_promote(
            champion={"f1": 0.70},
            challenger={"f1": 0.75, "accuracy": 0.90},
        )

    def test_missing_metric_defaults_to_zero_fails_floor(self) -> None:
        """If the challenger is missing a floor metric, it's treated as 0."""
        policy = PromotionPolicy(
            metric="f1",
            improvement_pct=0.0,
            min_metrics={"recall": 0.5},
        )
        assert not policy.should_promote(
            champion={"f1": 0.70},
            challenger={"f1": 0.80},  # recall absent → 0.0 < 0.5
        )


class TestRequireAllMetricsBetter:
    """Mode where every champion metric must be equalled or exceeded."""

    def test_all_better_promotes(self) -> None:
        policy = PromotionPolicy(require_all_metrics_better=True)
        assert policy.should_promote(
            champion={"f1": 0.80, "accuracy": 0.85},
            challenger={"f1": 0.82, "accuracy": 0.87},
        )

    def test_one_worse_blocks(self) -> None:
        policy = PromotionPolicy(require_all_metrics_better=True)
        assert not policy.should_promote(
            champion={"f1": 0.80, "accuracy": 0.85},
            challenger={"f1": 0.82, "accuracy": 0.84},
        )

    def test_empty_champion_promotes(self) -> None:
        """If champion has no metrics, all() over empty is True."""
        policy = PromotionPolicy(require_all_metrics_better=True)
        assert policy.should_promote(
            champion={},
            challenger={"f1": 0.5},
        )


class TestExplain:
    """explain() returns audit-friendly diffs."""

    def test_explain_includes_should_promote(self) -> None:
        policy = PromotionPolicy(metric="f1", improvement_pct=2.0)
        result = policy.explain(
            champion={"f1": 0.80, "accuracy": 0.85},
            challenger={"f1": 0.85, "accuracy": 0.86},
        )
        assert result["should_promote"] is True
        assert result["metric"] == "f1"
        assert "f1" in result["improvements"]
        assert "accuracy" in result["improvements"]

    def test_explain_delta_and_pct(self) -> None:
        policy = PromotionPolicy(metric="f1")
        result = policy.explain(
            champion={"f1": 0.50},
            challenger={"f1": 0.60},
        )
        f1_info = result["improvements"]["f1"]
        assert f1_info["delta"] == pytest.approx(0.10)
        assert f1_info["pct"] == pytest.approx(20.0)

    def test_explain_zero_champion(self) -> None:
        """Champion=0 gives pct=0 (no division error)."""
        policy = PromotionPolicy(metric="f1")
        result = policy.explain(
            champion={"f1": 0.0},
            challenger={"f1": 0.5},
        )
        assert result["improvements"]["f1"]["pct"] == 0.0

    def test_explain_with_extra_challenger_metrics(self) -> None:
        """Metrics only in challenger are still reported."""
        policy = PromotionPolicy(metric="f1")
        result = policy.explain(
            champion={"f1": 0.8},
            challenger={"f1": 0.85, "recall": 0.9},
        )
        assert "recall" in result["improvements"]
