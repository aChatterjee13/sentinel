"""Champion-challenger promotion rules."""

from __future__ import annotations

from typing import Any


class PromotionPolicy:
    """Decides when a challenger model is good enough to promote.

    Example:
        >>> policy = PromotionPolicy(metric="f1", improvement_pct=2.0)
        >>> policy.should_promote(champion={"f1": 0.85}, challenger={"f1": 0.88})
        True
    """

    def __init__(
        self,
        metric: str = "accuracy",
        improvement_pct: float = 0.0,
        require_all_metrics_better: bool = False,
        min_metrics: dict[str, float] | None = None,
    ):
        self.metric = metric
        self.improvement_pct = improvement_pct
        self.require_all_metrics_better = require_all_metrics_better
        self.min_metrics = min_metrics or {}

    def should_promote(
        self,
        champion: dict[str, float],
        challenger: dict[str, float],
    ) -> bool:
        """True when the challenger meets the promotion criteria."""
        # Floor checks
        for key, floor in self.min_metrics.items():
            if challenger.get(key, 0.0) < floor:
                return False

        if self.require_all_metrics_better:
            return all(challenger.get(k, 0.0) >= champion.get(k, 0.0) for k in champion)

        # Single-metric improvement gate
        c = champion.get(self.metric, 0.0)
        ch = challenger.get(self.metric, 0.0)
        if c == 0:
            return ch > 0
        improvement_pct = ((ch - c) / abs(c)) * 100
        return improvement_pct >= self.improvement_pct

    def explain(
        self,
        champion: dict[str, float],
        challenger: dict[str, float],
    ) -> dict[str, Any]:
        """Detailed reasoning for audit logging."""
        improvements = {}
        for k in set(champion) | set(challenger):
            c = champion.get(k, 0.0)
            ch = challenger.get(k, 0.0)
            delta = ch - c
            pct = (delta / abs(c) * 100) if c != 0 else 0.0
            improvements[k] = {"champion": c, "challenger": ch, "delta": delta, "pct": pct}
        return {
            "metric": self.metric,
            "required_improvement_pct": self.improvement_pct,
            "should_promote": self.should_promote(champion, challenger),
            "improvements": improvements,
        }
