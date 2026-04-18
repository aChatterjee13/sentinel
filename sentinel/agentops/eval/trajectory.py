"""Compare actual agent trajectories against optimal/golden references."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from sentinel.config.schema import AgentEvaluationConfig
from sentinel.core.types import AgentTrace, Span

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TrajectoryScore:
    """Result of comparing an actual trajectory to an optimal one."""

    score: float
    optimal_steps: int
    actual_steps: int
    extra_steps: int
    missing_steps: list[str] = field(default_factory=list)
    out_of_order: bool = False
    passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TrajectoryEvaluator:
    """Score an agent's trajectory against an optimal step sequence.

    The evaluator uses longest common subsequence (LCS) ordering plus a
    per-extra-step penalty configured in
    :class:`AgentEvaluationConfig`. A score of ``1.0`` means the agent
    followed the optimal path exactly; ``0.0`` means complete divergence.
    """

    def __init__(self, config: AgentEvaluationConfig | None = None):
        self.config = config or AgentEvaluationConfig()
        traj_cfg = self.config.trajectory or {}
        self.compare_against = traj_cfg.get("compare_against", "optimal")
        self.penalty_per_extra_step = float(traj_cfg.get("penalty_per_extra_step", 0.05))
        self.pass_threshold = float(traj_cfg.get("pass_threshold", 0.7))

    def score(
        self,
        actual_steps: Sequence[str],
        optimal_steps: Sequence[str],
    ) -> TrajectoryScore:
        if not optimal_steps:
            raw_score = 1.0 if not actual_steps else 0.0
            return TrajectoryScore(
                score=raw_score,
                optimal_steps=0,
                actual_steps=len(actual_steps),
                extra_steps=len(actual_steps),
                passed=raw_score >= self.pass_threshold,
            )

        actual_set = set(actual_steps)
        missing = [s for s in optimal_steps if s not in actual_set]
        coverage = (len(optimal_steps) - len(missing)) / len(optimal_steps)

        lcs_len = _lcs_length(list(actual_steps), list(optimal_steps))
        order_score = lcs_len / len(optimal_steps) if optimal_steps else 1.0
        out_of_order = lcs_len < len(optimal_steps) and len(missing) == 0

        extra = max(0, len(actual_steps) - len(optimal_steps))
        penalty = extra * self.penalty_per_extra_step

        raw = 0.5 * coverage + 0.5 * order_score - penalty
        bounded = max(0.0, min(1.0, raw))

        return TrajectoryScore(
            score=bounded,
            optimal_steps=len(optimal_steps),
            actual_steps=len(actual_steps),
            extra_steps=extra,
            missing_steps=missing,
            out_of_order=out_of_order,
            passed=bounded >= self.pass_threshold,
            metadata={
                "coverage": coverage,
                "order_score": order_score,
                "penalty": penalty,
            },
        )

    def score_trace(self, trace: AgentTrace, optimal_steps: Sequence[str]) -> TrajectoryScore:
        actual = _spans_to_step_names(trace.spans)
        return self.score(actual, optimal_steps)


def _spans_to_step_names(spans: Iterable[Span]) -> list[str]:
    return [span.name for span in spans]


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of ``a`` and ``b``."""
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            if x == y:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]
