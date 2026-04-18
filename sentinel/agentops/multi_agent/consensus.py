"""Multi-agent consensus and conflict detection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import structlog

from sentinel.config.schema import MultiAgentConfig
from sentinel.core.exceptions import EscalationRequired

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ConsensusResult:
    """Result of a multi-agent vote."""

    decision: Any
    agreement: float
    n_voters: int
    votes: dict[str, Any]
    has_consensus: bool
    conflict: bool
    conflict_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConsensusEvaluator:
    """Compute consensus across multiple agent responses.

    The evaluator implements three resolution strategies — ``majority_vote``,
    ``weighted_vote``, and ``escalate`` (raise on conflict). The strategy and
    minimum agreement threshold are configured via :class:`MultiAgentConfig`.
    """

    def __init__(self, config: MultiAgentConfig | None = None):
        self.config = config or MultiAgentConfig()
        self.consensus_cfg = self.config.consensus or {}
        self.min_agreement = float(self.consensus_cfg.get("min_agreement", 0.67))
        self.conflict_action = self.consensus_cfg.get("conflict_action", "escalate")
        self.enabled = bool(self.consensus_cfg.get("enabled", True))

    def evaluate(
        self,
        votes: dict[str, Any],
        weights: dict[str, float] | None = None,
    ) -> ConsensusResult:
        if not votes:
            return ConsensusResult(
                decision=None,
                agreement=0.0,
                n_voters=0,
                votes={},
                has_consensus=False,
                conflict=True,
                conflict_reason="no votes",
            )
        if weights and self.conflict_action == "weighted_vote":
            return self._weighted(votes, weights)
        return self._majority(votes)

    def _majority(self, votes: dict[str, Any]) -> ConsensusResult:
        counts = Counter(_canonical(v) for v in votes.values())
        decision_key, top_count = counts.most_common(1)[0]
        agreement = top_count / len(votes)
        decision = next(v for v in votes.values() if _canonical(v) == decision_key)
        has_consensus = agreement >= self.min_agreement
        conflict = not has_consensus
        if conflict and self.conflict_action == "escalate":
            log.warning(
                "multi_agent.consensus_conflict_escalating",
                agreement=agreement,
                min_agreement=self.min_agreement,
                method="majority",
            )
            raise EscalationRequired(
                f"consensus conflict: agreement {agreement:.2f} < {self.min_agreement}"
            )
        return ConsensusResult(
            decision=decision,
            agreement=agreement,
            n_voters=len(votes),
            votes=votes,
            has_consensus=has_consensus,
            conflict=conflict,
            conflict_reason=None
            if has_consensus
            else f"agreement {agreement:.2f} < {self.min_agreement}",
            metadata={"counts": dict(counts), "method": "majority"},
        )

    def _weighted(self, votes: dict[str, Any], weights: dict[str, float]) -> ConsensusResult:
        totals: dict[str, float] = {}
        canon_to_value: dict[str, Any] = {}
        total_weight = 0.0
        for agent, value in votes.items():
            w = float(weights.get(agent, 1.0))
            key = _canonical(value)
            totals[key] = totals.get(key, 0.0) + w
            canon_to_value.setdefault(key, value)
            total_weight += w
        winner_key = max(totals, key=lambda k: totals[k])
        agreement = totals[winner_key] / total_weight if total_weight else 0.0
        has_consensus = agreement >= self.min_agreement
        conflict = not has_consensus
        if conflict and self.conflict_action == "escalate":
            log.warning(
                "multi_agent.consensus_conflict_escalating",
                agreement=agreement,
                min_agreement=self.min_agreement,
                method="weighted",
            )
            raise EscalationRequired(
                f"consensus conflict: agreement {agreement:.2f} < {self.min_agreement}"
            )
        return ConsensusResult(
            decision=canon_to_value[winner_key],
            agreement=agreement,
            n_voters=len(votes),
            votes=votes,
            has_consensus=has_consensus,
            conflict=not has_consensus,
            conflict_reason=None
            if has_consensus
            else f"weighted agreement {agreement:.2f} < {self.min_agreement}",
            metadata={"totals": totals, "method": "weighted"},
        )

    def disagreement_rate(self, history: Iterable[ConsensusResult]) -> float:
        items = list(history)
        if not items:
            return 0.0
        return sum(1 for r in items if not r.has_consensus) / len(items)


def _canonical(value: Any) -> str:
    """Normalise a vote value to a hashable, comparable key."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float, bool)):
        return repr(value)
    return repr(value)
