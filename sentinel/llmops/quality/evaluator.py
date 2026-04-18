"""Response quality evaluator with heuristic + LLM-judge methods."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from typing import Any

import structlog

from sentinel.config.schema import LLMOpsConfig, QualityEvaluatorConfig
from sentinel.core.types import QualityScore
from sentinel.llmops.quality.base import BaseEvaluator

log = structlog.get_logger(__name__)


class ResponseEvaluator(BaseEvaluator):
    """Score LLM responses on configurable rubrics.

    Methods:
        - ``heuristic`` (default): cheap, no extra deps. Scores length,
          structure, and keyword coverage.
        - ``llm_judge``: calls an external judge model. Requires the
          caller to wire in a callable via ``judge_fn``.
        - ``reference_based``: BLEU/ROUGE-style overlap against a
          reference answer.
        - ``hybrid``: average of all available methods.
    """

    def __init__(
        self,
        config: QualityEvaluatorConfig | None = None,
        judge_fn: Callable[[str, str, str | None], dict[str, float]] | None = None,
    ):
        self.config = config or QualityEvaluatorConfig()
        self.method = self.config.method
        self.rubrics = self.config.rubrics or self._default_rubrics()
        self.sample_rate = self.config.sample_rate
        self._judge_fn = judge_fn

    @classmethod
    def from_config(cls, config: LLMOpsConfig | str | Any) -> ResponseEvaluator:
        if isinstance(config, str):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            return cls(cfg.llmops.quality.evaluator)
        if isinstance(config, LLMOpsConfig):
            return cls(config.quality.evaluator)
        return cls(config.llmops.quality.evaluator)  # type: ignore[union-attr]

    @staticmethod
    def _default_rubrics() -> dict[str, dict[str, Any]]:
        return {
            "relevance": {"weight": 0.3, "scale": 5},
            "completeness": {"weight": 0.3, "scale": 5},
            "clarity": {"weight": 0.2, "scale": 5},
            "safety": {"weight": 0.2, "scale": 5},
        }

    # ── Sampling ──────────────────────────────────────────────────

    def should_evaluate(self) -> bool:
        """Apply the configured sample rate."""
        return random.random() < self.sample_rate

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate(
        self,
        response: str,
        query: str | None = None,
        context: dict[str, Any] | None = None,
        reference: str | None = None,
    ) -> QualityScore:
        if self.method == "llm_judge" and self._judge_fn is not None:
            return self._llm_judge(response, query, context)
        if self.method == "reference_based" and reference:
            return self._reference_based(response, reference)
        if self.method == "hybrid":
            return self._hybrid(response, query, context, reference)
        return self._heuristic(response, query, context)

    # ── Heuristic ─────────────────────────────────────────────────

    def _heuristic(
        self,
        response: str,
        query: str | None,
        context: dict[str, Any] | None,
    ) -> QualityScore:
        scores: dict[str, float] = {}

        # relevance — keyword overlap with query
        if query:
            q_tokens = _tokenise(query)
            r_tokens = _tokenise(response)
            overlap = len(set(q_tokens) & set(r_tokens)) / max(1, len(set(q_tokens)))
            scores["relevance"] = min(1.0, overlap * 1.5)
        else:
            scores["relevance"] = 0.5

        # completeness — length sanity (not empty, not absurdly short)
        words = len(response.split())
        if words < 5:
            scores["completeness"] = 0.2
        elif words < 20:
            scores["completeness"] = 0.5
        else:
            scores["completeness"] = min(1.0, words / 200.0)

        # clarity — sentence count + average sentence length
        sentences = [s for s in re.split(r"[.!?]+", response) if s.strip()]
        if not sentences:
            scores["clarity"] = 0.0
        else:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            scores["clarity"] = max(0.0, min(1.0, 1.0 - abs(avg_len - 18) / 30))

        # safety — naive: no all-caps shouting, no obvious profanity
        caps_ratio = sum(1 for c in response if c.isupper()) / max(1, len(response))
        scores["safety"] = max(0.0, 1.0 - caps_ratio * 2)

        overall = self._weighted_average(scores)
        return QualityScore(
            overall=overall,
            rubric_scores=scores,
            method="heuristic",
            metadata={"word_count": words, "sentence_count": len(sentences)},
        )

    # ── LLM judge ─────────────────────────────────────────────────

    def _llm_judge(
        self,
        response: str,
        query: str | None,
        context: dict[str, Any] | None,
    ) -> QualityScore:
        try:
            scores = self._judge_fn(response, query or "", str(context or ""))  # type: ignore[misc]
            overall = self._weighted_average(scores)
            return QualityScore(overall=overall, rubric_scores=scores, method="llm_judge")
        except Exception as e:
            log.warning("evaluator.llm_judge_failed_fallback_to_heuristic", error=str(e))
            return self._heuristic(response, query, context)

    # ── Reference-based ───────────────────────────────────────────

    def _reference_based(self, response: str, reference: str) -> QualityScore:
        r_tokens = _tokenise(response)
        ref_tokens = _tokenise(reference)
        if not ref_tokens:
            return QualityScore(overall=0.0, rubric_scores={}, method="reference_based")
        intersect = len(set(r_tokens) & set(ref_tokens))
        precision = intersect / max(1, len(set(r_tokens)))
        recall = intersect / max(1, len(set(ref_tokens)))
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return QualityScore(
            overall=f1,
            rubric_scores={"precision": precision, "recall": recall, "f1": f1},
            method="reference_based",
        )

    # ── Hybrid ────────────────────────────────────────────────────

    def _hybrid(
        self,
        response: str,
        query: str | None,
        context: dict[str, Any] | None,
        reference: str | None,
    ) -> QualityScore:
        components = [self._heuristic(response, query, context)]
        if reference:
            components.append(self._reference_based(response, reference))
        if self._judge_fn is not None:
            components.append(self._llm_judge(response, query, context))
        overall = sum(c.overall for c in components) / len(components)
        merged: dict[str, float] = {}
        for c in components:
            merged.update({f"{c.method}.{k}": v for k, v in c.rubric_scores.items()})
        return QualityScore(overall=overall, rubric_scores=merged, method="hybrid")

    # ── Helpers ───────────────────────────────────────────────────

    def _weighted_average(self, scores: dict[str, float]) -> float:
        total = 0.0
        weight = 0.0
        for k, v in scores.items():
            w = float(self.rubrics.get(k, {}).get("weight", 1.0))
            total += v * w
            weight += w
        return total / weight if weight > 0 else 0.0


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 2]
