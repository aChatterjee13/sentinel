"""Toxicity guardrail — flag harmful or biased outputs."""

from __future__ import annotations

import re
from typing import Any, Literal

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail

# Lightweight word list for the heuristic fallback. Real deployments should
# wire in `detoxify` or a hosted classifier — these patterns are intentionally
# conservative to keep false positives low when nothing is installed.
_HEURISTIC_TERMS = {
    "slur",
    "hate",
    "kill yourself",
    "i will hurt",
    "racist",
    "violent threat",
}


class ToxicityGuardrail(BaseGuardrail):
    """Score outputs for toxicity / harmful content.

    Tries to use the ``detoxify`` package for production-quality scoring.
    Falls back to a small heuristic word list otherwise. Either way, the
    output is a ``[0, 1]`` toxicity score compared against ``threshold``.
    """

    name = "toxicity"
    direction = "output"

    def __init__(
        self,
        action: Literal["block", "warn", "redact"] = "block",
        threshold: float = 0.7,
        **kwargs: Any,
    ):
        super().__init__(action=action, **kwargs)
        self.threshold = threshold
        self._classifier = self._try_load()

    @staticmethod
    def _try_load() -> Any:
        try:
            from detoxify import Detoxify  # type: ignore[import-not-found]

            return Detoxify("original-small")
        except Exception:
            return None

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        score = self._score(content)
        if score >= self.threshold:
            if self.action == "redact":
                sanitised = self._redact(content)
                return self._result(
                    passed=False,
                    score=score,
                    reason=f"toxicity score {score:.2f} >= {self.threshold} (content redacted)",
                    sanitised=sanitised,
                    metadata={"score": score, "action": "redacted"},
                )
            return self._result(
                passed=False,
                score=score,
                reason=f"toxicity score {score:.2f} >= {self.threshold}",
                metadata={"score": score},
            )
        return self._result(passed=True, score=score)

    def _score(self, content: str) -> float:
        if self._classifier is not None:
            try:
                result = self._classifier.predict(content)
                return float(max(result.values()))
            except Exception:
                pass
        return self._heuristic_score(content)

    @staticmethod
    def _heuristic_score(content: str) -> float:
        text = content.lower()
        hits = sum(1 for term in _HEURISTIC_TERMS if re.search(rf"\b{term}\b", text))
        return min(1.0, hits * 0.4)

    def _redact(self, content: str) -> str:
        """Replace toxic spans with redaction markers.

        Uses detoxify per-sentence scoring when available; otherwise
        replaces heuristic-matched terms.
        """
        if self._classifier is not None:
            return self._redact_detoxify(content)
        return self._redact_heuristic(content)

    @staticmethod
    def _redact_heuristic(content: str) -> str:
        """Replace heuristic-matched toxic terms with ``[REDACTED]``."""
        result = content
        for term in _HEURISTIC_TERMS:
            result = re.sub(
                rf"\b{re.escape(term)}\b",
                "[REDACTED]",
                result,
                flags=re.IGNORECASE,
            )
        return result

    def _redact_detoxify(self, content: str) -> str:
        """Score each sentence; replace high-toxicity ones."""
        sentences = re.split(r"(?<=[.!?])\s+", content)
        out: list[str] = []
        for sentence in sentences:
            try:
                result = self._classifier.predict(sentence)  # type: ignore[union-attr]
                max_score = float(max(result.values()))
            except Exception:
                max_score = 0.0
            if max_score >= self.threshold:
                out.append("[Content removed: toxicity detected]")
            else:
                out.append(sentence)
        return " ".join(out)
