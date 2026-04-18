"""Jailbreak / prompt injection detection."""

from __future__ import annotations

import re
from typing import Any, Literal

import structlog

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail

log = structlog.get_logger(__name__)

# Common jailbreak patterns. This is intentionally a starting set — production
# deployments should override with their own threat-intel feed.
_HEURISTIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore (?:all |the )?previous instructions", re.IGNORECASE),
    re.compile(r"disregard (?:all |the )?(?:above|previous|earlier)", re.IGNORECASE),
    re.compile(r"you are now (?:dan|jailbroken|in developer mode)", re.IGNORECASE),
    re.compile(r"pretend (?:to be|you are) ", re.IGNORECASE),
    re.compile(r"system prompt[:\s]+(?:reveal|show|print)", re.IGNORECASE),
    re.compile(r"act as (?:if you are |an? )?evil", re.IGNORECASE),
    re.compile(r"do anything now", re.IGNORECASE),
    re.compile(r"\bDAN\b", re.IGNORECASE),
    re.compile(r"\\n\\nSystem:", re.IGNORECASE),
    re.compile(r"\bEND OF (?:PROMPT|INSTRUCTIONS)\b", re.IGNORECASE),
]


class JailbreakGuardrail(BaseGuardrail):
    """Detect prompt injection / jailbreak attempts.

    Combines heuristic regex patterns with optional embedding similarity
    against a corpus of known attack prompts.
    """

    name = "jailbreak_detection"
    direction = "input"

    def __init__(
        self,
        action: Literal["block", "warn", "redact"] = "block",
        method: Literal["heuristic", "embedding_similarity", "hybrid"] = "hybrid",
        threshold: float = 0.85,
        attack_corpus: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(action=action, **kwargs)
        self.method = method
        self.threshold = threshold
        self.attack_corpus = attack_corpus or []
        self._embedder = None
        self._degraded = False
        if method in ("embedding_similarity", "hybrid"):
            if not attack_corpus:
                log.warning(
                    "jailbreak.no_attack_corpus",
                    method=method,
                    msg="No attack corpus provided — embedding detection disabled, "
                    "falling back to heuristic-only.",
                )
                self._degraded = True
            else:
                self._embedder = self._try_load_embedder()
                if self._embedder is None:
                    log.warning(
                        "jailbreak.embedder_unavailable",
                        method=method,
                        msg="sentence-transformers not available — "
                        "falling back to heuristic-only.",
                    )
                    self._degraded = True

    @staticmethod
    def _try_load_embedder() -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        score, hits = self._heuristic_score(content)
        embedding_score = 0.0
        if self._embedder and self.attack_corpus:
            embedding_score = self._embedding_score(content)
            score = max(score, embedding_score)

        metadata: dict[str, Any] = {
            "hits": hits,
            "embedding_score": embedding_score,
        }
        if self._degraded:
            metadata["degraded"] = True
            metadata["effective_method"] = "heuristic"

        if score >= self.threshold:
            return self._result(
                passed=False,
                score=score,
                reason=f"jailbreak signal {score:.2f}: {hits[:3]}",
                metadata=metadata,
            )
        return self._result(passed=True, score=score, metadata=metadata)

    def _heuristic_score(self, content: str) -> tuple[float, list[str]]:
        hits = []
        for pattern in _HEURISTIC_PATTERNS:
            if pattern.search(content):
                hits.append(pattern.pattern)
        return min(1.0, len(hits) * 0.4), hits

    def _embedding_score(self, content: str) -> float:
        try:
            import numpy as np

            query = self._embedder.encode([content])[0]  # type: ignore[union-attr]
            corpus = self._embedder.encode(self.attack_corpus)  # type: ignore[union-attr]
            sims = np.dot(corpus, query) / (
                np.linalg.norm(corpus, axis=1) * np.linalg.norm(query) + 1e-9
            )
            return float(np.max(sims))
        except Exception:
            return 0.0
