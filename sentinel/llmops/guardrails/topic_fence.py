"""Topic fencing — keep agents inside their intended scope."""

from __future__ import annotations

from typing import Any, Literal

import structlog

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail

log = structlog.get_logger()


class TopicFenceGuardrail(BaseGuardrail):
    """Reject queries that fall outside the agent's allowed topics.

    Uses keyword-based matching by default; if sentence-transformers is
    available, swaps in embedding similarity against topic descriptors
    for higher precision.
    """

    name = "topic_fence"
    direction = "input"

    def __init__(
        self,
        action: Literal["block", "warn", "redact"] = "warn",
        allowed_topics: list[str] | None = None,
        blocked_topics: list[str] | None = None,
        threshold: float = 0.4,
        **kwargs: Any,
    ):
        super().__init__(action=action, **kwargs)
        self.allowed_topics = [t.lower() for t in (allowed_topics or [])]
        self.blocked_topics = [t.lower() for t in (blocked_topics or [])]
        self.threshold = threshold
        self._embedder = self._try_load_embedder()

    @staticmethod
    def _try_load_embedder() -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        text = content.lower()

        for blocked in self.blocked_topics:
            if blocked in text:
                return self._result(
                    passed=False,
                    score=1.0,
                    reason=f"contains blocked topic: {blocked}",
                )

        if not self.allowed_topics:
            return self._result(passed=True, score=0.0)

        if self._embedder is not None:
            score = self._embedding_match(content)
        else:
            score = self._keyword_match(text)

        if score < self.threshold:
            return self._result(
                passed=False,
                score=score,
                reason=f"off-topic (allowed-match score {score:.2f})",
                metadata={"allowed_topics": self.allowed_topics, "score": score},
            )
        return self._result(passed=True, score=score)

    def _keyword_match(self, text: str) -> float:
        hits = sum(1 for topic in self.allowed_topics if topic in text)
        return min(1.0, hits / max(1, len(self.allowed_topics)))

    def _embedding_match(self, content: str) -> float:
        try:
            import numpy as np

            query = self._embedder.encode([content])[0]  # type: ignore[union-attr]
            corpus = self._embedder.encode(self.allowed_topics)  # type: ignore[union-attr]
            sims = np.dot(corpus, query) / (
                np.linalg.norm(corpus, axis=1) * np.linalg.norm(query) + 1e-9
            )
            return float(np.max(sims))
        except Exception as e:
            log.error("topic_fence.embedding_failed", error=str(e))
            return 0.0
