"""Groundedness check for RAG responses."""

from __future__ import annotations

import re
from typing import Any, Literal

import structlog

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail

log = structlog.get_logger()


class GroundednessGuardrail(BaseGuardrail):
    """Verify that an LLM response is grounded in the retrieved context.

    Three back-end methods are supported:

    - ``chunk_overlap`` (default): token-level overlap with retrieved chunks.
    - ``nli``: requires a transformer NLI model.
    - ``llm_judge``: requires an LLM provider call (handled externally).
    """

    name = "groundedness"
    direction = "output"

    def __init__(
        self,
        action: Literal["block", "warn", "redact"] = "warn",
        method: Literal["chunk_overlap", "nli", "llm_judge"] = "chunk_overlap",
        min_score: float = 0.6,
        **kwargs: Any,
    ):
        super().__init__(action=action, **kwargs)
        self.method = method
        self.min_score = min_score
        self._nli = self._try_load_nli() if method == "nli" else None

    @staticmethod
    def _try_load_nli() -> Any:
        try:
            from transformers import pipeline  # type: ignore[import-not-found]

            return pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
        except Exception:
            return None

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        chunks = self._extract_chunks(context)
        if not chunks:
            # No retrieved context — we can't judge groundedness, pass through.
            return self._result(passed=True, score=1.0, metadata={"reason": "no context"})

        if self.method == "nli" and self._nli is not None:
            score = self._nli_score(content, chunks)
        else:
            score = self._overlap_score(content, chunks)

        if score < self.min_score:
            return self._result(
                passed=False,
                score=score,
                reason=f"groundedness {score:.2f} below {self.min_score}",
                metadata={"score": score, "method": self.method},
            )
        return self._result(passed=True, score=score)

    @staticmethod
    def _extract_chunks(context: dict[str, Any] | None) -> list[str]:
        if not context:
            return []
        chunks = context.get("chunks") or context.get("context") or context.get("retrieved")
        if isinstance(chunks, list):
            return [str(c) if not isinstance(c, dict) else str(c.get("text", c)) for c in chunks]
        if isinstance(chunks, str):
            return [chunks]
        return []

    @staticmethod
    def _overlap_score(content: str, chunks: list[str]) -> float:
        response_tokens = set(_tokenise(content))
        if not response_tokens:
            return 1.0
        chunk_tokens: set[str] = set()
        for c in chunks:
            chunk_tokens.update(_tokenise(c))
        return len(response_tokens & chunk_tokens) / max(1, len(response_tokens))

    def _nli_score(self, content: str, chunks: list[str]) -> float:
        try:
            joined = " ".join(chunks)
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
            if not sentences:
                return 1.0
            scores = []
            for sent in sentences:
                result = self._nli({"text": joined, "text_pair": sent})  # type: ignore[misc]
                # Maps entailment label to a positive score
                if isinstance(result, list):
                    result = result[0]
                label = result.get("label", "").lower()
                conf = float(result.get("score", 0.0))
                scores.append(conf if "entail" in label else 1.0 - conf)
            return sum(scores) / len(scores)
        except Exception as e:
            log.warning("groundedness.nli_fallback", error=str(e), method="chunk_overlap")
            return self._overlap_score(content, chunks)


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 2]
