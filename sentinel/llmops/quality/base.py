"""Abstract base class for response quality evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sentinel.core.types import QualityScore


class BaseEvaluator(ABC):
    """Pluggable response quality evaluator.

    Implementations might call an LLM judge, run heuristic rules, or
    compare the response against a reference answer. They all return a
    :class:`QualityScore` with the same shape so callers don't care.
    """

    method: str = "base"

    @abstractmethod
    def evaluate(
        self,
        response: str,
        query: str | None = None,
        context: dict[str, Any] | None = None,
        reference: str | None = None,
    ) -> QualityScore:
        """Score a single response.

        Args:
            response: The LLM output.
            query: The original user query, when available.
            context: Retrieved context (RAG) or conversation history.
            reference: A golden answer to compare against, if any.
        """
