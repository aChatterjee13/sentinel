"""Abstract base class for input/output guardrails."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from sentinel.core.types import AlertSeverity, GuardrailResult


class BaseGuardrail(ABC):
    """A pluggable safety check applied to LLM inputs or outputs.

    Subclasses implement :meth:`check` and return a
    :class:`GuardrailResult`. The pipeline short-circuits on the first
    ``blocked`` result, so order matters.
    """

    name: str = "base"
    direction: Literal["input", "output", "both"] = "both"

    def __init__(self, action: Literal["block", "warn", "redact"] = "warn", **kwargs: Any):
        self.action = action
        self._config = kwargs

    @abstractmethod
    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Run the guardrail on a string of content.

        Args:
            content: The text being checked (input or output).
            context: Optional context — retrieved chunks for groundedness,
                conversation history, etc.

        Returns:
            A :class:`GuardrailResult` describing pass/warn/block.
        """

    def _result(
        self,
        passed: bool,
        *,
        score: float = 0.0,
        reason: str = "",
        sanitised: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        if passed:
            severity = AlertSeverity.INFO
            blocked = False
        elif self.action == "block":
            severity = AlertSeverity.HIGH
            blocked = True
        elif self.action == "redact":
            severity = AlertSeverity.WARNING
            blocked = False
        else:
            severity = AlertSeverity.WARNING
            blocked = False
        return GuardrailResult(
            name=self.name,
            passed=passed,
            blocked=blocked,
            severity=severity,
            score=score,
            reason=reason or None,
            sanitised_content=sanitised,
            metadata=metadata or {},
        )
