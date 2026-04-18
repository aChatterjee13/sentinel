"""Human escalation triggers — confidence, failures, sensitive context."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from sentinel.config.schema import EscalationConfig, EscalationTrigger
from sentinel.core.exceptions import EscalationRequired

log = structlog.get_logger(__name__)


@dataclass
class EscalationDecision:
    """Result of evaluating escalation triggers."""

    triggered: bool
    reason: str | None = None
    action: str | None = None
    trigger_name: str | None = None


class EscalationManager:
    """Evaluate the configured escalation triggers on every reasoning step.

    Triggers fire when:

    - Confidence drops below a threshold
    - Consecutive tool failures exceed a count
    - Sensitive content is detected in planned actions
    - The agent enters a regulated context (financial advice, medical, legal)
    """

    def __init__(
        self,
        config: EscalationConfig | None = None,
        notification_callback: Callable[..., Any] | None = None,
    ):
        self.config = config or EscalationConfig()
        self._consecutive_failures: dict[str, int] = {}
        self._notification_callback = notification_callback
        self._lock = threading.Lock()

    def check(
        self,
        run_id: str,
        *,
        confidence: float | None = None,
        action_text: str | None = None,
        success: bool | None = None,
    ) -> EscalationDecision:
        with self._lock:
            if success is False:
                self._consecutive_failures[run_id] = self._consecutive_failures.get(run_id, 0) + 1
            elif success is True:
                self._consecutive_failures[run_id] = 0

            for trigger in self.config.triggers:
                decision = self._evaluate(trigger, run_id, confidence, action_text)
                if decision.triggered:
                    log.warning("escalation.triggered", run_id=run_id, trigger=trigger.condition)
                    if self._notification_callback is not None:
                        try:
                            self._notification_callback(
                                run_id=run_id,
                                trigger=trigger.condition,
                                action=decision.action,
                                reason=decision.reason,
                            )
                        except Exception as exc:
                            log.error("escalation.notification_failed", error=str(exc))
                    return decision
            return EscalationDecision(triggered=False)

    def raise_if_required(self, decision: EscalationDecision) -> None:
        if decision.triggered:
            raise EscalationRequired(decision.reason or "escalation required")

    # ── Trigger evaluators ────────────────────────────────────────

    def _evaluate(
        self,
        trigger: EscalationTrigger,
        run_id: str,
        confidence: float | None,
        action_text: str | None,
    ) -> EscalationDecision:
        cond = trigger.condition
        if (
            cond == "confidence_below"
            and confidence is not None
            and trigger.threshold is not None
            and confidence < trigger.threshold
        ):
            return EscalationDecision(
                triggered=True,
                reason=f"confidence {confidence:.2f} < {trigger.threshold}",
                action=trigger.action,
                trigger_name=cond,
            )

        if cond == "consecutive_tool_failures" and trigger.threshold is not None:
            failures = self._consecutive_failures.get(run_id, 0)
            if failures >= trigger.threshold:
                return EscalationDecision(
                    triggered=True,
                    reason=f"{failures} consecutive tool failures",
                    action=trigger.action,
                    trigger_name=cond,
                )

        if cond == "sensitive_data_detected" and action_text and _has_sensitive_data(action_text):
            return EscalationDecision(
                triggered=True,
                reason="sensitive data detected in action",
                action=trigger.action,
                trigger_name=cond,
            )

        if cond == "regulatory_context" and action_text and trigger.patterns:
            for pattern in trigger.patterns:
                if re.search(pattern, action_text, re.IGNORECASE):
                    return EscalationDecision(
                        triggered=True,
                        reason=f"regulated context: {pattern}",
                        action=trigger.action,
                        trigger_name=cond,
                    )

        return EscalationDecision(triggered=False)


def _has_sensitive_data(text: str) -> bool:
    patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN (US)
        r"\b(?:\d[ -]*?){13,16}\b",  # credit card number
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",  # email address
        r"\biban\s*:",  # IBAN reference
        r"\bpassword\s*:",  # password in payload
        r"\bapi[_-]?key\s*[:=]",  # API key assignment
        r"\b(?:secret|token)\s*[:=]",  # secret/token assignment
        r"\baccount[_\s-]?(?:number|num|no)\s*[:=]",  # account number
        r"\bsort[_\s-]?code\s*[:=]",  # UK sort code
        r"\brouting[_\s-]?(?:number|num)\s*[:=]",  # US routing number
        r"\b\d{2}/\d{2}/\d{4}\b.*\b(?:dob|birth)\b",  # date of birth near keyword
        r"\b(?:passport|license|licence)\s*(?:no|number|num)\s*[:=]",  # ID documents
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)
