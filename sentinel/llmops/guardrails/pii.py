"""PII detection and redaction guardrail.

Tries Microsoft Presidio first; falls back to regex patterns for the
common BFSI-relevant entity types so the guardrail works without any
extra dependencies.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail

_REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b"),
    "account_number": re.compile(r"\b\d{8,12}\b"),
}


class PIIGuardrail(BaseGuardrail):
    """Detect (and optionally redact) PII in text.

    Config:
        entities: list of entity types to detect (default: all known)
        redaction_strategy: "mask" | "hash" | "placeholder"
        action: "redact" | "block" | "warn"
    """

    name = "pii_detection"
    direction = "both"

    def __init__(
        self,
        action: Literal["block", "warn", "redact"] = "redact",
        entities: list[str] | None = None,
        redaction_strategy: Literal["mask", "hash", "placeholder"] = "mask",
        **kwargs: Any,
    ):
        super().__init__(action=action, **kwargs)
        self.entities = entities or list(_REGEX_PATTERNS.keys())
        self.strategy = redaction_strategy
        self._presidio = self._try_load_presidio()

    @staticmethod
    def _try_load_presidio() -> Any:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
            from presidio_anonymizer import AnonymizerEngine  # type: ignore[import-not-found]

            return (AnalyzerEngine(), AnonymizerEngine())
        except Exception:
            return None

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        findings = self._scan(content)
        if not findings:
            return self._result(passed=True, score=0.0)

        sanitised = content
        if self.action == "redact":
            sanitised = self._redact(content, findings)

        return self._result(
            passed=False,
            score=min(1.0, len(findings) / 10.0),
            reason=f"detected {len(findings)} PII entit(ies): {sorted({f[0] for f in findings})}",
            sanitised=sanitised,
            metadata={"findings": [{"type": t, "count": 1} for t, v, _, _ in findings]},
        )

    def _scan(self, content: str) -> list[tuple[str, str, int, int]]:
        if self._presidio:
            return self._presidio_scan(content)
        return self._regex_scan(content)

    def _regex_scan(self, content: str) -> list[tuple[str, str, int, int]]:
        out: list[tuple[str, str, int, int]] = []
        for entity in self.entities:
            pattern = _REGEX_PATTERNS.get(entity)
            if pattern is None:
                continue
            for m in pattern.finditer(content):
                out.append((entity, m.group(0), m.start(), m.end()))
        return out

    def _presidio_scan(self, content: str) -> list[tuple[str, str, int, int]]:
        analyzer, _ = self._presidio
        results = analyzer.analyze(text=content, language="en")
        return [(r.entity_type.lower(), content[r.start : r.end], r.start, r.end) for r in results]

    def _redact(self, content: str, findings: list[tuple[str, str, int, int]]) -> str:
        # Apply replacements in reverse so spans stay valid
        out = content
        for entity, value, start, end in sorted(findings, key=lambda f: -f[2]):
            replacement = self._replacement(entity, value)
            out = out[:start] + replacement + out[end:]
        return out

    def _replacement(self, entity: str, value: str) -> str:
        if self.strategy == "hash":
            return f"[{entity.upper()}:{self._hash(value)}]"
        if self.strategy == "placeholder":
            return f"[{entity.upper()}]"
        return "*" * len(value)

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()[:8]
