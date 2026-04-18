"""Regulatory language guardrail — block prohibited phrases."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail


class RegulatoryLanguageGuardrail(BaseGuardrail):
    """Detect prohibited phrases (e.g. financial advice, medical claims).

    The phrase list can be embedded in config or loaded from a YAML file
    that compliance teams maintain separately from code.
    """

    name = "regulatory_language"
    direction = "output"

    def __init__(
        self,
        action: Literal["block", "warn", "redact"] = "block",
        prohibited_phrases: list[str] | None = None,
        prohibited_phrases_file: str | None = None,
        case_sensitive: bool = False,
        **kwargs: Any,
    ):
        super().__init__(action=action, **kwargs)
        phrases = list(prohibited_phrases or [])
        if prohibited_phrases_file:
            phrases.extend(self._load_file(prohibited_phrases_file))
        self.case_sensitive = case_sensitive
        flags = 0 if case_sensitive else re.IGNORECASE
        self.patterns = [re.compile(re.escape(p), flags) for p in phrases]
        self.phrases = phrases

    @staticmethod
    def _load_file(path: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            return []
        try:
            import yaml  # type: ignore[import-not-found]

            data = yaml.safe_load(p.read_text())
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                return [str(x) for x in data.get("prohibited", [])]
        except Exception:
            return p.read_text().splitlines()
        return []

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        hits = [self.phrases[i] for i, p in enumerate(self.patterns) if p.search(content)]
        if hits:
            return self._result(
                passed=False,
                score=min(1.0, len(hits) * 0.3),
                reason=f"prohibited phrases: {hits[:3]}",
                metadata={"hits": hits},
            )
        return self._result(passed=True, score=0.0)
