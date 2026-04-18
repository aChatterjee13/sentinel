"""Format compliance — verify outputs match the expected schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail


class FormatComplianceGuardrail(BaseGuardrail):
    """Validate that responses parse as JSON and match an optional schema."""

    name = "format_compliance"
    direction = "output"

    def __init__(
        self,
        action: Literal["block", "warn", "redact"] = "warn",
        expected_schema: str | dict[str, Any] | None = None,
        require_json: bool = True,
        **kwargs: Any,
    ):
        super().__init__(action=action, **kwargs)
        self.require_json = require_json
        self.schema = self._load_schema(expected_schema) if expected_schema else None

    @staticmethod
    def _load_schema(schema: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(schema, dict):
            return schema
        path = Path(schema)
        if path.exists():
            return json.loads(path.read_text())
        try:
            return json.loads(schema)
        except json.JSONDecodeError:
            return {}

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        if not self.require_json and not self.schema:
            return self._result(passed=True, score=1.0)

        max_len = (self.schema or {}).get("maxLength")
        min_len = (self.schema or {}).get("minLength")
        if not self.require_json and (max_len or min_len):
            if max_len and len(content) > max_len:
                return self._result(
                    passed=False, score=0.5, reason=f"output too long: {len(content)} > {max_len}"
                )
            if min_len and len(content) < min_len:
                return self._result(
                    passed=False, score=0.5, reason=f"output too short: {len(content)} < {min_len}"
                )
            return self._result(passed=True, score=1.0)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return self._result(
                passed=False,
                score=0.0,
                reason=f"output not valid JSON: {e.msg}",
            )

        if self.schema:
            missing = [k for k in self.schema.get("required", []) if k not in data]
            if missing:
                return self._result(
                    passed=False,
                    score=0.5,
                    reason=f"missing required fields: {missing}",
                    metadata={"missing": missing},
                )
        return self._result(passed=True, score=1.0)
