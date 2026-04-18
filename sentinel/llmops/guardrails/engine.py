"""Guardrail pipeline — composes input + output guardrails."""

from __future__ import annotations

import threading
from typing import Any

import structlog

from sentinel.config.schema import GuardrailRuleConfig, GuardrailsConfig, LLMOpsConfig
from sentinel.core.exceptions import LLMOpsError
from sentinel.core.types import PipelineResult
from sentinel.llmops.guardrails.base import BaseGuardrail

log = structlog.get_logger(__name__)


class GuardrailPipeline:
    """Runs the configured input and output guardrails in order.

    Short-circuits on the first ``blocked`` result. Warnings are
    aggregated and returned for logging without halting execution.

    Pipeline execution is serialised with a :class:`threading.RLock` to
    protect guardrail implementations that may carry internal state
    (e.g. lazy-loaded classifiers). Guardrail implementations should be
    stateless where possible, but the pipeline does not assume this.
    """

    def __init__(
        self,
        input_guardrails: list[BaseGuardrail] | None = None,
        output_guardrails: list[BaseGuardrail] | None = None,
        audit_trail: Any | None = None,
    ):
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self._audit = audit_trail
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, config: LLMOpsConfig | str | Any) -> GuardrailPipeline:
        """Build a pipeline from an :class:`LLMOpsConfig` or YAML path."""
        if isinstance(config, str):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            llm_cfg: LLMOpsConfig = cfg.llmops
        elif isinstance(config, LLMOpsConfig):
            llm_cfg = config
        else:
            llm_cfg = config.llmops  # type: ignore[union-attr]

        return cls(
            input_guardrails=cls._build(llm_cfg.guardrails, "input"),
            output_guardrails=cls._build(llm_cfg.guardrails, "output"),
        )

    @staticmethod
    def _build(cfg: GuardrailsConfig, direction: str) -> list[BaseGuardrail]:
        from sentinel.llmops.guardrails import resolve_guardrail

        rules: list[GuardrailRuleConfig] = getattr(cfg, direction)
        out: list[BaseGuardrail] = []
        for rule in rules:
            try:
                cls = resolve_guardrail(rule.type)
                kwargs = rule.model_dump(
                    exclude={"type", "action", "critical"}, exclude_none=True
                )
                out.append(cls(action=rule.action, **kwargs))
            except Exception as e:
                if rule.critical:
                    raise LLMOpsError(
                        f"critical guardrail '{rule.type}' failed to initialize: {e}"
                    ) from e
                log.warning("guardrail.build_failed", type=rule.type, error=str(e))
        return out

    # ── Run ───────────────────────────────────────────────────────

    def check_input(self, content: str, context: dict[str, Any] | None = None) -> PipelineResult:
        with self._lock:
            return self._run(self.input_guardrails, content, context, audit=self._audit)

    def check_output(self, content: str, context: dict[str, Any] | None = None) -> PipelineResult:
        with self._lock:
            return self._run(self.output_guardrails, content, context, audit=self._audit)

    @staticmethod
    def _run(
        guardrails: list[BaseGuardrail],
        content: str,
        context: dict[str, Any] | None,
        audit: Any | None = None,
    ) -> PipelineResult:
        sanitised = content
        results = []
        warnings: list[str] = []
        for g in guardrails:
            result = g.check(sanitised, context=context)
            results.append(result)
            if audit is not None:
                audit.log(
                    event_type="guardrail_checked",
                    guardrail=result.name,
                    passed=result.passed,
                    blocked=result.blocked,
                    reason=result.reason,
                )
            if result.sanitised_content is not None:
                sanitised = result.sanitised_content
            if result.blocked:
                return PipelineResult(
                    blocked=True,
                    results=results,
                    sanitised_input=None,
                    reason=result.reason or f"{result.name} blocked",
                    warnings=warnings,
                )
            if not result.passed and result.reason:
                warnings.append(f"{result.name}: {result.reason}")
        return PipelineResult(
            blocked=False,
            results=results,
            sanitised_input=sanitised,
            warnings=warnings,
        )
