"""LLMOpsClient — convenience wrapper around the LLMOps subsystems."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import structlog

from sentinel.config.schema import LLMOpsConfig
from sentinel.core.types import PipelineResult, QualityScore
from sentinel.foundation.audit.trail import AuditTrail
from sentinel.llmops.guardrails.engine import GuardrailPipeline
from sentinel.llmops.prompt_drift import PromptDriftDetector
from sentinel.llmops.prompt_manager import Prompt, PromptManager
from sentinel.llmops.quality.evaluator import ResponseEvaluator
from sentinel.llmops.quality.retrieval_quality import RetrievalQualityMonitor
from sentinel.llmops.quality.semantic_drift import SemanticDriftMonitor
from sentinel.llmops.token_economics import TokenTracker

log = structlog.get_logger(__name__)


class LLMOpsClient:
    """Single-entry-point wrapper around the LLMOps modules.

    Created lazily by ``SentinelClient.llmops`` and exposes the prompt
    manager, guardrails, evaluator, semantic drift monitor, retrieval
    quality monitor, token tracker, and prompt drift detector through
    ``log_call()`` — a single helper that runs the whole pipeline.
    """

    def __init__(self, config: LLMOpsConfig, audit: AuditTrail | None = None):
        self.config = config
        self.audit = audit
        self.prompts = PromptManager(config)
        self.guardrails = GuardrailPipeline.from_config(config)
        from sentinel.llmops.quality.judge_factory import create_judge_fn

        judge_fn = create_judge_fn(config.quality.evaluator)
        self.evaluator = ResponseEvaluator(config.quality.evaluator, judge_fn=judge_fn)
        self.semantic_drift = SemanticDriftMonitor(config.quality.semantic_drift)
        self.retrieval_quality = RetrievalQualityMonitor(config.quality.retrieval_quality)
        self.token_tracker = TokenTracker(config.token_economics, audit_trail=audit)
        self.prompt_drift = PromptDriftDetector(config.prompt_drift)

    # ── High-level helper ────────────────────────────────────────

    def log_call(
        self,
        *,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        query: str = "",
        response: str = "",
        context_chunks: list[Any] | None = None,
        model: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        guardrail_results: dict[str, PipelineResult] | None = None,
        user_id: str | None = None,
        **dimensions: Any,
    ) -> dict[str, Any]:
        """Record a complete LLM interaction.

        Returns a dict with the per-call metrics so callers can attach
        them to their telemetry without having to introspect each
        sub-module.
        """
        usage = self.token_tracker.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            prompt_version=prompt_version,
            user_id=user_id,
            **dimensions,
        )

        quality: QualityScore | None = None
        if self.evaluator.should_evaluate():
            quality = self.evaluator.evaluate(
                response, query=query, context={"chunks": context_chunks}
            )

        retrieval = None
        if context_chunks:
            retrieval = self.retrieval_quality.evaluate(query, response, context_chunks)

        if response and self.semantic_drift._reference_centroid is not None:
            with contextlib.suppress(Exception):
                self.semantic_drift.observe(response)

        violations = 0
        if guardrail_results:
            violations = sum(1 for r in guardrail_results.values() for w in r.warnings) + sum(
                1 for r in guardrail_results.values() if r.blocked
            )

        if prompt_name and prompt_version:
            self.prompt_drift.observe(
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                quality_score=quality.overall if quality else None,
                guardrail_violations=violations,
                total_tokens=input_tokens + output_tokens,
            )
            self.prompts.log_result(
                prompt_name=prompt_name,
                version=prompt_version,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                quality_score=quality.overall if quality else None,
                latency_ms=latency_ms,
                guardrail_violations=list(guardrail_results or {}),
            )

        if self.audit is not None:
            self.audit.log(
                event_type="llm.call",
                model_name=prompt_name,
                model_version=prompt_version,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=usage.cost_usd,
                latency_ms=latency_ms,
                quality=quality.overall if quality else None,
                guardrail_violations=violations,
            )

        return {
            "usage": usage,
            "quality": quality,
            "retrieval_quality": retrieval,
            "guardrail_violations": violations,
            "logged_at": time.time(),
        }

    # ── Convenience accessors ─────────────────────────────────────

    def resolve_prompt(self, name: str, **context: Any) -> Prompt:
        return self.prompts.resolve(name, context=context)

    def check_input(self, content: str, context: dict[str, Any] | None = None) -> PipelineResult:
        return self.guardrails.check_input(content, context)

    def check_output(self, content: str, context: dict[str, Any] | None = None) -> PipelineResult:
        return self.guardrails.check_output(content, context)

    def fit_semantic_baseline(self, outputs: list[str]) -> None:
        """Initialize the semantic drift baseline from a corpus of reference outputs.

        Args:
            outputs: List of reference LLM responses representing the baseline
                distribution.

        Raises:
            ValueError: If *outputs* is empty.
        """
        self.semantic_drift.fit(outputs)

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return self.token_tracker.estimate_cost(model, input_tokens, output_tokens)
