"""Guardrail engine — input/output safety pipeline.

Each guardrail implements :class:`BaseGuardrail` and is registered in
``GUARDRAIL_REGISTRY`` so the YAML config can refer to it by name.
"""

from collections.abc import Callable

from sentinel.llmops.guardrails.base import BaseGuardrail
from sentinel.llmops.guardrails.engine import GuardrailPipeline


def _lazy_pii() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.pii import PIIGuardrail

    return PIIGuardrail


def _lazy_jailbreak() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.jailbreak import JailbreakGuardrail

    return JailbreakGuardrail


def _lazy_topic_fence() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.topic_fence import TopicFenceGuardrail

    return TopicFenceGuardrail


def _lazy_token_budget() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.token_budget import TokenBudgetGuardrail

    return TokenBudgetGuardrail


def _lazy_toxicity() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.toxicity import ToxicityGuardrail

    return ToxicityGuardrail


def _lazy_groundedness() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.groundedness import GroundednessGuardrail

    return GroundednessGuardrail


def _lazy_format() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.format_compliance import FormatComplianceGuardrail

    return FormatComplianceGuardrail


def _lazy_regulatory() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.regulatory import RegulatoryLanguageGuardrail

    return RegulatoryLanguageGuardrail


def _lazy_custom() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.custom import CustomGuardrail

    return CustomGuardrail


def _lazy_plugin() -> type[BaseGuardrail]:
    from sentinel.llmops.guardrails.plugin import PluginGuardrail

    return PluginGuardrail


GUARDRAIL_REGISTRY: dict[str, Callable[[], type[BaseGuardrail]]] = {
    "pii_detection": _lazy_pii,
    "jailbreak_detection": _lazy_jailbreak,
    "topic_fence": _lazy_topic_fence,
    "token_budget": _lazy_token_budget,
    "toxicity": _lazy_toxicity,
    "groundedness": _lazy_groundedness,
    "format_compliance": _lazy_format,
    "regulatory_language": _lazy_regulatory,
    "custom": _lazy_custom,
    "plugin": _lazy_plugin,
}


def resolve_guardrail(name: str) -> type[BaseGuardrail]:
    factory = GUARDRAIL_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"unknown guardrail: {name}")
    return factory()


__all__ = [
    "GUARDRAIL_REGISTRY",
    "BaseGuardrail",
    "GuardrailPipeline",
    "resolve_guardrail",
]
