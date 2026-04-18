"""Unit tests for PluginGuardrail — dynamic class loading."""

from __future__ import annotations

from typing import Any

import pytest

from sentinel.core.types import AlertSeverity, GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail
from sentinel.llmops.guardrails.plugin import (
    PluginGuardrail,
    PluginLoadError,
)

# ── Test fixtures (inline classes) ─────────────────────────────────


class _PassthroughGuardrail(BaseGuardrail):
    """Always passes — used as a loadable fixture."""

    name = "passthrough"

    def __init__(self, multiplier: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.multiplier = multiplier

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        return self._result(passed=True, score=0.0)


class _AlwaysBlockGuardrail(BaseGuardrail):
    """Always blocks — used as a loadable fixture."""

    name = "always_block"

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        return self._result(passed=False, score=1.0, reason="always blocked")


class _DuckTypedChecker:
    """Not a BaseGuardrail subclass but has a check() method."""

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        return GuardrailResult(
            name="duck",
            passed=True,
            blocked=False,
            severity=AlertSeverity.INFO,
            score=0.0,
        )


class _NoCheckMethod:
    """Has no check() method — should be rejected."""

    pass


# ── Loading valid classes ──────────────────────────────────────────


# Trusted prefix that allows loading test-local fixtures.
_TEST_PREFIXES = ("sentinel.", "tests.",)


class TestPluginLoading:
    def test_load_from_toxicity_module(self) -> None:
        """Load a real guardrail from the sentinel package."""
        g = PluginGuardrail(
            module="sentinel.llmops.guardrails.toxicity",
            class_name="ToxicityGuardrail",
            action="warn",
            config={"threshold": 0.9},
        )
        result = g.check("hello world")
        assert result.passed

    def test_load_with_config_forwarding(self) -> None:
        """Config dict is forwarded as kwargs to the loaded class."""
        g = PluginGuardrail(
            module=__name__,
            class_name="_PassthroughGuardrail",
            config={"multiplier": 42},
            trusted_prefixes=_TEST_PREFIXES,
        )
        assert g._inner.multiplier == 42  # type: ignore[attr-defined]

    def test_always_block_plugin(self) -> None:
        g = PluginGuardrail(
            module=__name__,
            class_name="_AlwaysBlockGuardrail",
            action="block",
            trusted_prefixes=_TEST_PREFIXES,
        )
        result = g.check("anything")
        assert not result.passed

    def test_duck_typed_class_accepted(self) -> None:
        """A class with check() but not extending BaseGuardrail works."""
        g = PluginGuardrail(
            module=__name__,
            class_name="_DuckTypedChecker",
            trusted_prefixes=_TEST_PREFIXES,
        )
        result = g.check("hello")
        assert result.passed


# ── Error handling ─────────────────────────────────────────────────


class TestPluginErrors:
    def test_invalid_module_path(self) -> None:
        with pytest.raises(PluginLoadError, match="Cannot import module"):
            PluginGuardrail(
                module="sentinel.nonexistent.module.path",
                class_name="Foo",
            )

    def test_invalid_class_name(self) -> None:
        with pytest.raises(PluginLoadError, match="has no attribute"):
            PluginGuardrail(
                module="sentinel.llmops.guardrails.toxicity",
                class_name="NonexistentClass",
            )

    def test_class_without_check(self) -> None:
        with pytest.raises(PluginLoadError, match="does not expose a check"):
            PluginGuardrail(
                module=__name__,
                class_name="_NoCheckMethod",
                trusted_prefixes=_TEST_PREFIXES,
            )

    def test_non_callable_attribute(self) -> None:
        """Trying to load a module-level constant raises PluginLoadError."""
        with pytest.raises(PluginLoadError, match="is not callable"):
            PluginGuardrail(
                module=__name__,
                class_name="pytest",  # pytest is a module, not callable ctor
                trusted_prefixes=_TEST_PREFIXES,
            )

    def test_untrusted_module_prefix_rejected(self) -> None:
        """Loading from a module not in trusted_prefixes raises PluginLoadError."""
        with pytest.raises(PluginLoadError, match="not in trusted prefixes"):
            PluginGuardrail(
                module="some_random_package.guardrails",
                class_name="SomeClass",
            )

    def test_blocked_stdlib_module_rejected(self) -> None:
        """Dangerous stdlib modules are blocked even with matching prefix."""
        with pytest.raises(PluginLoadError, match="blocked for security"):
            PluginGuardrail(
                module="os",
                class_name="path",
                trusted_prefixes=("os",),
            )


# ── Registry integration ──────────────────────────────────────────


class TestPluginRegistry:
    def test_plugin_in_registry(self) -> None:
        from sentinel.llmops.guardrails import resolve_guardrail

        cls = resolve_guardrail("plugin")
        assert cls is PluginGuardrail

    def test_pipeline_builds_plugin(self) -> None:
        from sentinel.config.schema import GuardrailRuleConfig, GuardrailsConfig, LLMOpsConfig
        from sentinel.llmops.guardrails.engine import GuardrailPipeline

        rule = GuardrailRuleConfig(
            type="plugin",
            action="warn",
            module=__name__,
            class_name="_PassthroughGuardrail",
            trusted_prefixes=("sentinel.", "tests."),
        )
        cfg = LLMOpsConfig(guardrails=GuardrailsConfig(input=[rule]))
        pipeline = GuardrailPipeline.from_config(cfg)
        assert len(pipeline.input_guardrails) == 1
        result = pipeline.check_input("hello")
        assert result.passed


# ── Context forwarding ─────────────────────────────────────────────


class TestContextForwarding:
    def test_context_passed_through(self) -> None:
        g = PluginGuardrail(
            module=__name__,
            class_name="_PassthroughGuardrail",
            trusted_prefixes=_TEST_PREFIXES,
        )
        result = g.check("content", context={"key": "value"})
        assert result.passed
