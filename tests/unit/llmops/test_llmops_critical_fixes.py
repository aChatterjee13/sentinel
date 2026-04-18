"""Tests for the four critical LLMOps fixes.

1. Judge factory — provider auto-detection and callable interface
2. PromptManager.get_stats / get_all_stats
3. Critical guardrail init failure mode
4. Toxicity redaction with sanitised_content
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel.config.schema import (
    GuardrailRuleConfig,
    GuardrailsConfig,
    LLMOpsConfig,
    QualityEvaluatorConfig,
)
from sentinel.core.exceptions import LLMOpsError
from sentinel.llmops.guardrails.engine import GuardrailPipeline
from sentinel.llmops.guardrails.toxicity import ToxicityGuardrail
from sentinel.llmops.prompt_manager import PromptManager
from sentinel.llmops.quality.judge_factory import (
    _build_system_prompt,
    _build_user_prompt,
    _parse_scores,
    create_judge_fn,
)

# ── Fix 1: Judge factory ────────────────────────────────────────────


class TestJudgeFactory:
    """create_judge_fn returns None when no LLM package is available,
    and a working callable when mocked."""

    def test_returns_none_when_method_is_heuristic(self) -> None:
        cfg = QualityEvaluatorConfig(method="heuristic", judge_model="gpt-4o-mini")
        assert create_judge_fn(cfg) is None

    def test_returns_none_when_no_judge_model(self) -> None:
        cfg = QualityEvaluatorConfig(method="llm_judge", judge_model=None)
        assert create_judge_fn(cfg) is None

    def test_returns_none_when_openai_not_installed(self) -> None:
        cfg = QualityEvaluatorConfig(method="llm_judge", judge_model="gpt-4o-mini")
        with patch.dict("sys.modules", {"openai": None}):
            assert create_judge_fn(cfg) is None

    def test_returns_none_when_anthropic_not_installed(self) -> None:
        cfg = QualityEvaluatorConfig(method="llm_judge", judge_model="claude-3-haiku")
        with patch.dict("sys.modules", {"anthropic": None}):
            assert create_judge_fn(cfg) is None

    def test_openai_callable_interface(self) -> None:
        """Mock the openai package and verify the callable returns rubric scores."""
        cfg = QualityEvaluatorConfig(
            method="llm_judge",
            judge_model="gpt-4o-mini",
            rubrics={
                "relevance": {"weight": 0.5, "scale": 5},
                "clarity": {"weight": 0.5, "scale": 5},
            },
        )
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"relevance": 4.5, "clarity": 3.0})
        mock_openai.OpenAI.return_value.chat.completions.create.return_value.choices = [
            mock_choice
        ]

        with patch.dict("sys.modules", {"openai": mock_openai}):
            fn = create_judge_fn(cfg)

        assert fn is not None
        scores = fn("Answer text", "Question?", None)
        assert "relevance" in scores
        assert "clarity" in scores
        assert scores["relevance"] == 4.5
        assert scores["clarity"] == 3.0

    def test_anthropic_callable_interface(self) -> None:
        cfg = QualityEvaluatorConfig(
            method="llm_judge",
            judge_model="claude-3-haiku",
            rubrics={
                "relevance": {"weight": 0.5, "scale": 5},
                "safety": {"weight": 0.5, "scale": 5},
            },
        )
        mock_anthropic = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = json.dumps({"relevance": 3.0, "safety": 5.0})
        mock_anthropic.Anthropic.return_value.messages.create.return_value.content = [
            mock_content_block
        ]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            fn = create_judge_fn(cfg)

        assert fn is not None
        scores = fn("Safe answer", "Is this safe?", "Some context")
        assert scores["relevance"] == 3.0
        assert scores["safety"] == 5.0

    def test_scores_are_clamped_to_scale(self) -> None:
        rubrics = {"relevance": {"weight": 1.0, "scale": 5}}
        raw = '{"relevance": 99}'
        scores = _parse_scores(raw, rubrics)
        assert scores["relevance"] == 5.0

    def test_scores_clamped_non_negative(self) -> None:
        rubrics = {"relevance": {"weight": 1.0, "scale": 5}}
        raw = '{"relevance": -2}'
        scores = _parse_scores(raw, rubrics)
        assert scores["relevance"] == 0.0

    def test_parse_scores_bad_json_raises(self) -> None:
        with pytest.raises(ValueError, match="did not return valid JSON"):
            _parse_scores("no json here", {"r": {"scale": 5}})

    def test_build_system_prompt_contains_rubrics(self) -> None:
        rubrics = {"relevance": {"scale": 5}, "clarity": {"scale": 10}}
        prompt = _build_system_prompt(rubrics)
        assert "relevance" in prompt
        assert "clarity" in prompt
        assert "0 to 5" in prompt
        assert "0 to 10" in prompt

    def test_build_user_prompt_includes_context(self) -> None:
        prompt = _build_user_prompt("resp", "query", "ctx")
        assert "resp" in prompt
        assert "query" in prompt
        assert "ctx" in prompt

    def test_build_user_prompt_without_context(self) -> None:
        prompt = _build_user_prompt("resp", "query", None)
        assert "Context" not in prompt


# ── Fix 2: PromptManager.get_stats / get_all_stats ──────────────────


class TestPromptManagerStats:
    """Stats are populated by log_result and retrievable via get_stats."""

    @pytest.fixture()
    def manager(self, tmp_path: Any) -> PromptManager:
        cfg = LLMOpsConfig()
        pm = PromptManager(cfg, root=tmp_path / "prompts")
        pm.register(
            name="summarise",
            version="1.0",
            system_prompt="You are a summariser.",
            template="Summarise: {{ text }}",
        )
        pm.register(
            name="summarise",
            version="2.0",
            system_prompt="You are a summariser v2.",
            template="Summarise v2: {{ text }}",
        )
        return pm

    def test_get_stats_empty_before_log(self, manager: PromptManager) -> None:
        stats = manager.get_stats("summarise", "1.0")
        assert stats == {}

    def test_get_stats_after_log(self, manager: PromptManager) -> None:
        manager.log_result("summarise", "1.0", input_tokens=100, output_tokens=50, quality_score=0.9, latency_ms=200.0)
        stats = manager.get_stats("summarise", "1.0")
        assert stats["calls"] == 1
        assert stats["avg_input_tokens"] == 100.0
        assert stats["avg_output_tokens"] == 50.0
        assert stats["avg_quality"] == 0.9
        assert stats["avg_latency_ms"] == 200.0

    def test_get_stats_rolling_average(self, manager: PromptManager) -> None:
        manager.log_result("summarise", "1.0", input_tokens=100, output_tokens=50)
        manager.log_result("summarise", "1.0", input_tokens=200, output_tokens=100)
        stats = manager.get_stats("summarise", "1.0")
        assert stats["calls"] == 2
        assert stats["avg_input_tokens"] == pytest.approx(150.0)
        assert stats["avg_output_tokens"] == pytest.approx(75.0)

    def test_get_stats_unknown_prompt_raises(self, manager: PromptManager) -> None:
        with pytest.raises(LLMOpsError, match="not found"):
            manager.get_stats("nonexistent", "1.0")

    def test_get_all_stats(self, manager: PromptManager) -> None:
        manager.log_result("summarise", "1.0", input_tokens=100, output_tokens=50)
        manager.log_result("summarise", "2.0", input_tokens=200, output_tokens=80)
        all_stats = manager.get_all_stats()
        assert "summarise" in all_stats
        assert "1.0" in all_stats["summarise"]
        assert "2.0" in all_stats["summarise"]
        assert all_stats["summarise"]["1.0"]["calls"] == 1
        assert all_stats["summarise"]["2.0"]["calls"] == 1

    def test_get_all_stats_empty(self, tmp_path: Any) -> None:
        pm = PromptManager(LLMOpsConfig(), root=tmp_path / "empty_prompts")
        assert pm.get_all_stats() == {}

    def test_get_stats_returns_copy(self, manager: PromptManager) -> None:
        """Mutations on the returned dict must not affect internal state."""
        manager.log_result("summarise", "1.0", input_tokens=100, output_tokens=50)
        stats = manager.get_stats("summarise", "1.0")
        stats["calls"] = 999
        assert manager.get_stats("summarise", "1.0")["calls"] == 1


# ── Fix 3: Critical guardrail failure ───────────────────────────────


class TestCriticalGuardrailInit:
    """critical=True raises; critical=False logs and skips."""

    def test_critical_true_raises_on_init_failure(self) -> None:
        rules = GuardrailsConfig(
            input=[
                GuardrailRuleConfig(type="nonexistent_guardrail", action="block", critical=True)
            ],
            output=[],
        )
        cfg = LLMOpsConfig(guardrails=rules)
        with pytest.raises(LLMOpsError, match="critical guardrail"):
            GuardrailPipeline.from_config(cfg)

    def test_critical_false_skips_on_init_failure(self) -> None:
        rules = GuardrailsConfig(
            input=[
                GuardrailRuleConfig(type="nonexistent_guardrail", action="block", critical=False)
            ],
            output=[],
        )
        cfg = LLMOpsConfig(guardrails=rules)
        pipeline = GuardrailPipeline.from_config(cfg)
        assert pipeline.input_guardrails == []

    def test_critical_default_is_false(self) -> None:
        rule = GuardrailRuleConfig(type="toxicity", action="warn")
        assert rule.critical is False

    def test_valid_guardrail_loads_regardless_of_critical(self) -> None:
        rules = GuardrailsConfig(
            input=[],
            output=[
                GuardrailRuleConfig(type="toxicity", action="warn", threshold=0.5, critical=True)
            ],
        )
        cfg = LLMOpsConfig(guardrails=rules)
        pipeline = GuardrailPipeline.from_config(cfg)
        assert len(pipeline.output_guardrails) == 1


# ── Fix 4: Toxicity redaction ───────────────────────────────────────


class TestToxicityRedaction:
    """action='redact' returns sanitised_content with toxic terms replaced."""

    def test_redact_replaces_heuristic_terms(self) -> None:
        guardrail = ToxicityGuardrail(action="redact", threshold=0.3)
        result = guardrail.check("This contains hate speech")
        assert not result.passed
        assert result.sanitised_content is not None
        assert "[REDACTED]" in result.sanitised_content
        assert "hate" not in result.sanitised_content.lower()

    def test_redact_metadata_has_action(self) -> None:
        guardrail = ToxicityGuardrail(action="redact", threshold=0.3)
        result = guardrail.check("This contains hate speech")
        assert result.metadata.get("action") == "redacted"

    def test_redact_reason_mentions_redacted(self) -> None:
        guardrail = ToxicityGuardrail(action="redact", threshold=0.3)
        result = guardrail.check("This is hate")
        assert result.reason is not None
        assert "redacted" in result.reason.lower()

    def test_block_does_not_return_sanitised_content(self) -> None:
        guardrail = ToxicityGuardrail(action="block", threshold=0.3)
        result = guardrail.check("This contains hate speech")
        assert result.blocked is True
        assert result.sanitised_content is None

    def test_redact_not_blocked(self) -> None:
        """Redaction should sanitise, not block."""
        guardrail = ToxicityGuardrail(action="redact", threshold=0.3)
        result = guardrail.check("This contains hate speech")
        assert result.blocked is False

    def test_clean_content_passes(self) -> None:
        guardrail = ToxicityGuardrail(action="redact", threshold=0.3)
        result = guardrail.check("This is perfectly fine content")
        assert result.passed is True
        assert result.sanitised_content is None

    def test_multiple_toxic_terms_all_redacted(self) -> None:
        guardrail = ToxicityGuardrail(action="redact", threshold=0.3)
        result = guardrail.check("hate and violent threat together")
        assert result.sanitised_content is not None
        assert "hate" not in result.sanitised_content.lower()
        assert "violent threat" not in result.sanitised_content.lower()
        assert result.sanitised_content.count("[REDACTED]") >= 2

    def test_redact_preserves_clean_words(self) -> None:
        guardrail = ToxicityGuardrail(action="redact", threshold=0.3)
        result = guardrail.check("I love dogs but hate cats")
        assert result.sanitised_content is not None
        assert "dogs" in result.sanitised_content
        assert "cats" in result.sanitised_content
        assert "hate" not in result.sanitised_content.lower()

    def test_redact_detoxify_path(self) -> None:
        """Simulate detoxify classifier scoring sentences."""
        guardrail = ToxicityGuardrail(action="redact", threshold=0.5)
        # Mock the classifier to make the first sentence toxic
        mock_classifier = MagicMock()
        mock_classifier.predict.side_effect = lambda s: (
            {"toxicity": 0.9} if "bad" in s.lower() else {"toxicity": 0.1}
        )
        guardrail._classifier = mock_classifier
        result = guardrail.check("Bad sentence here. Good sentence here.")
        assert result.sanitised_content is not None
        assert "[Content removed: toxicity detected]" in result.sanitised_content
        assert "Good sentence here." in result.sanitised_content
