"""Unit tests for CustomGuardrail — DSL rule engine."""

from __future__ import annotations

import json

import pytest

from sentinel.llmops.guardrails.custom import CustomGuardrail

# ── Helpers ────────────────────────────────────────────────────────


def _make(
    rules: list[dict],
    combine: str = "any",
    action: str = "block",
    name: str = "test",
) -> CustomGuardrail:
    return CustomGuardrail(name=name, action=action, rules=rules, combine=combine)


# ── regex_match ────────────────────────────────────────────────────


class TestRegexMatch:
    def test_pass(self) -> None:
        g = _make([{"rule": "regex_match", "pattern": r"\d{3}"}])
        assert g.check("order 123").passed

    def test_fail(self) -> None:
        g = _make([{"rule": "regex_match", "pattern": r"\d{3}"}])
        assert not g.check("no numbers").passed

    def test_case_insensitive(self) -> None:
        g = _make([{"rule": "regex_match", "pattern": "hello", "case_insensitive": True}])
        assert g.check("HELLO world").passed


# ── regex_absent ───────────────────────────────────────────────────


class TestRegexAbsent:
    def test_pass(self) -> None:
        g = _make([{"rule": "regex_absent", "pattern": r"secret"}])
        assert g.check("nothing here").passed

    def test_fail(self) -> None:
        g = _make([{"rule": "regex_absent", "pattern": r"secret"}])
        assert not g.check("my secret key").passed

    def test_case_insensitive(self) -> None:
        g = _make([{"rule": "regex_absent", "pattern": "secret", "case_insensitive": True}])
        assert not g.check("my SECRET key").passed


# ── keyword_present ────────────────────────────────────────────────


class TestKeywordPresent:
    def test_pass(self) -> None:
        g = _make([{"rule": "keyword_present", "keywords": ["hello", "world"]}])
        assert g.check("hello there").passed

    def test_fail(self) -> None:
        g = _make([{"rule": "keyword_present", "keywords": ["hello", "world"]}])
        assert not g.check("goodbye").passed


# ── keyword_absent ─────────────────────────────────────────────────


class TestKeywordAbsent:
    def test_pass(self) -> None:
        g = _make([{"rule": "keyword_absent", "keywords": ["spam", "junk"]}])
        assert g.check("good content").passed

    def test_fail(self) -> None:
        g = _make([{"rule": "keyword_absent", "keywords": ["spam", "junk"]}])
        assert not g.check("this is spam").passed


# ── min_length ─────────────────────────────────────────────────────


class TestMinLength:
    def test_pass(self) -> None:
        g = _make([{"rule": "min_length", "min_chars": 5}])
        assert g.check("hello world").passed

    def test_fail(self) -> None:
        g = _make([{"rule": "min_length", "min_chars": 50}])
        assert not g.check("short").passed


# ── max_length ─────────────────────────────────────────────────────


class TestMaxLength:
    def test_pass(self) -> None:
        g = _make([{"rule": "max_length", "max_chars": 100}])
        assert g.check("short").passed

    def test_fail(self) -> None:
        g = _make([{"rule": "max_length", "max_chars": 5}])
        result = g.check("this is way too long")
        assert not result.passed


# ── json_schema ────────────────────────────────────────────────────


class TestJsonSchema:
    def test_valid_json_passes(self) -> None:
        schema = {"type": "object", "required": ["name"]}
        content = json.dumps({"name": "Alice"})
        g = _make([{"rule": "json_schema", "schema": schema}])
        assert g.check(content).passed

    def test_invalid_json_fails(self) -> None:
        g = _make([{"rule": "json_schema", "schema": {"type": "object"}}])
        assert not g.check("not json at all").passed

    def test_missing_required_key(self) -> None:
        schema = {"type": "object", "required": ["name", "age"]}
        content = json.dumps({"name": "Alice"})
        g = _make([{"rule": "json_schema", "schema": schema}])
        # May pass via jsonschema or fallback; both should catch missing key
        result = g.check(content)
        assert not result.passed


# ── sentiment ──────────────────────────────────────────────────────


class TestSentiment:
    def test_positive_in_range(self) -> None:
        g = _make([{"rule": "sentiment", "min_score": 0.0, "max_score": 1.0}])
        assert g.check("great excellent wonderful").passed

    def test_negative_out_of_range(self) -> None:
        g = _make([{"rule": "sentiment", "min_score": 0.0, "max_score": 1.0}])
        assert not g.check("terrible horrible awful").passed

    def test_neutral_content(self) -> None:
        g = _make([{"rule": "sentiment", "min_score": -0.5, "max_score": 0.5}])
        assert g.check("the cat sat on the mat").passed


# ── language ───────────────────────────────────────────────────────


class TestLanguage:
    def test_english_detected(self) -> None:
        g = _make([{"rule": "language", "allowed": ["en"]}])
        assert g.check("the cat is on the table and of course to").passed

    def test_non_allowed_language(self) -> None:
        g = _make([{"rule": "language", "allowed": ["fr"]}])
        assert not g.check("the cat is on the table and of course to").passed

    def test_spanish_detected(self) -> None:
        g = _make([{"rule": "language", "allowed": ["es"]}])
        assert g.check("el gato de la casa en el jardín y la cocina").passed


# ── word_count ─────────────────────────────────────────────────────


class TestWordCount:
    def test_pass(self) -> None:
        g = _make([{"rule": "word_count", "min_words": 2, "max_words": 10}])
        assert g.check("hello world").passed

    def test_too_few(self) -> None:
        g = _make([{"rule": "word_count", "min_words": 5}])
        assert not g.check("hi").passed

    def test_too_many(self) -> None:
        g = _make([{"rule": "word_count", "max_words": 2}])
        assert not g.check("one two three four").passed


# ── not_empty ──────────────────────────────────────────────────────


class TestNotEmpty:
    def test_pass(self) -> None:
        g = _make([{"rule": "not_empty"}])
        assert g.check("content").passed

    def test_blank_fails(self) -> None:
        g = _make([{"rule": "not_empty"}])
        assert not g.check("").passed

    def test_whitespace_only_fails(self) -> None:
        g = _make([{"rule": "not_empty"}])
        assert not g.check("   \n\t  ").passed


# ── Combine logic ─────────────────────────────────────────────────


class TestCombineLogic:
    def test_any_blocks_on_first_failure(self) -> None:
        """combine='any' (strict): one failure is enough."""
        g = _make(
            [
                {"rule": "min_length", "min_chars": 1},  # passes
                {"rule": "max_length", "max_chars": 3},  # fails
            ],
            combine="any",
        )
        result = g.check("hello world")
        assert not result.passed

    def test_any_passes_when_all_pass(self) -> None:
        g = _make(
            [
                {"rule": "min_length", "min_chars": 1},
                {"rule": "max_length", "max_chars": 100},
            ],
            combine="any",
        )
        assert g.check("hello").passed

    def test_all_blocks_only_when_every_rule_fails(self) -> None:
        """combine='all' (lenient): block only if ALL rules fail."""
        g = _make(
            [
                {"rule": "min_length", "min_chars": 100},  # fails
                {"rule": "max_length", "max_chars": 100},  # passes
            ],
            combine="all",
        )
        # One rule passed, so lenient mode should NOT trigger
        assert g.check("short text").passed

    def test_all_blocks_when_everything_fails(self) -> None:
        g = _make(
            [
                {"rule": "min_length", "min_chars": 100},  # fails
                {"rule": "max_length", "max_chars": 1},  # fails
            ],
            combine="all",
        )
        result = g.check("hello")
        assert not result.passed
        assert result.blocked  # action=block

    def test_all_with_single_rule(self) -> None:
        """Single rule — combine mode is irrelevant; failure triggers."""
        g = _make([{"rule": "not_empty"}], combine="all")
        assert not g.check("").passed


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_content_with_min_length(self) -> None:
        g = _make([{"rule": "min_length", "min_chars": 1}])
        assert not g.check("").passed

    def test_unicode_content(self) -> None:
        g = _make([{"rule": "keyword_present", "keywords": ["café"]}])
        assert g.check("Welcome to the café!").passed

    def test_very_long_content(self) -> None:
        long_text = "word " * 10_000
        g = _make([{"rule": "max_length", "max_chars": 1_000_000}])
        assert g.check(long_text).passed

    def test_unknown_rule_type(self) -> None:
        g = _make([{"rule": "nonexistent_rule"}])
        result = g.check("anything")
        assert not result.passed

    def test_warn_action_does_not_block(self) -> None:
        g = _make([{"rule": "not_empty"}], action="warn")
        result = g.check("")
        assert not result.passed
        assert not result.blocked

    def test_block_action_blocks(self) -> None:
        g = _make([{"rule": "not_empty"}], action="block")
        result = g.check("")
        assert not result.passed
        assert result.blocked

    def test_metadata_includes_guardrail_name(self) -> None:
        g = _make([{"rule": "not_empty"}], name="my_guard")
        result = g.check("")
        assert result.metadata.get("guardrail") == "my_guard"

    def test_no_rules_passes(self) -> None:
        g = CustomGuardrail(name="empty", action="block", rules=[])
        assert g.check("anything").passed


# ── Multiple rules combined ────────────────────────────────────────


class TestMultipleRules:
    def test_all_pass_happy_path(self) -> None:
        g = _make(
            [
                {"rule": "not_empty"},
                {"rule": "min_length", "min_chars": 3},
                {"rule": "keyword_present", "keywords": ["hello"]},
            ],
            combine="any",
        )
        assert g.check("hello world").passed

    def test_mixed_rules_any_triggers(self) -> None:
        g = _make(
            [
                {"rule": "not_empty"},  # passes
                {"rule": "keyword_absent", "keywords": ["bad"]},  # fails
                {"rule": "min_length", "min_chars": 1},  # passes
            ],
            combine="any",
        )
        assert not g.check("this is bad").passed

    def test_mixed_rules_all_lenient(self) -> None:
        g = _make(
            [
                {"rule": "not_empty"},  # passes
                {"rule": "keyword_absent", "keywords": ["bad"]},  # fails
                {"rule": "min_length", "min_chars": 1},  # passes
            ],
            combine="all",
        )
        # Only 1 of 3 fails → lenient mode does NOT trigger
        assert g.check("this is bad").passed

    def test_score_reflects_failure_ratio(self) -> None:
        g = _make(
            [
                {"rule": "min_length", "min_chars": 100},  # fails
                {"rule": "max_length", "max_chars": 1},  # fails
            ],
            combine="all",
        )
        result = g.check("hello")
        assert not result.passed
        assert result.score is not None
        assert result.score == pytest.approx(1.0)


# ── Registry integration ──────────────────────────────────────────


class TestRegistryIntegration:
    def test_custom_in_registry(self) -> None:
        from sentinel.llmops.guardrails import resolve_guardrail

        cls = resolve_guardrail("custom")
        assert cls is CustomGuardrail

    def test_pipeline_builds_custom(self) -> None:
        from sentinel.config.schema import GuardrailRuleConfig, GuardrailsConfig, LLMOpsConfig
        from sentinel.llmops.guardrails.engine import GuardrailPipeline

        rule = GuardrailRuleConfig(
            type="custom",
            action="warn",
            name="test_custom",
            rules=[{"rule": "not_empty"}],
            combine="any",
        )
        cfg = LLMOpsConfig(guardrails=GuardrailsConfig(input=[rule]))
        pipeline = GuardrailPipeline.from_config(cfg)
        assert len(pipeline.input_guardrails) == 1
        result = pipeline.check_input("hello")
        assert result.passed
