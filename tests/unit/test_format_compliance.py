"""Unit tests for FormatComplianceGuardrail."""

from __future__ import annotations

import json

from sentinel.llmops.guardrails.format_compliance import FormatComplianceGuardrail


class TestFormatComplianceGuardrail:
    """Tests for the format_compliance guardrail."""

    # ── valid JSON matching schema ────────────────────────────────

    def test_valid_json_passes(self) -> None:
        g = FormatComplianceGuardrail()
        result = g.check('{"key": "value"}')
        assert result.passed
        assert not result.blocked
        assert result.score == 1.0

    def test_valid_json_with_schema_required_fields(self) -> None:
        schema = {"required": ["name", "age"]}
        g = FormatComplianceGuardrail(expected_schema=schema)
        result = g.check('{"name": "Alice", "age": 30}')
        assert result.passed
        assert result.score == 1.0

    # ── invalid JSON ──────────────────────────────────────────────

    def test_invalid_json_fails(self) -> None:
        g = FormatComplianceGuardrail()
        result = g.check("not json at all")
        assert not result.passed
        assert result.score == 0.0
        assert "not valid JSON" in (result.reason or "")

    def test_partial_json_fails(self) -> None:
        g = FormatComplianceGuardrail()
        result = g.check('{"key": ')
        assert not result.passed
        assert result.score == 0.0

    # ── valid JSON but wrong schema ───────────────────────────────

    def test_missing_required_fields(self) -> None:
        schema = {"required": ["name", "age"]}
        g = FormatComplianceGuardrail(expected_schema=schema)
        result = g.check('{"name": "Alice"}')
        assert not result.passed
        assert result.score == 0.5
        assert "missing required fields" in (result.reason or "")
        assert result.metadata.get("missing") == ["age"]

    def test_all_required_fields_missing(self) -> None:
        schema = {"required": ["x", "y"]}
        g = FormatComplianceGuardrail(expected_schema=schema)
        result = g.check("{}")
        assert not result.passed
        assert set(result.metadata.get("missing", [])) == {"x", "y"}

    # ── no schema configured ──────────────────────────────────────

    def test_no_schema_require_json_passes_valid_json(self) -> None:
        g = FormatComplianceGuardrail(expected_schema=None, require_json=True)
        result = g.check('{"hello": "world"}')
        assert result.passed

    def test_no_schema_no_require_json_passes_anything(self) -> None:
        g = FormatComplianceGuardrail(expected_schema=None, require_json=False)
        result = g.check("just some plain text")
        assert result.passed
        assert result.score == 1.0

    # ── action types ──────────────────────────────────────────────

    def test_action_block_blocks_on_failure(self) -> None:
        g = FormatComplianceGuardrail(action="block")
        result = g.check("not json")
        assert not result.passed
        assert result.blocked

    def test_action_warn_does_not_block(self) -> None:
        g = FormatComplianceGuardrail(action="warn")
        result = g.check("not json")
        assert not result.passed
        assert not result.blocked

    def test_action_redact_does_not_block(self) -> None:
        g = FormatComplianceGuardrail(action="redact")
        result = g.check("not json")
        assert not result.passed
        assert not result.blocked

    # ── schema from dict ──────────────────────────────────────────

    def test_schema_from_dict(self) -> None:
        schema = {"required": ["id"], "properties": {"id": {"type": "integer"}}}
        g = FormatComplianceGuardrail(expected_schema=schema)
        result = g.check('{"id": 42}')
        assert result.passed

    # ── schema from JSON string ───────────────────────────────────

    def test_schema_from_json_string(self) -> None:
        schema_str = json.dumps({"required": ["a"]})
        g = FormatComplianceGuardrail(expected_schema=schema_str)
        result = g.check('{"a": 1}')
        assert result.passed

    # ── schema from file path ─────────────────────────────────────

    def test_schema_from_file(self, tmp_path) -> None:
        schema = {"required": ["foo"]}
        p = tmp_path / "schema.json"
        p.write_text(json.dumps(schema))
        g = FormatComplianceGuardrail(expected_schema=str(p))
        result = g.check('{"foo": "bar"}')
        assert result.passed

    def test_schema_file_not_found_fallback(self) -> None:
        # Non-existent file and not valid JSON → empty dict schema
        g = FormatComplianceGuardrail(expected_schema="/nonexistent/path.json")
        result = g.check('{"anything": true}')
        assert result.passed

    # ── length-based checks with require_json=False ───────────────

    def test_max_length_exceeded(self) -> None:
        schema = {"maxLength": 10}
        g = FormatComplianceGuardrail(expected_schema=schema, require_json=False)
        result = g.check("a" * 20)
        assert not result.passed
        assert "too long" in (result.reason or "")

    def test_min_length_not_met(self) -> None:
        schema = {"minLength": 10}
        g = FormatComplianceGuardrail(expected_schema=schema, require_json=False)
        result = g.check("hi")
        assert not result.passed
        assert "too short" in (result.reason or "")

    def test_length_within_bounds(self) -> None:
        schema = {"minLength": 2, "maxLength": 50}
        g = FormatComplianceGuardrail(expected_schema=schema, require_json=False)
        result = g.check("hello world")
        assert result.passed

    # ── guardrail metadata ────────────────────────────────────────

    def test_name_and_direction(self) -> None:
        g = FormatComplianceGuardrail()
        assert g.name == "format_compliance"
        assert g.direction == "output"
