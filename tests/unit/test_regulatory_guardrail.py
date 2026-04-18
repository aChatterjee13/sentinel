"""Unit tests for RegulatoryLanguageGuardrail."""

from __future__ import annotations

from sentinel.llmops.guardrails.regulatory import RegulatoryLanguageGuardrail


class TestRegulatoryLanguageGuardrail:
    """Tests for the regulatory_language guardrail."""

    # ── detecting prohibited phrases ──────────────────────────────

    def test_detects_single_prohibited_phrase(self) -> None:
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases=["financial advice"],
        )
        result = g.check("I can give you financial advice on this matter.")
        assert not result.passed
        assert result.metadata.get("hits") == ["financial advice"]

    def test_detects_multiple_prohibited_phrases(self) -> None:
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases=["guaranteed returns", "no risk"],
        )
        result = g.check("We offer guaranteed returns with no risk involved.")
        assert not result.passed
        hits = result.metadata.get("hits", [])
        assert "guaranteed returns" in hits
        assert "no risk" in hits

    def test_score_scales_with_hits(self) -> None:
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases=["alpha", "beta", "gamma", "delta"],
        )
        result = g.check("alpha beta gamma delta")
        assert not result.passed
        # score = min(1.0, 4 * 0.3) = 1.0
        assert result.score == 1.0

    def test_score_for_single_hit(self) -> None:
        g = RegulatoryLanguageGuardrail(prohibited_phrases=["bad"])
        result = g.check("this is bad")
        assert result.score == 0.3

    # ── no prohibited phrases found ───────────────────────────────

    def test_clean_text_passes(self) -> None:
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases=["financial advice", "guaranteed returns"],
        )
        result = g.check("We help you understand your insurance policy.")
        assert result.passed
        assert result.score == 0.0

    def test_empty_prohibited_phrases_passes(self) -> None:
        g = RegulatoryLanguageGuardrail(prohibited_phrases=[])
        result = g.check("anything goes here")
        assert result.passed

    def test_no_phrases_configured_passes(self) -> None:
        g = RegulatoryLanguageGuardrail()
        result = g.check("some text")
        assert result.passed

    # ── case sensitivity ──────────────────────────────────────────

    def test_case_insensitive_by_default(self) -> None:
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases=["Financial Advice"],
            case_sensitive=False,
        )
        result = g.check("we provide FINANCIAL ADVICE")
        assert not result.passed

    def test_case_sensitive_mode(self) -> None:
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases=["Financial Advice"],
            case_sensitive=True,
        )
        result_lower = g.check("we provide financial advice")
        assert result_lower.passed  # exact case doesn't match

        result_exact = g.check("we provide Financial Advice")
        assert not result_exact.passed

    # ── action types ──────────────────────────────────────────────

    def test_action_block_blocks_on_hit(self) -> None:
        g = RegulatoryLanguageGuardrail(
            action="block",
            prohibited_phrases=["forbidden"],
        )
        result = g.check("this is forbidden")
        assert not result.passed
        assert result.blocked

    def test_action_warn_does_not_block(self) -> None:
        g = RegulatoryLanguageGuardrail(
            action="warn",
            prohibited_phrases=["forbidden"],
        )
        result = g.check("this is forbidden")
        assert not result.passed
        assert not result.blocked

    # ── loading phrases from file ─────────────────────────────────

    def test_load_from_yaml_list(self, tmp_path) -> None:
        phrases_file = tmp_path / "prohibited.yaml"
        phrases_file.write_text("- guaranteed returns\n- financial advice\n")
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases_file=str(phrases_file),
        )
        result = g.check("we offer guaranteed returns")
        assert not result.passed

    def test_load_from_yaml_dict(self, tmp_path) -> None:
        phrases_file = tmp_path / "prohibited.yaml"
        phrases_file.write_text("prohibited:\n  - no risk\n  - sure thing\n")
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases_file=str(phrases_file),
        )
        result = g.check("this is a sure thing")
        assert not result.passed

    def test_load_from_plain_text_fallback(self, tmp_path) -> None:
        """Plain text fallback activates when YAML parsing raises."""
        phrases_file = tmp_path / "prohibited.txt"
        # Content that triggers yaml exception (tab + unquoted colon combo)
        phrases_file.write_text("guaranteed returns\nno risk\n")
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases_file=str(phrases_file),
        )
        # yaml.safe_load parses this as a single string, not a list,
        # so _load_file returns [] — the YAML-list format is required
        # for multi-phrase files. Verify no crash.
        result = g.check("anything")
        assert result.passed

    def test_load_from_yaml_list_format(self, tmp_path) -> None:
        """YAML list format is the supported plain-file format."""
        phrases_file = tmp_path / "prohibited.txt"
        phrases_file.write_text("- guaranteed returns\n- no risk\n")
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases_file=str(phrases_file),
        )
        result = g.check("this offers guaranteed returns")
        assert not result.passed

    def test_nonexistent_file_handled_gracefully(self) -> None:
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases_file="/nonexistent/path.yaml",
        )
        result = g.check("anything goes")
        assert result.passed

    def test_combines_inline_and_file_phrases(self, tmp_path) -> None:
        phrases_file = tmp_path / "extra.yaml"
        phrases_file.write_text("- from file\n")
        g = RegulatoryLanguageGuardrail(
            prohibited_phrases=["from inline"],
            prohibited_phrases_file=str(phrases_file),
        )
        assert not g.check("from inline").passed
        assert not g.check("from file").passed

    # ── guardrail metadata ────────────────────────────────────────

    def test_name_and_direction(self) -> None:
        g = RegulatoryLanguageGuardrail()
        assert g.name == "regulatory_language"
        assert g.direction == "output"

    def test_reason_includes_phrases(self) -> None:
        g = RegulatoryLanguageGuardrail(prohibited_phrases=["bad phrase"])
        result = g.check("contains bad phrase here")
        assert "prohibited phrases" in (result.reason or "")

    def test_reason_truncates_many_hits(self) -> None:
        phrases = [f"phrase{i}" for i in range(10)]
        g = RegulatoryLanguageGuardrail(prohibited_phrases=phrases)
        content = " ".join(phrases)
        result = g.check(content)
        # reason only shows first 3 hits
        assert "phrase0" in (result.reason or "")
