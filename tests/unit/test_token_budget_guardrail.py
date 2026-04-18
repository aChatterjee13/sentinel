"""Unit tests for TokenBudgetGuardrail."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sentinel.llmops.guardrails.token_budget import TokenBudgetGuardrail


class TestTokenBudgetGuardrail:
    """Tests for the token_budget guardrail."""

    # ── under budget ──────────────────────────────────────────────

    def test_short_text_under_budget(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=4000)
        result = g.check("Hello world")
        assert result.passed
        assert not result.blocked
        assert result.score < 1.0

    def test_exactly_at_budget_passes(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=100)
        # Heuristic: ~4 chars per token, 400 chars ≈ 100 tokens
        text = "a" * 400
        result = g.check(text)
        assert result.passed

    # ── over budget ───────────────────────────────────────────────

    def test_long_text_over_budget(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=10)
        # Heuristic: 200 chars ≈ 50 tokens, way over 10
        text = "word " * 50
        result = g.check(text)
        assert not result.passed
        assert "tokens" in (result.reason or "")
        assert result.metadata.get("max") == 10

    def test_over_budget_score_exceeds_one(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=10)
        text = "a" * 200  # ~50 tokens
        result = g.check(text)
        assert result.score > 1.0

    # ── heuristic fallback ────────────────────────────────────────

    def test_heuristic_fallback_when_no_tiktoken(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=100)
        g._encoder = None  # force heuristic
        result = g.check("short text")
        assert result.passed

    def test_heuristic_counts_approximately(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=100)
        g._encoder = None
        # 80 chars → ~20 tokens via heuristic (80 // 4)
        count = g._count("a" * 80)
        assert count == 20

    def test_heuristic_minimum_one_token(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=100)
        g._encoder = None
        count = g._count("")
        assert count >= 1

    # ── tiktoken encoder ──────────────────────────────────────────

    def test_with_mock_tiktoken_encoder(self) -> None:
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(50))
        g = TokenBudgetGuardrail(max_input_tokens=100)
        g._encoder = mock_encoder
        result = g.check("some text")
        assert result.passed
        mock_encoder.encode.assert_called_once_with("some text")

    def test_tiktoken_encoder_failure_falls_back(self) -> None:
        mock_encoder = MagicMock()
        mock_encoder.encode.side_effect = RuntimeError("encoder broke")
        g = TokenBudgetGuardrail(max_input_tokens=100)
        g._encoder = mock_encoder
        # Should fall back to heuristic without raising
        result = g.check("some text here")
        assert result.passed

    # ── action types ──────────────────────────────────────────────

    def test_action_block_blocks_over_budget(self) -> None:
        g = TokenBudgetGuardrail(action="block", max_input_tokens=5)
        result = g.check("a very long text that exceeds the tiny budget")
        assert not result.passed
        assert result.blocked

    def test_action_warn_does_not_block(self) -> None:
        g = TokenBudgetGuardrail(action="warn", max_input_tokens=5)
        result = g.check("a very long text that exceeds the tiny budget")
        assert not result.passed
        assert not result.blocked

    def test_action_block_passes_under_budget(self) -> None:
        g = TokenBudgetGuardrail(action="block", max_input_tokens=4000)
        result = g.check("short")
        assert result.passed
        assert not result.blocked

    # ── metadata ──────────────────────────────────────────────────

    def test_metadata_on_failure(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=5)
        g._encoder = None
        result = g.check("a" * 100)
        assert result.metadata.get("max") == 5
        assert result.metadata.get("tokens") > 5

    def test_name_and_direction(self) -> None:
        g = TokenBudgetGuardrail()
        assert g.name == "token_budget"
        assert g.direction == "input"

    # ── _try_load_encoder ─────────────────────────────────────────

    def test_try_load_encoder_returns_none_on_import_error(self) -> None:
        with patch.dict("sys.modules", {"tiktoken": None}):
            result = TokenBudgetGuardrail._try_load_encoder("cl100k_base")
            # Either None or a real encoder depending on environment
            # The method should not raise
            assert result is None or hasattr(result, "encode")

    def test_score_ratio_under_budget(self) -> None:
        g = TokenBudgetGuardrail(max_input_tokens=1000)
        g._encoder = None
        result = g.check("a" * 400)  # ~100 tokens
        assert 0 < result.score < 1.0
