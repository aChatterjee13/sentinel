"""End-to-end integration tests for the LLMOps subsystem (fix 15).

Verifies that the high-level ``LLMOpsClient.log_call()`` pipeline works
correctly when all sub-modules are wired together, including:

- Token tracking
- Prompt stats recording
- Guardrail + log_call flow
- Semantic drift without fit()
- fit_semantic_baseline() from both LLMOpsClient and SentinelClient
- Audit trail event logging
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sentinel.config.schema import LLMOpsConfig
from sentinel.core.types import PipelineResult
from sentinel.llmops.client import LLMOpsClient

# ── Helpers ───────────────────────────────────────────────────────

def _make_client(*, audit: Any | None = None) -> LLMOpsClient:
    """Create an LLMOpsClient with minimal config, no external deps."""
    return LLMOpsClient(LLMOpsConfig(enabled=True), audit=audit)


# ── Test: full pipeline ──────────────────────────────────────────


class TestLogCallFullPipeline:
    """log_call() should orchestrate token tracking, quality evaluation,
    retrieval quality, semantic drift observation, prompt drift, and audit."""

    def test_basic_log_call(self) -> None:
        """A minimal log_call with only model + tokens should succeed."""
        client = _make_client()
        result = client.log_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
        )
        assert "usage" in result
        assert result["usage"].input_tokens == 100
        assert result["usage"].output_tokens == 50

    def test_log_call_with_all_params(self) -> None:
        """log_call with all parameters should not raise."""
        audit = MagicMock()
        client = _make_client(audit=audit)

        result = client.log_call(
            prompt_name="claims_qa",
            prompt_version="1.0",
            query="What is the coverage?",
            response="The coverage is 100k.",
            context_chunks=["Coverage for this policy is 100k dollars."],
            model="gpt-4o",
            input_tokens=200,
            output_tokens=80,
            latency_ms=500.0,
            guardrail_results=None,
            user_id="user-42",
        )
        assert result["usage"].model == "gpt-4o"
        assert result["retrieval_quality"] is not None
        assert result["retrieval_quality"].chunks_retrieved == 1
        # Audit event should be logged
        audit.log.assert_called()

    def test_log_call_records_prompt_stats(self) -> None:
        """When prompt_name + version are provided, prompt drift is observed."""
        client = _make_client()

        # Register the prompt so log_result doesn't fail
        client.prompts.register(
            name="summariser",
            version="2.0",
            system_prompt="You summarise text.",
            template="Summarise: {{text}}",
        )

        client.log_call(
            prompt_name="summariser",
            prompt_version="2.0",
            query="Summarise this.",
            response="Here is the summary.",
            model="gpt-4o-mini",
            input_tokens=50,
            output_tokens=30,
        )

        # Prompt drift should have observations
        key = "summariser@2.0"
        assert key in client.prompt_drift._stats
        stats = client.prompt_drift._stats[key]
        assert len(stats.token_usage) == 1

    def test_log_call_with_guardrail_results(self) -> None:
        """Guardrail results should be counted for violations."""
        client = _make_client()

        client.prompts.register(
            name="qa", version="1.0",
            system_prompt="QA bot", template="{{q}}",
        )

        guardrail_results = {
            "input": PipelineResult(
                blocked=False,
                warnings=["pii: possible PII detected"],
            ),
            "output": PipelineResult(
                blocked=False,
                warnings=["toxicity: borderline content"],
            ),
        }
        result = client.log_call(
            prompt_name="qa",
            prompt_version="1.0",
            response="answer",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=10,
            guardrail_results=guardrail_results,
        )
        assert result["guardrail_violations"] == 2  # 2 warnings

    def test_audit_trail_event_logged(self) -> None:
        """The audit trail should receive an 'llm.call' event."""
        audit = MagicMock()
        client = _make_client(audit=audit)

        client.log_call(model="gpt-4o", input_tokens=10, output_tokens=5)

        audit.log.assert_called_once()
        call_kwargs = audit.log.call_args
        assert call_kwargs.kwargs.get("event_type") == "llm.call" or (
            call_kwargs[1].get("event_type") == "llm.call"
        )


# ── Test: guardrail + log_call flow ─────────────────────────────


class TestGuardrailAndLogCallFlow:
    """check_input() and log_call() should work together."""

    def test_check_then_log(self) -> None:
        """Calling check_input then log_call should not raise."""
        client = _make_client()

        input_result = client.check_input("What is coverage?")
        assert not input_result.blocked

        log_result = client.log_call(
            query="What is coverage?",
            response="Coverage is 100k.",
            model="gpt-4o",
            input_tokens=20,
            output_tokens=10,
            guardrail_results={"input": input_result},
        )
        assert log_result is not None


# ── Test: semantic drift without fit() ───────────────────────────


class TestSemanticDriftWithoutFit:
    """log_call() must gracefully handle unfitted semantic drift monitor."""

    def test_no_crash_without_fit(self) -> None:
        """log_call with response but without fit() should not crash."""
        client = _make_client()
        assert client.semantic_drift._reference_centroid is None

        result = client.log_call(
            response="The model produced this output.",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=20,
        )
        assert result is not None

    def test_observe_not_called_without_fit(self) -> None:
        """When baseline isn't fitted, observe() should be skipped."""
        client = _make_client()

        with patch.object(client.semantic_drift, "observe") as mock:
            client.log_call(response="test", model="m", input_tokens=1, output_tokens=1)
            mock.assert_not_called()


# ── Test: fit_semantic_baseline() ────────────────────────────────


class TestFitSemanticBaseline:
    """fit_semantic_baseline() on both LLMOpsClient and SentinelClient."""

    def test_llmops_client_fit(self) -> None:
        """LLMOpsClient.fit_semantic_baseline should delegate to semantic_drift.fit."""
        client = _make_client()

        with patch.object(client.semantic_drift, "fit") as mock_fit:
            client.fit_semantic_baseline(["output1", "output2"])
            mock_fit.assert_called_once_with(["output1", "output2"])

    def test_llmops_client_fit_empty_raises(self) -> None:
        """Empty list should raise ValueError."""
        client = _make_client()
        with pytest.raises(ValueError):
            client.fit_semantic_baseline([])

    def test_sentinel_client_proxy(self) -> None:
        """SentinelClient.fit_semantic_baseline should proxy to llmops."""
        from sentinel.config.schema import SentinelConfig
        from sentinel.core.client import SentinelClient

        cfg = SentinelConfig(
            model={"name": "test_model"},
            llmops=LLMOpsConfig(enabled=True),
        )
        sentinel = SentinelClient(cfg)

        with patch.object(sentinel.llmops, "fit_semantic_baseline") as mock_fit:
            sentinel.fit_semantic_baseline(["a", "b"])
            mock_fit.assert_called_once_with(["a", "b"])

    def test_fit_then_log_call_observes(self) -> None:
        """After fitting, log_call should call observe()."""
        client = _make_client()

        # Mock embed to avoid needing sentence-transformers
        fake_embed = lambda texts: np.random.default_rng(42).random((len(texts), 8))  # noqa: E731
        client.semantic_drift._embed_fn = fake_embed

        client.fit_semantic_baseline(["baseline output one", "baseline output two"])
        assert client.semantic_drift._reference_centroid is not None

        with patch.object(client.semantic_drift, "observe") as mock_obs:
            client.log_call(
                response="new response",
                model="gpt-4o",
                input_tokens=10,
                output_tokens=10,
            )
            mock_obs.assert_called_once_with("new response")
