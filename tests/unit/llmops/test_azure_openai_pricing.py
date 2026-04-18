"""Tests for Azure OpenAI pricing + provider classification."""

from __future__ import annotations

import pytest

from sentinel.config.defaults import DEFAULT_PRICING, get_default_pricing
from sentinel.config.schema import TokenEconomicsConfig
from sentinel.llmops.token_economics import TokenTracker, provider_from_model


class TestDefaultPricingCoverage:
    """The default pricing table must carry Azure OpenAI entries."""

    @pytest.mark.parametrize(
        "model",
        [
            "azure/gpt-4o",
            "azure/gpt-4o-mini",
            "azure/gpt-4-turbo",
            "azure/gpt-35-turbo",
            "azure/text-embedding-3-small",
            "azure/text-embedding-3-large",
        ],
    )
    def test_azure_model_present(self, model: str) -> None:
        entry = DEFAULT_PRICING.get(model)
        assert entry is not None, f"{model} missing from DEFAULT_PRICING"
        assert "input" in entry and "output" in entry
        assert entry["input"] >= 0.0
        assert entry["output"] >= 0.0

    def test_get_default_pricing_returns_azure_entries(self) -> None:
        pricing = get_default_pricing("azure/gpt-4o")
        assert pricing is not None
        assert pricing == {"input": 0.005, "output": 0.015}


class TestEstimateCostForAzureModels:
    def test_estimate_cost_azure_gpt_4o(self) -> None:
        tracker = TokenTracker()
        # 1000 input + 500 output @ $0.005/$0.015 per 1k →
        # (1000 * 0.005 + 500 * 0.015) / 1000 = 0.0125
        cost = tracker.estimate_cost("azure/gpt-4o", 1000, 500)
        assert cost == pytest.approx(0.0125)

    def test_estimate_cost_azure_embedding(self) -> None:
        tracker = TokenTracker()
        # 10_000 input tokens, no output tokens
        cost = tracker.estimate_cost("azure/text-embedding-3-small", 10_000, 0)
        assert cost == pytest.approx(0.0002)

    def test_custom_pricing_overrides_default(self) -> None:
        config = TokenEconomicsConfig(pricing={"azure/gpt-4o": {"input": 0.001, "output": 0.002}})
        tracker = TokenTracker(config=config)
        # Overridden price: (1000 * 0.001 + 500 * 0.002) / 1000 = 0.002
        cost = tracker.estimate_cost("azure/gpt-4o", 1000, 500)
        assert cost == pytest.approx(0.002)

    def test_unknown_model_zero_cost(self) -> None:
        tracker = TokenTracker()
        cost = tracker.estimate_cost("imaginary-foo-v9", 1000, 500)
        assert cost == 0.0


class TestProviderFromModel:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("azure/gpt-4o", "azure"),
            ("azure/gpt-35-turbo", "azure"),
            ("azure/text-embedding-3-small", "azure"),
            ("gpt-4o", "openai"),
            ("gpt-4o-mini", "openai"),
            ("gpt-4-turbo", "openai"),
            ("text-embedding-3-small", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("claude-opus-4-6", "anthropic"),
            ("claude-sonnet-4-6", "anthropic"),
            ("claude-haiku-4-5", "anthropic"),
            ("llama-3-70b", "unknown"),
            ("mistral-large", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_classification(self, model: str, expected: str) -> None:
        assert provider_from_model(model) == expected


class TestRecordTagsProvider:
    def test_record_aggregates_by_provider(self) -> None:
        tracker = TokenTracker()
        tracker.record("azure/gpt-4o", 1000, 500)
        tracker.record("azure/gpt-4o-mini", 1000, 200)
        tracker.record("gpt-4o", 500, 250)

        totals = tracker.totals()
        assert "provider:azure" in totals
        assert "provider:openai" in totals
        # Azure calls: 2 records, 2000 input + 700 output tokens
        azure = totals["provider:azure"]
        assert azure["calls"] == 2
        assert azure["input_tokens"] == 2000
        assert azure["output_tokens"] == 700
        # OpenAI calls: 1 record
        assert totals["provider:openai"]["calls"] == 1

    def test_record_unknown_provider_still_tagged(self) -> None:
        tracker = TokenTracker()
        tracker.record("mistral-7b", 1000, 500)
        totals = tracker.totals()
        assert "provider:unknown" in totals
        assert totals["provider:unknown"]["calls"] == 1
