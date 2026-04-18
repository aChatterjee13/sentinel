"""Tests for KPILinker."""

from __future__ import annotations

import pytest

from sentinel.config.schema import BusinessKPIConfig, KPIMapping
from sentinel.intelligence.kpi_linker import KPILinker

# ── Fixtures ───────────────────────────────────────────────────────


def _make_config(mappings: list[dict[str, str]]) -> BusinessKPIConfig:
    return BusinessKPIConfig(
        mappings=[KPIMapping(**m) for m in mappings],
    )


@pytest.fixture()
def single_mapping_config() -> BusinessKPIConfig:
    return _make_config(
        [
            {
                "model_metric": "precision",
                "business_kpi": "fraud_catch_rate",
                "data_source": "warehouse://fraud_metrics",
            },
        ]
    )


@pytest.fixture()
def multi_mapping_config() -> BusinessKPIConfig:
    return _make_config(
        [
            {
                "model_metric": "precision",
                "business_kpi": "fraud_catch_rate",
                "data_source": "warehouse://fraud",
            },
            {
                "model_metric": "recall",
                "business_kpi": "false_positive_rate",
                "data_source": "warehouse://fraud",
            },
        ]
    )


# ── Core behaviour ─────────────────────────────────────────────────


class TestReport:
    def test_report_with_no_fetcher(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        result = linker.report({"precision": 0.92})

        assert result["n_links"] == 1
        assert result["linked_kpis"][0]["metric_value"] == 0.92
        assert result["linked_kpis"][0]["kpi_value"] is None

    def test_report_with_cached_kpis(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        linker._cached_kpis["fraud_catch_rate"] = 0.87

        result = linker.report({"precision": 0.92})
        assert result["linked_kpis"][0]["kpi_value"] == 0.87

    def test_report_missing_metric(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        result = linker.report({"recall": 0.80})  # precision not provided

        assert result["linked_kpis"][0]["metric_value"] is None

    def test_report_multiple_mappings(self, multi_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(multi_mapping_config)
        result = linker.report({"precision": 0.9, "recall": 0.85})

        assert result["n_links"] == 2
        assert result["linked_kpis"][0]["metric_value"] == 0.9
        assert result["linked_kpis"][1]["metric_value"] == 0.85

    def test_empty_mappings(self) -> None:
        config = _make_config([])
        linker = KPILinker(config)
        result = linker.report({"precision": 0.9})

        assert result["n_links"] == 0
        assert result["linked_kpis"] == []


# ── Fetcher / refresh ──────────────────────────────────────────────


class TestRefresh:
    def test_refresh_caches_values(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        linker.set_fetcher(lambda source: 0.95)

        cached = linker.refresh()
        assert cached["fraud_catch_rate"] == 0.95

    def test_refresh_with_no_fetcher(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        cached = linker.refresh()
        assert cached == {}

    def test_refresh_skips_exceptions(self, single_mapping_config: BusinessKPIConfig) -> None:
        def _bad_fetcher(source: str) -> float:
            raise ConnectionError("timeout")

        linker = KPILinker(single_mapping_config)
        linker.set_fetcher(_bad_fetcher)

        cached = linker.refresh()
        assert "fraud_catch_rate" not in cached

    def test_refresh_skips_nan(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        linker.set_fetcher(lambda source: float("nan"))

        cached = linker.refresh()
        assert "fraud_catch_rate" not in cached

    def test_refresh_skips_inf(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        linker.set_fetcher(lambda source: float("inf"))

        cached = linker.refresh()
        assert "fraud_catch_rate" not in cached

    def test_refresh_skips_none(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        linker.set_fetcher(lambda source: None)

        cached = linker.refresh()
        assert "fraud_catch_rate" not in cached

    def test_refresh_skips_non_numeric(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        linker.set_fetcher(lambda source: "not_a_number")  # type: ignore[return-value]

        cached = linker.refresh()
        assert "fraud_catch_rate" not in cached

    def test_refresh_multiple_mappings(self, multi_mapping_config: BusinessKPIConfig) -> None:
        calls: list[str] = []

        def _fetcher(source: str) -> float:
            calls.append(source)
            return 0.88

        linker = KPILinker(multi_mapping_config)
        linker.set_fetcher(_fetcher)

        cached = linker.refresh()
        assert len(calls) == 2
        assert cached["fraud_catch_rate"] == 0.88
        assert cached["false_positive_rate"] == 0.88


# ── set_fetcher ────────────────────────────────────────────────────


class TestSetFetcher:
    def test_set_fetcher(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        assert linker._fetcher is None

        linker.set_fetcher(lambda s: 1.0)
        assert linker._fetcher is not None

    def test_replace_fetcher(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        linker.set_fetcher(lambda s: 1.0)
        linker.set_fetcher(lambda s: 2.0)
        linker.refresh()

        assert linker._cached_kpis["fraud_catch_rate"] == 2.0


# ── Integration: refresh then report ───────────────────────────────


class TestIntegration:
    def test_refresh_then_report(self, single_mapping_config: BusinessKPIConfig) -> None:
        linker = KPILinker(single_mapping_config)
        linker.set_fetcher(lambda s: 0.92)
        linker.refresh()

        result = linker.report({"precision": 0.90})
        assert result["linked_kpis"][0]["metric_value"] == 0.90
        assert result["linked_kpis"][0]["kpi_value"] == 0.92
