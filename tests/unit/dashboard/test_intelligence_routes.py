"""Tests for intelligence dashboard routes — WS-E."""

from __future__ import annotations

from unittest.mock import MagicMock

from sentinel.config.schema import (
    BusinessKPIConfig,
    DashboardConfig,
    DashboardUIConfig,
    KPIMapping,
    ModelConfig,
    ModelGraphConfig,
    ModelGraphEdge,
    SentinelConfig,
)
from sentinel.dashboard.state import DashboardState
from sentinel.dashboard.views.intelligence import build


def _make_state(
    graph_edges: list[ModelGraphEdge] | None = None,
    kpi_mappings: list[KPIMapping] | None = None,
    explainability_configured: bool = False,
) -> DashboardState:
    """Build a mock DashboardState for intelligence view tests."""
    from sentinel.intelligence.kpi_linker import KPILinker
    from sentinel.intelligence.model_graph import ModelGraph

    config = SentinelConfig(
        model=ModelConfig(name="fraud_v2"),
        model_graph=ModelGraphConfig(
            dependencies=graph_edges or [],
            cascade_alerts=True,
        ),
        business_kpi=BusinessKPIConfig(
            mappings=kpi_mappings or [],
        ),
    )

    client = MagicMock()
    client.model_name = "fraud_v2"
    client.current_version = "1.0.0"
    client.config = config
    client.model_graph = ModelGraph(config.model_graph)
    client.kpi_linker = KPILinker(config.business_kpi)
    client._explainability = MagicMock() if explainability_configured else None
    client.explainability_engine = client._explainability

    # Registry mock
    mv = MagicMock()
    mv.metrics = {"accuracy": 0.92, "f1": 0.88}
    client.registry.get_latest.return_value = mv

    state = DashboardState(
        client=client,
        config=DashboardConfig(
            ui=DashboardUIConfig(show_modules=["intelligence"]),
        ),
    )
    return state


class TestIntelligenceView:
    def test_build_empty_graph(self) -> None:
        state = _make_state()
        data = build(state)
        assert data["model_graph"]["nodes"] == []
        assert data["model_graph"]["edges"] == []
        assert data["cascade_impact"].get("downstream_affected", []) == []
        assert data["explainability"]["configured"] is False

    def test_build_with_graph(self) -> None:
        edges = [
            ModelGraphEdge(upstream="feature_pipeline", downstream="fraud_v2"),
            ModelGraphEdge(upstream="fraud_v2", downstream="auto_adjudication"),
        ]
        state = _make_state(graph_edges=edges)
        data = build(state)
        assert len(data["model_graph"]["nodes"]) == 3
        assert len(data["model_graph"]["edges"]) == 2

    def test_cascade_impact_with_downstream(self) -> None:
        edges = [
            ModelGraphEdge(upstream="fraud_v2", downstream="auto_adjudication"),
            ModelGraphEdge(upstream="fraud_v2", downstream="risk_scorer"),
        ]
        state = _make_state(graph_edges=edges)
        data = build(state)
        assert "auto_adjudication" in data["cascade_impact"]["downstream_affected"]
        assert "risk_scorer" in data["cascade_impact"]["downstream_affected"]

    def test_kpi_linkage(self) -> None:
        mappings = [
            KPIMapping(
                model_metric="accuracy",
                business_kpi="fraud_catch_rate",
                data_source="warehouse://analytics",
            ),
        ]
        state = _make_state(kpi_mappings=mappings)
        data = build(state)
        assert data["kpi_report"]["n_links"] == 1
        link = data["kpi_report"]["linked_kpis"][0]
        assert link["model_metric"] == "accuracy"
        assert link["business_kpi"] == "fraud_catch_rate"
        assert link["metric_value"] == 0.92

    def test_explainability_configured(self) -> None:
        state = _make_state(explainability_configured=True)
        data = build(state)
        assert data["explainability"]["configured"] is True
        assert data["explainability"]["method"] == "shap"

    def test_explainability_not_configured(self) -> None:
        state = _make_state(explainability_configured=False)
        data = build(state)
        assert data["explainability"]["configured"] is False
        assert data["explainability"]["method"] == "not configured"


class TestCascadeAlertInClient:
    """Test that cascade alerts fire in _fire_drift_alert."""

    def test_cascade_alert_dispatched(self) -> None:
        from sentinel.config.schema import ModelGraphConfig, ModelGraphEdge
        from sentinel.core.types import AlertSeverity, DriftReport
        from sentinel.intelligence.model_graph import ModelGraph

        client = MagicMock()
        client.model_name = "fraud_v2"
        client.model_graph = ModelGraph(
            ModelGraphConfig(
                dependencies=[
                    ModelGraphEdge(upstream="fraud_v2", downstream="downstream_model"),
                ],
                cascade_alerts=True,
            )
        )
        client.notifications = MagicMock()
        client.audit = MagicMock()

        # Call the actual method under test — import and bind
        from sentinel.core.client import SentinelClient

        # Use the unbound method pattern
        report = MagicMock(spec=DriftReport)
        report.method = "psi"
        report.severity = AlertSeverity.HIGH
        report.test_statistic = 0.35
        report.drifted_features = ["feature_1"]

        SentinelClient._fire_drift_alert(client, report)

        # Should have dispatched 2 alerts: primary + cascade
        assert client.notifications.dispatch.call_count == 2
        cascade_call = client.notifications.dispatch.call_args_list[1]
        cascade_alert = cascade_call[0][0]
        assert cascade_alert.source == "model_graph"

    def test_no_cascade_when_no_downstream(self) -> None:
        from sentinel.core.types import AlertSeverity, DriftReport
        from sentinel.intelligence.model_graph import ModelGraph

        client = MagicMock()
        client.model_name = "fraud_v2"
        client.model_graph = ModelGraph()  # empty graph
        client.notifications = MagicMock()

        from sentinel.core.client import SentinelClient

        report = MagicMock(spec=DriftReport)
        report.method = "psi"
        report.severity = AlertSeverity.HIGH
        report.test_statistic = 0.35
        report.drifted_features = ["feature_1"]

        SentinelClient._fire_drift_alert(client, report)

        # Only primary alert dispatched
        assert client.notifications.dispatch.call_count == 1


class TestKPIInOverview:
    def test_overview_includes_kpi_report(self) -> None:
        from sentinel.dashboard.views.overview import build as overview_build
        from sentinel.intelligence.kpi_linker import KPILinker

        config = SentinelConfig(
            model=ModelConfig(name="fraud_v2"),
            business_kpi=BusinessKPIConfig(
                mappings=[
                    KPIMapping(model_metric="f1", business_kpi="detection_rate"),
                ],
            ),
        )

        client = MagicMock()
        client.model_name = "fraud_v2"
        client.current_version = "1.0.0"
        client.config = config
        client.kpi_linker = KPILinker(config.business_kpi)

        mv = MagicMock()
        mv.metrics = {"f1": 0.88}
        client.registry.get_latest.return_value = mv
        client.status.return_value = {"model": "fraud_v2"}
        client.audit.latest.return_value = []
        client.deployment_manager.list_active.return_value = []

        state = DashboardState(client=client)
        data = overview_build(state)
        assert "kpi_report" in data
        assert data["kpi_report"]["n_links"] == 1
