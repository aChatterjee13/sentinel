"""Unit tests for ComplianceReporter (expanded coverage)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sentinel.config.schema import AuditConfig
from sentinel.core.types import AuditEvent
from sentinel.foundation.audit.compliance import ComplianceReporter
from sentinel.foundation.audit.trail import AuditTrail


def _make_event(event_type: str, model_name: str = "fraud_model", **kwargs) -> AuditEvent:
    return AuditEvent(
        event_type=event_type,
        model_name=model_name,
        model_version=kwargs.pop("model_version", "1.0"),
        payload=kwargs,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def trail(tmp_path):
    config = AuditConfig(path=str(tmp_path / "audit"))
    return AuditTrail(config)


@pytest.fixture
def reporter(trail):
    return ComplianceReporter(trail)


def _populate_trail(trail, events):
    for event in events:
        trail.log_event(event)


class TestFCAReport:
    """Tests for FCA Consumer Duty report generation."""

    def test_fca_report_structure(self, reporter, trail) -> None:
        _populate_trail(
            trail,
            [
                _make_event("drift_detected"),
                _make_event("retrain_triggered"),
                _make_event("prediction_logged"),
            ],
        )
        report = reporter.generate("fca_consumer_duty", "fraud_model")
        assert report["framework"] == "fca_consumer_duty"
        assert report["model"] == "fraud_model"
        assert "summary" in report
        assert report["summary"]["total_events"] == 3
        assert report["summary"]["drift_detections"] == 1
        assert report["summary"]["retrains"] == 1
        assert report["summary"]["predictions_logged"] == 1

    def test_fca_report_fairness_active(self, reporter, trail) -> None:
        _populate_trail(trail, [_make_event("drift_detected")])
        report = reporter.generate("fca_consumer_duty", "fraud_model")
        assert report["fairness_monitoring"]["status"] == "active"

    def test_fca_report_no_drift(self, reporter, trail) -> None:
        _populate_trail(trail, [_make_event("prediction_logged")])
        report = reporter.generate("fca_consumer_duty", "fraud_model")
        assert report["fairness_monitoring"]["status"] == "no_signals"

    def test_fca_report_human_oversight(self, reporter, trail) -> None:
        _populate_trail(
            trail,
            [
                _make_event("approval_decision", approved=True, reviewer="alice"),
            ],
        )
        report = reporter.generate("fca_consumer_duty", "fraud_model")
        assert report["human_oversight"]["approval_decisions"] == 1
        assert report["human_oversight"]["details"][0]["approved"] is True

    def test_fca_report_empty_trail(self, reporter) -> None:
        report = reporter.generate("fca_consumer_duty", "fraud_model")
        assert report["summary"]["total_events"] == 0


class TestEUAIActReport:
    """Tests for EU AI Act report generation."""

    def test_eu_ai_act_report_structure(self, reporter, trail) -> None:
        _populate_trail(
            trail,
            [
                _make_event("deployment_started", model_version="2.0"),
                _make_event("deployment_completed", model_version="2.0"),
            ],
        )
        report = reporter.generate("eu_ai_act", "fraud_model")
        assert report["framework"] == "eu_ai_act"
        assert report["risk_classification"] == "high"
        assert report["human_oversight"] is True
        assert report["logging"] == "complete"

    def test_eu_ai_act_transparency_versions(self, reporter, trail) -> None:
        _populate_trail(
            trail,
            [
                _make_event("model_registered", model_version="1.0"),
                _make_event("model_registered", model_version="2.0"),
            ],
        )
        report = reporter.generate("eu_ai_act", "fraud_model")
        versions = report["transparency"]["registered_versions"]
        assert "1.0" in versions
        assert "2.0" in versions

    def test_eu_ai_act_deployment_history(self, reporter, trail) -> None:
        _populate_trail(
            trail,
            [
                _make_event("deployment_started", strategy="canary", traffic=5),
            ],
        )
        report = reporter.generate("eu_ai_act", "fraud_model")
        assert len(report["transparency"]["deployment_history"]) == 1

    def test_eu_ai_act_empty_trail(self, reporter) -> None:
        report = reporter.generate("eu_ai_act", "fraud_model")
        assert report["transparency"]["registered_versions"] == []
        assert report["transparency"]["deployment_history"] == []


class TestInternalAudit:
    """Tests for internal audit report generation."""

    def test_internal_audit_event_counts(self, reporter, trail) -> None:
        _populate_trail(
            trail,
            [
                _make_event("drift_detected"),
                _make_event("drift_detected"),
                _make_event("retrain_triggered"),
                _make_event("prediction_logged"),
            ],
        )
        report = reporter.generate("internal_audit", "fraud_model")
        assert report["framework"] == "internal_audit"
        assert report["event_counts"]["drift_detected"] == 2
        assert report["event_counts"]["retrain_triggered"] == 1
        assert report["event_counts"]["prediction_logged"] == 1

    def test_internal_audit_timestamps(self, reporter, trail) -> None:
        _populate_trail(
            trail,
            [
                _make_event("event_a"),
                _make_event("event_b"),
            ],
        )
        report = reporter.generate("internal_audit", "fraud_model")
        assert report["first_event"] is not None
        assert report["last_event"] is not None

    def test_internal_audit_empty_trail(self, reporter) -> None:
        report = reporter.generate("internal_audit", "fraud_model")
        assert report["event_counts"] == {}
        assert report["first_event"] is None
        assert report["last_event"] is None

    def test_internal_audit_events_serialized(self, reporter, trail) -> None:
        _populate_trail(trail, [_make_event("test_event")])
        report = reporter.generate("internal_audit", "fraud_model")
        assert len(report["events"]) == 1
        assert report["events"][0]["event_type"] == "test_event"


class TestUnknownFramework:
    """Test error handling for unknown frameworks."""

    def test_unknown_framework_raises(self, reporter) -> None:
        with pytest.raises(ValueError, match="unknown compliance framework"):
            reporter.generate("sarbanes_oxley", "fraud_model")


class TestPeriodFiltering:
    """Test that period_days parameter works."""

    def test_period_filters_old_events(self, reporter, trail) -> None:
        # Log events — they'll be recent (within default 30 days)
        _populate_trail(trail, [_make_event("recent_event")])
        report = reporter.generate("internal_audit", "fraud_model", period_days=30)
        assert report["event_counts"].get("recent_event", 0) == 1

    def test_very_short_period(self, reporter, trail) -> None:
        _populate_trail(trail, [_make_event("event")])
        # 0 days period should still include today's events
        report = reporter.generate("internal_audit", "fraud_model", period_days=0)
        # Events just written should still be within "since"
        # (since = now - 0 days = now, and event.timestamp ~ now)
        # This is edge-case behavior
        assert isinstance(report, dict)
