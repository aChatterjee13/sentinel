"""Tests for CSV export endpoints and deployment rollback."""

from __future__ import annotations

import csv
import io

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


class TestExportAuditCSV:
    def test_returns_csv_with_correct_headers(self, client: TestClient) -> None:
        resp = client.get("/api/export/audit.csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "audit_export.csv" in resp.headers.get("content-disposition", "")

        reader = csv.reader(io.StringIO(resp.text))
        header = next(reader)
        assert header == [
            "timestamp",
            "event_type",
            "model_name",
            "model_version",
            "actor",
            "payload",
        ]
        rows = list(reader)
        assert len(rows) >= 3  # conftest seeds 3 events

    def test_filter_by_event_type(self, client: TestClient) -> None:
        resp = client.get("/api/export/audit.csv?event_type=drift_checked")
        assert resp.status_code == 200

        reader = csv.reader(io.StringIO(resp.text))
        next(reader)  # skip header
        rows = list(reader)
        assert len(rows) >= 2
        assert all(row[1] == "drift_checked" for row in rows)

    def test_limit_parameter(self, client: TestClient) -> None:
        resp = client.get("/api/export/audit.csv?limit=1")
        assert resp.status_code == 200

        reader = csv.reader(io.StringIO(resp.text))
        next(reader)
        rows = list(reader)
        assert len(rows) == 1


class TestExportDriftCSV:
    def test_returns_csv_with_drift_headers(self, client: TestClient) -> None:
        resp = client.get("/api/export/drift.csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "drift_export.csv" in resp.headers.get("content-disposition", "")

        reader = csv.reader(io.StringIO(resp.text))
        header = next(reader)
        assert header == [
            "timestamp",
            "model_name",
            "method",
            "is_drifted",
            "severity",
            "test_statistic",
        ]
        rows = list(reader)
        assert len(rows) >= 2  # conftest seeds 2 drift_checked events


class TestExportMetricsCSV:
    def test_returns_csv_with_metrics_headers(self, client: TestClient) -> None:
        resp = client.get("/api/export/metrics.csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "metrics_export.csv" in resp.headers.get("content-disposition", "")

        reader = csv.reader(io.StringIO(resp.text))
        header = next(reader)
        assert header == ["timestamp", "model_name", "model_version", "payload"]

    def test_empty_result_still_has_header(self, client: TestClient) -> None:
        # No prediction_logged events are seeded, so body is header-only
        resp = client.get("/api/export/metrics.csv")
        assert resp.status_code == 200
        reader = csv.reader(io.StringIO(resp.text))
        header = next(reader)
        assert header[0] == "timestamp"


class TestRollbackEndpoint:
    def test_rollback_returns_result(self, client: TestClient) -> None:
        # GET first to obtain the CSRF cookie
        get_resp = client.get("/api/health/live")
        csrf_token = get_resp.cookies.get("sentinel_csrf", "")
        resp = client.post(
            "/api/deployments/rollback?version=1.0",
            headers={"X-CSRF-Token": csrf_token},
        )
        # The deployment may fail because no model is actually registered,
        # but the endpoint itself should be wired correctly (200 or 400).
        assert resp.status_code in (200, 400)

    def test_rollback_without_version(self, client: TestClient) -> None:
        get_resp = client.get("/api/health/live")
        csrf_token = get_resp.cookies.get("sentinel_csrf", "")
        resp = client.post(
            "/api/deployments/rollback",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code in (200, 400)
