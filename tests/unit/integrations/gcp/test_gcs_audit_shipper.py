"""Tests for ``GcsShipper`` against a fake ``google.cloud.storage`` module.

Same pattern as the S3 and Azure Blob shipper tests — we stub the GCS SDK
into ``sys.modules`` so the real SDK is never imported.
"""

from __future__ import annotations

import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.core.exceptions import AuditError


def _install_fake_gcs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bucket: MagicMock | None = None,
    reload_raises: Exception | None = None,
) -> MagicMock:
    """Install a fake ``google.cloud.storage`` module."""
    if bucket is None:
        bucket = MagicMock(name="Bucket")
        bucket.name = "test-bucket"
    if reload_raises is not None:
        bucket.reload.side_effect = reload_raises

    fake_blob = MagicMock(name="Blob")
    bucket.blob.return_value = fake_blob

    client_instance = MagicMock(name="StorageClient")
    client_instance.bucket.return_value = bucket

    storage_mod = types.ModuleType("google.cloud.storage")
    storage_mod.Client = MagicMock(return_value=client_instance)  # type: ignore[attr-defined]

    # Build the google.cloud package chain
    google_mod = types.ModuleType("google")
    google_cloud_mod = types.ModuleType("google.cloud")

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_mod)

    return bucket


def _fresh_gcs_audit() -> Any:
    if "sentinel.integrations.gcp.gcs_audit" in sys.modules:
        del sys.modules["sentinel.integrations.gcp.gcs_audit"]
    import sentinel.integrations.gcp.gcs_audit as mod

    return mod


class TestGcsShipperConstruction:
    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "google.cloud.storage", None)
        monkeypatch.setitem(sys.modules, "google.cloud", None)
        monkeypatch.setitem(sys.modules, "google", None)
        mod = _fresh_gcs_audit()
        with pytest.raises(AuditError, match="gcp extra"):
            mod.GcsShipper(bucket="audit-bucket")

    def test_constructs_with_fake_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcs(monkeypatch)
        mod = _fresh_gcs_audit()
        shipper = mod.GcsShipper(bucket="audit-bucket", prefix="sentinel")
        try:
            assert shipper is not None
        finally:
            shipper.close()


class TestGcsShipperShipping:
    def test_ship_uploads_rotated_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bucket = _install_fake_gcs(monkeypatch)
        mod = _fresh_gcs_audit()
        shipper = mod.GcsShipper(bucket="audit-bucket", prefix="sentinel")
        try:
            rotated = tmp_path / "audit-2026-01-01.jsonl"
            rotated.write_text('{"event_type":"x"}\n')
            shipper.ship(rotated)
            for _ in range(50):
                if bucket.blob.called:
                    break
                time.sleep(0.02)
            assert bucket.blob.called
            blob_name = bucket.blob.call_args.args[0]
            assert blob_name == "sentinel/audit-2026-01-01.jsonl"
        finally:
            shipper.close()

    def test_ship_failure_does_not_kill_worker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bucket = _install_fake_gcs(monkeypatch)
        fake_blob = bucket.blob.return_value
        fake_blob.upload_from_filename.side_effect = [RuntimeError("throttled"), None, None]
        mod = _fresh_gcs_audit()
        shipper = mod.GcsShipper(bucket="audit-bucket")
        try:
            first = tmp_path / "audit-2026-01-01.jsonl"
            first.write_text('{"event_type":"x"}\n')
            second = tmp_path / "audit-2026-01-02.jsonl"
            second.write_text('{"event_type":"x"}\n')
            shipper.ship(first)
            shipper.ship(second)
            # With retry: first file fail+retry=2, second file=1 → total 3
            for _ in range(100):
                if fake_blob.upload_from_filename.call_count >= 3:
                    break
                time.sleep(0.05)
            assert fake_blob.upload_from_filename.call_count == 3
        finally:
            shipper.close()


class TestGcsShipperRetention:
    def test_enforce_retention_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcs(monkeypatch)
        mod = _fresh_gcs_audit()
        shipper = mod.GcsShipper(bucket="audit-bucket")
        try:
            assert shipper.enforce_retention(datetime.now(timezone.utc)) == 0
        finally:
            shipper.close()


class TestGcsShipperHealthCheck:
    def test_health_check_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcs(monkeypatch)
        mod = _fresh_gcs_audit()
        shipper = mod.GcsShipper(bucket="audit-bucket")
        try:
            assert shipper.health_check() is True
        finally:
            shipper.close()

    def test_health_check_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcs(monkeypatch, reload_raises=RuntimeError("AccessDenied"))
        mod = _fresh_gcs_audit()
        shipper = mod.GcsShipper(bucket="audit-bucket")
        try:
            assert shipper.health_check() is False
        finally:
            shipper.close()


class TestGcsAuditConfig:
    def test_gcs_config_validation(self) -> None:
        from sentinel.config.schema import GcsAuditConfig

        cfg = GcsAuditConfig(bucket="my-bucket")
        assert cfg.bucket == "my-bucket"
        assert cfg.prefix == "sentinel-audit"
        assert cfg.project is None
        assert cfg.delete_local_after_ship is False

    def test_audit_config_requires_gcs_block(self) -> None:
        from sentinel.config.schema import AuditConfig

        with pytest.raises(ValueError, match=r"audit\.storage=gcs requires audit\.gcs"):
            AuditConfig(storage="gcs")

    def test_audit_config_with_gcs_block(self) -> None:
        from sentinel.config.schema import AuditConfig, GcsAuditConfig

        cfg = AuditConfig(
            storage="gcs",
            gcs=GcsAuditConfig(bucket="my-bucket", project="my-project"),
        )
        assert cfg.gcs is not None
        assert cfg.gcs.bucket == "my-bucket"
