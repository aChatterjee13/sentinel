"""Tests for ``S3Shipper`` against a fake ``boto3`` module.

Same pattern as the Azure Blob shipper tests — we stub boto3 into
``sys.modules`` so the real SDK is never imported.
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


def _install_fake_boto3(
    monkeypatch: pytest.MonkeyPatch,
    *,
    s3_client: MagicMock | None = None,
    head_bucket_raises: Exception | None = None,
) -> MagicMock:
    if s3_client is None:
        s3_client = MagicMock(name="S3Client")
    if head_bucket_raises is not None:
        s3_client.head_bucket.side_effect = head_bucket_raises

    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = MagicMock(return_value=s3_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    return s3_client


def _fresh_s3_audit() -> Any:
    if "sentinel.integrations.aws.s3_audit" in sys.modules:
        del sys.modules["sentinel.integrations.aws.s3_audit"]
    import sentinel.integrations.aws.s3_audit as mod

    return mod


class TestS3ShipperConstruction:
    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)
        mod = _fresh_s3_audit()
        with pytest.raises(AuditError, match="aws extra"):
            mod.S3Shipper(bucket="audit-bucket")

    def test_constructs_with_fake_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch)
        mod = _fresh_s3_audit()
        shipper = mod.S3Shipper(bucket="audit-bucket", prefix="sentinel", region="us-east-1")
        try:
            factory = sys.modules["boto3"].client  # type: ignore[attr-defined]
            factory.assert_called_once_with("s3", region_name="us-east-1")
        finally:
            shipper.close()


class TestS3ShipperShipping:
    def test_ship_uploads_rotated_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        s3 = _install_fake_boto3(monkeypatch)
        mod = _fresh_s3_audit()
        shipper = mod.S3Shipper(bucket="audit-bucket", prefix="sentinel")
        try:
            rotated = tmp_path / "audit-2026-01-01.jsonl"
            rotated.write_text('{"event_type":"x"}\n')
            shipper.ship(rotated)
            for _ in range(50):
                if s3.upload_file.called:
                    break
                time.sleep(0.02)
            assert s3.upload_file.called
            args = s3.upload_file.call_args.args
            # upload_file(file_path, bucket, key)
            assert args[0] == str(rotated)
            assert args[1] == "audit-bucket"
            assert args[2] == "sentinel/audit-2026-01-01.jsonl"
        finally:
            shipper.close()

    def test_ship_failure_does_not_kill_worker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        s3 = _install_fake_boto3(monkeypatch)
        s3.upload_file.side_effect = [RuntimeError("throttled"), None, None]
        mod = _fresh_s3_audit()
        shipper = mod.S3Shipper(bucket="audit-bucket")
        try:
            first = tmp_path / "audit-2026-01-01.jsonl"
            first.write_text('{"event_type":"x"}\n')
            second = tmp_path / "audit-2026-01-02.jsonl"
            second.write_text('{"event_type":"x"}\n')
            shipper.ship(first)
            shipper.ship(second)
            for _ in range(100):
                if s3.upload_file.call_count >= 3:
                    break
                time.sleep(0.05)
            # First file: fail + retry success = 2, second file: success = 1 → total 3
            assert s3.upload_file.call_count == 3
        finally:
            shipper.close()


class TestS3ShipperRetention:
    def test_enforce_retention_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch)
        mod = _fresh_s3_audit()
        shipper = mod.S3Shipper(bucket="audit-bucket")
        try:
            assert shipper.enforce_retention(datetime.now(timezone.utc)) == 0
        finally:
            shipper.close()


class TestS3ShipperHealthCheck:
    def test_health_check_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch)
        mod = _fresh_s3_audit()
        shipper = mod.S3Shipper(bucket="audit-bucket")
        try:
            assert shipper.health_check() is True
        finally:
            shipper.close()

    def test_health_check_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch, head_bucket_raises=RuntimeError("AccessDenied"))
        mod = _fresh_s3_audit()
        shipper = mod.S3Shipper(bucket="audit-bucket")
        try:
            assert shipper.health_check() is False
        finally:
            shipper.close()
