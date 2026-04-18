"""Tests for ``AzureBlobShipper`` against a fake ``azure.storage.blob``.

The real SDK is never imported. We stub ``azure.identity`` and
``azure.storage.blob`` into ``sys.modules`` before constructing the
shipper, then verify that it:

* constructs a BlobServiceClient with ``DefaultAzureCredential``
* passes a rotated file through to ``upload_blob`` with the right name
* swallows ship failures on the worker thread without killing it
* reports a clean ``health_check`` and a dirty one
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


def _install_fake_azure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    container_client: MagicMock | None = None,
    health_raises: Exception | None = None,
) -> MagicMock:
    """Install fake azure.identity + azure.storage.blob modules."""
    if container_client is None:
        container_client = MagicMock(name="ContainerClient")
        container_client.url = "https://acct.blob.core.windows.net/audit"
    if health_raises is not None:
        container_client.get_container_properties.side_effect = health_raises

    blob_service = MagicMock(name="BlobServiceClient")
    blob_service.get_container_client.return_value = container_client

    identity_mod = types.ModuleType("azure.identity")
    identity_mod.DefaultAzureCredential = MagicMock(  # type: ignore[attr-defined]
        name="DefaultAzureCredential"
    )

    blob_mod = types.ModuleType("azure.storage.blob")
    blob_mod.BlobServiceClient = MagicMock(  # type: ignore[attr-defined]
        return_value=blob_service
    )

    storage_mod = types.ModuleType("azure.storage")
    azure_mod = types.ModuleType("azure")

    monkeypatch.setitem(sys.modules, "azure", azure_mod)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_mod)
    monkeypatch.setitem(sys.modules, "azure.storage", storage_mod)
    monkeypatch.setitem(sys.modules, "azure.storage.blob", blob_mod)

    return container_client


def _fresh_blob_audit() -> Any:
    if "sentinel.integrations.azure.blob_audit" in sys.modules:
        del sys.modules["sentinel.integrations.azure.blob_audit"]
    import sentinel.integrations.azure.blob_audit as mod

    return mod


class TestAzureBlobShipperConstruction:
    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "azure.identity", None)
        monkeypatch.setitem(sys.modules, "azure.storage.blob", None)
        mod = _fresh_blob_audit()
        with pytest.raises(AuditError, match="azure extra"):
            mod.AzureBlobShipper(
                account_url="https://acct.blob.core.windows.net",
                container_name="audit",
            )

    def test_constructs_with_fake_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure(monkeypatch)
        mod = _fresh_blob_audit()
        shipper = mod.AzureBlobShipper(
            account_url="https://acct.blob.core.windows.net",
            container_name="audit",
            prefix="sentinel",
        )
        try:
            # BlobServiceClient was called once with our account URL.
            factory = sys.modules["azure.storage.blob"].BlobServiceClient  # type: ignore[attr-defined]
            factory.assert_called_once()
            kwargs = factory.call_args.kwargs
            assert kwargs["account_url"] == "https://acct.blob.core.windows.net"
        finally:
            shipper.close()


class TestAzureBlobShipperShipping:
    def test_ship_uploads_rotated_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        container = _install_fake_azure(monkeypatch)
        mod = _fresh_blob_audit()
        shipper = mod.AzureBlobShipper(
            account_url="https://acct.blob.core.windows.net",
            container_name="audit",
            prefix="sentinel",
        )
        try:
            rotated = tmp_path / "audit-2026-01-01.jsonl"
            rotated.write_text('{"event_type":"x"}\n')
            shipper.ship(rotated)
            # Give the worker thread a moment to pick up the item.
            for _ in range(50):
                if container.upload_blob.called:
                    break
                time.sleep(0.02)
            assert container.upload_blob.called
            # The blob name should use the configured prefix.
            kwargs = container.upload_blob.call_args.kwargs
            assert kwargs["name"] == "sentinel/audit-2026-01-01.jsonl"
        finally:
            shipper.close()

    def test_ship_failure_does_not_kill_worker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        container = _install_fake_azure(monkeypatch)
        container.upload_blob.side_effect = [RuntimeError("network"), None, None]
        mod = _fresh_blob_audit()
        shipper = mod.AzureBlobShipper(
            account_url="https://acct.blob.core.windows.net",
            container_name="audit",
        )
        try:
            first = tmp_path / "audit-2026-01-01.jsonl"
            first.write_text('{"event_type":"x"}\n')
            second = tmp_path / "audit-2026-01-02.jsonl"
            second.write_text('{"event_type":"x"}\n')
            shipper.ship(first)
            shipper.ship(second)
            # Both ships should have been attempted despite the failure.
            # With retry: first file fail+retry=2, second file=1 → total 3
            for _ in range(100):
                if container.upload_blob.call_count >= 3:
                    break
                time.sleep(0.05)
            assert container.upload_blob.call_count == 3
        finally:
            shipper.close()


class TestAzureBlobShipperRetention:
    def test_enforce_retention_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure(monkeypatch)
        mod = _fresh_blob_audit()
        shipper = mod.AzureBlobShipper(
            account_url="https://acct.blob.core.windows.net",
            container_name="audit",
        )
        try:
            assert shipper.enforce_retention(datetime.now(timezone.utc)) == 0
        finally:
            shipper.close()


class TestAzureBlobShipperHealthCheck:
    def test_health_check_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure(monkeypatch)
        mod = _fresh_blob_audit()
        shipper = mod.AzureBlobShipper(
            account_url="https://acct.blob.core.windows.net",
            container_name="audit",
        )
        try:
            assert shipper.health_check() is True
        finally:
            shipper.close()

    def test_health_check_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure(monkeypatch, health_raises=RuntimeError("403"))
        mod = _fresh_blob_audit()
        shipper = mod.AzureBlobShipper(
            account_url="https://acct.blob.core.windows.net",
            container_name="audit",
        )
        try:
            assert shipper.health_check() is False
        finally:
            shipper.close()
