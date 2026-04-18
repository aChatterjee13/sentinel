"""Azure Blob Storage audit log uploader."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from sentinel.core.exceptions import AuditError
from sentinel.foundation.audit.shipper import ThreadedShipper

log = structlog.get_logger(__name__)


class AzureBlobAuditStorage:
    """Uploads rotated audit log files to an Azure Blob container.

    The :class:`AuditTrail` writes daily JSON-Lines files locally; this
    helper streams them to Blob Storage so the local copy can be pruned
    while retaining a tamper-evident archive.

    Requires the `azure` extra: ``pip install sentinel-mlops[azure]``.
    """

    def __init__(
        self,
        account_url: str,
        container_name: str,
        prefix: str = "sentinel-audit",
        credential: Any = None,
    ):
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            raise AuditError(
                "azure extra not installed — `pip install sentinel-mlops[azure]`"
            ) from e
        self._service = BlobServiceClient(
            account_url=account_url,
            credential=credential or DefaultAzureCredential(),
        )
        self._container = self._service.get_container_client(container_name)
        self._prefix = prefix.rstrip("/")
        # Container may already exist — creation is best-effort here so
        # that operators can pre-provision with finer-grained RBAC.
        with contextlib.suppress(Exception):
            self._container.create_container()

    def upload_file(self, file_path: Path, *, delete_local: bool = False) -> str:
        """Upload a single audit file. Returns the blob URL."""
        if not file_path.exists():
            raise AuditError(f"audit file not found: {file_path}")
        blob_name = f"{self._prefix}/{file_path.name}"
        with file_path.open("rb") as fh:
            self._container.upload_blob(name=blob_name, data=fh, overwrite=True)
        log.info("audit.uploaded", file=str(file_path), blob=blob_name)
        if delete_local:
            file_path.unlink(missing_ok=True)
        return f"{self._container.url}/{blob_name}"

    def upload_directory(self, directory: Path, *, delete_local: bool = False) -> list[str]:
        """Upload every JSONL file in `directory` (one per day)."""
        urls: list[str] = []
        for path in sorted(directory.glob("audit-*.jsonl")):
            try:
                urls.append(self.upload_file(path, delete_local=delete_local))
            except Exception as e:
                log.error("audit.upload_failed", file=str(path), error=str(e))
        return urls

    def write_marker(self, model_name: str, payload: dict[str, Any]) -> None:
        """Drop a small marker blob — useful for compliance acknowledgements."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        blob_name = f"{self._prefix}/markers/{model_name}-{ts}.json"
        import json

        self._container.upload_blob(
            name=blob_name,
            data=json.dumps(payload, default=str).encode("utf-8"),
            overwrite=True,
        )

    def health_check(self) -> bool:
        """Return True if the container is reachable.

        Used by ``sentinel cloud test`` for preflight validation — any
        failure is surfaced to the operator with the underlying SDK
        error so Azure RBAC / network issues can be diagnosed.
        """
        try:
            self._container.get_container_properties()
            return True
        except Exception as e:
            log.warning("audit.blob.health_check_failed", error=str(e))
            return False


class AzureBlobShipper(ThreadedShipper):
    """Async shipper wrapping :class:`AzureBlobAuditStorage`.

    Used by :class:`AuditTrail` when ``audit.storage = "azure_blob"``.
    Enqueues rotated audit files on a background thread so the hot
    write path never blocks on Azure network latency.

    Args:
        account_url: Blob service URL (e.g. ``https://acct.blob.core.windows.net``).
        container_name: Destination container; auto-created if missing.
        prefix: Blob name prefix under which files are stored.
        delete_local_after_ship: Whether to remove the local file
            after a successful upload. Local retention still runs
            independently via ``AuditTrail.enforce_retention``.
        credential: Optional Azure credential; defaults to
            :class:`DefaultAzureCredential`.
        queue_size: Max number of pending ship requests; excess
            requests are dropped with a warning.
    """

    def __init__(
        self,
        *,
        account_url: str,
        container_name: str,
        prefix: str = "sentinel-audit",
        delete_local_after_ship: bool = False,
        credential: Any = None,
        queue_size: int = 128,
    ) -> None:
        self._storage = AzureBlobAuditStorage(
            account_url=account_url,
            container_name=container_name,
            prefix=prefix,
            credential=credential,
        )
        self._delete_local = delete_local_after_ship
        super().__init__(queue_size=queue_size)

    def _ship_sync(self, file_path: Path) -> None:
        self._storage.upload_file(file_path, delete_local=self._delete_local)

    def enforce_retention(self, cutoff: datetime) -> int:
        # Remote retention is a separate workstream — we only ship,
        # we do not reach back into Blob to delete shipped archives.
        # This contract keeps us wire-compatible with immutable
        # storage policies (WORM) that many compliance teams require.
        return 0

    def health_check(self) -> bool:
        return self._storage.health_check()
