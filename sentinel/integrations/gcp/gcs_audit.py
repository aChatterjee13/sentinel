"""GCS audit log uploader (mirrors :class:`S3AuditStorage`)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import structlog

from sentinel.core.exceptions import AuditError
from sentinel.foundation.audit.shipper import ThreadedShipper

log = structlog.get_logger(__name__)


class GcsAuditStorage:
    """Upload rotated audit logs to a GCS bucket.

    Requires the `gcp` extra: ``pip install sentinel-mlops[gcp]``.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "sentinel-audit",
        project: str | None = None,
    ):
        try:
            from google.cloud import storage as gcs_storage  # type: ignore[import-not-found]
        except ImportError as e:
            raise AuditError("gcp extra not installed — `pip install sentinel-mlops[gcp]`") from e
        client = gcs_storage.Client(project=project) if project else gcs_storage.Client()
        self._bucket_obj = client.bucket(bucket)
        self._prefix = prefix.rstrip("/")

    def upload_file(self, file_path: Path, *, delete_local: bool = False) -> str:
        """Upload a single audit file. Returns the ``gs://`` URI."""
        if not file_path.exists():
            raise AuditError(f"audit file not found: {file_path}")
        blob_name = f"{self._prefix}/{file_path.name}"
        blob = self._bucket_obj.blob(blob_name)
        blob.upload_from_filename(str(file_path))
        log.info("audit.gcs_uploaded", file=str(file_path), blob=blob_name)
        if delete_local:
            file_path.unlink(missing_ok=True)
        return f"gs://{self._bucket_obj.name}/{blob_name}"

    def upload_directory(self, directory: Path, *, delete_local: bool = False) -> list[str]:
        """Upload every JSONL file in ``directory`` (one per day)."""
        urls: list[str] = []
        for path in sorted(directory.glob("audit-*.jsonl")):
            try:
                urls.append(self.upload_file(path, delete_local=delete_local))
            except Exception as e:
                log.error("audit.gcs_upload_failed", file=str(path), error=str(e))
        return urls

    def health_check(self) -> bool:
        """Return True if the bucket is reachable."""
        try:
            self._bucket_obj.reload()
            return True
        except Exception as e:
            log.warning("audit.gcs.health_check_failed", error=str(e))
            return False


class GcsShipper(ThreadedShipper):
    """Async shipper wrapping :class:`GcsAuditStorage`.

    Used by :class:`AuditTrail` when ``audit.storage = "gcs"``. Same
    contract as :class:`AzureBlobShipper` and :class:`S3Shipper` —
    fire-and-forget enqueue, background worker does the upload.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "sentinel-audit",
        project: str | None = None,
        delete_local_after_ship: bool = False,
        queue_size: int = 128,
    ) -> None:
        self._storage = GcsAuditStorage(bucket=bucket, prefix=prefix, project=project)
        self._delete_local = delete_local_after_ship
        super().__init__(queue_size=queue_size)

    def _ship_sync(self, file_path: Path) -> None:
        self._storage.upload_file(file_path, delete_local=self._delete_local)

    def enforce_retention(self, cutoff: datetime) -> int:
        # Same immutable-bucket contract as AzureBlobShipper / S3Shipper —
        # we never reach into GCS to delete shipped files.
        return 0

    def health_check(self) -> bool:
        return self._storage.health_check()
