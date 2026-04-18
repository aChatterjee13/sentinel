"""S3 audit log uploader (mirrors :class:`AzureBlobAuditStorage`)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import structlog

from sentinel.core.exceptions import AuditError
from sentinel.foundation.audit.shipper import ThreadedShipper

log = structlog.get_logger(__name__)


class S3AuditStorage:
    """Upload rotated audit logs to an S3 bucket.

    Requires the `aws` extra: ``pip install sentinel-mlops[aws]``.
    """

    def __init__(self, bucket: str, prefix: str = "sentinel-audit", region: str | None = None):
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as e:
            raise AuditError("aws extra not installed — `pip install sentinel-mlops[aws]`") from e
        self._client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def upload_file(self, file_path: Path, *, delete_local: bool = False) -> str:
        if not file_path.exists():
            raise AuditError(f"audit file not found: {file_path}")
        key = f"{self._prefix}/{file_path.name}"
        self._client.upload_file(str(file_path), self._bucket, key)
        log.info("audit.s3_uploaded", file=str(file_path), key=key)
        if delete_local:
            file_path.unlink(missing_ok=True)
        return f"s3://{self._bucket}/{key}"

    def upload_directory(self, directory: Path, *, delete_local: bool = False) -> list[str]:
        urls: list[str] = []
        for path in sorted(directory.glob("audit-*.jsonl")):
            try:
                urls.append(self.upload_file(path, delete_local=delete_local))
            except Exception as e:
                log.error("audit.s3_upload_failed", file=str(path), error=str(e))
        return urls

    def health_check(self) -> bool:
        """Return True if the bucket is reachable via ``HeadBucket``."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return True
        except Exception as e:
            log.warning("audit.s3.health_check_failed", error=str(e))
            return False


class S3Shipper(ThreadedShipper):
    """Async shipper wrapping :class:`S3AuditStorage`.

    Used by :class:`AuditTrail` when ``audit.storage = "s3"``. Same
    contract as :class:`AzureBlobShipper` — fire-and-forget enqueue,
    background worker does the upload.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "sentinel-audit",
        region: str | None = None,
        delete_local_after_ship: bool = False,
        queue_size: int = 128,
    ) -> None:
        self._storage = S3AuditStorage(bucket=bucket, prefix=prefix, region=region)
        self._delete_local = delete_local_after_ship
        super().__init__(queue_size=queue_size)

    def _ship_sync(self, file_path: Path) -> None:
        self._storage.upload_file(file_path, delete_local=self._delete_local)

    def enforce_retention(self, cutoff: datetime) -> int:
        # Same immutable-bucket contract as AzureBlobShipper — we
        # never reach into S3 to delete shipped files.
        return 0

    def health_check(self) -> bool:
        return self._storage.health_check()
