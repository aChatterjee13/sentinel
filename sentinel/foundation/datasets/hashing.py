"""Content hashing utilities for dataset integrity verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_REMOTE_PREFIXES = ("s3://", "az://", "gs://", "http://", "https://")


def compute_hash(path: str, algorithm: str = "sha256") -> str | None:
    """Compute a content hash of a local file.

    Args:
        path: File path (local or remote URI).
        algorithm: Hash algorithm name accepted by :mod:`hashlib`.

    Returns:
        Hex-digest string for local files, ``None`` for remote URIs or
        if the file cannot be read.

    Example:
        >>> h = compute_hash("data/train.parquet")
        >>> assert h is not None and len(h) == 64
    """
    if any(path.startswith(prefix) for prefix in _REMOTE_PREFIXES):
        log.debug("datasets.hash_skipped_remote", path=path)
        return None

    file = Path(path)
    if not file.is_file():
        log.warning("datasets.hash_file_not_found", path=path)
        return None

    hasher = hashlib.new(algorithm)
    try:
        with file.open("rb") as fh:
            while chunk := fh.read(8192):
                hasher.update(chunk)
    except OSError as exc:
        log.warning("datasets.hash_read_error", path=path, error=str(exc))
        return None

    return hasher.hexdigest()
