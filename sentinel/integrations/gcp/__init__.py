"""GCP integrations — Cloud Storage audit shipper.

Requires the ``gcp`` extra: ``pip install sentinel-mlops[gcp]``.
"""

from sentinel.integrations.gcp.gcs_audit import GcsAuditStorage, GcsShipper

__all__ = ["GcsAuditStorage", "GcsShipper"]
