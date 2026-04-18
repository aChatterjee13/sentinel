"""AWS integrations — SageMaker, S3, CloudWatch (stubs).

These are intentionally thin wrappers; the bulk of the implementation is
deferred until there is concrete customer demand. The interface is fixed
so SDK consumers can build against it without breaking later.
"""

from sentinel.integrations.aws.s3_audit import S3AuditStorage, S3Shipper

__all__ = ["S3AuditStorage", "S3Shipper"]
