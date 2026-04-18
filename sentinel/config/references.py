"""File-reference validation for Sentinel configs.

A Sentinel config can point at several on-disk paths — baseline
datasets, JSON Schema files, holdout datasets, audit directories.
``sentinel validate`` traditionally only checked Pydantic schema
correctness, so a typo in ``model.baseline_dataset`` would pass
validation cleanly and only blow up when the SDK actually tried to
read the file at first prediction time. This module surfaces those
issues at validation time.

URI-style locations (``s3://``, ``azure://``, ``azureml://``,
``gs://``, ``http://``, ``https://``) are skipped — they're not local
paths and need cloud-credential pre-flight that lives in workstream #2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sentinel.config.schema import SentinelConfig

#: URI schemes that the local-path validator should ignore.
_REMOTE_SCHEMES: tuple[str, ...] = (
    "s3://",
    "azure://",
    "azureml://",
    "gs://",
    "http://",
    "https://",
    "abfs://",
    "abfss://",
    "wasbs://",
    "warehouse://",
)


ReferenceSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class ReferenceIssue:
    """A single problem found while validating file references.

    Attributes:
        field: Dotted JSON path to the offending field
            (e.g. ``model.baseline_dataset``).
        path: The path string as it appeared in the config.
        message: Human-friendly explanation.
        severity: ``"error"`` (fail strict validation) or ``"warning"``
            (informational).
    """

    field: str
    path: str
    message: str
    severity: ReferenceSeverity = "error"

    def format(self) -> str:
        """Return a one-line summary suitable for CLI output."""
        return f"[{self.severity}] {self.field}: {self.message} ({self.path!r})"


def _is_remote_uri(path: str) -> bool:
    """Return True if ``path`` is a URI we shouldn't try to stat locally."""
    return any(path.startswith(scheme) for scheme in _REMOTE_SCHEMES)


def _resolve(base_dir: Path, raw: str) -> Path:
    """Resolve ``raw`` against ``base_dir`` if it isn't absolute."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def validate_file_references(
    cfg: SentinelConfig,
    base_dir: Path,
) -> list[ReferenceIssue]:
    """Walk the config and check that referenced paths exist.

    Args:
        cfg: A loaded :class:`~sentinel.config.schema.SentinelConfig`.
        base_dir: Directory used as the root for relative path resolution
            — typically the parent directory of the config file.

    Returns:
        A list of :class:`ReferenceIssue` records. An empty list means
        every reference resolved cleanly.

    Example:
        >>> from sentinel.config import load_config, validate_file_references
        >>> cfg = load_config("sentinel.yaml")
        >>> issues = validate_file_references(cfg, Path("."))
        >>> for issue in issues:
        ...     print(issue.format())
    """
    issues: list[ReferenceIssue] = []

    # ── model.baseline_dataset ──────────────────────────────────
    if cfg.model.baseline_dataset:
        _check_existing_file(
            "model.baseline_dataset",
            cfg.model.baseline_dataset,
            base_dir,
            issues,
        )

    # ── data_quality.schema.path ────────────────────────────────
    schema_path = cfg.data_quality.schema_.path
    if schema_path:
        _check_existing_file(
            "data_quality.schema.path",
            schema_path,
            base_dir,
            issues,
        )

    # ── retraining.validation.holdout_dataset ───────────────────
    holdout = cfg.retraining.validation.holdout_dataset
    if holdout:
        _check_existing_file(
            "retraining.validation.holdout_dataset",
            holdout,
            base_dir,
            issues,
        )

    # ── retraining.pipeline (only when path-like) ───────────────
    pipeline = cfg.retraining.pipeline
    if pipeline and not _is_remote_uri(pipeline):
        # ``pipeline`` is sometimes a script path; warn (not error) if it's
        # missing because plenty of pipelines are remote-only.
        resolved = _resolve(base_dir, pipeline)
        if not resolved.exists():
            issues.append(
                ReferenceIssue(
                    field="retraining.pipeline",
                    path=pipeline,
                    message=f"pipeline path does not exist: {resolved}",
                    severity="warning",
                )
            )

    # ── audit.path (parent dir must be writeable) ───────────────
    audit_path_str = cfg.audit.path
    if cfg.audit.storage == "local" and audit_path_str:
        audit_path = _resolve(base_dir, audit_path_str)
        parent = audit_path.parent if not audit_path.exists() else audit_path
        # We can't reliably mkdir at validation time — that's a side
        # effect — but we can verify the existing parent is writable.
        target = parent if parent.exists() else parent.parent
        if not target.exists():
            issues.append(
                ReferenceIssue(
                    field="audit.path",
                    path=audit_path_str,
                    message=f"neither the audit dir nor its parent exists: {audit_path}",
                )
            )
        elif not os.access(target, os.W_OK):
            issues.append(
                ReferenceIssue(
                    field="audit.path",
                    path=audit_path_str,
                    message=f"audit directory is not writable: {target}",
                )
            )

    return issues


def _check_existing_file(
    field: str,
    raw: str,
    base_dir: Path,
    issues: list[ReferenceIssue],
) -> None:
    """Append an issue if ``raw`` should be a local file but isn't there."""
    if _is_remote_uri(raw):
        return
    resolved = _resolve(base_dir, raw)
    if not resolved.exists():
        issues.append(
            ReferenceIssue(
                field=field,
                path=raw,
                message=f"file not found: {resolved}",
            )
        )
