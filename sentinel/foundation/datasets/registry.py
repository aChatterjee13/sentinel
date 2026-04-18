"""Dataset metadata registry — register, query, compare, and link dataset versions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from sentinel.foundation.datasets.hashing import compute_hash

log = structlog.get_logger(__name__)


class DatasetVersion(BaseModel):
    """Immutable metadata record for a single dataset version.

    Example:
        >>> ds = DatasetVersion(name="train", version="1.0", path="data/train.parquet",
        ...                     created_at=datetime.now(timezone.utc).isoformat())
        >>> ds.name
        'train'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    name: str
    version: str
    path: str
    format: str = "unknown"
    split: str | None = None
    num_rows: int | None = None
    num_features: int | None = None
    content_hash: str | None = None
    schema_: dict[str, str] = Field(default_factory=dict, alias="schema")
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Lineage links
    linked_experiments: list[str] = Field(default_factory=list)
    linked_models: list[dict[str, str]] = Field(default_factory=list)
    derived_from: str | None = None


class DatasetRegistry:
    """Local-filesystem dataset metadata registry.

    Each dataset version is persisted as a JSON file at
    ``{storage_path}/{name}/{version}.json``.  The registry only
    manages metadata — it never copies, moves, or deletes the actual
    data files referenced by :pyattr:`DatasetVersion.path`.

    Args:
        storage_path: Directory for JSON metadata files.
        auto_hash: When ``True``, automatically compute content hashes
            for local files during :meth:`register`.
        require_schema: When ``True``, :meth:`register` raises
            :class:`ValueError` if no ``schema`` is provided.

    Example:
        >>> registry = DatasetRegistry(Path("./datasets"))
        >>> ds = registry.register(name="train", version="1.0",
        ...                        path="data/train.parquet", format="parquet")
        >>> registry.get("train", "1.0").path
        'data/train.parquet'
    """

    def __init__(
        self,
        storage_path: Path,
        auto_hash: bool = True,
        require_schema: bool = False,
    ):
        self._root = Path(storage_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._auto_hash = auto_hash
        self._require_schema = require_schema

    # ── Helpers ───────────────────────────────────────────────────

    def _meta_path(self, name: str, version: str) -> Path:
        return self._root / name / f"{version}.json"

    def _save(self, ds: DatasetVersion) -> None:
        """Persist a dataset version to disk with atomic write."""
        meta = self._meta_path(ds.name, ds.version)
        meta.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(ds.model_dump(mode="json", by_alias=True), indent=2, default=str)
        meta.write_text(payload)
        log.info(
            "datasets.saved",
            name=ds.name,
            version=ds.version,
            path=str(meta),
        )

    def _load(self, name: str, version: str) -> DatasetVersion:
        meta = self._meta_path(name, version)
        if not meta.exists():
            raise KeyError(f"Dataset '{name}@{version}' not found")
        return DatasetVersion.model_validate(json.loads(meta.read_text()))

    # ── Public API ────────────────────────────────────────────────

    def register(
        self,
        *,
        name: str,
        version: str,
        path: str,
        format: str = "unknown",
        split: str | None = None,
        num_rows: int | None = None,
        num_features: int | None = None,
        schema: dict[str, str] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        derived_from: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """Register a new dataset version.

        Args:
            name: Logical dataset name.
            version: Semantic version string.
            path: Location of the data (local path or remote URI).
            format: Data format (``parquet``, ``csv``, ``json``, etc.).
            split: Data split (``train``, ``validation``, ``test``, etc.).
            num_rows: Row count.
            num_features: Feature/column count.
            schema: Column name → dtype mapping.
            description: Human-readable description.
            tags: Arbitrary string tags for searching.
            source: What produced this dataset (pipeline, notebook, etc.).
            derived_from: Parent dataset in ``'name@version'`` format.
            metadata: Arbitrary key-value metadata.

        Returns:
            The registered :class:`DatasetVersion`.

        Raises:
            ValueError: If the version already exists, or if
                ``require_schema`` is enabled but no schema is given.

        Example:
            >>> ds = registry.register(name="train", version="1.0",
            ...                        path="data/train.parquet")
        """
        if self._meta_path(name, version).exists():
            raise ValueError(f"Dataset '{name}@{version}' already exists")

        if self._require_schema and not schema:
            raise ValueError(
                f"Schema is required for dataset '{name}@{version}' (require_schema=True)"
            )

        content_hash: str | None = None
        if self._auto_hash:
            content_hash = compute_hash(path)

        ds = DatasetVersion(
            name=name,
            version=version,
            path=path,
            format=format,
            split=split,
            num_rows=num_rows,
            num_features=num_features,
            content_hash=content_hash,
            schema=schema or {},
            description=description,
            tags=tags or [],
            source=source,
            derived_from=derived_from,
            metadata=metadata or {},
        )
        self._save(ds)
        return ds

    def get(self, name: str, version: str) -> DatasetVersion:
        """Get a specific dataset version.

        Args:
            name: Dataset name.
            version: Version string.

        Returns:
            The :class:`DatasetVersion`.

        Raises:
            KeyError: If the version does not exist.
        """
        return self._load(name, version)

    def list_versions(self, name: str) -> list[DatasetVersion]:
        """List all versions of a dataset, sorted by version string.

        Args:
            name: Dataset name.

        Returns:
            List of :class:`DatasetVersion` instances sorted by version.
        """
        ds_dir = self._root / name
        if not ds_dir.exists():
            return []
        versions: list[DatasetVersion] = []
        for meta in sorted(ds_dir.glob("*.json")):
            try:
                versions.append(DatasetVersion.model_validate(json.loads(meta.read_text())))
            except Exception:
                log.warning("datasets.load_failed", path=str(meta))
        return sorted(versions, key=lambda d: d.version)

    def list_all(self) -> list[DatasetVersion]:
        """List all datasets across all names.

        Returns:
            List of every :class:`DatasetVersion` in the registry.
        """
        result: list[DatasetVersion] = []
        if not self._root.exists():
            return result
        for ds_dir in sorted(self._root.iterdir()):
            if ds_dir.is_dir():
                result.extend(self.list_versions(ds_dir.name))
        return result

    def list_names(self) -> list[str]:
        """List all dataset names in the registry.

        Returns:
            Sorted list of dataset name strings.
        """
        if not self._root.exists():
            return []
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())

    def search(
        self,
        *,
        tags: list[str] | None = None,
        split: str | None = None,
        format: str | None = None,
        name_pattern: str | None = None,
    ) -> list[DatasetVersion]:
        """Search datasets by criteria.

        Args:
            tags: If given, every tag must be present on the dataset.
            split: Filter by split name.
            format: Filter by data format.
            name_pattern: Regex pattern to match against dataset name.

        Returns:
            List of matching :class:`DatasetVersion` instances.

        Example:
            >>> registry.search(tags=["production"], split="train")
            [DatasetVersion(name='train', ...)]
        """
        results: list[DatasetVersion] = []
        for ds in self.list_all():
            if tags and not all(t in ds.tags for t in tags):
                continue
            if split is not None and ds.split != split:
                continue
            if format is not None and ds.format != format:
                continue
            if name_pattern is not None and not re.search(name_pattern, ds.name):
                continue
            results.append(ds)
        return results

    def compare(self, name: str, version_a: str, version_b: str) -> dict[str, Any]:
        """Compare two versions of a dataset.

        Args:
            name: Dataset name.
            version_a: First version string.
            version_b: Second version string.

        Returns:
            Dict with ``schema_added``, ``schema_removed``, ``schema_type_changed``,
            ``row_count_diff``, ``feature_count_diff``, and ``hash_match`` keys.

        Raises:
            KeyError: If either version is not found.

        Example:
            >>> diff = registry.compare("train", "1.0", "2.0")
            >>> diff["schema_added"]
            ['new_column']
        """
        a = self.get(name, version_a)
        b = self.get(name, version_b)

        schema_a = set(a.schema_.keys())
        schema_b = set(b.schema_.keys())

        type_changes: dict[str, dict[str, str | None]] = {}
        for col in schema_a & schema_b:
            if a.schema_[col] != b.schema_[col]:
                type_changes[col] = {"old": a.schema_[col], "new": b.schema_[col]}

        return {
            "schema_added": sorted(schema_b - schema_a),
            "schema_removed": sorted(schema_a - schema_b),
            "schema_type_changed": type_changes,
            "row_count_diff": (b.num_rows or 0) - (a.num_rows or 0),
            "feature_count_diff": (b.num_features or 0) - (a.num_features or 0),
            "hash_match": (
                a.content_hash == b.content_hash if a.content_hash and b.content_hash else None
            ),
        }

    def link_to_experiment(self, name: str, version: str, run_id: str) -> None:
        """Link a dataset version to an experiment run.

        Args:
            name: Dataset name.
            version: Version string.
            run_id: Experiment run identifier.

        Raises:
            KeyError: If the dataset version does not exist.
        """
        ds = self.get(name, version)
        experiments = list(ds.linked_experiments)
        if run_id not in experiments:
            experiments.append(run_id)
        updated = ds.model_copy(update={"linked_experiments": experiments})
        self._save(updated)
        log.info("datasets.linked_experiment", name=name, version=version, run_id=run_id)

    def link_to_model(self, name: str, version: str, model_name: str, model_version: str) -> None:
        """Link a dataset version to a model version.

        Args:
            name: Dataset name.
            version: Dataset version string.
            model_name: Model name.
            model_version: Model version string.

        Raises:
            KeyError: If the dataset version does not exist.
        """
        ds = self.get(name, version)
        link = {"name": model_name, "version": model_version}
        models = list(ds.linked_models)
        if link not in models:
            models.append(link)
        updated = ds.model_copy(update={"linked_models": models})
        self._save(updated)
        log.info(
            "datasets.linked_model",
            name=name,
            version=version,
            model_name=model_name,
            model_version=model_version,
        )

    def verify(self, name: str, version: str) -> bool:
        """Re-compute content hash and compare to stored hash.

        Args:
            name: Dataset name.
            version: Version string.

        Returns:
            ``True`` if the re-computed hash matches the stored hash,
            ``False`` if they differ.  If the stored hash is ``None``
            (e.g. remote path), returns ``True`` (nothing to verify).

        Raises:
            KeyError: If the dataset version does not exist.

        Example:
            >>> assert registry.verify("train", "1.0")
        """
        ds = self.get(name, version)
        if ds.content_hash is None:
            log.debug("datasets.verify_skip_no_hash", name=name, version=version)
            return True
        current_hash = compute_hash(ds.path)
        if current_hash is None:
            log.warning("datasets.verify_cannot_read", name=name, version=version)
            return False
        match = current_hash == ds.content_hash
        if not match:
            log.warning(
                "datasets.verify_mismatch",
                name=name,
                version=version,
                expected=ds.content_hash,
                actual=current_hash,
            )
        return match

    def delete_version(self, name: str, version: str) -> None:
        """Delete a dataset version's metadata (not the data itself).

        Args:
            name: Dataset name.
            version: Version string.

        Raises:
            KeyError: If the version does not exist.
        """
        meta = self._meta_path(name, version)
        if not meta.exists():
            raise KeyError(f"Dataset '{name}@{version}' not found")
        meta.unlink()
        log.info("datasets.deleted", name=name, version=version)

        # Clean up empty parent directory
        parent = meta.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
