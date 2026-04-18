"""Data quality checks — schema, freshness, nulls, duplicates, outliers."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from sentinel.config.schema import DataQualityConfig
from sentinel.core.types import AlertSeverity, QualityIssue, QualityReport

log = structlog.get_logger(__name__)

# Maximum unique values for a string column to be treated as categorical.
_ENUM_CARDINALITY_LIMIT = 20

# Timeout in seconds for regex pattern matching.
_PATTERN_TIMEOUT_SECS = 1.0


class DataQualityChecker:
    """Validates input data against the configured quality policies.

    Example:
        >>> checker = DataQualityChecker(config, model_name="claims_fraud")
        >>> report = checker.check({"age": 35, "amount": 1200})
        >>> if report.has_critical_issues:
        ...     raise ValueError("input rejected")
    """

    def __init__(self, config: DataQualityConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self._schema: dict[str, Any] | None = None
        self._reference_stats: dict[str, dict[str, Any]] | None = None
        self._last_seen: datetime | None = None
        if config.schema_.path:
            schema_path = Path(config.schema_.path)
            if schema_path.exists():
                self._schema = json.loads(schema_path.read_text())

    # ── Public API ────────────────────────────────────────────────

    def infer_schema(
        self,
        data: list[dict[str, Any]] | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Infer a JSON Schema from reference data.

        Args:
            data: Reference rows as a list of dicts or a 2-D numpy array.
            feature_names: Column names when *data* is a numpy array.  If
                omitted for a numpy array, synthetic names ``feature_0``,
                ``feature_1``, … are generated.

        Returns:
            A valid JSON Schema dict.  Also stored as ``self._schema`` and
            optionally saved to disk at the configured ``schema_.path``.

        Raises:
            ValueError: If *data* is a numpy array and *feature_names* is not
                provided or has the wrong length.
        """
        rows = self._to_rows(data, feature_names)
        if not rows:
            schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            self._schema = schema
            return schema

        all_keys: list[str] = list(dict.fromkeys(k for r in rows for k in r))
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for key in all_keys:
            values = [r.get(key) for r in rows]
            non_null = [v for v in values if v is not None]
            spec: dict[str, Any] = {}

            if non_null:
                spec["type"] = self._infer_type(non_null)
            else:
                spec["type"] = "string"

            # Required if zero nulls
            if all(v is not None for v in values):
                required.append(key)

            # Numeric ranges
            if spec["type"] in ("integer", "number"):
                numeric = [v for v in non_null if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if numeric:
                    spec["minimum"] = min(numeric)
                    spec["maximum"] = max(numeric)

            # Low-cardinality string enums
            if spec["type"] == "string":
                str_vals = [v for v in non_null if isinstance(v, str)]
                unique = list(dict.fromkeys(str_vals))
                if 0 < len(unique) <= _ENUM_CARDINALITY_LIMIT:
                    spec["enum"] = unique

            properties[key] = spec

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        self._schema = schema

        if self.config.schema_.path:
            out = Path(self.config.schema_.path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(schema, indent=2, default=str))
            log.info("schema_saved", path=str(out))

        return schema

    def fit(
        self,
        reference_data: list[dict[str, Any]] | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fit the quality checker on reference data.

        If no schema file exists (or ``self._schema`` is ``None``), a schema
        is inferred from *reference_data*.  If a schema file **does** exist
        the inferred schema is merged with the on-disk version so that manual
        overrides always win.

        Computed per-feature statistics (mean, std, min, max, null_rate) are
        stored internally for profile reporting.

        Args:
            reference_data: Reference rows.
            feature_names: Column names when *reference_data* is a numpy array.

        Returns:
            The final merged schema dict.
        """
        rows = self._to_rows(reference_data, feature_names)

        # Compute reference statistics
        self._reference_stats = self._compute_stats(rows)

        # Read manual schema from disk *before* infer_schema can overwrite it.
        existing_schema: dict[str, Any] | None = None
        if self.config.schema_.path:
            schema_path = Path(self.config.schema_.path)
            if schema_path.exists():
                try:
                    existing_schema = json.loads(schema_path.read_text())
                except (json.JSONDecodeError, OSError):
                    existing_schema = None

        inferred = self.infer_schema(rows)

        if existing_schema is not None:
            merged = self._merge_schemas(inferred, existing_schema)
        else:
            merged = inferred

        self._schema = merged
        log.info("data_quality_fit", model=self.model_name, features=len(merged.get("properties", {})))
        return merged

    def check(self, data: dict[str, Any] | list[dict[str, Any]]) -> QualityReport:
        """Run the configured quality checks against a single row or batch."""
        rows = [data] if isinstance(data, dict) else list(data)
        issues: list[QualityIssue] = []

        if self.config.schema_.enforce and self._schema is not None:
            issues.extend(self._validate_schema(rows))

        issues.extend(self._check_nulls(rows))
        issues.extend(self._check_duplicates(rows))
        issues.extend(self._check_outliers(rows))
        issues.extend(self._check_freshness())

        profile = self._compute_profile(rows)

        self._last_seen = datetime.now(timezone.utc)
        return QualityReport(
            model_name=self.model_name,
            is_valid=not any(
                i.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL) for i in issues
            ),
            issues=issues,
            rows_checked=len(rows),
            profile=profile,
        )

    def mark_fresh(self) -> None:
        """Manually update the freshness timestamp."""
        self._last_seen = datetime.now(timezone.utc)

    # ── Internal checks ───────────────────────────────────────────

    def _validate_schema(self, rows: list[dict[str, Any]]) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if self._schema is None:
            return issues
        required = self._schema.get("required", [])
        properties = self._schema.get("properties", {})
        for row in rows:
            for field in required:
                if field not in row or row[field] is None:
                    issues.append(
                        QualityIssue(
                            feature=field,
                            rule="schema.required",
                            severity=AlertSeverity.CRITICAL,
                            message=f"required field '{field}' missing",
                            count=1,
                        )
                    )
            for key, value in row.items():
                spec = properties.get(key)
                if spec is None:
                    continue
                expected_type = spec.get("type")
                if expected_type and not self._matches_type(value, expected_type):
                    issues.append(
                        QualityIssue(
                            feature=key,
                            rule="schema.type",
                            severity=AlertSeverity.HIGH,
                            message=f"{key}={value!r} expected type {expected_type}",
                            count=1,
                        )
                    )
                if (
                    "minimum" in spec
                    and isinstance(value, (int, float))
                    and value < spec["minimum"]
                ):
                    issues.append(
                        QualityIssue(
                            feature=key,
                            rule="schema.minimum",
                            severity=AlertSeverity.WARNING,
                            message=f"{key}={value} below min {spec['minimum']}",
                            count=1,
                        )
                    )
                if (
                    "maximum" in spec
                    and isinstance(value, (int, float))
                    and value > spec["maximum"]
                ):
                    issues.append(
                        QualityIssue(
                            feature=key,
                            rule="schema.maximum",
                            severity=AlertSeverity.WARNING,
                            message=f"{key}={value} above max {spec['maximum']}",
                            count=1,
                        )
                    )
                if "enum" in spec and value is not None and value not in spec["enum"]:
                    issues.append(
                        QualityIssue(
                            feature=key,
                            rule="schema.enum",
                            severity=AlertSeverity.WARNING,
                            message=f"{key}={value!r} not in allowed values {spec['enum']}",
                            count=1,
                        )
                    )
                if (
                    "pattern" in spec
                    and value is not None
                    and not self._matches_pattern(spec["pattern"], str(value))
                ):
                    issues.append(
                        QualityIssue(
                            feature=key,
                            rule="schema.pattern",
                            severity=AlertSeverity.WARNING,
                            message=f"{key}={value!r} does not match pattern {spec['pattern']!r}",
                            count=1,
                        )
                    )
        return issues

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        if value is None:
            return False
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_type = type_map.get(expected, object)
        return isinstance(value, expected_type)

    @staticmethod
    def _matches_pattern(pattern: str, value: str) -> bool:
        """Attempt a regex match with a timeout guard."""
        result: list[bool] = [False]

        def _do_match() -> None:
            result[0] = re.match(pattern, value) is not None

        t = threading.Thread(target=_do_match, daemon=True)
        t.start()
        t.join(timeout=_PATTERN_TIMEOUT_SECS)
        if t.is_alive():
            log.warning("pattern_match_timeout", pattern=pattern, value=value[:50])
            return False
        return result[0]

    # ── Data conversion helpers ────────────────────────────────────

    @staticmethod
    def _to_rows(
        data: list[dict[str, Any]] | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Normalise *data* into a list of dicts."""
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(f"Expected a 2-D array, got {data.ndim}-D")
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(data.shape[1])]
            if data.shape[1] != len(feature_names):
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) must match "
                    f"array columns ({data.shape[1]})"
                )
            return [
                {name: data[i, j].item() for j, name in enumerate(feature_names)}
                for i in range(data.shape[0])
            ]
        return list(data)

    @staticmethod
    def _infer_type(non_null_values: list[Any]) -> str:
        """Pick the best JSON Schema type for a collection of non-null values."""
        has_bool = any(isinstance(v, bool) for v in non_null_values)
        has_int = any(isinstance(v, int) and not isinstance(v, bool) for v in non_null_values)
        has_float = any(isinstance(v, float) for v in non_null_values)
        has_str = any(isinstance(v, str) for v in non_null_values)

        if has_str:
            return "string"
        if has_bool and not has_int and not has_float:
            return "boolean"
        if has_float:
            return "number"
        if has_int:
            return "integer"
        return "string"

    @staticmethod
    def _merge_schemas(
        inferred: dict[str, Any], manual: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge *inferred* with *manual* — manual overrides win per-field."""
        merged_props = dict(inferred.get("properties", {}))
        for key, spec in manual.get("properties", {}).items():
            merged_props[key] = spec  # manual override wins entirely

        merged_required = list(
            dict.fromkeys(
                inferred.get("required", []) + manual.get("required", [])
            )
        )

        return {
            "type": "object",
            "properties": merged_props,
            "required": merged_required,
        }

    @staticmethod
    def _compute_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Compute per-feature reference statistics."""
        if not rows:
            return {}
        all_keys = list(dict.fromkeys(k for r in rows for k in r))
        stats: dict[str, dict[str, Any]] = {}
        for key in all_keys:
            values = [r.get(key) for r in rows]
            non_null = [v for v in values if v is not None]
            null_rate = 1.0 - (len(non_null) / len(values)) if values else 0.0
            numeric = [
                float(v) for v in non_null
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            entry: dict[str, Any] = {
                "null_rate": null_rate,
                "count": len(values),
            }
            if numeric:
                arr = np.array(numeric, dtype=float)
                entry["mean"] = float(arr.mean())
                entry["std"] = float(arr.std())
                entry["min"] = float(arr.min())
                entry["max"] = float(arr.max())
            stats[key] = entry
        return stats

    @staticmethod
    def _compute_profile(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Build a per-feature profile dict for the QualityReport."""
        if not rows:
            return {}
        all_keys = list(dict.fromkeys(k for r in rows for k in r))
        profile: dict[str, dict[str, Any]] = {}
        for key in all_keys:
            values = [r.get(key) for r in rows]
            non_null = [v for v in values if v is not None]
            null_rate = 1.0 - (len(non_null) / len(values)) if values else 0.0
            unique_count = len({repr(v) for v in non_null})

            # Determine predominant type
            inferred_type = "string"
            if non_null:
                has_bool = any(isinstance(v, bool) for v in non_null)
                has_int = any(isinstance(v, int) and not isinstance(v, bool) for v in non_null)
                has_float = any(isinstance(v, float) for v in non_null)
                has_str = any(isinstance(v, str) for v in non_null)
                if has_str:
                    inferred_type = "string"
                elif has_bool and not has_int and not has_float:
                    inferred_type = "boolean"
                elif has_float:
                    inferred_type = "number"
                elif has_int:
                    inferred_type = "integer"

            numeric = [
                float(v) for v in non_null
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            entry: dict[str, Any] = {
                "type": inferred_type,
                "null_rate": null_rate,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "unique_count": unique_count,
            }
            if numeric:
                arr = np.array(numeric, dtype=float)
                entry["mean"] = float(arr.mean())
                entry["std"] = float(arr.std())
                entry["min"] = float(arr.min())
                entry["max"] = float(arr.max())
            profile[key] = entry
        return profile

    def _check_nulls(self, rows: list[dict[str, Any]]) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if not rows:
            return issues
        keys = {k for r in rows for k in r}
        for k in keys:
            null_count = sum(1 for r in rows if r.get(k) is None)
            null_rate = null_count / len(rows)
            if null_rate > self.config.null_threshold:
                issues.append(
                    QualityIssue(
                        feature=k,
                        rule="nulls",
                        severity=AlertSeverity.WARNING,
                        message=f"null rate {null_rate:.1%} exceeds threshold {self.config.null_threshold:.1%}",
                        count=null_count,
                    )
                )
        return issues

    def _check_duplicates(self, rows: list[dict[str, Any]]) -> list[QualityIssue]:
        if len(rows) < 2:
            return []
        seen: set[str] = set()
        dupes = 0
        for r in rows:
            try:
                key = json.dumps(r, sort_keys=True, default=str)
            except TypeError:
                continue
            if key in seen:
                dupes += 1
            else:
                seen.add(key)
        rate = dupes / len(rows)
        if rate > self.config.duplicate_threshold:
            return [
                QualityIssue(
                    feature=None,
                    rule="duplicates",
                    severity=AlertSeverity.WARNING,
                    message=f"{rate:.1%} duplicate rows (threshold {self.config.duplicate_threshold:.1%})",
                    count=dupes,
                )
            ]
        return []

    def _check_outliers(self, rows: list[dict[str, Any]]) -> list[QualityIssue]:
        if not rows:
            return []
        if self.config.outlier_detection.method == "isolation_forest":
            # IF requires sklearn — only run if scikit-learn available
            return self._isolation_forest_outliers(rows)

        issues: list[QualityIssue] = []
        keys = {k for r in rows for k in r if isinstance(r.get(k), (int, float))}
        for k in keys:
            values = np.array(
                [r.get(k) for r in rows if isinstance(r.get(k), (int, float))], dtype=float
            )
            if values.size < 5:
                continue
            n_outliers = 0
            if self.config.outlier_detection.method == "zscore":
                z = np.abs((values - values.mean()) / (values.std() + 1e-9))
                n_outliers = int(np.sum(z > 3))
            elif self.config.outlier_detection.method == "iqr":
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_outliers = int(np.sum((values < lo) | (values > hi)))
            if n_outliers > 0:
                rate = n_outliers / len(values)
                if rate > self.config.outlier_detection.contamination:
                    issues.append(
                        QualityIssue(
                            feature=k,
                            rule="outliers",
                            severity=AlertSeverity.WARNING,
                            message=f"{n_outliers} outliers ({rate:.1%})",
                            count=n_outliers,
                        )
                    )
        return issues

    def _isolation_forest_outliers(self, rows: list[dict[str, Any]]) -> list[QualityIssue]:
        try:
            from sklearn.ensemble import IsolationForest  # type: ignore[import-not-found]
        except ImportError:
            return []  # silently skip if sklearn not installed
        if len(rows) < 10:
            return []
        keys = sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float))})
        if not keys:
            return []
        X = np.array([[r.get(k, 0.0) or 0.0 for k in keys] for r in rows])
        clf = IsolationForest(
            contamination=self.config.outlier_detection.contamination, random_state=42
        )
        labels = clf.fit_predict(X)
        n_outliers = int(np.sum(labels == -1))
        if n_outliers == 0:
            return []
        return [
            QualityIssue(
                feature=None,
                rule="outliers.isolation_forest",
                severity=AlertSeverity.WARNING,
                message=f"isolation forest flagged {n_outliers} of {len(rows)} rows",
                count=n_outliers,
            )
        ]

    def _check_freshness(self) -> list[QualityIssue]:
        if self._last_seen is None:
            return []
        age_hours = (datetime.now(timezone.utc) - self._last_seen).total_seconds() / 3600
        if age_hours > self.config.freshness.max_age_hours:
            return [
                QualityIssue(
                    feature=None,
                    rule="freshness",
                    severity=AlertSeverity.HIGH,
                    message=f"data age {age_hours:.1f}h exceeds {self.config.freshness.max_age_hours}h",
                )
            ]
        return []
