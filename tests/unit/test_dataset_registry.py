"""Tests for the dataset metadata registry, hashing, and lineage modules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel.foundation.datasets.hashing import compute_hash
from sentinel.foundation.datasets.lineage import get_lineage_graph, resolve_dataset_ref
from sentinel.foundation.datasets.registry import DatasetRegistry, DatasetVersion

# ── DatasetVersion model ──────────────────────────────────────────


class TestDatasetVersion:
    """Tests for the DatasetVersion Pydantic model."""

    def test_minimal_creation(self) -> None:
        ds = DatasetVersion(name="train", version="1.0", path="/data/train.csv")
        assert ds.name == "train"
        assert ds.version == "1.0"
        assert ds.path == "/data/train.csv"
        assert ds.format == "unknown"
        assert ds.content_hash is None
        assert ds.schema_ == {}
        assert ds.tags == []
        assert ds.linked_experiments == []
        assert ds.linked_models == []

    def test_frozen(self) -> None:
        ds = DatasetVersion(name="train", version="1.0", path="/data/train.csv")
        with pytest.raises(ValidationError):
            ds.name = "other"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DatasetVersion(
                name="train",
                version="1.0",
                path="/data/train.csv",
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_full_creation(self) -> None:
        ds = DatasetVersion(
            name="features",
            version="2.0",
            path="s3://bucket/features.parquet",
            format="parquet",
            split="train",
            num_rows=50000,
            num_features=42,
            content_hash="abc123",
            schema={"age": "int64", "income": "float64"},
            description="Feature set v2",
            tags=["production", "fraud"],
            source="pipeline:feature_eng",
            metadata={"author": "ml-team"},
            linked_experiments=["run-001"],
            linked_models=[{"name": "fraud", "version": "1.0"}],
            derived_from="raw@1.0",
        )
        assert ds.num_rows == 50000
        assert ds.schema_["age"] == "int64"
        assert ds.tags == ["production", "fraud"]
        assert ds.derived_from == "raw@1.0"

    def test_model_dump_roundtrip(self) -> None:
        ds = DatasetVersion(
            name="test",
            version="1.0",
            path="/data/test.csv",
            tags=["a"],
        )
        payload = ds.model_dump(mode="json")
        restored = DatasetVersion.model_validate(payload)
        assert restored == ds

    def test_created_at_auto_populated(self) -> None:
        ds = DatasetVersion(name="t", version="1.0", path="/x")
        assert ds.created_at is not None
        assert len(ds.created_at) > 10  # ISO format timestamp


# ── DatasetRegistry ───────────────────────────────────────────────


class TestDatasetRegistryRegister:
    """Tests for DatasetRegistry.register()."""

    def test_register_creates_metadata_file(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/data/train.csv")
        meta = tmp_path / "ds" / "train" / "1.0.json"
        assert meta.exists()
        data = json.loads(meta.read_text())
        assert data["name"] == "train"
        assert data["version"] == "1.0"

    def test_register_returns_dataset_version(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        ds = reg.register(
            name="train",
            version="1.0",
            path="/data/train.csv",
            format="csv",
            num_rows=1000,
        )
        assert isinstance(ds, DatasetVersion)
        assert ds.name == "train"
        assert ds.format == "csv"
        assert ds.num_rows == 1000

    def test_register_duplicate_raises(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/data/train.csv")
        with pytest.raises(ValueError, match="already exists"):
            reg.register(name="train", version="1.0", path="/data/train.csv")

    def test_register_auto_hash_local_file(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b,c\n1,2,3\n")
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=True)
        ds = reg.register(name="my_data", version="1.0", path=str(data_file))
        assert ds.content_hash is not None
        assert len(ds.content_hash) == 64  # SHA-256 hex digest

    def test_register_auto_hash_remote_path(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=True)
        ds = reg.register(name="remote", version="1.0", path="s3://bucket/data.parquet")
        assert ds.content_hash is None

    def test_register_auto_hash_disabled(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b,c\n")
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=False)
        ds = reg.register(name="my_data", version="1.0", path=str(data_file))
        assert ds.content_hash is None

    def test_register_require_schema_enforced(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds", require_schema=True)
        with pytest.raises(ValueError, match="Schema is required"):
            reg.register(name="train", version="1.0", path="/data/train.csv")

    def test_register_require_schema_with_schema(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds", require_schema=True)
        ds = reg.register(
            name="train",
            version="1.0",
            path="/data/train.csv",
            schema={"col1": "int64"},
        )
        assert ds.schema_ == {"col1": "int64"}

    def test_register_with_all_optional_fields(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        ds = reg.register(
            name="full",
            version="1.0",
            path="/data/full.parquet",
            format="parquet",
            split="train",
            num_rows=10000,
            num_features=50,
            schema={"a": "int64", "b": "float64"},
            description="Full feature set",
            tags=["production", "v1"],
            source="pipeline:etl",
            derived_from="raw@0.1",
            metadata={"team": "ml"},
        )
        assert ds.split == "train"
        assert ds.description == "Full feature set"
        assert ds.derived_from == "raw@0.1"
        assert ds.metadata == {"team": "ml"}


class TestDatasetRegistryGet:
    """Tests for DatasetRegistry.get()."""

    def test_get_returns_correct_version(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        ds = reg.get("train", "1.0")
        assert ds.name == "train"
        assert ds.version == "1.0"

    def test_get_raises_key_error_for_missing(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent", "1.0")

    def test_get_raises_key_error_wrong_version(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        with pytest.raises(KeyError, match="not found"):
            reg.get("train", "2.0")


class TestDatasetRegistryList:
    """Tests for list_versions(), list_all(), list_names()."""

    def test_list_versions_returns_sorted(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="data", version="2.0", path="/a")
        reg.register(name="data", version="1.0", path="/b")
        reg.register(name="data", version="1.5", path="/c")
        versions = reg.list_versions("data")
        assert [v.version for v in versions] == ["1.0", "1.5", "2.0"]

    def test_list_versions_empty(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        assert reg.list_versions("nonexistent") == []

    def test_list_all(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.register(name="test", version="1.0", path="/b")
        all_ds = reg.list_all()
        assert len(all_ds) == 2
        names = {d.name for d in all_ds}
        assert names == {"train", "test"}

    def test_list_all_empty(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        assert reg.list_all() == []

    def test_list_names(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="beta", version="1.0", path="/a")
        reg.register(name="alpha", version="1.0", path="/b")
        assert reg.list_names() == ["alpha", "beta"]


class TestDatasetRegistrySearch:
    """Tests for DatasetRegistry.search()."""

    @pytest.fixture()
    def populated_registry(self, tmp_path: Path) -> DatasetRegistry:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(
            name="train",
            version="1.0",
            path="/a",
            format="parquet",
            split="train",
            tags=["production", "v1"],
        )
        reg.register(
            name="test",
            version="1.0",
            path="/b",
            format="csv",
            split="test",
            tags=["production"],
        )
        reg.register(
            name="holdout",
            version="1.0",
            path="/c",
            format="parquet",
            split="holdout",
            tags=["evaluation"],
        )
        return reg

    def test_search_by_tags(self, populated_registry: DatasetRegistry) -> None:
        results = populated_registry.search(tags=["production"])
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"train", "test"}

    def test_search_by_multiple_tags(self, populated_registry: DatasetRegistry) -> None:
        results = populated_registry.search(tags=["production", "v1"])
        assert len(results) == 1
        assert results[0].name == "train"

    def test_search_by_split(self, populated_registry: DatasetRegistry) -> None:
        results = populated_registry.search(split="test")
        assert len(results) == 1
        assert results[0].name == "test"

    def test_search_by_format(self, populated_registry: DatasetRegistry) -> None:
        results = populated_registry.search(format="parquet")
        assert len(results) == 2

    def test_search_by_name_pattern(self, populated_registry: DatasetRegistry) -> None:
        results = populated_registry.search(name_pattern="^t")
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"train", "test"}

    def test_search_combined_filters(self, populated_registry: DatasetRegistry) -> None:
        results = populated_registry.search(tags=["production"], format="parquet")
        assert len(results) == 1
        assert results[0].name == "train"

    def test_search_no_results(self, populated_registry: DatasetRegistry) -> None:
        results = populated_registry.search(tags=["nonexistent"])
        assert results == []


class TestDatasetRegistryCompare:
    """Tests for DatasetRegistry.compare()."""

    def test_compare_schema_diff(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(
            name="data",
            version="1.0",
            path="/a",
            schema={"a": "int64", "b": "float64", "c": "str"},
            num_rows=100,
            num_features=3,
        )
        reg.register(
            name="data",
            version="2.0",
            path="/b",
            schema={"a": "int64", "b": "int32", "d": "str"},
            num_rows=200,
            num_features=3,
        )
        diff = reg.compare("data", "1.0", "2.0")
        assert diff["schema_added"] == ["d"]
        assert diff["schema_removed"] == ["c"]
        assert "b" in diff["schema_type_changed"]
        assert diff["schema_type_changed"]["b"]["old"] == "float64"
        assert diff["schema_type_changed"]["b"]["new"] == "int32"
        assert diff["row_count_diff"] == 100
        assert diff["feature_count_diff"] == 0

    def test_compare_hash_match(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=True)
        reg.register(name="data", version="1.0", path=str(data_file))
        reg.register(name="data", version="2.0", path=str(data_file))
        diff = reg.compare("data", "1.0", "2.0")
        assert diff["hash_match"] is True

    def test_compare_no_hash(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=False)
        reg.register(name="data", version="1.0", path="/a")
        reg.register(name="data", version="2.0", path="/b")
        diff = reg.compare("data", "1.0", "2.0")
        assert diff["hash_match"] is None

    def test_compare_raises_on_missing(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="data", version="1.0", path="/a")
        with pytest.raises(KeyError):
            reg.compare("data", "1.0", "2.0")


class TestDatasetRegistryLinks:
    """Tests for link_to_experiment() and link_to_model()."""

    def test_link_to_experiment(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.link_to_experiment("train", "1.0", "run-001")
        ds = reg.get("train", "1.0")
        assert "run-001" in ds.linked_experiments

    def test_link_to_experiment_idempotent(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.link_to_experiment("train", "1.0", "run-001")
        reg.link_to_experiment("train", "1.0", "run-001")
        ds = reg.get("train", "1.0")
        assert ds.linked_experiments.count("run-001") == 1

    def test_link_to_model(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.link_to_model("train", "1.0", "fraud_model", "2.0")
        ds = reg.get("train", "1.0")
        assert {"name": "fraud_model", "version": "2.0"} in ds.linked_models

    def test_link_to_model_idempotent(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.link_to_model("train", "1.0", "fraud", "1.0")
        reg.link_to_model("train", "1.0", "fraud", "1.0")
        ds = reg.get("train", "1.0")
        assert len(ds.linked_models) == 1

    def test_link_to_experiment_missing_dataset(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        with pytest.raises(KeyError):
            reg.link_to_experiment("ghost", "1.0", "run-001")

    def test_link_to_model_missing_dataset(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        with pytest.raises(KeyError):
            reg.link_to_model("ghost", "1.0", "model", "1.0")

    def test_multiple_links(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.link_to_experiment("train", "1.0", "run-001")
        reg.link_to_experiment("train", "1.0", "run-002")
        reg.link_to_model("train", "1.0", "model_a", "1.0")
        reg.link_to_model("train", "1.0", "model_b", "2.0")
        ds = reg.get("train", "1.0")
        assert len(ds.linked_experiments) == 2
        assert len(ds.linked_models) == 2


class TestDatasetRegistryVerify:
    """Tests for DatasetRegistry.verify()."""

    def test_verify_matching_hash(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b,c\n1,2,3\n")
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=True)
        reg.register(name="train", version="1.0", path=str(data_file))
        assert reg.verify("train", "1.0") is True

    def test_verify_mismatched_hash(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b,c\n1,2,3\n")
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=True)
        reg.register(name="train", version="1.0", path=str(data_file))
        # Modify the file after registration
        data_file.write_text("a,b,c\n4,5,6\n")
        assert reg.verify("train", "1.0") is False

    def test_verify_no_stored_hash(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=False)
        reg.register(name="train", version="1.0", path="/remote/data.csv")
        assert reg.verify("train", "1.0") is True

    def test_verify_missing_file(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.write_text("content")
        reg = DatasetRegistry(tmp_path / "ds", auto_hash=True)
        reg.register(name="train", version="1.0", path=str(data_file))
        # Delete the data file
        data_file.unlink()
        assert reg.verify("train", "1.0") is False

    def test_verify_missing_dataset(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        with pytest.raises(KeyError):
            reg.verify("nonexistent", "1.0")


class TestDatasetRegistryDelete:
    """Tests for DatasetRegistry.delete_version()."""

    def test_delete_removes_metadata(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.delete_version("train", "1.0")
        with pytest.raises(KeyError):
            reg.get("train", "1.0")

    def test_delete_missing_raises(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        with pytest.raises(KeyError, match="not found"):
            reg.delete_version("ghost", "1.0")

    def test_delete_cleans_empty_directory(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.delete_version("train", "1.0")
        assert not (tmp_path / "ds" / "train").exists()

    def test_delete_preserves_other_versions(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.register(name="train", version="2.0", path="/b")
        reg.delete_version("train", "1.0")
        assert reg.get("train", "2.0").version == "2.0"
        assert (tmp_path / "ds" / "train").exists()


# ── Hashing ───────────────────────────────────────────────────────


class TestComputeHash:
    """Tests for the compute_hash utility."""

    def test_local_file_produces_sha256(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("hello world\n")
        h = compute_hash(str(f))
        assert h is not None
        assert len(h) == 64

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("identical")
        f2.write_text("identical")
        assert compute_hash(str(f1)) == compute_hash(str(f2))

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("content a")
        f2.write_text("content b")
        assert compute_hash(str(f1)) != compute_hash(str(f2))

    def test_remote_s3_returns_none(self) -> None:
        assert compute_hash("s3://bucket/data.parquet") is None

    def test_remote_az_returns_none(self) -> None:
        assert compute_hash("az://container/data.csv") is None

    def test_remote_gs_returns_none(self) -> None:
        assert compute_hash("gs://bucket/data.csv") is None

    def test_remote_http_returns_none(self) -> None:
        assert compute_hash("https://example.com/data.csv") is None

    def test_file_not_found_returns_none(self) -> None:
        assert compute_hash("/nonexistent/path/data.csv") is None


# ── Lineage ───────────────────────────────────────────────────────


class TestResolveDatasetRef:
    """Tests for resolve_dataset_ref()."""

    def test_basic_parsing(self) -> None:
        name, version = resolve_dataset_ref("train@1.0")
        assert name == "train"
        assert version == "1.0"

    def test_name_with_special_chars(self) -> None:
        name, version = resolve_dataset_ref("my-dataset_v2@3.1.0")
        assert name == "my-dataset_v2"
        assert version == "3.1.0"

    def test_multiple_at_signs(self) -> None:
        name, version = resolve_dataset_ref("data@set@2.0")
        assert name == "data@set"
        assert version == "2.0"

    def test_missing_at_raises(self) -> None:
        with pytest.raises(ValueError, match="name@version"):
            resolve_dataset_ref("train_1.0")

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name@version"):
            resolve_dataset_ref("@1.0")

    def test_empty_version_raises(self) -> None:
        with pytest.raises(ValueError, match="name@version"):
            resolve_dataset_ref("train@")


class TestGetLineageGraph:
    """Tests for get_lineage_graph()."""

    def test_simple_lineage(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="raw", version="1.0", path="/a")
        reg.register(name="features", version="1.0", path="/b", derived_from="raw@1.0")
        graph = get_lineage_graph(reg, "features", "1.0")
        assert graph["dataset"] == "features@1.0"
        assert graph["derived_from"] == "raw@1.0"
        assert graph["ancestry"] == ["raw@1.0"]

    def test_chain_lineage(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="raw", version="1.0", path="/a")
        reg.register(name="cleaned", version="1.0", path="/b", derived_from="raw@1.0")
        reg.register(name="features", version="1.0", path="/c", derived_from="cleaned@1.0")
        graph = get_lineage_graph(reg, "features", "1.0")
        assert graph["ancestry"] == ["cleaned@1.0", "raw@1.0"]

    def test_lineage_with_missing_parent(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="features", version="1.0", path="/a", derived_from="missing@1.0")
        graph = get_lineage_graph(reg, "features", "1.0")
        assert graph["ancestry"] == ["missing@1.0"]

    def test_lineage_no_parent(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="raw", version="1.0", path="/a")
        graph = get_lineage_graph(reg, "raw", "1.0")
        assert graph["derived_from"] is None
        assert graph["ancestry"] == []

    def test_lineage_includes_links(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="train", version="1.0", path="/a")
        reg.link_to_experiment("train", "1.0", "run-001")
        reg.link_to_model("train", "1.0", "fraud", "2.0")
        graph = get_lineage_graph(reg, "train", "1.0")
        assert "run-001" in graph["linked_experiments"]
        assert {"name": "fraud", "version": "2.0"} in graph["linked_models"]

    def test_lineage_missing_dataset_raises(self, tmp_path: Path) -> None:
        reg = DatasetRegistry(tmp_path / "ds")
        with pytest.raises(KeyError):
            get_lineage_graph(reg, "nonexistent", "1.0")

    def test_lineage_circular_protection(self, tmp_path: Path) -> None:
        """Ensure lineage traversal doesn't loop forever on circular refs."""
        reg = DatasetRegistry(tmp_path / "ds")
        reg.register(name="a", version="1.0", path="/a", derived_from="b@1.0")
        reg.register(name="b", version="1.0", path="/b", derived_from="a@1.0")
        graph = get_lineage_graph(reg, "a", "1.0")
        # Should terminate and include both
        assert "b@1.0" in graph["ancestry"]
        assert "a@1.0" in graph["ancestry"]
        assert len(graph["ancestry"]) == 2  # no infinite loop
