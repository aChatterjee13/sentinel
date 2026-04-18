"""Tests for sentinel.foundation.registry.serializers — WS-A."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentinel.foundation.registry.serializers import (
    BaseModelSerializer,
    JoblibSerializer,
    ONNXSerializer,
    PickleSerializer,
    resolve_serializer,
)


class TestPickleSerializer:
    def test_roundtrip(self, tmp_path: Path) -> None:
        s = PickleSerializer()
        obj = {"key": "value", "nums": [1, 2, 3]}
        path = tmp_path / "model.pkl"
        s.serialize(obj, path)
        assert path.exists()
        loaded = s.deserialize(path)
        assert loaded == obj

    def test_extension(self) -> None:
        assert PickleSerializer().extension == ".pkl"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        s = PickleSerializer()
        deep = tmp_path / "a" / "b" / "model.pkl"
        s.serialize({"data": 1}, deep)
        assert deep.exists()


class TestJoblibSerializer:
    def test_roundtrip(self, tmp_path: Path) -> None:
        s = JoblibSerializer()
        obj = {"hello": "world"}
        path = tmp_path / f"model{s.extension}"
        s.serialize(obj, path)
        loaded = s.deserialize(path)
        assert loaded == obj

    def test_extension_is_string(self) -> None:
        s = JoblibSerializer()
        assert s.extension in (".joblib", ".pkl")


class TestONNXSerializer:
    def test_serialize_bytes(self, tmp_path: Path) -> None:
        s = ONNXSerializer()
        data = b"fake-onnx-model-bytes"
        path = tmp_path / "model.onnx"
        s.serialize(data, path)
        assert path.read_bytes() == data

    def test_serialize_proto(self, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.SerializeToString.return_value = b"proto-bytes"
        s = ONNXSerializer()
        path = tmp_path / "model.onnx"
        s.serialize(mock_model, path)
        assert path.read_bytes() == b"proto-bytes"

    def test_serialize_invalid_type(self, tmp_path: Path) -> None:
        s = ONNXSerializer()
        with pytest.raises(TypeError, match="ONNXSerializer expects"):
            s.serialize(42, tmp_path / "model.onnx")

    def test_extension(self) -> None:
        assert ONNXSerializer().extension == ".onnx"


class TestResolveSerializer:
    def test_explicit_pickle(self) -> None:
        s = resolve_serializer(name="pickle")
        assert isinstance(s, PickleSerializer)

    def test_explicit_joblib(self) -> None:
        s = resolve_serializer(name="joblib")
        assert isinstance(s, JoblibSerializer)

    def test_explicit_onnx(self) -> None:
        s = resolve_serializer(name="onnx")
        assert isinstance(s, ONNXSerializer)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown serializer"):
            resolve_serializer(name="unknown")

    def test_auto_default_is_joblib(self) -> None:
        s = resolve_serializer(name="auto")
        assert isinstance(s, JoblibSerializer)

    def test_auto_none_is_joblib(self) -> None:
        s = resolve_serializer()
        assert isinstance(s, JoblibSerializer)

    def test_auto_detects_sklearn(self) -> None:
        model = MagicMock()
        model.__class__.__module__ = "sklearn.ensemble._forest"
        s = resolve_serializer(model=model)
        assert isinstance(s, JoblibSerializer)

    def test_auto_detects_bytes_as_onnx(self) -> None:
        s = resolve_serializer(model=b"onnx-bytes")
        assert isinstance(s, ONNXSerializer)

    def test_auto_detects_proto_as_onnx(self) -> None:
        model = MagicMock(spec=["SerializeToString"])
        s = resolve_serializer(model=model)
        assert isinstance(s, ONNXSerializer)

    def test_is_base_class(self) -> None:
        s = resolve_serializer()
        assert isinstance(s, BaseModelSerializer)
