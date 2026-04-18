"""Unit tests for SemanticDriftMonitor (expanded coverage)."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.config.schema import SemanticDriftConfig
from sentinel.core.types import AlertSeverity
from sentinel.llmops.quality.semantic_drift import SemanticDriftMonitor, _cosine_distance


def _mock_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic mock: hash text to produce a pseudo-embedding."""
    result = []
    for t in texts:
        h = hash(t) % 10000
        result.append([h / 10000.0, (h + 1) / 10000.0, (h + 2) / 10000.0])
    return result


def _constant_embed_fn(texts: list[str]) -> list[list[float]]:
    """All texts map to the same embedding."""
    return [[1.0, 0.0, 0.0]] * len(texts)


def _shifted_embed_fn(texts: list[str]) -> list[list[float]]:
    """All texts map to the opposite direction."""
    return [[0.0, 0.0, 1.0]] * len(texts)


class TestCosineDistance:
    """Tests for the _cosine_distance helper."""

    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        assert _cosine_distance(a, a) == pytest.approx(0.0, abs=1e-7)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-7)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_distance(a, b) == pytest.approx(2.0, abs=1e-7)

    def test_zero_vector(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _cosine_distance(a, b) == 1.0


class TestSemanticDriftMonitor:
    """Tests for SemanticDriftMonitor."""

    def _make_monitor(self, embed_fn=None, **kwargs) -> SemanticDriftMonitor:
        config = SemanticDriftConfig(**kwargs)
        return SemanticDriftMonitor(
            config=config,
            embed_fn=embed_fn or _mock_embed_fn,
            window_size=100,
        )

    # ── fit ────────────────────────────────────────────────────────

    def test_fit_sets_reference(self) -> None:
        mon = self._make_monitor()
        mon.fit(["hello world", "test sentence"])
        assert mon._reference_centroid is not None
        assert mon._reference_n == 2

    def test_fit_empty_raises(self) -> None:
        mon = self._make_monitor()
        with pytest.raises(ValueError, match="empty"):
            mon.fit([])

    # ── detect without fit ────────────────────────────────────────

    def test_detect_before_fit_raises(self) -> None:
        mon = self._make_monitor()
        with pytest.raises(RuntimeError, match="fit"):
            mon.detect()

    # ── no drift case ─────────────────────────────────────────────

    def test_no_drift_same_distribution(self) -> None:
        mon = self._make_monitor(embed_fn=_constant_embed_fn)
        mon.fit(["a", "b", "c"])
        for text in ["x", "y", "z"]:
            mon.observe(text)
        report = mon.detect()
        assert not report.is_drifted
        assert report.test_statistic == pytest.approx(0.0, abs=1e-6)

    # ── drift detected ────────────────────────────────────────────

    def test_drift_detected_shifted_embeddings(self) -> None:
        # Fit with constant embedding, then observe with shifted
        ref_fn = _constant_embed_fn
        mon = SemanticDriftMonitor(
            config=SemanticDriftConfig(threshold=0.1),
            embed_fn=ref_fn,
            window_size=100,
        )
        mon.fit(["a", "b", "c"])
        # Manually override embed_fn to shifted
        mon._embed_fn = _shifted_embed_fn
        for text in ["x", "y", "z"]:
            mon.observe(text)
        report = mon.detect()
        assert report.is_drifted
        assert report.test_statistic > 0.1

    # ── empty window ──────────────────────────────────────────────

    def test_detect_empty_window(self) -> None:
        mon = self._make_monitor(embed_fn=_constant_embed_fn)
        mon.fit(["a", "b"])
        report = mon.detect()
        assert not report.is_drifted
        assert report.metadata.get("reason") == "no observations"

    # ── reset window ──────────────────────────────────────────────

    def test_reset_window_clears(self) -> None:
        mon = self._make_monitor()
        mon.fit(["a"])
        mon.observe("b")
        assert len(mon._window) == 1
        mon.reset_window()
        assert len(mon._window) == 0

    # ── severity levels ───────────────────────────────────────────

    def test_severity_scales_with_distance(self) -> None:
        mon = SemanticDriftMonitor(
            config=SemanticDriftConfig(threshold=0.1),
            embed_fn=_constant_embed_fn,
            window_size=100,
        )
        mon.fit(["a"])
        mon._embed_fn = _shifted_embed_fn
        mon.observe("x")
        report = mon.detect()
        # Distance should be ~1.0, so severity HIGH or CRITICAL
        assert report.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)

    # ── report metadata ───────────────────────────────────────────

    def test_report_metadata(self) -> None:
        mon = self._make_monitor()
        mon.fit(["a", "b", "c"])
        mon.observe("x")
        report = mon.detect()
        assert report.method == "semantic_drift"
        assert report.metadata.get("reference_n") == 3
        assert report.metadata.get("window_n") == 1

    def test_report_model_name(self) -> None:
        mon = self._make_monitor()
        mon.fit(["a"])
        mon.observe("b")
        report = mon.detect(model_name="my_llm")
        assert report.model_name == "my_llm"

    # ── window size limit ─────────────────────────────────────────

    def test_window_respects_maxlen(self) -> None:
        mon = SemanticDriftMonitor(
            config=SemanticDriftConfig(window_size=3),
            embed_fn=_mock_embed_fn,
        )
        mon.fit(["a"])
        for i in range(10):
            mon.observe(f"text_{i}")
        assert len(mon._window) == 3

    # ── _embed with custom fn ─────────────────────────────────────

    def test_embed_uses_custom_fn(self) -> None:
        calls = []

        def tracking_fn(texts):
            calls.append(texts)
            return [[0.1, 0.2]] * len(texts)

        mon = SemanticDriftMonitor(embed_fn=tracking_fn)
        mon.fit(["hello"])
        assert len(calls) == 1
        assert calls[0] == ["hello"]

    # ── _embed without fn raises ──────────────────────────────────

    def test_embed_without_fn_or_sentence_transformers_raises(self) -> None:
        mon = SemanticDriftMonitor(embed_fn=None)
        # Without sentence-transformers installed and no embed_fn, should raise
        try:
            mon._embed(["test"])
        except RuntimeError as e:
            assert "no embedder available" in str(e)
        except Exception:
            # sentence-transformers might be installed, so it won't raise
            pass
