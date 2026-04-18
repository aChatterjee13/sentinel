"""Graph ML quality metrics — link prediction, KG completion, classification."""

from __future__ import annotations

from collections.abc import Sequence


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute AUC-ROC using rank-based formulation (no sklearn required)."""
    if not scores or len(scores) != len(labels):
        return float("nan")
    ranked = sorted(zip(scores, labels, strict=False), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        import structlog

        _log = structlog.get_logger(__name__)
        _log.warning("graph.auc_undefined", n_pos=n_pos, n_neg=n_neg)
        return float("nan")
    rank_sum = 0
    for i, (_, label) in enumerate(ranked, 1):
        if label == 1:
            rank_sum += i
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def hits_at_k(rankings: Sequence[int], k: int = 10) -> float:
    """Fraction of true entities that appear in the top-k predictions."""
    if not rankings:
        return 0.0
    return float(sum(1 for r in rankings if 0 < r <= k) / len(rankings))


def mrr(rankings: Sequence[int]) -> float:
    """Mean reciprocal rank — robust score for KG completion."""
    if not rankings:
        return 0.0
    return float(sum(1.0 / r for r in rankings if r > 0) / len(rankings))


def node_classification_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-F1 across node classes."""
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return 0.0
    f1_sum = 0.0
    for cls in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred, strict=False) if a == cls and b == cls)
        fp = sum(1 for a, b in zip(y_true, y_pred, strict=False) if a != cls and b == cls)
        fn = sum(1 for a, b in zip(y_true, y_pred, strict=False) if a == cls and b != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_sum += f1
    return float(f1_sum / len(classes))


def embedding_isotropy(embeddings) -> float:
    """Crude isotropy proxy: ratio of smallest to largest singular value."""
    import numpy as np

    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 1.0
    centred = arr - arr.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        return 0.0
    return float(singular[-1] / singular[0])
