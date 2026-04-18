"""NLP quality metrics — token F1, span match, classification metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class ClassificationQualityResult:
    """Aggregated classification quality metrics."""

    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class: dict[str, dict[str, float]]


def token_f1(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    """Per-instance token-level F1 between two token sequences."""
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred = Counter(pred_tokens)
    gold = Counter(gold_tokens)
    common = sum((pred & gold).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(gold.values())
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


def span_exact_match(
    pred_spans: Sequence[tuple[int, int, str]],
    gold_spans: Sequence[tuple[int, int, str]],
) -> float:
    """Fraction of gold spans recovered exactly (start, end, label)."""
    if not gold_spans:
        return 1.0 if not pred_spans else 0.0
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)
    return len(gold_set & pred_set) / len(gold_set)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> ClassificationQualityResult:
    """Compute accuracy + per-class precision/recall/F1 + macro/micro averages."""
    if not y_true or not y_pred:
        return ClassificationQualityResult(0.0, 0.0, 0.0, {})

    n = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred, strict=False) if a == b)
    accuracy = correct / n

    classes = sorted(set(y_true) | set(y_pred))
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for actual, predicted in zip(y_true, y_pred, strict=False):
        if actual == predicted:
            tp[actual] += 1
        else:
            fp[predicted] += 1
            fn[actual] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_sum = 0.0
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    for cls in classes:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": tp[cls] + fn[cls]}
        f1_sum += f1
        micro_tp += tp[cls]
        micro_fp += fp[cls]
        micro_fn += fn[cls]

    macro_f1 = f1_sum / len(classes) if classes else 0.0
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) else 0.0
    )

    return ClassificationQualityResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        per_class=per_class,
    )
