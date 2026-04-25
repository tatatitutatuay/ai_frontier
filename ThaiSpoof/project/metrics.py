from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float
    eer_threshold: float


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def confusion_counts(y_true: list[int], scores: list[float], threshold: float = 0.5) -> tuple[int, int, int, int]:
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if not y_true:
        raise ValueError("at least one score is required")

    tn = fp = fn = tp = 0
    for target, score in zip(y_true, scores):
        pred = 1 if score >= threshold else 0
        if target == 0 and pred == 0:
            tn += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 1 and pred == 0:
            fn += 1
        elif target == 1 and pred == 1:
            tp += 1
        else:
            raise ValueError("labels must be 0 for genuine or 1 for spoof")
    return tn, fp, fn, tp


def compute_eer(y_true: list[int], scores: list[float]) -> tuple[float, float]:
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if not y_true:
        raise ValueError("at least one score is required")
    if not any(label == 0 for label in y_true) or not any(label == 1 for label in y_true):
        raise ValueError("EER requires both genuine and spoof labels")

    thresholds = sorted(set(float(score) for score in scores), reverse=True)
    best_eer = 1.0
    best_threshold = thresholds[0]
    best_gap = float("inf")

    for threshold in thresholds:
        tn, fp, fn, tp = confusion_counts(y_true, scores, threshold)
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        gap = abs(fpr - fnr)
        eer = (fpr + fnr) / 2.0
        if gap < best_gap or (gap == best_gap and eer < best_eer):
            best_gap = gap
            best_eer = eer
            best_threshold = threshold

    return best_eer, best_threshold


def calculate_binary_metrics(y_true: list[int], scores: list[float], threshold: float = 0.5) -> BinaryMetrics:
    tn, fp, fn, tp = confusion_counts(y_true, scores, threshold)

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    precision = _safe_div(tp, tp + fp)
    recall = sensitivity
    f1 = _safe_div(2 * precision * recall, precision + recall)
    eer, eer_threshold = compute_eer(y_true, scores)

    return BinaryMetrics(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        eer=eer,
        eer_threshold=eer_threshold,
    )
