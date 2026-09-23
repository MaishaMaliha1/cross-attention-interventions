from __future__ import annotations

import numpy as np


def binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(inter / union) if union else 1.0


def best_threshold_iou(heatmap: np.ndarray, target: np.ndarray, thresholds=None) -> float:
    h = np.asarray(heatmap, dtype=np.float32)
    lo, hi = float(h.min()), float(h.max())
    if hi > lo:
        h = (h - lo) / (hi - lo)
    else:
        h = np.zeros_like(h)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    return max(binary_iou(h >= t, target) for t in thresholds)


def percentile_iou(heatmap: np.ndarray, target: np.ndarray, percentile: float = 80.0) -> float:
    h = np.asarray(heatmap, dtype=np.float32)
    threshold = np.percentile(h, percentile)
    return binary_iou(h >= threshold, target)


def mean(values) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")
