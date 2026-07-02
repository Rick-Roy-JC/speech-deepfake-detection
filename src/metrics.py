"""Evaluation metrics for anti-spoofing."""
import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate.

    labels: 1 = bonafide, 0 = spoof.
    scores: higher = more likely bonafide (e.g. softmax prob of class 1).
    Returns (eer, threshold).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])