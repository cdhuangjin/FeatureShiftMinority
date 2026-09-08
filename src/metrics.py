"""Performance metrics for a binary imbalanced classifier (minority == class 1)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


EPS = 1.0e-12


def _safe_ratio(num: float, den: float) -> float:
    return float(num / (den + EPS))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Return the full metric dictionary for one prediction."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    pos = y_true == 1
    neg = y_true == 0
    min_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    maj_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    min_precision = _safe_ratio(tp, tp + fp)
    min_f1 = _safe_ratio(2 * min_precision * min_recall, min_precision + min_recall)
    maj_f1 = _safe_ratio(
        2 * maj_recall * _safe_ratio(neg.sum(), neg.sum() + tp),
        maj_recall + _safe_ratio(neg.sum(), neg.sum() + tp),
    )

    if pos.sum() == 0:
        min_recall = 0.0
    if neg.sum() == 0:
        maj_recall = 0.0

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    balanced_acc = _safe_ratio(min_recall + maj_recall, 2)
    gmean = float(np.sqrt(max(min_recall * maj_recall, 0.0)))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    return {
        "AUROC": round(auroc, 6),
        "AUPRC": round(auprc, 6),
        "minority_recall": round(min_recall, 6),
        "minority_f1": round(min_f1, 6),
        "minority_precision": round(min_precision, 6),
        "majority_recall": round(maj_recall, 6),
        "majority_f1": round(maj_f1, 6),
        "balanced_accuracy": round(balanced_acc, 6),
        "MCC": round(mcc, 6),
        "Gmean": round(gmean, 6),
    }


def minority_recall(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def majority_recall(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))

