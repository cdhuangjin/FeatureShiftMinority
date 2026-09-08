"""Permutation-based feature importance: global, minority, majority, and MSI."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .metrics import minority_recall, majority_recall


def _permute_group(X: np.ndarray, cols: List[int], rng: np.random.Generator) -> np.ndarray:
    """Return a copy of X with the feature group jointly row-permuted."""
    Xp = X.copy()
    perm = rng.permutation(X.shape[0])
    for c in cols:
        Xp[:, c] = X[perm, c]
    return Xp


def _importance_for_metric(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_map: Dict[str, dict],
    base_score: float,
    metric_fn,
    repeats: int,
    rng: np.random.Generator,
) -> Tuple[float, ...]:
    scores = []
    for _ in range(repeats):
        Xp = _permute_group(X, feature_map["cols"], rng)
        yp = model.predict_proba(Xp)[:, 1]
        scores.append(base_score - metric_fn(y, yp))
    return tuple(float(s) for s in scores)


def compute_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    feature_map: Dict[str, dict],
    config: dict,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return (importance DataFrame, {global/minority/majority base scores})."""
    repeats = config["importance"]["permutation_repeats"]
    rng = np.random.default_rng(seed)

    base_proba = model.predict_proba(X_val)[:, 1]
    global_base = float(average_precision_score(y_val, base_proba))
    min_base = minority_recall(y_val, base_proba)
    maj_base = majority_recall(y_val, base_proba)

    rows = []
    for feat in feature_names:
        cols = feature_map[feat]["cols"]
        # Global (AUPRC)
        g_scores = _importance_for_metric(
            model, X_val, y_val, feature_map[feat], global_base,
            lambda y, p: average_precision_score(y, p), repeats, rng,
        )
        # Minority (minority recall)
        mi_scores = _importance_for_metric(
            model, X_val, y_val, feature_map[feat], min_base,
            minority_recall, repeats, rng,
        )
        # Majority (majority recall)
        ma_scores = _importance_for_metric(
            model, X_val, y_val, feature_map[feat], maj_base,
            majority_recall, repeats, rng,
        )
        g = float(np.mean(g_scores))
        mi = float(np.mean(mi_scores))
        ma = float(np.mean(ma_scores))
        rows.append(
            {
                "feature": feat,
                "cols": cols,
                "global_importance": round(g, 6),
                "minority_importance": round(mi, 6),
                "majority_importance": round(ma, 6),
                "MSI": round(mi - ma, 6),
            }
        )

    imp = pd.DataFrame(rows)
    # Lower rank = more important (rank 1 = highest). AUPRC/MSI higher is better.
    imp["global_rank"] = imp["global_importance"].rank(ascending=False, method="min").astype(int)
    imp["MSI_rank"] = imp["MSI"].rank(ascending=False, method="min").astype(int)

    bases = {"global": global_base, "minority": min_base, "majority": maj_base}
    return imp, bases

