"""MAFR-v0: class-specific robust feature scores and subset selection.

This module contains the *definition* of the MAFR-v0 scoring used by Phase C3.
It is deliberately lightweight and CPU-friendly: no neural nets, no generative
reconstruction, no target-label tuning and no test-based selection.  All scoring
inputs are computed on source train / validation only (see mafr_selection.py for
that orchestration).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


SCORE_COMPONENTS = ["I_minority", "class_specificity", "redundancy", "availability_robustness"]


def minmax_normalize(values: Sequence[float]) -> np.ndarray:
    """Min-max normalise a vector to [0, 1]. Constant vectors map to 0.5."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = hi - lo
    out = np.full(arr.shape, 0.5, dtype=float)
    if span > 1.0e-12:
        out = (arr - lo) / span
    # Propagate any NaN values.
    out = np.where(np.isnan(arr), np.nan, out)
    return out


def class_specificity(minority_importance: Sequence[float], majority_importance: Sequence[float]) -> np.ndarray:
    """Minority importance minus majority importance."""
    return np.asarray(minority_importance, dtype=float) - np.asarray(majority_importance, dtype=float)


def mafr_score(
    norm_minority: Sequence[float],
    norm_specificity: Sequence[float],
    norm_redundancy: Sequence[float],
    norm_robustness: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    """Weighted MAFR score: w1*I_min + w2*specificity + w3*redundancy + w4*robustness."""
    w1, w2, w3, w4 = weights
    a = np.asarray(norm_minority, dtype=float)
    b = np.asarray(norm_specificity, dtype=float)
    c = np.asarray(norm_redundancy, dtype=float)
    d = np.asarray(norm_robustness, dtype=float)
    score = w1 * a + w2 * b + w3 * c + w4 * d
    return np.where(np.isnan(score), -np.inf, score)


def _tiebreak_order(feature_names: Sequence[str]) -> Dict[str, int]:
    """Stable feature ordering used for deterministic tie-breaking."""
    return {name: idx for idx, name in enumerate(feature_names)}


def compute_feature_scores(
    feature_names: Sequence[str],
    global_importance: Sequence[float],
    minority_importance: Sequence[float],
    majority_importance: Sequence[float],
    redundancy: Sequence[float],
    robustness: Sequence[float],
    weights_variants: Dict[str, Sequence[float]],
) -> pd.DataFrame:
    """Return a per-feature score DataFrame (normalised components + per-variant scores)."""
    names = list(feature_names)
    n = len(names)
    spec = class_specificity(minority_importance, majority_importance)

    norm_min = minmax_normalize(minority_importance)
    norm_spec = minmax_normalize(spec)
    norm_red = minmax_normalize(redundancy)
    norm_rob = minmax_normalize(robustness)

    df = pd.DataFrame(
        {
            "feature": names,
            "global_importance": np.asarray(global_importance, dtype=float),
            "minority_importance": np.asarray(minority_importance, dtype=float),
            "majority_importance": np.asarray(majority_importance, dtype=float),
            "class_specificity": spec,
            "redundancy": np.asarray(redundancy, dtype=float),
            "availability_robustness": np.asarray(robustness, dtype=float),
            "norm_minority": norm_min,
            "norm_class_specificity": norm_spec,
            "norm_redundancy": norm_red,
            "norm_availability_robustness": norm_rob,
        }
    )

    for variant, weights in weights_variants.items():
        score = mafr_score(norm_min, norm_spec, norm_red, norm_rob, weights)
        df[f"{variant}_score"] = score
        # lower rank = better (rank 1 = highest score).
        df[f"{variant}_rank"] = df[f"{variant}_score"].rank(ascending=False, method="min").astype(int)

    return df


def select_subset(
    scores: pd.DataFrame,
    variant: str,
    retention: float,
    n_features: int,
    min_features: int = 3,
) -> Tuple[List[str], List[str]]:
    """Return (selected_features, fallback_priority) for one score variant/retention.

    ``selected_features`` = the robust subset kept at the given retention ratio.
    ``fallback_priority`` = the full feature ranking by score (highest first), used as
    the availability fallback order (last entries are dropped first).
    """
    score_col = f"{variant}_score"
    if score_col not in scores.columns:
        raise ValueError(f"Variant {variant!r} not present in score frame.")
    order_idx = _tiebreak_order(scores["feature"].tolist())
    # Feature-string tie-break is not numeric-safe, so re-sort by a stable integer key.
    ranked = scores.copy()
    ranked["_order"] = ranked["feature"].map(order_idx)
    ranked = ranked.sort_values([score_col, "_order"], ascending=[False, True], na_position="last")
    ranked = ranked.drop(columns=["_order"])

    n_keep = int(round(retention * n_features))
    n_keep = max(min_features, min(n_keep, n_features))
    selected = ranked["feature"].tolist()[:n_keep]
    fallback = ranked["feature"].tolist()
    return selected, fallback


def retention_options(retention_ratios: Sequence[float], default_retention: float) -> List[float]:
    """Return the retention ratios to search, default first for reproducibility."""
    ratios = list(retention_ratios)
    if default_retention not in ratios:
        ratios.append(default_retention)
    return ratios
