"""Train-only feature grouping by absolute Spearman correlation clustering."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def cluster_features(X_train: np.ndarray, feature_names: List[str], threshold: float = 0.7) -> Dict[str, int]:
    """Return feature -> group id using average-linkage on 1 - |Spearman|."""
    X = pd.DataFrame(X_train, columns=feature_names)
    n = X.shape[1]
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if X.iloc[:, i].std() == 0 or X.iloc[:, j].std() == 0:
                r = 0.0
            else:
                r, _ = spearmanr(X.iloc[:, i], X.iloc[:, j])
            corr[i, j] = corr[j, i] = abs(float(r)) if not np.isnan(r) else 0.0
    dist = 1.0 - corr
    # Small jitter keeps linkage stable when distance is exactly zero.
    dist = dist + np.eye(n) * 1e-9
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return {feature_names[i]: int(labels[i]) for i in range(n)}


def groups_by_size(group_map: Dict[str, int]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for feat, g in group_map.items():
        out.setdefault(g, []).append(feat)
    return out


def ordered_groups_by_feature_loss(
    group_map: Dict[str, int],
    feature_names: List[str],
    budget: float,
    min_removed: int = 1,
    max_removed: int = 999,
) -> Tuple[List[str], int, float]:
    """Greedily remove whole groups until the feature-loss budget is met."""
    groups = groups_by_size(group_map)
    # Remove larger groups first (simulates losing a whole subsystem early).
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    removed: List[str] = []
    n_total = len(feature_names)
    budget_n = max(min_removed, int(round(budget * n_total)))
    for group in sorted_groups:
        if len(removed) >= max_removed:
            break
        removed.extend(group)
        if len(removed) >= budget_n:
            break
    # Clamp to available features.
    removed = list(dict.fromkeys(removed))[: n_total]
    removed = removed[: min(len(removed), max_removed)]
    return removed, len(removed), len(removed) / max(n_total, 1)


def largest_group(group_map: Dict[str, int]) -> List[str]:
    groups = groups_by_size(group_map)
    if not groups:
        return []
    return max(groups.values(), key=len)
