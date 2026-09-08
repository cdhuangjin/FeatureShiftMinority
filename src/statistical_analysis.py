"""Bootstrap CI, Wilcoxon signed-rank, rank-biserial, FDR, Spearman."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": np.nan, "median": np.nan, "CI_low": np.nan, "CI_high": np.nan}
    rng = np.random.default_rng(seed)
    n = arr.size
    resampled = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        resampled[i] = arr[idx].mean()
    lo = (1 - confidence) / 2
    hi = 1 - lo
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "CI_low": float(np.percentile(resampled, lo * 100)),
        "CI_high": float(np.percentile(resampled, hi * 100)),
    }


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return {"statistic": np.nan, "p_value": np.nan}
    d = a - b
    nz = np.count_nonzero(d)
    if nz == 0:
        return {"statistic": np.nan, "p_value": 1.0}
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return {"statistic": np.nan, "p_value": np.nan}
    return {"statistic": float(stat), "p_value": float(p)}


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation for the signed-rank test."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    nz = d[d != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    if total == 0:
        return 0.0
    return float((r_plus - r_minus) / total)


def bh_correction(p_values: Sequence[float], alpha: float = 0.05) -> Dict:
    """Benjamini-Hochberg FDR. Returns adjusted p-values and significance flags."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return {"adjusted": [], "significant": []}
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        threshold = ranked[i] * n / (i + 1)
        adjusted[order[i]] = min(prev, threshold)
        prev = adjusted[order[i]]
    return {
        "adjusted": [float(x) for x in adjusted],
        "significant": [bool(x < alpha) for x in adjusted],
    }


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"rho": np.nan, "p_value": np.nan}
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p_value": float(p)}

