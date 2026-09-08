"""Removal-rate sensitivity slopes and critical (accelerated) degradation interval."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def fit_slope(remove_rates: Sequence[float], recalls: Sequence[float]) -> Tuple[float, float, float]:
    """Return (slope, intercept, R^2) of recall on removal_rate."""
    x = np.asarray(remove_rates, dtype=float)
    y = np.asarray(recalls, dtype=float)
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0, 0.0
    coeffs = np.polyfit(x, y, 1)  # [slope, intercept]
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    yhat = intercept + slope * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def critical_interval(remove_rates: Sequence[float], recalls: Sequence[float]) -> str:
    """Interval with the largest absolute local minority-recall drop."""
    rates = list(remove_rates)
    recs = list(recalls)
    best = (0.0, "")
    for i in range(len(rates) - 1):
        drop = recs[i] - recs[i + 1]
        if drop > best[0]:
            best = (drop, f"{int(rates[i]*100)}->{int(rates[i+1]*100)}")
    return best[1]


def sensitivity_summary(
    remove_rates: Sequence[float],
    minority_recalls: Sequence[float],
    majority_recalls: Sequence[float],
    eps: float,
) -> Dict[str, float]:
    b_min, a_min, r2_min = fit_slope(remove_rates, minority_recalls)
    b_maj, a_maj, r2_maj = fit_slope(remove_rates, majority_recalls)
    slope_gap = abs(b_min) - abs(b_maj)
    slope_ratio = abs(b_min) / (abs(b_maj) + eps)
    return {
        "minority_slope": round(b_min, 6),
        "majority_slope": round(b_maj, 6),
        "slope_gap": round(slope_gap, 6),
        "slope_ratio": round(slope_ratio, 6),
        "R2_minority": round(r2_min, 6),
        "R2_majority": round(r2_maj, 6),
        "critical_interval": critical_interval(remove_rates, minority_recalls),
    }
