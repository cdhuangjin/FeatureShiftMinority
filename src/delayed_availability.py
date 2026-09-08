"""Synthetic delayed-availability stress test (availability fractions)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def rank_features_by_global(importance: pd.DataFrame) -> List[str]:
    return importance.sort_values("global_importance", ascending=False)["feature"].tolist()


def build_delayed_envs(
    feature_names: List[str], importance: pd.DataFrame, fractions: List[float]
) -> List[Dict[str, object]]:
    """Return availability environments: fraction + the unavailable feature set."""
    ranked = rank_features_by_global(importance)
    envs = []
    for frac in fractions:
        n_avail = max(1, int(round(frac * len(feature_names))))
        available = set(ranked[:n_avail])
        unavailable = [f for f in feature_names if f not in available]
        envs.append(
            {
                "availability_fraction": frac,
                "n_available": len(available),
                "unavailable_features": unavailable,
            }
        )
    return envs


def recovery_availability(
    full_recall: float, points: List[Tuple[float, float]], target_fraction: float = 0.90
) -> float:
    """Smallest availability fraction whose recall reaches target_fraction * full."""
    if full_recall <= 0:
        return 1.0
    target = target_fraction * full_recall
    for frac, recall in sorted(points, key=lambda x: x[0]):
        if recall >= target:
            return frac
    return max((p[0] for p in points), default=1.0)


def recovery_metrics(
    full_minority_recall: float,
    full_majority_recall: float,
    points_minority: List[Tuple[float, float]],
    points_majority: List[Tuple[float, float]],
) -> Dict[str, float]:
    mra_min = recovery_availability(full_minority_recall, points_minority)
    mra_maj = recovery_availability(full_majority_recall, points_majority)
    return {
        "MinorityRecoveryAvailability": mra_min,
        "MajorityRecoveryAvailability": mra_maj,
        "RecoveryGap": mra_min - mra_maj,
    }

