"""Delayed availability: monotonic availability and recovery metrics."""

import pandas as pd

from src.delayed_availability import build_delayed_envs, recovery_availability, recovery_metrics


def _imp():
    return pd.DataFrame(
        {"feature": [f"f{i}" for i in range(10)], "global_importance": list(range(10, 0, -1))}
    )


def test_availability_monotonic():
    envs = build_delayed_envs([f"f{i}" for i in range(10)], _imp(), [0.50, 0.75, 1.00])
    unavail = [len(e["unavailable_features"]) for e in envs]
    assert unavail[0] > unavail[1] > unavail[2]
    assert envs[2]["unavailable_features"] == []


def test_recovery_metrics_positive_gap_when_minority_needs_more():
    full_min = 0.8
    full_maj = 0.9
    min_points = [(0.50, 0.30), (0.75, 0.60), (1.00, 0.80)]
    maj_points = [(0.50, 0.80), (0.75, 0.88), (1.00, 0.90)]
    rm = recovery_metrics(full_min, full_maj, min_points, maj_points)
    # Recovery is defined as reaching 90% of the full-feature recall.
    # Minority target = 0.9*0.8 = 0.72, only reached at 1.00 availability.
    # Majority target = 0.9*0.9 = 0.81, reached at 0.75 availability.
    assert rm["MinorityRecoveryAvailability"] == 1.00
    assert rm["MajorityRecoveryAvailability"] == 0.75
    assert rm["RecoveryGap"] > 0
