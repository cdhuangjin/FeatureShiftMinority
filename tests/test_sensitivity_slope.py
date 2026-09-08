"""Hand-computed sensitivity slope, gap, ratio, critical interval."""

import math

from src.sensitivity_analysis import fit_slope, critical_interval, sensitivity_summary


def test_fit_slope_exact():
    rates = [0.0, 0.1, 0.2]
    recalls = [0.8, 0.7, 0.6]
    slope, intercept, r2 = fit_slope(rates, recalls)
    assert math.isclose(slope, -1.0, rel_tol=1e-9)
    assert math.isclose(intercept, 0.8, rel_tol=1e-9)
    assert math.isclose(r2, 1.0, rel_tol=1e-9)


def test_sensitivity_summary():
    rates = [0.0, 0.05, 0.10, 0.20]
    minority = [0.60, 0.50, 0.40, 0.20]
    majority = [0.95, 0.94, 0.93, 0.91]
    s = sensitivity_summary(rates, minority, majority, 1e-12)
    assert abs(s["minority_slope"]) > abs(s["majority_slope"])
    assert s["slope_gap"] > 0
    assert s["slope_ratio"] > 1
    assert s["critical_interval"]  # non-empty string

