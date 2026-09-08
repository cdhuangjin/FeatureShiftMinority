"""Bootstrap, Wilcoxon, rank-biserial, FDR, Spearman."""

import numpy as np

from src.statistical_analysis import (
    bootstrap_ci,
    wilcoxon_paired,
    rank_biserial,
    bh_correction,
    spearman_corr,
)


def test_bootstrap_ci_ordering():
    vals = np.random.default_rng(0).normal(0.5, 0.1, 50)
    ci = bootstrap_ci(vals, iterations=500, confidence=0.95, seed=0)
    assert ci["CI_low"] < ci["mean"] < ci["CI_high"]


def test_wilcoxon_significant_when_difference():
    a = np.arange(1.0, 11.0)
    b = np.arange(0.0, 10.0)
    w = wilcoxon_paired(a, b)
    assert w["p_value"] < 0.05


def test_rank_biserial_range():
    a = np.arange(1.0, 11.0)
    b = np.arange(0.0, 10.0)
    es = rank_biserial(a, b)
    assert -1 <= es <= 1
    assert es > 0


def test_bh_monotone_and_bounds():
    p = np.array([0.001, 0.02, 0.03, 0.5, 0.8])
    out = bh_correction(p, alpha=0.05)
    # Adjusted p-values are >= raw p-values and <= 1.
    for raw, adj in zip(p, out["adjusted"]):
        assert adj >= raw - 1e-12
        assert adj <= 1.0 + 1e-12


def test_spearman_perfect():
    x = np.arange(1, 11)
    y = 3 * x + 1
    s = spearman_corr(x, y)
    assert abs(s["rho"] - 1.0) < 1e-9
