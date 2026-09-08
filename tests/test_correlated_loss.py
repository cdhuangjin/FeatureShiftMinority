"""Correlated loss uses the largest correlated cluster; matched random has equal size."""

import numpy as np

from src.feature_grouping import cluster_features
from src.structured_shift import build_structured_envs


def _cfg():
    return {
        "structured": {
            "group_removal_budget": [0.10, 0.30],
            "group_min_features": 1,
            "group_max_removed": 5,
            "correlation_threshold": 0.7,
            "correlated_loss_budget": 0.10,
        }
    }


def test_correlated_and_matched_random_same_size():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(80, 1))
    X = np.hstack([base + 0.01 * rng.normal(size=(80, 1)) for _ in range(3)] + [rng.normal(size=(80, 4))])
    names = [f"f{i}" for i in range(7)]
    groups = cluster_features(X, names, threshold=0.8)
    envs = build_structured_envs(names, groups, _cfg(), seed=42, dataset="d", model="m")
    corr = [e for e in envs if e["mechanism"] == "correlated_loss"][0]
    match = [e for e in envs if e["mechanism"] == "matched_random" and e["pair_id"] == "corr"][0]
    assert corr["n_removed"] == len(corr["removed_features"])
    assert match["n_removed"] == corr["n_removed"]
    assert len(corr["removed_features"]) >= 1

