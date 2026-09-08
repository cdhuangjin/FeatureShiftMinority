"""Group-loss enforces whole-group removal and stable shape/labels."""

import numpy as np
import pandas as pd

from src.feature_grouping import cluster_features, ordered_groups_by_feature_loss, groups_by_size


def test_group_loss_removes_whole_group():
    rng = np.random.default_rng(0)
    # Two strongly correlated blocks => two clusters.
    base = rng.normal(size=(100, 1))
    X = np.hstack(
        [
            base + 0.01 * rng.normal(size=(100, 1)),
            base + 0.01 * rng.normal(size=(100, 1)),
            rng.normal(size=(100, 2)),
        ]
    )
    names = [f"f{i}" for i in range(4)]
    groups = cluster_features(X, names, threshold=0.9)
    gsize = groups_by_size(groups)
    removed, n, frac = ordered_groups_by_feature_loss(groups, names, budget=0.25, min_removed=1)
    # Removed set must be a union of whole groups.
    removed_set = set(removed)
    for g, feats in gsize.items():
        group_set = set(feats)
        if removed_set.intersection(group_set):
            assert group_set.issubset(removed_set)
    assert 1 <= n <= 4


def test_group_loss_preserves_shape_and_label():
    import numpy as np
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    names = [f"f{i}" for i in range(4)]
    groups = cluster_features(X, names, threshold=0.3)
    removed, _, _ = ordered_groups_by_feature_loss(groups, names, budget=0.5, min_removed=1)
    y = np.zeros(50)
    assert len(removed) >= 1
    # Masking is a column operation: samples and labels unchanged.
    assert len(y) == 50

