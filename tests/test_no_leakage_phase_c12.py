"""Phase C1/C2 leakage invariants: split disjoint, grouping uses train only."""

import numpy as np
import pandas as pd

from src.data import make_splits, DatasetProfile
from src.feature_grouping import cluster_features
from src.delayed_availability import build_delayed_envs


def _cfg():
    return {"split": {"train": 0.60, "validation": 0.20, "test": 0.20}}


def test_split_disjoint():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] < 0).astype(int)
    prof = DatasetProfile("d", X, y, [f"f{i}" for i in range(4)], "t", "s")
    s = make_splits(prof, _cfg(), 42)
    assert not set(s["train"]).intersection(s["validation"])
    assert not set(s["train"]).intersection(s["test"])
    assert not set(s["validation"]).intersection(s["test"])


def test_grouping_does_not_use_labels():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 6))
    g1 = cluster_features(X, [f"f{i}" for i in range(6)], threshold=0.5)
    g2 = cluster_features(X, [f"f{i}" for i in range(6)], threshold=0.5)
    assert g1 == g2  # deterministic, does not depend on y


def test_delayed_ordering_uses_importance_only():
    imp = pd.DataFrame({"feature": [f"f{i}" for i in range(6)], "global_importance": [6, 5, 4, 3, 2, 1]})
    envs = build_delayed_envs([f"f{i}" for i in range(6)], imp, [0.50])
    # The 3 lowest-importance features are unavailable at 50% availability.
    assert sorted(envs[0]["unavailable_features"]) == ["f3", "f4", "f5"]

