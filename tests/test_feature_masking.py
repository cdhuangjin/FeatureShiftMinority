"""Feature masking invariants."""

import numpy as np
import pandas as pd

from src.preprocessing import Preprocessor
from src.feature_shift import apply_mask, n_remove_features, select_removed_features


def test_apply_mask_shape_and_fill():
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "num2": [10.0, 20.0, 30.0, 40.0],
            "cat": ["a", "b", "a", "b"],
        }
    )
    prep = Preprocessor().fit(X)
    Xp = prep.transform(X)
    masked = apply_mask(Xp, ["num"], prep.feature_map)
    assert masked.shape == Xp.shape
    assert np.allclose(masked[:, 0], prep.medians["num"])
    assert np.allclose(masked[:, 1], Xp[:, 1])


def test_apply_mask_onehot_to_zero():
    X = pd.DataFrame({"cat": ["a", "b", "a", "b", "a"]})
    prep = Preprocessor().fit(X)
    Xp = prep.transform(X)
    spec = prep.feature_map["cat"]
    assert spec["type"] == "onehot"
    masked = apply_mask(Xp, ["cat"], prep.feature_map)
    for col in spec["cols"]:
        assert np.all(masked[:, col] == 0)


def test_n_remove_bounds():
    config = {"feature_shift": {"removal_fraction": 0.10, "min_removed": 1, "max_removed": 5}}
    assert n_remove_features(6, config) == 1
    assert n_remove_features(36, config) == 3
    assert n_remove_features(100, config) == 5


def test_select_random_is_identity_of_count():
    config = {"feature_shift": {"removal_fraction": 0.10, "min_removed": 1, "max_removed": 5}}
    imp = pd.DataFrame(
        {"feature": [f"f{i}" for i in range(10)], "global_importance": range(10, 0, -1), "MSI": range(1, 11)}
    )
    removed = select_removed_features(
        "random", imp, [f"f{i}" for i in range(10)], 3, seed=42, dataset="d", model="m"
    )
    assert len(removed) == 3
    assert len(set(removed)) == 3
