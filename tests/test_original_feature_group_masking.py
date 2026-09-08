"""Removing one original feature masks all of its one-hot dummy columns."""

import numpy as np
import pandas as pd

from src.preprocessing import Preprocessor
from src.feature_shift import apply_mask


def test_onehot_group_masked_together():
    X = pd.DataFrame({"cat": ["a", "b", "c", "a", "b", "c"], "num": [1.0, 2, 3, 4, 5, 6]})
    prep = Preprocessor().fit(X)
    Xp = prep.transform(X)
    spec = prep.feature_map["cat"]
    assert spec["type"] == "onehot"
    assert len(spec["cols"]) == 3
    masked = apply_mask(Xp, ["cat"], prep.feature_map)
    for col in spec["cols"]:
        assert np.all(masked[:, col] == 0)
    # The other original feature ("num") is left untouched.
    for col in prep.feature_map["num"]["cols"]:
        assert np.allclose(masked[:, col], Xp[:, col])
