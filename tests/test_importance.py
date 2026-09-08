"""Importance views on a synthetic toy dataset."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.feature_importance import compute_importance
from src.preprocessing import Preprocessor


def _make_toy():
    rng = np.random.default_rng(7)
    n = 2000
    x0 = np.where(rng.random(n) < 0.10, rng.normal(-3, 0.3, n), rng.normal(3, 0.5, n))
    x1 = rng.normal(size=n)
    y = (x0 < 0).astype(int)
    X = pd.DataFrame({"f0": x0, "f1": x1, "f2": rng.normal(size=n)})
    return X, y


def _fit():
    X, y = _make_toy()
    prep = Preprocessor().fit(X)
    Xp = prep.transform(X)
    model = XGBClassifier(
        n_estimators=100, max_depth=4, eval_metric="logloss", tree_method="hist", random_state=0
    )
    model.fit(Xp, y)
    return model, prep, Xp, y


def test_minority_feature_gets_high_minority_importance():
    model, prep, Xp, y = _fit()
    imp, _ = compute_importance(
        model, Xp, y, list(prep.feature_names), prep.feature_map,
        {"importance": {"permutation_repeats": 5}}, seed=1,
    )
    row0 = imp[imp["feature"] == "f0"].iloc[0]
    assert row0["minority_importance"] > 0
    assert row0["MSI"] > 0


def test_global_and_msi_ranks_exist_and_can_differ():
    model, prep, Xp, y = _fit()
    imp, _ = compute_importance(
        model, Xp, y, list(prep.feature_names), prep.feature_map,
        {"importance": {"permutation_repeats": 5}}, seed=1,
    )
    assert {"global_rank", "MSI_rank"}.issubset(imp.columns)
    assert imp["global_rank"].dtype.kind == "i"
    assert imp["MSI_rank"].dtype.kind == "i"

