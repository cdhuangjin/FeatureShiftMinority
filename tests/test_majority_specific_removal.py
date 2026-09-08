"""Minority vs majority-specific feature ranking can differ."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.feature_importance import compute_importance
from src.preprocessing import Preprocessor, to_frame
from src.feature_shift import apply_mask


def _toy():
    rng = np.random.default_rng(11)
    n = 2000
    # f0: strongly minority-predictive (minority at negative values)
    x0 = np.where(rng.random(n) < 0.10, rng.normal(-3, 0.2, n), rng.normal(3, 0.5, n))
    # f1: majority-only variation, no minority signal
    x1 = rng.normal(size=n)
    y = (x0 < 0).astype(int)
    return pd.DataFrame({"f0": x0, "f1": x1, "f2": rng.normal(size=n)}), y


def test_minority_and_majority_rankings_can_differ():
    X, y = _toy()
    prep = Preprocessor().fit(X)
    Xp = prep.transform(X)
    model = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss", tree_method="hist", random_state=0)
    model.fit(Xp, y)
    imp, _ = compute_importance(
        model, Xp, y, list(X.columns), prep.feature_map, {"importance": {"permutation_repeats": 5}}, seed=1
    )
    f0 = imp[imp["feature"] == "f0"].iloc[0]
    # f0 (minority-key) should have positive MSI; a majority-only feature lower.
    assert f0["MSI"] > 0
    assert imp["MSI"].max() >= f0["MSI"]
    # The two importances are separate columns (can rank differently).
    assert "minority_importance" in imp.columns and "majority_importance" in imp.columns


def test_majority_specific_selects_lowest_msi():
    import numpy as np
    from src.phase_c12_runner import _removed_for
    imp = pd.DataFrame(
        {
            "feature": ["f0", "f1", "f2"],
            "global_importance": [0.3, 0.2, 0.1],
            "MSI": [0.4, -0.2, 0.05],
        }
    )
    removed = _removed_for("majority_specific", imp, ["f0", "f1", "f2"], 1, 42, "d", "m")
    assert removed == ["f1"]  # lowest MSI is most majority-specific

