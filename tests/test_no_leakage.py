"""Leakage checks: importance/preprocessing must never touch the test split."""

import numpy as np
import pytest

from src.data import DatasetProfile, make_splits
from src.preprocessing import Preprocessor, to_frame


@pytest.fixture
def config():
    return {"split": {"train": 0.60, "validation": 0.20, "test": 0.20}}


@pytest.fixture
def fake_profile():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    X[:, 4] = rng.choice([0.0, 1.0], size=400)
    y = (X[:, 0] < -0.5).astype(int)
    # Force imbalance while keeping both classes present.
    y[:50] = 1
    return DatasetProfile(
        name="toy",
        X=X,
        y=y,
        feature_names=[f"f{i}" for i in range(5)],
        source="test",
        replacement_reason="synthetic",
    )


def test_splits_are_disjoint(fake_profile, config):
    split = make_splits(fake_profile, config, seed=42)
    tr, va, te = split["train"], split["validation"], split["test"]
    assert len(tr) + len(va) + len(te) == fake_profile.n_samples
    assert not set(tr).intersection(va)
    assert not set(tr).intersection(te)
    assert not set(va).intersection(te)
    y = fake_profile.y
    for idx in (tr, va, te):
        assert 0 < y[idx].sum() < len(idx)


def test_preprocessor_fit_does_not_touch_test(fake_profile, config):
    split = make_splits(fake_profile, config, seed=42)
    X = fake_profile.X
    Xtr = to_frame(X[split["train"]], fake_profile.feature_names)
    Xte = to_frame(X[split["test"]], fake_profile.feature_names)
    prep = Preprocessor().fit(Xtr)
    medians_before = dict(prep.medians)
    out_te = prep.transform(Xte)
    assert prep.medians == medians_before
    assert out_te.shape[0] == len(split["test"])
    assert out_te.shape[1] == fake_profile.n_features


def test_masking_fill_comes_from_train_median(fake_profile, config):
    from src.feature_shift import apply_mask

    split = make_splits(fake_profile, config, seed=42)
    X = fake_profile.X
    Xtr = to_frame(X[split["train"]], fake_profile.feature_names)
    Xte = to_frame(X[split["test"]], fake_profile.feature_names)
    prep = Preprocessor().fit(Xtr)
    te_p = prep.transform(Xte)
    masked = apply_mask(te_p, ["f0"], prep.feature_map)
    train_median = prep.medians["f0"]
    assert np.allclose(masked[:, 0], train_median)
    assert not np.allclose(masked[:, 0], np.median(Xte["f0"]))

