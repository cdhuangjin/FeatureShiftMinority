"""BMO calculation, observability labels, floor-risk handling."""

from src.baseline_observability import observability_label, screen_datasets
from src.dataset_registry import PhaseDataset
from src.data import DatasetProfile
import numpy as np


def _cfg():
    return {
        "observability": {"strong_minority_recall": 0.30, "minimum_primary_recall": 0.20},
        "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "seed": 42,
    }


def test_observability_labels():
    cfg = _cfg()
    assert observability_label(0.35, cfg) == "strongly_observable"
    assert observability_label(0.25, cfg) == "weakly_observable"
    assert observability_label(0.05, cfg) == "floor_risk"


def test_floor_risk_datasets_are_retained():
    # A dataset with a floor-risk BMO must still be reported (not dropped).
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] < 0).astype(int)
    y[0:5] = 1
    prof = DatasetProfile("d", X, y, ["f0", "f1", "f2"], "test", "synthetic")
    phase = PhaseDataset(profile=prof, source="test", n_numeric=3, n_categorical=0)
    bmo = screen_datasets({"d": phase}, seed=42, config=_cfg())
    assert len(bmo) == 1
    assert bmo["observability_class"].iloc[0] in {"strongly_observable", "weakly_observable", "floor_risk"}

