"""Dataset freeze + floor-risk retention checks."""

import numpy as np

from src.dataset_registry import PhaseDataset, freeze_datasets
from src.data import DatasetProfile


def _candidate(name, n=100, d=5, minority_frac=0.1):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] < 0).astype(int)
    y[: max(1, int(n * minority_frac))] = 1
    prof = DatasetProfile(
        name=name, X=X, y=y, feature_names=[f"f{i}" for i in range(d)],
        source="test", replacement_reason="synthetic",
    )
    return PhaseDataset(profile=prof, source="test", n_numeric=d, n_categorical=0)


def test_freeze_is_deterministic_and_keeps_all(tmp_path):
    cands = {n: _candidate(n) for n in ["a", "b", "c"]}
    config = tmp_path / "ds.yaml"
    f1 = freeze_datasets(cands, config)
    f2 = freeze_datasets(cands, config)
    assert f1["dataset_freeze_hash"] == f2["dataset_freeze_hash"]
    assert sorted(f1["dataset_list"]) == ["a", "b", "c"]


def test_freeze_does_not_drop_floor_risk():
    # Floor-risk datasets must be kept in the frozen list.
    cands = {"extreme": _candidate("extreme", n=100, minority_frac=0.02)}
    config = None
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        f = freeze_datasets(cands, Path(td) / "ds.yaml")
    assert "extreme" in f["dataset_list"]

