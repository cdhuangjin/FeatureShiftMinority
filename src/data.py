"""Dataset loading and deterministic stratified splitting for Project C Gate A.

The prompt recommends Credit Card Fraud / Give Me Some Credit / one of
Mammography-Satimage-Oil-Yeast.  The first two require Kaggle credentials and
are not needed to answer the research question, so we substitute three
imbalanced public benchmarks that are downloadable from imblearn's Zenodo
cache.  The replacement is recorded in `DatasetProfile.replacement_reason`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import StratifiedShuffleSplit


REPLACEMENT_REASON = (
    "The prompt-first choices Credit Card Fraud and Give Me Some Credit are "
    "published on Kaggle and require an authenticated Kaggle API token, which "
    "is not available in this environment.  To keep the run fully reproducible "
    "on CPU we use three public, clearly licensed imbalanced binary benchmarks "
    "from imblearn's Zenodo cache instead (satimage, mammography, abalone_19). "
    "All three remain imbalanced (none is converted to a balanced dataset)."
)


@dataclass
class DatasetProfile:
    """Metadata + raw arrays for one dataset."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    source: str
    replacement_reason: str
    n_samples: int = field(init=False)
    n_features: int = field(init=False)
    minority_count: int = field(init=False)
    majority_count: int = field(init=False)
    imbalance_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        self.n_samples, self.n_features = self.X.shape
        vals, counts = np.unique(self.y, return_counts=True)
        if len(vals) != 2:
            raise ValueError(f"Dataset {self.name} is not binary (classes={vals}).")
        if 1 not in vals:
            raise ValueError(f"Minority class 1 missing in dataset {self.name}.")
        self.minority_count = int((self.y == 1).sum())
        self.majority_count = int((self.y == 0).sum())
        self.imbalance_ratio = self.majority_count / max(self.minority_count, 1)

    def describe(self) -> Dict[str, object]:
        return {
            "dataset": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "minority_count": self.minority_count,
            "majority_count": self.majority_count,
            "imbalance_ratio": round(self.imbalance_ratio, 4),
            "source": self.source,
        }


def _normalise_labels(raw: np.ndarray) -> np.ndarray:
    """Map arbitrary binary labels so minority == 1 and majority == 0."""
    # The Zenodo benchmarks use {-1, +1} with +1 as minority.
    if set(np.unique(raw)) == {-1.0, 1.0} or set(np.unique(raw)) == {-1, 1}:
        return (raw == 1).astype(int)
    if set(np.unique(raw)) == {0.0, 1.0} or set(np.unique(raw)) == {0, 1}:
        return raw.astype(int)
    # Generic fallback: the minority class becomes 1.
    vals, counts = np.unique(raw, return_counts=True)
    minority_val = vals[int(np.argmin(counts))]
    return (raw == minority_val).astype(int)


def load_datasets(config: dict, data_root: Path) -> Dict[str, DatasetProfile]:
    """Load the configured datasets from imblearn's Zenodo cache."""
    names = config["datasets"]
    names = [names] if isinstance(names, str) else list(names)
    data_home = str(data_root / "raw")
    loaded = fetch_datasets(data_home=data_home, filter_data=tuple(names))

    profiles: Dict[str, DatasetProfile] = {}
    for name in names:
        bunch = loaded[name]
        X = np.asarray(bunch.data, dtype=float)
        y = _normalise_labels(np.asarray(bunch.target, dtype=float).astype(int))
        feat_names = [f"f{i}" for i in range(X.shape[1])]
        profiles[name] = DatasetProfile(
            name=name,
            X=X,
            y=y,
            feature_names=feat_names,
            source=config["dataset_sources"][name],
            replacement_reason=REPLACEMENT_REASON,
        )

    # Persist the raw features/labels so downstream runs do not re-download.
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, prof in profiles.items():
        np.savez_compressed(raw_dir / f"{name}.npz", X=prof.X, y=prof.y)
    return profiles


def make_splits(profile: DatasetProfile, config: dict, seed: int) -> Dict[str, np.ndarray]:
    """Stratified 60/20/20 train/validation/test split for a single seed."""
    split = config["split"]
    tr = split["train"]
    va = split["validation"]
    te = split["test"]
    # First carve out an exclusive test set.
    outer = StratifiedShuffleSplit(n_splits=1, test_size=te, random_state=seed)
    train_idx, test_idx = next(outer.split(profile.X, profile.y))
    # Then split the train portion into train and validation.
    inner = StratifiedShuffleSplit(
        n_splits=1, test_size=va / (tr + va), random_state=seed
    )
    inner_train, val_idx = next(inner.split(profile.X[train_idx], profile.y[train_idx]))
    val_idx = train_idx[val_idx]
    train_idx = train_idx[inner_train]
    return {
        "train": np.asarray(train_idx),
        "validation": np.asarray(val_idx),
        "test": np.asarray(test_idx),
    }


def dataset_profiles_table(profiles: Dict[str, DatasetProfile]) -> pd.DataFrame:
    rows = [p.describe() for p in profiles.values()]
    return pd.DataFrame(rows)

