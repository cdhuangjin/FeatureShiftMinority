"""Median imputation + one-hot encoding, fitted only on the training split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump, load


class Preprocessor:
    """Fit-on-train preprocessor that returns a fixed-width numeric matrix.

    Attributes
    ----------
    feature_map : dict
        Maps each logical (pre-encoding) feature to the processed column indices
        and the fill value used when that feature is unavailable:
        ``{"cols": [i, ...], "type": "numeric"|"onehot", "fill": float}``.
    group_of_column : dict
        Maps a processed column index to its logical feature name.
    """

    def __init__(self) -> None:
        self.medians: Dict[str, float] = {}
        self.onehot_columns: List[str] = []
        self.feature_map: Dict[str, Dict[str, object]] = {}
        self.group_of_column: Dict[int, str] = {}
        self.feature_names: List[str] = []

    def _is_numeric(self, series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series)

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """Learn medians and one-hot categories from the training frame."""
        numeric_cols = [c for c in X.columns if self._is_numeric(X[c])]
        categorical_cols = [c for c in X.columns if not self._is_numeric(X[c])]

        self.medians = {c: float(X[c].median()) for c in numeric_cols}

        self.onehot_columns = []
        for c in categorical_cols:
            cats = [str(v) for v in X[c].dropna().unique()]
            self.onehot_columns.extend([f"{c}__{v}" for v in sorted(cats)])

        col = 0
        for c in numeric_cols:
            self.feature_map[c] = {"cols": [col], "type": "numeric", "fill": self.medians[c]}
            self.group_of_column[col] = c
            col += 1
        for c in categorical_cols:
            n = sum(1 for name in self.onehot_columns if name.startswith(f"{c}__"))
            cols = list(range(col, col + n))
            self.feature_map[c] = {"cols": cols, "type": "onehot", "fill": 0.0}
            for col_i in cols:
                self.group_of_column[col_i] = c
            col += n

        self.feature_names = numeric_cols + categorical_cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted imputation/one-hot and return a float matrix."""
        if not self.feature_map:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        numeric_cols = [c for c in X.columns if self._is_numeric(X[c])]
        categorical_cols = [c for c in X.columns if not self._is_numeric(X[c])]

        num_block = X[numeric_cols].copy()
        for c in numeric_cols:
            num_block[c] = num_block[c].fillna(self.medians[c]).astype(float)

        if categorical_cols:
            cat_frame = X[categorical_cols].copy()
            for c in categorical_cols:
                cat_frame[c] = cat_frame[c].fillna("missing")
            cat_dummies = pd.get_dummies(cat_frame, prefix=categorical_cols)
            # Align to the columns seen during fit.
            missing = [c for c in self.onehot_columns if c not in cat_dummies.columns]
            for c in missing:
                cat_dummies[c] = 0
            cat_dummies = cat_dummies.reindex(columns=self.onehot_columns).fillna(0)
            cat_block = cat_dummies.to_numpy(dtype=float)
        else:
            cat_block = np.empty((X.shape[0], 0), dtype=float)

        return np.hstack([num_block.to_numpy(dtype=float), cat_block])


def to_frame(X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(X, columns=feature_names)


def save_preprocessor(prep: Preprocessor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(prep, path)


def load_preprocessor(path: Path) -> Preprocessor:
    return load(path)


def save_feature_names(feature_names: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)

