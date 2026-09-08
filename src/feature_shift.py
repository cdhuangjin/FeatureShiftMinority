"""Feature unavailability masking (train-time full / test-time partial features)."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _seed_from(key: str) -> int:
    """Deterministic integer seed derived from a string key."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


def n_remove_features(n_features: int, config: dict) -> int:
    frac = config["feature_shift"]["removal_fraction"]
    raw = int(frac * n_features)
    return max(config["feature_shift"]["min_removed"], min(config["feature_shift"]["max_removed"], raw))


def select_removed_features(
    strategy: str,
    importance: pd.DataFrame,
    feature_names: List[str],
    n_remove: int,
    seed: int,
    dataset: str,
    model: str,
) -> List[str]:
    """Choose the feature set to make unavailable for one environment."""
    if n_remove <= 0:
        return []
    if strategy == "random":
        rng = np.random.default_rng(_seed_from(f"{dataset}|{seed}|random"))
        return sorted(rng.choice(feature_names, size=n_remove, replace=False).tolist())
    if strategy == "global_importance":
        ranked = importance.sort_values("global_importance", ascending=False)["feature"].tolist()
        return sorted(ranked[:n_remove])
    if strategy == "minority_specific":
        ranked = importance.sort_values("MSI", ascending=False)["feature"].tolist()
        return sorted(ranked[:n_remove])
    raise ValueError(f"Unknown shift strategy: {strategy}")


def apply_mask(
    X: np.ndarray,
    removed_features: List[str],
    feature_map: Dict[str, dict],
) -> np.ndarray:
    """Replace unavailable feature columns with their stored fill value."""
    Xm = X.copy()
    for feat in removed_features:
        spec = feature_map[feat]
        for col in spec["cols"]:
            Xm[:, col] = spec["fill"]
    return Xm


def build_environments(
    X_test: np.ndarray,
    feature_names: List[str],
    feature_map: Dict[str, dict],
    importance: pd.DataFrame,
    config: dict,
    seed: int,
    dataset: str,
    model: str,
) -> List[Tuple[str, np.ndarray, List[str]]]:
    """Return [(env_name, masked_X, dropped_features)] for E0..E3."""
    n = n_remove_features(len(feature_names), config)
    envs = [("full", X_test, [])]
    for strat in config["feature_shift"]["strategies"]:
        dropped = select_removed_features(strat, importance, feature_names, n, seed, dataset, model)
        envs.append((strat, apply_mask(X_test, dropped, feature_map), dropped))
    return envs
