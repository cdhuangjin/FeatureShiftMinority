"""Structured feature-loss environments: group loss, correlated loss, matched random."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .feature_grouping import largest_group, ordered_groups_by_feature_loss
from .feature_shift import _seed_from


def _random_remove(feature_names: List[str], n: int, seed_key: str) -> List[str]:
    if n <= 0:
        return []
    rng = np.random.default_rng(_seed_from(seed_key))
    return sorted(rng.choice(feature_names, size=min(n, len(feature_names)), replace=False).tolist())


def build_structured_envs(
    feature_names: List[str],
    group_map: Dict[str, int],
    config: dict,
    seed: int,
    dataset: str,
    model: str,
) -> List[Dict[str, object]]:
    """Return structured environment descriptors (mechanism, severity, removed features)."""
    st = config["structured"]
    min_removed = st.get("group_min_features", 1)
    max_removed = st.get("group_max_removed", len(feature_names))
    n_total = len(feature_names)
    envs: List[Dict[str, object]] = []

    # Group loss at two budgets, each with a matched random control.
    for budget in st["group_removal_budget"]:
        removed, n, frac = ordered_groups_by_feature_loss(
            group_map, feature_names, budget, min_removed, max_removed
        )
        envs.append(
            {
                "mechanism": "group_loss",
                "pair_id": f"group_{budget}",
                "severity": budget,
                "n_removed": n,
                "removal_fraction": frac,
                "removed_features": removed,
            }
        )
        rand = _random_remove(feature_names, n, f"{dataset}|{seed}|{model}|matched_{budget}")
        envs.append(
            {
                "mechanism": "matched_random",
                "pair_id": f"group_{budget}",
                "severity": budget,
                "n_removed": n,
                "removal_fraction": frac,
                "removed_features": rand,
            }
        )

    # Correlated loss: the largest correlated cluster (capped to the budget).
    corr_budget = st.get("correlated_loss_budget", 0.10)
    corr_group = largest_group(group_map)
    budget_n = max(min_removed, int(round(corr_budget * n_total)))
    corr_removed = corr_group[: min(len(corr_group), budget_n)]
    if corr_removed:
        envs.append(
            {
                "mechanism": "correlated_loss",
                "pair_id": "corr",
                "severity": corr_budget,
                "n_removed": len(corr_removed),
                "removal_fraction": len(corr_removed) / max(n_total, 1),
                "removed_features": corr_removed,
            }
        )
        rand = _random_remove(feature_names, len(corr_removed), f"{dataset}|{seed}|{model}|corr_matched")
        envs.append(
            {
                "mechanism": "matched_random",
                "pair_id": "corr",
                "severity": corr_budget,
                "n_removed": len(corr_removed),
                "removal_fraction": len(corr_removed) / max(n_total, 1),
                "removed_features": rand,
            }
        )
    return envs
