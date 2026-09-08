"""Phase C3 MAFR-v0 selection and per-cell runner.

The MAFR-v0 method is small, CPU-only and uses only source train/validation.
Important invariant: the robust subset, the chosen variant and the chosen retention
are frozen *before* any test-time evaluation.  Test data is never used to score,
rank or select features.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import make_splits
from .dataset_registry import PhaseDataset
from .preprocessing import Preprocessor, to_frame
from .models import build_model
from .feature_importance import compute_importance
from .feature_shift import apply_mask, _seed_from
from .metrics import compute_metrics, minority_recall
from .mafr import compute_feature_scores, select_subset, retention_options


# (strategy, fraction) mapping for the fixed test-time shift environments.
TEST_SHIFT_SPEC = {
    "random_10": ("random", 0.10),
    "global_importance_10": ("global_importance", 0.10),
    "minority_specific_10": ("minority_specific", 0.10),
    "majority_specific_10": ("majority_specific", 0.10),
    "global_importance_20": ("global_importance", 0.20),
    "minority_specific_20": ("minority_specific", 0.20),
}


def n_budget(n_features: int, config: dict) -> int:
    cfg = config["methods"]
    retention = cfg.get("budget_retention", config["mafr"].get("default_retention", 0.80))
    return budget_from_retention(n_features, retention, cfg.get("budget_min_features", config["mafr"].get("min_features", 3)))


def budget_from_retention(n_features: int, retention: float, min_features: int = 3) -> int:
    """Number of features kept at a retention ratio, clamped to [min_features, n_features]."""
    budget = int(round(retention * n_features))
    budget = max(min_features, budget)
    return min(budget, n_features)


def compute_corr_matrix(X_train: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Train-only absolute Spearman correlation matrix."""
    X = pd.DataFrame(X_train, columns=feature_names)
    n = X.shape[1]
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if X.iloc[:, i].std() == 0 or X.iloc[:, j].std() == 0:
                r = 0.0
            else:
                r, _ = _spearman(X.iloc[:, i], X.iloc[:, j])
                r = abs(float(r)) if not np.isnan(r) else 0.0
            corr[i, j] = corr[j, i] = r
    return corr


def compute_redundancy(X_train: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Train-only per-feature substitutability = max absolute Spearman with another feature."""
    corr = compute_corr_matrix(X_train, feature_names)
    n = corr.shape[0]
    out = {}
    for i in range(n):
        others = np.delete(corr[i], i)
        out[feature_names[i]] = float(others.max()) if others.size else 0.0
    return out


def _spearman(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    from scipy.stats import spearmanr

    return spearmanr(a, b)


def compute_availability_robustness(
    est,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    feature_map: Dict[str, dict],
    config: dict,
) -> Dict[str, float]:
    """Validation-only leave-one-out robustness: fraction of full minority recall retained
    when a single feature is unavailable.  Higher means the feature is expendable."""
    eps = config["epsilon"]
    full_proba = est.predict_proba(X_val)[:, 1]
    full_min = minority_recall(y_val, full_proba)
    out = {}
    for feat in feature_names:
        cols = feature_map[feat]["cols"]
        fill = feature_map[feat]["fill"]
        Xm = X_val.copy()
        for c in cols:
            Xm[:, c] = fill
        proba = est.predict_proba(Xm)[:, 1]
        m_min = minority_recall(y_val, proba)
        out[feat] = float(m_min / (full_min + eps))
    return out


def build_validation_stress_envs(
    feature_names: List[str],
    importance: pd.DataFrame,
    config: dict,
    seed: int,
    dataset: str,
    model: str,
) -> List[Dict[str, object]]:
    """Fixed validation-only availability-stress environments used to pick the MAFR variant."""
    stress = config["mafr"]["validation_stress"]
    envs: List[Dict[str, object]] = []
    for rate in stress.get("random_rates", []):
        n = max(1, int(round(rate * len(feature_names))))
        rng = np.random.default_rng(_seed_from(f"{dataset}|{seed}|{model}|vstress|{rate}"))
        removed = sorted(rng.choice(feature_names, size=n, replace=False).tolist())
        envs.append({"name": f"random_{int(rate*100)}", "removed_features": removed})
    g_rate = stress.get("global_rate", 0.10)
    envs.append(
        {
            "name": f"global_important_{int(g_rate*100)}",
            "removed_features": _top_n(importance, "global_importance", g_rate, len(feature_names)),
        }
    )
    m_rate = stress.get("minority_rate", 0.10)
    envs.append(
        {
            "name": f"minority_specific_{int(m_rate*100)}",
            "removed_features": _top_n(importance, "MSI", m_rate, len(feature_names)),
        }
    )
    return envs


def _top_n(importance: pd.DataFrame, col: str, fraction: float, n_features: int) -> List[str]:
    n = max(1, int(round(fraction * n_features)))
    ranked = importance.sort_values(col, ascending=False)["feature"].tolist()
    return ranked[:n]


def select_mafr_variant(
    est_b0,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    feature_map: Dict[str, dict],
    scores: pd.DataFrame,
    stress_envs: List[Dict[str, object]],
    config: dict,
    seed: int,
    dataset: str,
    model: str,
) -> Dict[str, object]:
    """Pick the (variant, retention) with the best source-validation stress AUPRC.

    Selection uses ONLY train + validation.  The returned subset is then frozen and
    evaluated on test without further tuning.
    """
    variants = config["mafr"]["weights_variants"]
    retentions = retention_options(config["mafr"]["retention_ratios"], config["mafr"]["default_retention"])
    n_features = len(feature_names)
    min_features = config["mafr"].get("min_features", 3)

    best = None
    for variant in variants:
        for retention in retentions:
            selected, _fallback = select_subset(scores, variant, retention, n_features, min_features)
            cols = method_column_indices(selected, feature_names, feature_map)
            est = build_model(model, seed, config["model_config"])
            est.fit(X_train[:, cols], y_train)
            stress_scores = []
            for env in stress_envs:
                Xm = apply_mask(X_val, env["removed_features"], feature_map)
                proba = est.predict_proba(Xm[:, cols])[:, 1]
                stress_scores.append(average_precision_score(y_val, proba))
            stress_score = float(np.mean(stress_scores))
            if best is None or stress_score > best["stress_score"]:
                best = {
                    "variant": variant,
                    "retention": retention,
                    "selected_features": selected,
                    "stress_score": stress_score,
                    "n_selected": len(selected),
                }
    assert best is not None
    _sel, fallback = select_subset(scores, best["variant"], best["retention"], n_features, min_features)
    best["fallback_priority"] = fallback
    return best


def build_baseline_subsets(
    importance: pd.DataFrame,
    feature_names: List[str],
    budget: int,
    config: dict,
    seed: int,
    dataset: str,
    model: str,
    corr_matrix: np.ndarray | None = None,
) -> Dict[str, List[str]]:
    """B1 global-importance, B2 minority-only, B3 redundancy-aware global, B4 random."""
    order_index = {name: i for i, name in enumerate(feature_names)}
    by_order = sorted(feature_names, key=lambda f: order_index[f])
    budget = min(budget, len(feature_names))

    b1 = importance.sort_values("global_importance", ascending=False)["feature"].tolist()[:budget]
    b2 = importance.sort_values("minority_importance", ascending=False)["feature"].tolist()[:budget]
    b3 = _redundancy_aware_global(
        importance, feature_names, budget, config["mafr"].get("correlation_threshold", 0.7),
        order_index, corr_matrix,
    )

    rng = np.random.default_rng(_seed_from(f"{dataset}|{seed}|{model}|B4"))
    b4 = sorted(rng.choice(feature_names, size=budget, replace=False).tolist())
    # Sort each subset by original feature order for a stable column layout.
    return {
        "B1": sorted(b1, key=lambda f: order_index[f]),
        "B2": sorted(b2, key=lambda f: order_index[f]),
        "B3": sorted(b3, key=lambda f: order_index[f]),
        "B4": sorted(b4, key=lambda f: order_index[f]),
    }


def _redundancy_aware_global(
    importance: pd.DataFrame,
    feature_names: List[str],
    budget: int,
    threshold: float,
    order_index: Dict[str, int],
    corr_matrix: np.ndarray | None,
) -> List[str]:
    """Greedy global-importance selection with a redundancy skip.

    A candidate is skipped when its absolute Spearman correlation to any already
    selected feature exceeds ``threshold`` (i.e. it is a near-duplicate).  If the
    greedy pass cannot fill the budget we fall back to the highest global-importance
    features not yet selected, so the subset always has the required budget.
    """
    ranked = importance.sort_values("global_importance", ascending=False)["feature"].tolist()
    index_of = {name: i for i, name in enumerate(feature_names)}
    selected: List[str] = []
    for f in ranked:
        if len(selected) >= budget:
            break
        redundant = False
        if corr_matrix is not None:
            fi = index_of[f]
            for s in selected:
                si = index_of[s]
                if corr_matrix[fi, si] > threshold:
                    redundant = True
                    break
        if not redundant:
            selected.append(f)
    # Fill to budget from highest global importance not already selected.
    for f in ranked:
        if len(selected) >= budget:
            break
        if f not in selected:
            selected.append(f)
    return selected


def method_column_indices(
    subset: List[str], feature_names: List[str], feature_map: Dict[str, dict]
) -> List[int]:
    """Processed column indices for a subset, ordered by the original feature order."""
    order = {name: i for i, name in enumerate(feature_names)}
    ordered = sorted(subset, key=lambda f: order[f])
    cols: List[int] = []
    for f in ordered:
        cols.extend(feature_map[f]["cols"])
    return cols


def select_removed_for_shift(
    shift_name: str,
    importance: pd.DataFrame,
    feature_names: List[str],
    seed: int,
    dataset: str,
    model: str,
) -> List[str]:
    if shift_name not in TEST_SHIFT_SPEC:
        return []
    strategy, fraction = TEST_SHIFT_SPEC[shift_name]
    n = max(1, int(round(fraction * len(feature_names))))
    if strategy == "random":
        rng = np.random.default_rng(_seed_from(f"{dataset}|{seed}|{model}|shift|{shift_name}"))
        return sorted(rng.choice(feature_names, size=n, replace=False).tolist())
    if strategy == "global_importance":
        ranked = importance.sort_values("global_importance", ascending=False)["feature"].tolist()
        return sorted(ranked[:n])
    if strategy == "minority_specific":
        ranked = importance.sort_values("MSI", ascending=False)["feature"].tolist()
        return sorted(ranked[:n])
    if strategy == "majority_specific":
        ranked = importance.sort_values("MSI", ascending=True)["feature"].tolist()
        return sorted(ranked[:n])
    raise ValueError(f"Unknown shift strategy {strategy}")


def build_test_shift_envs(
    feature_names: List[str],
    importance: pd.DataFrame,
    config: dict,
    seed: int,
    dataset: str,
    model: str,
) -> List[Dict[str, object]]:
    envs = [{"name": "full", "removed_features": [], "n_removed": 0}]
    for shift_name in config["test_shifts"]:
        if shift_name == "full":
            continue
        removed = select_removed_for_shift(shift_name, importance, feature_names, seed, dataset, model)
        envs.append({"name": shift_name, "removed_features": removed, "n_removed": len(removed)})
    return envs


def run_cell(
    phase_ds: PhaseDataset,
    seed: int,
    model: str,
    config: dict,
    data_root,
):
    prof = phase_ds.profile
    split_idx = make_splits(prof, config, seed)
    X, y = prof.X, prof.y
    Xtr = to_frame(X[split_idx["train"]], prof.feature_names)
    Xva = to_frame(X[split_idx["validation"]], prof.feature_names)
    Xte = to_frame(X[split_idx["test"]], prof.feature_names)
    ytr = y[split_idx["train"]]
    yva = y[split_idx["validation"]]
    yte = y[split_idx["test"]]

    prep = Preprocessor().fit(Xtr)
    Xtr_p = prep.transform(Xtr)
    Xva_p = prep.transform(Xva)
    Xte_p = prep.transform(Xte)

    feature_names = prof.feature_names
    feature_map = prep.feature_map
    n_features = len(feature_names)
    eps = config["epsilon"]
    threshold = config["threshold"]
    gate_cfg = config["method_gate"]

    est_b0 = build_model(model, seed, config["model_config"])
    est_b0.fit(Xtr_p, ytr)
    importance, bases = compute_importance(est_b0, Xva_p, yva, feature_names, feature_map, config, seed)

    redundancy = compute_redundancy(Xtr_p, feature_names)
    corr_matrix = compute_corr_matrix(Xtr_p, feature_names)
    robustness = compute_availability_robustness(est_b0, Xva_p, yva, feature_names, feature_map, config)

    scores = compute_feature_scores(
        feature_names,
        importance["global_importance"].tolist(),
        importance["minority_importance"].tolist(),
        importance["majority_importance"].tolist(),
        [redundancy[f] for f in feature_names],
        [robustness[f] for f in feature_names],
        config["mafr"]["weights_variants"],
    )

    stress_envs = build_validation_stress_envs(feature_names, importance, config, seed, prof.name, model)
    min_features = config["mafr"].get("min_features", 3)
    sel = select_mafr_variant(
        est_b0, Xtr_p, ytr, Xva_p, yva, feature_names, feature_map, scores,
        stress_envs, config, seed, prof.name, model,
    )
    mafr_subset = sel["selected_features"]
    mafr_fallback = sel["fallback_priority"]
    # B1--M1 must share an identical feature budget.  The MAFR-chosen retention
    # (validation-based) is the shared budget applied to every subset method.
    shared_budget = budget_from_retention(n_features, sel["retention"], min_features)

    baseline_subsets = build_baseline_subsets(
        importance, feature_names, shared_budget, config, seed, prof.name, model, corr_matrix
    )
    methods = {
        "B0": list(feature_names),
        "B1": baseline_subsets["B1"],
        "B2": baseline_subsets["B2"],
        "B3": baseline_subsets["B3"],
        "B4": baseline_subsets["B4"],
        "M1": mafr_subset,
    }

    shift_envs = build_test_shift_envs(feature_names, importance, config, seed, prof.name, model)

    results: List[dict] = []
    for method, subset in methods.items():
        subset_cols = method_column_indices(subset, feature_names, feature_map)
        subset_set = set(subset)
        est = est_b0 if method == "B0" else (
            build_model(model, seed, config["model_config"]).fit(Xtr_p[:, subset_cols], ytr)
        )
        full_proba = est.predict_proba(Xte_p[:, subset_cols])[:, 1]
        full_metrics = compute_metrics(yte, full_proba, threshold)
        for env in shift_envs:
            removed = env["removed_features"]
            if env["name"] == "full":
                proba = full_proba
            else:
                Xte_masked = apply_mask(Xte_p, removed, feature_map)
                proba = est.predict_proba(Xte_masked[:, subset_cols])[:, 1]
            m = compute_metrics(yte, proba, threshold)
            effective = len([f for f in removed if f in subset_set])
            min_full = full_metrics["minority_recall"]
            maj_full = full_metrics["majority_recall"]
            min_sh = m["minority_recall"]
            maj_sh = m["majority_recall"]
            mrd_min = (min_full - min_sh) / (min_full + eps)
            mard_maj = (maj_full - maj_sh) / (maj_full + eps)
            mvg = mrd_min - mard_maj
            auroc_drop = full_metrics["AUROC"] - m["AUROC"]
            auprc_drop = full_metrics["AUPRC"] - m["AUPRC"]
            rec_drop = min_full - min_sh
            hidden_auroc = bool(
                auroc_drop <= gate_cfg["hidden_auroc_max"] and rec_drop >= gate_cfg["hidden_recall_drop_min"]
            )
            results.append(
                {
                    "dataset": prof.name,
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "shift_env": env["name"],
                    "n_features": n_features,
                    "budget": shared_budget,
                    "n_selected": len(subset),
                    "n_removed_original": env["n_removed"],
                    "effective_affected": effective,
                    "AUROC": m["AUROC"],
                    "AUPRC": m["AUPRC"],
                    "minority_recall": m["minority_recall"],
                    "minority_f1": m["minority_f1"],
                    "majority_recall": m["majority_recall"],
                    "majority_f1": m["majority_f1"],
                    "full_AUROC": full_metrics["AUROC"],
                    "full_AUPRC": full_metrics["AUPRC"],
                    "full_minority_recall": min_full,
                    "full_minority_f1": full_metrics["minority_f1"],
                    "full_majority_recall": maj_full,
                    "full_majority_f1": full_metrics["majority_f1"],
                    "MVG_recall": round(mvg, 6),
                    "MRD_recall": round(mrd_min, 6),
                    "MaRD_recall": round(mard_maj, 6),
                    "AUROC_drop": round(auroc_drop, 6),
                    "AUPRC_drop": round(auprc_drop, 6),
                    "minority_recall_drop": round(rec_drop, 6),
                    "hidden_auroc": hidden_auroc,
                }
            )

    # Feature score rows.
    score_rows: List[dict] = []
    for _, r in scores.iterrows():
        score_rows.append(
            {
                "dataset": prof.name,
                "model": model,
                "seed": seed,
                "feature": r["feature"],
                "global_importance": r["global_importance"],
                "minority_importance": r["minority_importance"],
                "majority_importance": r["majority_importance"],
                "class_specificity": r["class_specificity"],
                "redundancy": r["redundancy"],
                "availability_robustness": r["availability_robustness"],
                "norm_minority": r["norm_minority"],
                "norm_class_specificity": r["norm_class_specificity"],
                "norm_redundancy": r["norm_redundancy"],
                "norm_availability_robustness": r["norm_availability_robustness"],
                **{f"{v}_score": r[f"{v}_score"] for v in config["mafr"]["weights_variants"]},
                **{f"{v}_rank": int(r[f"{v}_rank"]) for v in config["mafr"]["weights_variants"]},
            }
        )

    selection_records = [
        {
            "dataset": prof.name,
            "model": model,
            "seed": seed,
            "variant": sel["variant"],
            "retention": sel["retention"],
            "n_selected": len(mafr_subset),
            "n_features": n_features,
            "budget": shared_budget,
            "stress_score": sel["stress_score"],
            "selected_features": mafr_subset,
            "fallback_priority": mafr_fallback,
        }
    ]

    return {"results": results, "feature_scores": score_rows, "selection": selection_records}


def build_cell_plan(datasets, seeds, models) -> List[Tuple[str, int, str]]:
    return [(ds, seed, model) for ds in datasets for seed in seeds for model in models]


def coalesce_checkpoint_payloads(payloads: List[dict]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for key in ("results", "feature_scores", "selection"):
        recs = []
        for p in payloads:
            recs.extend(p.get(key, []))
        out[key] = pd.DataFrame(recs)
    return out
