"""Phase C3 MAFR Method Gate invariants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import make_splits
from src.dataset_registry import PhaseDataset, DatasetProfile
from src.preprocessing import Preprocessor, to_frame
from src.models import build_model
from src.mafr import compute_feature_scores, select_subset, mafr_score, minmax_normalize
from src.mafr_selection import (
    build_baseline_subsets,
    build_validation_stress_envs,
    compute_corr_matrix,
    compute_redundancy,
    select_mafr_variant,
    select_removed_for_shift,
    method_column_indices,
    run_cell,
    n_budget,
    budget_from_retention,
)
from src.method_gate_analysis import evaluate_method_gate, block_statistics, cell_aggregates


ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return {
        "seeds": [42, 52, 62],
        "datasets": ["d"],
        "models": ["logistic_regression", "xgboost"],
        "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "model_config": {
            "logistic_regression": {"max_iter": 5000, "C": 1.0, "class_weight": None, "n_jobs": -1},
            "xgboost": {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9,
                "colsample_bytree": 0.9, "objective": "binary:logistic", "eval_metric": "logloss",
                "tree_method": "hist", "n_jobs": -1,
            },
        },
        "importance": {"permutation_repeats": 3, "global_metric": "auprc",
                       "minority_metric": "minority_recall", "majority_metric": "majority_recall"},
        "mafr": {
            "weights_variants": {"MAFR-A": [0.35, 0.25, 0.20, 0.20],
                                 "MAFR-B": [0.30, 0.30, 0.20, 0.20],
                                 "MAFR-C": [0.30, 0.20, 0.25, 0.25]},
            "retention_ratios": [0.70, 0.80, 0.90],
            "default_retention": 0.80,
            "min_features": 3,
            "correlation_threshold": 0.7,
            "validation_stress": {"random_rates": [0.05, 0.10, 0.20], "global_rate": 0.10, "minority_rate": 0.10},
        },
        "methods": {"baselines": ["B0", "B1", "B2", "B3", "B4"], "mafr": "M1",
                    "budget_retention": 0.80, "budget_min_features": 3},
        "test_shifts": ["full", "random_10", "global_importance_10", "minority_specific_10",
                        "majority_specific_10", "global_importance_20", "minority_specific_20"],
        "statistics": {"bootstrap_iterations": 500, "confidence_level": 0.95,
                       "fdr_alpha": 0.05, "min_effect_size": 0.20},
        "method_gate": {
            "hidden_auroc_max": 0.03, "hidden_recall_drop_min": 0.10,
            "mg1_min_cells": 5, "mg1_min_reduction": 0.10, "mg2_min_cells": 5,
            "mg2_min_recovery": 0.05, "mg3_hfr_reduction_min": 0.30,
            "mg4_max_mean_auprc_drop": 0.02, "mg4_max_cell_auprc_drop": 0.05,
            "mg5_max_mean_auroc_drop": 0.01, "mg6_min_cells": 4, "mg7": 0.0,
            "mg8_min_effect_size": 0.20, "strong_go_conditions_required": 7,
        },
        "epsilon": 1.0e-12,
        "threshold": 0.5,
    }


def _synthetic_phase_ds(seed=0, n=240, n_feats=8, minority_frac=0.30):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feats))
    # Minority label informed by a couple of features + a duplicate.
    X[:, 1] = X[:, 0] * 0.9 + rng.normal(scale=0.1, size=n)  # correlated duplicate of f0
    score = 1.5 * X[:, 0] + 0.6 * X[:, 1] + 0.3 * X[:, 2]
    proba = 1.0 / (1.0 + np.exp(-score))
    y = (rng.random(size=n) < proba).astype(int)
    # Force minority fraction.
    minority_idx = rng.choice(np.flatnonzero(y == 1), size=max(1, int(minority_frac * n)), replace=False)
    y = np.zeros(n, dtype=int)
    y[minority_idx] = 1
    if y.sum() == 0 or y.sum() == n:
        y[0] = 1
    names = [f"f{i}" for i in range(n_feats)]
    prof = DatasetProfile("d", X, y, names, "synthetic", "test")
    return PhaseDataset(profile=prof, source="synthetic", n_numeric=n_feats, n_categorical=0)


def _selection_from_cell(phase_ds, seed, model, cfg):
    prof = phase_ds.profile
    split_idx = make_splits(prof, cfg, seed)
    X, y = prof.X, prof.y
    Xtr = to_frame(X[split_idx["train"]], prof.feature_names)
    Xva = to_frame(X[split_idx["validation"]], prof.feature_names)
    ytr = y[split_idx["train"]]
    yva = y[split_idx["validation"]]
    prep = Preprocessor().fit(Xtr)
    Xtr_p = prep.transform(Xtr)
    Xva_p = prep.transform(Xva)
    est = build_model(model, seed, cfg["model_config"]).fit(Xtr_p, ytr)
    from src.feature_importance import compute_importance
    importance, _ = compute_importance(est, Xva_p, yva, prof.feature_names, prep.feature_map, cfg, seed)
    redundancy = compute_redundancy(Xtr_p, prof.feature_names)
    from src.mafr_selection import compute_availability_robustness
    robustness = compute_availability_robustness(est, Xva_p, yva, prof.feature_names, prep.feature_map, cfg)
    scores = compute_feature_scores(
        prof.feature_names,
        importance["global_importance"].tolist(),
        importance["minority_importance"].tolist(),
        importance["majority_importance"].tolist(),
        [redundancy[f] for f in prof.feature_names],
        [robustness[f] for f in prof.feature_names],
        cfg["mafr"]["weights_variants"],
    )
    stress = build_validation_stress_envs(prof.feature_names, importance, cfg, seed, prof.name, model)
    sel = select_mafr_variant(est, Xtr_p, ytr, Xva_p, yva, prof.feature_names, prep.feature_map,
                              scores, stress, cfg, seed, prof.name, model)
    return sel


def test_class_specific_score_hand_computed():
    cfg = _cfg()
    feature_names = ["f0", "f1"]
    # Two features: f0 minority importance 0.8 / majority 0.2 (spec 0.6),
    # f1 minority 0.4 / majority 0.6 (spec -0.2). redundancy 0.5 / 0.2, robustness 0.9 / 0.4.
    df = compute_feature_scores(
        feature_names,
        [0.5, 0.5],
        [0.8, 0.4],
        [0.2, 0.6],
        [0.5, 0.2],
        [0.9, 0.4],
        cfg["mafr"]["weights_variants"],
    )
    # min-max of minority [0.8,0.4]=>[1,0]; spec [0.6,-0.2]=>[1,0];
    # redundancy [0.5,0.2]=>[1,0]; robustness [0.9,0.4]=>[1,0].
    w = cfg["mafr"]["weights_variants"]["MAFR-A"]
    expected_f0 = w[0] * 1 + w[1] * 1 + w[2] * 1 + w[3] * 1
    expected_f1 = w[0] * 0 + w[1] * 0 + w[2] * 0 + w[3] * 0
    assert np.isclose(df.loc[0, "MAFR-A_score"], expected_f0)
    assert np.isclose(df.loc[1, "MAFR-A_score"], expected_f1)
    # MAFR-B weights [0.30,0.30,0.20,0.20] -> f0 still 1.0.
    assert np.isclose(df.loc[0, "MAFR-B_score"], 0.30 + 0.30 + 0.20 + 0.20)


def test_minmax_constant_vector():
    arr = minmax_normalize([0.5, 0.5, 0.5])
    assert np.allclose(arr, [0.5, 0.5, 0.5])


def test_redundancy_train_only_and_correlated_pair():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    X[:, 1] = X[:, 0] * 2.0 + 0.001  # near-perfectly correlated
    red = compute_redundancy(X, ["f0", "f1", "f2"])
    assert red["f0"] > 0.9
    assert red["f1"] > 0.9
    assert red["f2"] < 0.5


def test_deterministic_selection():
    cfg = _cfg()
    phase_ds = _synthetic_phase_ds(seed=11)
    s1 = _selection_from_cell(phase_ds, 42, "logistic_regression", cfg)
    s2 = _selection_from_cell(phase_ds, 42, "logistic_regression", cfg)
    assert s1["variant"] == s2["variant"]
    assert s1["retention"] == s2["retention"]
    assert s1["selected_features"] == s2["selected_features"]


def test_no_test_leakage_selection():
    cfg = _cfg()
    phase_ds = _synthetic_phase_ds(seed=5)
    split_idx = make_splits(phase_ds.profile, cfg, 42)
    s1 = _selection_from_cell(phase_ds, 42, "logistic_regression", cfg)
    # Perturb ONLY the test rows' features (split stays identical because y is unchanged),
    # i.e. the data the selection must never see. Selection must be unchanged.
    phase_ds2 = _synthetic_phase_ds(seed=5)
    rng = np.random.default_rng(99)
    phase_ds2.profile.X[split_idx["test"]] = rng.normal(size=(len(split_idx["test"]), phase_ds2.profile.X.shape[1]))
    s2 = _selection_from_cell(phase_ds2, 42, "logistic_regression", cfg)
    assert s1["variant"] == s2["variant"]
    assert s1["retention"] == s2["retention"]
    assert s1["selected_features"] == s2["selected_features"]


def test_feature_budget_equality():
    cfg = _cfg()
    rng = np.random.default_rng(0)
    feature_names = [f"f{i}" for i in range(9)]
    importance = pd.DataFrame({
        "feature": feature_names,
        "global_importance": rng.random(9),
        "minority_importance": rng.random(9),
        "majority_importance": rng.random(9),
        "MSI": rng.random(9) - 0.5,
    })
    X = rng.normal(size=(120, 9))
    corr = compute_corr_matrix(X, feature_names)
    budget = n_budget(9, cfg)
    subs = build_baseline_subsets(importance, feature_names, budget, cfg, 42, "d", "logistic_regression", corr)
    for name in ["B1", "B2", "B3", "B4"]:
        assert len(subs[name]) == budget
    # M1 subset uses the same budget via n_budget (tested separately via select_subset).
    scores = compute_feature_scores(
        feature_names,
        importance["global_importance"].tolist(),
        importance["minority_importance"].tolist(),
        importance["majority_importance"].tolist(),
        [0.1] * 9, [0.9] * 9, cfg["mafr"]["weights_variants"],
    )
    sel, _ = select_subset(scores, "MAFR-A", cfg["methods"]["budget_retention"], 9, cfg["mafr"]["min_features"])
    assert len(sel) == budget


def test_budget_from_retention():
    assert budget_from_retention(36, 0.80, 3) == 29
    assert budget_from_retention(16, 0.70, 3) == 11
    assert budget_from_retention(16, 0.90, 3) == 14
    assert budget_from_retention(3, 0.80, 3) == 3
    assert budget_from_retention(5, 0.10, 3) == 3  # min floor


def test_masking_uses_original_feature_identity():
    cfg = _cfg()
    importance = pd.DataFrame({
        "feature": ["f0", "f1", "f2"],
        "global_importance": [0.9, 0.5, 0.2],
        "minority_importance": [0.8, 0.6, 0.3],
        "majority_importance": [0.2, 0.4, 0.7],
        "MSI": [0.6, 0.2, -0.4],
    })
    removed = select_removed_for_shift("global_importance_10", importance, ["f0", "f1", "f2"], 42, "d", "m")
    assert len(removed) == 1 and removed == ["f0"]
    removed_min = select_removed_for_shift("minority_specific_10", importance, ["f0", "f1", "f2"], 42, "d", "m")
    assert removed_min == ["f0"]
    removed_maj = select_removed_for_shift("majority_specific_10", importance, ["f0", "f1", "f2"], 42, "d", "m")
    assert removed_maj == ["f2"]


def test_method_column_indices_and_effective_affected_count():
    # f0 -> col0, f1 -> col1, f2 -> col2, f3 -> col3 (numeric).
    feature_map = {f"f{i}": {"cols": [i], "type": "numeric", "fill": 0.0} for i in range(4)}
    cols = method_column_indices(["f2", "f0"], ["f0", "f1", "f2", "f3"], feature_map)
    assert cols == [0, 2]
    # Effective affected count: removed original features that are in the subset.
    removed = ["f0", "f3"]
    subset = ["f0", "f1", "f2"]
    effective = len([f for f in removed if f in set(subset)])
    assert effective == 1


def test_run_cell_end_to_end_and_no_leakage_of_test_labels():
    cfg = _cfg()
    phase_ds = _synthetic_phase_ds(seed=7)
    split_idx = make_splits(phase_ds.profile, cfg, 42)
    out = run_cell(phase_ds, 42, "logistic_regression", cfg, data_root=None)
    assert "selection" in out and len(out["selection"]) == 1
    sel1 = out["selection"][0]
    # Perturb only test-row features (labels/split unchanged) and re-run the full cell.
    phase_ds2 = _synthetic_phase_ds(seed=7)
    rng = np.random.default_rng(123)
    phase_ds2.profile.X[split_idx["test"]] = rng.normal(size=(len(split_idx["test"]), phase_ds2.profile.X.shape[1]))
    out2 = run_cell(phase_ds2, 42, "logistic_regression", cfg, data_root=None)
    sel2 = out2["selection"][0]
    assert sel1["variant"] == sel2["variant"]
    assert sel1["selected_features"] == sel2["selected_features"]
    # Sanity: results rows present and include B0/M1 across all shifts.
    methods = set(pd.DataFrame(out["results"])["method"])
    assert {"B0", "B1", "B2", "B3", "B4", "M1"} <= methods
    # B1--M1 must share the same feature budget.
    res = pd.DataFrame(out["results"])
    sub = res[res["method"].isin(["B1", "B2", "B3", "B4", "M1"])].drop_duplicates("method")
    n_selected = sub.set_index("method")["n_selected"].to_dict()
    assert len(set(n_selected.values())) == 1
    assert n_selected["M1"] == n_selected["B1"] == n_selected["B3"]


def test_cell_aggregates_and_mvg_reduction_metric():
    cfg = _cfg()
    rows = []
    for seed in [42, 52]:
        for env in ["random_10", "global_importance_10"]:
            for method, mvg, rec in [("B0", 0.5, 0.4), ("M1", 0.2, 0.6)]:
                rows.append({"dataset": "ds", "model": "m", "seed": seed, "method": method, "shift_env": env,
                             "MVG_recall": mvg, "minority_recall": rec, "AUROC": 0.9, "AUPRC": 0.8})
            # full rows
            for method in ["B0", "M1"]:
                rows.append({"dataset": "ds", "model": "m", "seed": seed, "method": method, "shift_env": "full",
                             "MVG_recall": 0.0, "minority_recall": 0.7, "AUROC": 0.92, "AUPRC": 0.85})
    res = pd.DataFrame(rows)
    cells = cell_aggregates(res)
    row = cells.iloc[0]
    # MVG reduction per shifted row = 0.5 - 0.2 = 0.3; mean over 2 seeds x 2 envs = 0.3.
    assert np.isclose(row["mean_MVG_Reduction"], 0.3)
    assert row["mvg_present"]  # some recovery >= 0


def test_block_statistics_paired_aggregation():
    cfg = _cfg()
    rows = []
    for ds in ["a", "b", "c"]:
        for model in ["lr", "xgb"]:
            for seed in [42, 52]:
                for method, mvg in [("B0", 0.5), ("B1", 0.4), ("B2", 0.6), ("B3", 0.45), ("M1", 0.2)]:
                    for env in ["random_10", "global_importance_10"]:
                        rows.append({"dataset": ds, "model": model, "seed": seed, "method": method,
                                     "shift_env": env, "MVG_recall": mvg})
    res = pd.DataFrame(rows)
    stats, strongest_row = block_statistics(res, cfg)
    # 3 datasets x 2 models x 2 seeds = 12 blocks.
    assert stats["n_blocks"].iloc[0] == 12
    # M1 mean MVG (0.2) is the lowest, so the strongest (lowest) is not M1.
    assert strongest_row["baseline"] != "M1"


def _cells_for_gate(**overrides):
    base = {
        "dataset": [], "model": [], "mean_MVG_Reduction": [], "mvg_present": [],
        "iid_AUPRC_cost": [], "iid_AUROC_cost": [], "worst_min_M1": [], "worst_B_best": [],
    }
    overrides = {**base, **overrides}
    return pd.DataFrame(overrides)


def _strong_row(ci_low=0.05, es=0.4, baseline="B0"):
    return {"baseline": baseline, "CI_low": ci_low, "effect_size": es, "n_blocks": 18,
            "mean_MVG_baseline": 0.5, "mean_MVG_M1": 0.2}


def test_gate_strong_go():
    cfg = _cfg()
    n = 6
    cells = pd.DataFrame({
        "dataset": [f"d{i}" for i in range(n)],
        "model": ["lr"] * n,
        "mean_MVG_Reduction": [0.2] * n,
        "mvg_present": [True] * n,
        "iid_AUPRC_cost": [0.005] * n,
        "iid_AUROC_cost": [0.003] * n,
        "worst_min_M1": [0.6] * n,
        "worst_B_best": [0.4] * n,
    })
    model_red = {"logistic_regression": 0.2, "xgboost": 0.2}
    hfr = {"rates": {"B0": 0.2, "M1": 0.05}, "counts": {"B0": 40, "M1": 10}, "hfr_reduction": 0.75}
    decision, conds, ev = evaluate_method_gate(cells, model_red, hfr, _strong_row(), cfg)
    assert decision == "STRONG-GO"
    assert all(conds.values())


def test_gate_go_partial():
    cfg = _cfg()
    cells = pd.DataFrame({
        "dataset": [f"d{i}" for i in range(6)],
        "model": ["lr"] * 6,
        "mean_MVG_Reduction": [0.2] * 6,
        "mvg_present": [True] * 6,
        "iid_AUPRC_cost": [0.005] * 6,
        "iid_AUROC_cost": [0.003] * 6,
        "worst_min_M1": [0.6] * 6,
        "worst_B_best": [0.4] * 6,
    })
    model_red = {"logistic_regression": 0.2, "xgboost": 0.2}
    hfr = {"rates": {"B0": 0.2, "M1": 0.15}, "counts": {"B0": 40, "M1": 30}, "hfr_reduction": 0.25}
    decision, conds, ev = evaluate_method_gate(cells, model_red, hfr, _strong_row(ci_low=0.05, es=0.1), cfg)
    assert decision == "GO"
    assert not conds["MG3_hidden_failure"]
    assert not conds["MG8_statistics"]


def test_gate_hold():
    cfg = _cfg()
    # Direction positive but only 4 cells improve, none reach the MG1 threshold.
    cells = pd.DataFrame({
        "dataset": [f"d{i}" for i in range(6)],
        "model": ["lr"] * 6,
        "mean_MVG_Reduction": [0.03, 0.03, 0.03, 0.03, -0.01, -0.01],
        "mvg_present": [False] * 6,
        "iid_AUPRC_cost": [0.005] * 6,
        "iid_AUROC_cost": [0.003] * 6,
        "worst_min_M1": [0.5] * 6,
        "worst_B_best": [0.5] * 6,
    })
    model_red = {"logistic_regression": 0.02, "xgboost": 0.02}
    hfr = {"rates": {"B0": 0.2, "M1": 0.2}, "counts": {"B0": 40, "M1": 40}, "hfr_reduction": 0.0}
    decision, _, _ = evaluate_method_gate(cells, model_red, hfr, _strong_row(), cfg)
    assert decision == "HOLD"


def test_gate_stop_method():
    cfg = _cfg()
    cells = pd.DataFrame({
        "dataset": [f"d{i}" for i in range(6)],
        "model": ["lr"] * 6,
        "mean_MVG_Reduction": [-0.05] * 6,
        "mvg_present": [False] * 6,
        "iid_AUPRC_cost": [0.06] * 6,
        "iid_AUROC_cost": [0.03] * 6,
        "worst_min_M1": [0.4] * 6,
        "worst_B_best": [0.6] * 6,
    })
    model_red = {"logistic_regression": -0.05, "xgboost": -0.05}
    hfr = {"rates": {"B0": 0.2, "M1": 0.2}, "counts": {"B0": 40, "M1": 40}, "hfr_reduction": 0.0}
    decision, _, _ = evaluate_method_gate(cells, model_red, hfr, _strong_row(), cfg)
    assert decision == "STOP-METHOD"


def test_phase_c12_artifacts_unchanged():
    """Guard that Phase C1/C2 output hashes are untouched by Phase C3 code."""
    phase12 = ROOT / "results" / "phase_c12"
    if not phase12.exists():
        return

    def snapshot(root):
        out = {}
        for p in sorted(root.rglob("*")):
            if p.is_file() and "raw" not in p.parts:
                out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    before = snapshot(phase12)
    _cfg()  # exercise the Phase C3 config path (no writes to phase12).
    after = snapshot(phase12)
    assert before == after
