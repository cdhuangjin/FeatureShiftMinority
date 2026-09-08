"""Paper Gate decision: STRONG-GO / GO / HOLD / STOP-PIVOT."""

from __future__ import annotations

from typing import Dict, List, Tuple


def evaluate_paper_gate(summary: Dict, config: dict) -> Tuple[str, Dict]:
    pg = config["paper_gate"]
    datasets = summary["dataset_mvg"]
    observable = {
        ds: v
        for ds, v in datasets.items()
        if v["observability_class"] != "floor_risk"
    }
    models = summary["model_mvg"]
    strong_mvg = pg["strong_mvg"]

    conds: Dict[str, bool] = {}
    # 1. >= min observable datasets with mean MVG_recall > 0
    conds["c1_positive_mvg_datasets"] = (
        sum(1 for v in observable.values() if v["mean_MVG_recall"] > 0)
        >= pg["min_observable_mvg_positive_datasets"]
    )
    # 2. >= min observable datasets mean MVG_recall >= 0.10
    conds["c2_strong_mvg_datasets"] = (
        sum(1 for v in observable.values() if v["mean_MVG_recall"] >= strong_mvg)
        >= pg["min_observable_mvg_strong_datasets"]
    )
    # 3. >= 3/4 models with positive MVG
    conds["c3_model_consistency"] = (
        sum(1 for v in models.values() if v > 0) >= pg["min_models_positive"]
    )
    # 4. >= 5 datasets with steeper minority slope
    conds["c4_slope"] = (
        sum(1 for v in datasets.values() if v.get("slope_gap", 0) > 0) >= 5
    )
    # 5. core Wilcoxon significant after FDR
    conds["c5_wilcoxon"] = (
        summary.get("wilcoxon_fdr_p", 1.0) < 0.05
        and summary.get("wilcoxon", {}).get("p_value", 1.0) < 0.05
    )
    # 6. effect size moderate or larger
    conds["c6_effect_size"] = abs(summary.get("effect_size", 0.0)) >= 0.3
    # 7. >= 3 datasets hidden failure
    conds["c7_hidden_failure"] = (
        sum(
            1
            for v in datasets.values()
            if v.get("hidden_failure_count", 0) >= pg.get("min_hidden_consistency", 2)
        )
        >= pg["min_hidden_failure_datasets"]
    )
    # 8. >= 4 observable datasets DeltaMVG_class > 0
    conds["c8_class_control"] = (
        sum(1 for v in observable.values() if v.get("DeltaMVG_class", 0) > 0)
        >= pg["min_class_control_datasets"]
    )
    # 9. >= 3 datasets with positive structured penalty
    conds["c9_structured"] = (
        sum(
            1
            for v in datasets.values()
            if v.get("group_penalty", 0) > 0 or v.get("corr_penalty", 0) > 0
        )
        >= pg["min_structured_positive_datasets"]
    )
    # 10. effect > seed noise (bootstrap CI excludes zero)
    conds["c10_seed_noise"] = summary.get("mvg_ci_low", 0.0) > 0.0

    n_met = sum(1 for v in conds.values() if v)
    reasons: List[str] = []
    for k, v in conds.items():
        if v:
            reasons.append(f"{k}: True")

    # Single-dataset dominance detector.
    total_positive = sum(max(v["mean_MVG_recall"], 0) for v in observable.values())
    dominance = 0.0
    if total_positive > 0:
        top_ds = max(observable, key=lambda d: observable[d]["mean_MVG_recall"])
        dominance = max(observable[top_ds]["mean_MVG_recall"], 0) / total_positive

    if n_met >= pg["strong_go_conditions_required"] and conds["c1_positive_mvg_datasets"] and conds["c3_model_consistency"] and conds["c5_wilcoxon"]:
        decision = "STRONG-GO"
    elif conds["c1_positive_mvg_datasets"] and conds["c3_model_consistency"] and conds["c5_wilcoxon"]:
        decision = "GO"
    else:
        n_positive_ds = sum(1 for v in observable.values() if v["mean_MVG_recall"] > 0)
        n_models_positive = sum(1 for v in models.values() if v > 0)
        ci_crosses_zero = summary.get("mvg_ci_low", 0.0) <= 0.0
        if n_positive_ds >= 3:
            decision = "HOLD"
        elif n_models_positive >= 2 and (ci_crosses_zero or dominance > 0.6):
            decision = "HOLD"
        else:
            decision = "STOP-PIVOT"

    evidence = {
        "decision": decision,
        "conditions": conds,
        "n_conditions_met": n_met,
        "conditions_required": pg["strong_go_conditions_required"],
        "n_observable_datasets": len(observable),
        "n_datasets_positive_mvg": sum(1 for v in observable.values() if v["mean_MVG_recall"] > 0),
        "n_datasets_strong_mvg": sum(1 for v in observable.values() if v["mean_MVG_recall"] >= strong_mvg),
        "n_models_positive": sum(1 for v in models.values() if v > 0),
        "dominance_top_dataset": dominance,
        "reasons": reasons,
    }
    return decision, evidence
