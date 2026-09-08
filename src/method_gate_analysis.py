"""Phase C3 method-gate aggregation, statistics, figures and decision."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .statistical_analysis import bootstrap_ci, wilcoxon_paired, rank_biserial


SHIFTED_ENVS = [
    "random_10",
    "global_importance_10",
    "minority_specific_10",
    "majority_specific_10",
    "global_importance_20",
    "minority_specific_20",
]
CORE_MG2_ENVS = ["global_importance_10", "minority_specific_10", "global_importance_20", "minority_specific_20"]
NON_MAFR_METHODS = ["B0", "B1", "B2", "B3", "B4"]


def _shifted(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["shift_env"].ne("full")].copy()


def _full(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["shift_env"].eq("full")].copy()


def cell_aggregates(results: pd.DataFrame, mg2_min_recovery: float = 0.05) -> pd.DataFrame:
    """Per (dataset, model) cell aggregates over seeds and shifted environments."""
    sh = _shifted(results)
    full = _full(results)
    rows = []
    for (ds, model), cell in sh.groupby(["dataset", "model"]):
        seed_list = sorted(cell["seed"].unique())
        # MVG reduction = B0 - M1 per (seed, shift).
        b0 = cell[cell["method"].eq("B0")]
        m1 = cell[cell["method"].eq("M1")]
        reds = []
        for seed in seed_list:
            for env in SHIFTED_ENVS:
                b0v = b0[(b0["seed"].eq(seed)) & (b0["shift_env"].eq(env))]["MVG_recall"].values
                m1v = m1[(m1["seed"].eq(seed)) & (m1["shift_env"].eq(env))]["MVG_recall"].values
                if b0v.size and m1v.size:
                    reds.append(float(b0v[0] - m1v[0]))
        mean_red = float(np.mean(reds)) if reds else np.nan

        # MR recovery metrics for MG2.
        recoveries = []
        for seed in seed_list:
            for env in CORE_MG2_ENVS:
                b0v = b0[(b0["seed"].eq(seed)) & (b0["shift_env"].eq(env))]["minority_recall"].values
                m1v = m1[(m1["seed"].eq(seed)) & (m1["shift_env"].eq(env))]["minority_recall"].values
                if b0v.size and m1v.size:
                    recoveries.append(float(m1v[0] - b0v[0]))
        mvg_present = bool(np.any(np.asarray(recoveries) >= mg2_min_recovery)) if recoveries else False

        # Worst-case minority recall per method.
        worst: Dict[str, float] = {}
        for method in ["B0", "B1", "B2", "B3", "B4", "M1"]:
            vals = cell[cell["method"].eq(method)]["minority_recall"].values
            worst[method] = float(np.min(vals)) if vals.size else np.nan
        worst_B_best = float(max(worst["B1"], worst["B2"], worst["B3"]))

        # IID costs.
        fcell = full[(full["dataset"].eq(ds)) & (full["model"].eq(model))]
        b0_full = fcell[fcell["method"].eq("B0")]
        m1_full = fcell[fcell["method"].eq("M1")]
        iid_auprc_cost = float((b0_full["AUPRC"].mean() - m1_full["AUPRC"].mean())) if not b0_full.empty and not m1_full.empty else np.nan
        iid_auroc_cost = float((b0_full["AUROC"].mean() - m1_full["AUROC"].mean())) if not b0_full.empty and not m1_full.empty else np.nan
        iid_auprc_m1 = float(m1_full["AUPRC"].mean()) if not m1_full.empty else np.nan
        iid_auprc_b0 = float(b0_full["AUPRC"].mean()) if not b0_full.empty else np.nan
        iid_auroc_m1 = float(m1_full["AUROC"].mean()) if not m1_full.empty else np.nan
        iid_auroc_b0 = float(b0_full["AUROC"].mean()) if not b0_full.empty else np.nan

        rows.append(
            {
                "dataset": ds,
                "model": model,
                "n_seeds": len(seed_list),
                "mean_MVG_Reduction": round(mean_red, 6) if not np.isnan(mean_red) else np.nan,
                "mean_MVG_B0": round(float(cell[cell["method"].eq("B0")]["MVG_recall"].mean()), 6),
                "mean_MVG_M1": round(float(cell[cell["method"].eq("M1")]["MVG_recall"].mean()), 6),
                "mvg_present": mvg_present,
                "worst_min_B0": worst["B0"],
                "worst_min_B1": worst["B1"],
                "worst_min_B2": worst["B2"],
                "worst_min_B3": worst["B3"],
                "worst_min_B4": worst["B4"],
                "worst_min_M1": worst["M1"],
                "worst_B_best": worst_B_best,
                "iid_AUPRC_cost": round(iid_auprc_cost, 6) if not np.isnan(iid_auprc_cost) else np.nan,
                "iid_AUROC_cost": round(iid_auroc_cost, 6) if not np.isnan(iid_auroc_cost) else np.nan,
                "iid_AUPRC_M1": round(iid_auprc_m1, 6) if not np.isnan(iid_auprc_m1) else np.nan,
                "iid_AUPRC_B0": round(iid_auprc_b0, 6) if not np.isnan(iid_auprc_b0) else np.nan,
                "iid_AUROC_M1": round(iid_auroc_m1, 6) if not np.isnan(iid_auroc_m1) else np.nan,
                "iid_AUROC_B0": round(iid_auroc_b0, 6) if not np.isnan(iid_auroc_b0) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def model_mvg_reduction(cells: pd.DataFrame, config: dict) -> Dict[str, float]:
    out = {}
    for model in config["models"]:
        c = cells[cells["model"].eq(model)]["mean_MVG_Reduction"].dropna()
        out[model] = float(c.mean()) if len(c) else np.nan
    return out


def hidden_failure_summary(results: pd.DataFrame, config: dict) -> Dict[str, object]:
    sh = _shifted(results)
    rates = {}
    counts = {}
    for method in ["B0", "B1", "B2", "B3", "B4", "M1"]:
        d = sh[sh["method"].eq(method)]
        counts[method] = int(d["hidden_auroc"].sum()) if not d.empty else 0
        rates[method] = float(d["hidden_auroc"].mean()) if not d.empty else 0.0
    eps = config["epsilon"]
    hfr_reduction = 0.0
    if rates["B0"] > eps:
        hfr_reduction = 1.0 - rates["M1"] / (rates["B0"] + eps)
    return {"rates": rates, "counts": counts, "hfr_reduction": float(hfr_reduction)}


def block_statistics(results: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Paired statistics at the dataset x model x seed block level (anti-pseudo-replication)."""
    sh = _shifted(results)
    agg = sh.groupby(["dataset", "model", "seed", "method"])["MVG_recall"].mean().unstack("method")
    blocks = list(agg.index)
    methods = [m for m in ["B0", "B1", "B2", "B3"] if m in agg.columns]
    mean_by_method = {m: float(agg[m].mean()) for m in methods}
    strongest = min(methods, key=lambda m: mean_by_method[m])

    rows = []
    m1 = agg["M1"].reindex(blocks).to_numpy(dtype=float)
    for m in methods:
        base = agg[m].reindex(blocks).to_numpy(dtype=float)
        diff = base - m1
        ci = bootstrap_ci(diff, config["statistics"]["bootstrap_iterations"], config["statistics"]["confidence_level"], seed=len(rows))
        es = rank_biserial(base, m1)
        w = wilcoxon_paired(base, m1)
        rows.append(
            {
                "baseline": m,
                "comparison": f"{m}_vs_M1",
                "n_blocks": int(len(blocks)),
                "mean_MVG_baseline": round(float(base.mean()), 6),
                "mean_MVG_M1": round(float(m1.mean()), 6),
                "mean_effect": round(float(diff.mean()), 6),
                "median_effect": round(float(np.median(diff)), 6),
                "CI_low": round(ci["CI_low"], 6),
                "CI_high": round(ci["CI_high"], 6),
                "wilcoxon_p": w["p_value"],
                "effect_size": round(es, 6),
                "strongest": bool(m == strongest),
            }
        )
    strongest_row = [r for r in rows if r["strongest"]][0]
    return pd.DataFrame(rows), strongest_row


def evaluate_method_gate(cells: pd.DataFrame, model_red: Dict[str, float], hfr: Dict[str, object],
                         strongest_row: dict, config: dict) -> Tuple[str, Dict[str, bool], dict]:
    mg = config["method_gate"]
    cells_f = cells[cells["mean_MVG_Reduction"].notna()].copy()
    n_cells = len(cells)
    conds: Dict[str, bool] = {}

    cells_mg1 = int((cells_f["mean_MVG_Reduction"] >= mg["mg1_min_reduction"]).sum())
    conds["MG1_minority_robustness"] = cells_mg1 >= mg["mg1_min_cells"]
    cells_mg2 = int(cells_f["mvg_present"].sum())
    conds["MG2_minority_recall"] = cells_mg2 >= mg["mg2_min_cells"]
    conds["MG3_hidden_failure"] = hfr["hfr_reduction"] >= mg["mg3_hfr_reduction_min"]
    mean_auprc_cost = float(cells_f["iid_AUPRC_cost"].mean())
    max_auprc_cost = float(cells_f["iid_AUPRC_cost"].max())
    conds["MG4_iid_auprc"] = (mean_auprc_cost <= mg["mg4_max_mean_auprc_drop"]) and (
        max_auprc_cost <= mg["mg4_max_cell_auprc_drop"]
    )
    mean_auroc_cost = float(cells_f["iid_AUROC_cost"].mean())
    conds["MG5_iid_auroc"] = mean_auroc_cost <= mg["mg5_max_mean_auroc_drop"]
    cells_mg6 = int((cells_f["worst_min_M1"] > cells_f["worst_B_best"]).sum())
    conds["MG6_strong_baseline"] = cells_mg6 >= mg["mg6_min_cells"]
    conds["MG7_cross_model"] = all(v > mg["mg7"] for v in model_red.values())
    ci_low = strongest_row["CI_low"]
    es = strongest_row["effect_size"]
    conds["MG8_statistics"] = (ci_low > 0) and (es >= mg["mg8_min_effect_size"])

    mean_red = float(cells_f["mean_MVG_Reduction"].mean())
    cells_improve = int((cells_f["mean_MVG_Reduction"] > 0).sum())
    mean_cost = mean_auprc_cost
    evidence = {
        "n_cells": n_cells,
        "cells_mg1": cells_mg1,
        "cells_mg2": cells_mg2,
        "cells_mg6": cells_mg6,
        "cells_improve": cells_improve,
        "mean_MVG_Reduction": round(mean_red, 6),
        "mean_iid_AUPRC_cost": round(mean_auprc_cost, 6),
        "max_iid_AUPRC_cost": round(max_auprc_cost, 6),
        "mean_iid_AUROC_cost": round(mean_auroc_cost, 6),
        "hfr_reduction": round(hfr["hfr_reduction"], 6),
        "strongest_baseline": strongest_row["baseline"],
        "ci_low": round(ci_low, 6),
        "effect_size": round(es, 6),
    }

    m1_mg1 = conds["MG1_minority_robustness"]
    m1_mg2 = conds["MG2_minority_recall"]
    m1_mg4 = conds["MG4_iid_auprc"]
    m1_mg6 = conds["MG6_strong_baseline"]
    m1_mg7 = conds["MG7_cross_model"]
    go_core = all([m1_mg1, m1_mg2, m1_mg4, m1_mg6, m1_mg7])
    go_partial = (
        (0.20 <= hfr["hfr_reduction"] < mg["mg3_hfr_reduction_min"])
        or (ci_low > 0 and es < mg["mg8_min_effect_size"])
        or (ci_low > 0 and conds["MG8_statistics"] is False)
    )

    if all(conds.values()):
        decision = "STRONG-GO"
    elif go_core and go_partial:
        decision = "GO"
    elif go_core:
        decision = "GO"
    elif mean_red <= 0 or cells_improve <= 2 or mean_cost > mg["mg4_max_cell_auprc_drop"]:
        decision = "STOP-METHOD"
    elif 3 <= cells_improve <= 4 or mean_cost > mg["mg4_max_mean_auprc_drop"]:
        decision = "HOLD"
    else:
        decision = "HOLD"
    return decision, conds, evidence


def make_figures(results: pd.DataFrame, cells: pd.DataFrame, hfr: Dict[str, object], config: dict, out_dir: Path) -> List[str]:
    sns.set_theme(style="whitegrid", font_scale=0.85)
    plt.rcParams["figure.dpi"] = 110
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    sh = _shifted(results)

    # 1. MVG baseline B0 vs M1.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    agg = sh.groupby(["dataset", "model", "seed", "method"])["MVG_recall"].mean().reset_index()
    bx = agg[agg["method"].eq("B0")]
    mx = agg[agg["method"].eq("M1")]
    ax.scatter(bx["MVG_recall"], mx["MVG_recall"], s=45, c="#4C72B0", alpha=0.8)
    lims = [0, max(float(bx["MVG_recall"].max()), float(mx["MVG_recall"].max()), 0.1)]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("B0 MVG (vulnerability)")
    ax.set_ylabel("M1 MVG")
    ax.set_title("Figure 1: MVG baseline vs MAFR")
    fig.tight_layout(); fig.savefig(out_dir / "figure1_mvg_baseline_vs_mafr.png", bbox_inches="tight"); plt.close(fig); names.append("figure1_mvg_baseline_vs_mafr.png")

    # 2. Minority recall recovery (M1 vs B0 per shifted env).
    fig, ax = plt.subplots(figsize=(7, 5))
    for env in SHIFTED_ENVS:
        d = sh[sh["shift_env"].eq(env)]
        b = d[d["method"].eq("B0")]
        m = d[d["method"].eq("M1")]
        common = set(zip(b["dataset"], b["model"], b["seed"])) & set(zip(m["dataset"], m["model"], m["seed"]))
        pts = [(ds, md, sd) for ds, md, sd in common]
        vals = [float(m[(m["dataset"].eq(ds)) & (m["model"].eq(md)) & (m["seed"].eq(sd))]["minority_recall"].iloc[0]
                     - b[(b["dataset"].eq(ds)) & (b["model"].eq(md)) & (b["seed"].eq(sd))]["minority_recall"].iloc[0])
                for ds, md, sd in pts]
        if vals:
            ax.scatter([env] * len(vals), vals, s=28, alpha=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Shift environment")
    ax.set_ylabel("Minority recall recovery (M1 - B0)")
    ax.set_title("Figure 2: minority recall recovery")
    fig.tight_layout(); fig.savefig(out_dir / "figure2_minority_recall_recovery.png", bbox_inches="tight"); plt.close(fig); names.append("figure2_minority_recall_recovery.png")

    # 3. Hidden failure rate per method.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    methods = list(hfr["rates"].keys())
    rates = [hfr["rates"][m] for m in methods]
    sns.barplot(x=methods, y=rates, ax=ax)
    ax.set_xlabel("Method")
    ax.set_ylabel("Hidden failure rate")
    ax.set_title("Figure 3: hidden failure rate")
    fig.tight_layout(); fig.savefig(out_dir / "figure3_hidden_failure_rate.png", bbox_inches="tight"); plt.close(fig); names.append("figure3_hidden_failure_rate.png")

    # 4. IID AUPRC trade-off (full AUPRC B0 vs M1).
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    full = _full(results)
    fagg = full.groupby(["dataset", "model", "method"])["AUPRC"].mean().reset_index()
    fb = fagg[fagg["method"].eq("B0")]
    fm = fagg[fagg["method"].eq("M1")]
    ax.scatter(fb["AUPRC"], fm["AUPRC"], s=70, c="#DD8452")
    lims = [min(float(fb["AUPRC"].min()), float(fm["AUPRC"].min())), max(float(fb["AUPRC"].max()), float(fm["AUPRC"].max()))]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("B0 full AUPRC")
    ax.set_ylabel("M1 full AUPRC")
    ax.set_title("Figure 4: IID AUPRC trade-off")
    fig.tight_layout(); fig.savefig(out_dir / "figure4_iid_auprc.png", bbox_inches="tight"); plt.close(fig); names.append("figure4_iid_auprc.png")

    # 5. Worst-case minority recall (per method).
    fig, ax = plt.subplots(figsize=(6.5, 5))
    wcols = ["worst_min_B0", "worst_min_B1", "worst_min_B2", "worst_min_B3", "worst_min_B4", "worst_min_M1"]
    wvals = [cells[c].mean() if cells[c].notna().any() else np.nan for c in wcols]
    sns.barplot(x=wcols, y=wvals, ax=ax)
    ax.set_xlabel("Method (worst-case)")
    ax.set_ylabel("Worst-case minority recall")
    ax.set_title("Figure 5: worst-case minority recall")
    fig.tight_layout(); fig.savefig(out_dir / "figure5_worst_case_recall.png", bbox_inches="tight"); plt.close(fig); names.append("figure5_worst_case_recall.png")

    # 6. Per-cell gate heatmap (MVG reduction).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    hm = cells.pivot(index="dataset", columns="model", values="mean_MVG_Reduction")
    sns.heatmap(hm, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Figure 6: per-cell mean MVG reduction")
    fig.tight_layout(); fig.savefig(out_dir / "figure6_cell_gate_heatmap.png", bbox_inches="tight"); plt.close(fig); names.append("figure6_cell_gate_heatmap.png")
    return names
