"""Post-run analysis: summaries, statistics, figures, paper-gate inputs, report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .statistical_analysis import (
    bootstrap_ci,
    wilcoxon_paired,
    rank_biserial,
    bh_correction,
    spearman_corr,
)
from .paper_gate import evaluate_paper_gate
from .sensitivity_analysis import sensitivity_summary


ENV_LABELS = {
    "full": "Full",
    "random": "Random",
    "global_importance": "Global-important",
    "minority_specific": "Minority-specific",
    "majority_specific": "Majority-specific",
}


def recompute_slopes(perf: pd.DataFrame) -> pd.DataFrame:
    """Rebuild slope results from stored performance frames (corrected slope fit)."""
    full = perf[perf["shift_type"].eq("full")][
        ["dataset", "model", "seed", "minority_recall", "majority_recall"]
    ].rename(columns={"minority_recall": "full_min", "majority_recall": "full_maj"})
    per = perf[~perf["shift_type"].eq("full")][
        ["dataset", "model", "seed", "shift_type", "removal_fraction", "minority_recall", "majority_recall"]
    ]
    merged = per.merge(full, on=["dataset", "model", "seed"])
    rows = []
    for (ds, model, seed, st), g in merged.groupby(["dataset", "model", "seed", "shift_type"]):
        rates = [0.0]
        mins = [g["full_min"].iloc[0]]
        majs = [g["full_maj"].iloc[0]]
        for _, r in g.sort_values("removal_fraction").iterrows():
            rates.append(float(r["removal_fraction"]))
            mins.append(float(r["minority_recall"]))
            majs.append(float(r["majority_recall"]))
        ss = sensitivity_summary(rates, mins, majs, 1e-12)
        rows.append({"dataset": ds, "model": model, "seed": seed, "shift_type": st, **ss})
    return pd.DataFrame(rows)


def dataset_summary(vul, slope, class_control, hidden, structured, obs_classes) -> pd.DataFrame:
    rows = []
    for ds in sorted(vul["dataset"].unique()):
        d = vul[vul["dataset"].eq(ds)]
        mean_mvg = d["MVG_recall"].mean()
        slope_d = slope[slope["dataset"].eq(ds)]["slope_gap"].mean()
        # stable hidden cells (>=2/3 seeds) per (model, shift_type, severity)
        hd = hidden[hidden["dataset"].eq(ds)]
        stable = 0
        if not hd.empty:
            for (model, st, sev), g in hd.groupby(["model", "shift_type", "severity"]):
                if g["hidden_auroc"].sum() >= 2:
                    stable += 1
        class_d = class_control[class_control["dataset"].eq(ds)]
        delta = class_d["DeltaMVG_class"].mean() if not class_d.empty else np.nan
        struct_d = structured[structured["dataset"].eq(ds)]
        gp = struct_d[struct_d["mechanism"].eq("group_loss_penalty")]["StructuredPenalty"].mean() if not struct_d.empty else np.nan
        cp = struct_d[struct_d["mechanism"].eq("correlated_loss_penalty")]["StructuredPenalty"].mean() if not struct_d.empty else np.nan
        obs_class = obs_classes.get(ds, "unknown")
        is_floor = obs_class == "floor_risk"
        rows.append(
            {
                "dataset": ds,
                "mean_MVG_recall": round(float(mean_mvg), 6) if not is_floor else np.nan,
                "mean_MVG_f1": round(float(d["MVG_f1"].mean()), 6) if not is_floor else np.nan,
                "slope_gap": round(float(slope_d), 6) if not np.isnan(slope_d) else np.nan,
                "hidden_failure_count": stable,
                "DeltaMVG_class": round(float(delta), 6) if not pd.isna(delta) and not is_floor else np.nan,
                "group_penalty": round(float(gp), 6) if not pd.isna(gp) and not is_floor else np.nan,
                "corr_penalty": round(float(cp), 6) if not pd.isna(cp) and not is_floor else np.nan,
                "observability_class": obs_class,
            }
        )
    return pd.DataFrame(rows)


def model_summary(vul, obs_classes) -> pd.DataFrame:
    rows = []
    observable = {ds for ds, c in obs_classes.items() if c != "floor_risk"}
    for model in sorted(vul["model"].unique()):
        d = vul[vul["model"].eq(model)]
        d_obs = d[d["dataset"].isin(observable)]
        vals = d_obs["MVG_recall"].dropna()
        ci = (
            bootstrap_ci(vals.values, 2000, 0.95)
            if len(vals)
            else {"mean": np.nan, "median": np.nan, "CI_low": np.nan, "CI_high": np.nan}
        )
        rows.append(
            {
                "model": model,
                "mean_MVG": round(float(vals.mean()), 6) if len(vals) else np.nan,
                "median_MVG": round(float(np.median(vals)), 6) if len(vals) else np.nan,
                "positive_fraction": round(float((vals > 0).mean()), 4) if len(vals) else np.nan,
                "CI_low": round(ci["CI_low"], 6),
                "CI_high": round(ci["CI_high"], 6),
            }
        )
    return pd.DataFrame(rows)


def mechanism_summary(vul, obs_classes) -> pd.DataFrame:
    observable = {ds for ds, c in obs_classes.items() if c != "floor_risk"}
    rows = []
    for st in ["random", "global_importance", "minority_specific", "majority_specific"]:
        d = vul[vul["shift_type"].eq(st) & vul["dataset"].isin(observable)]["MVG_recall"].dropna()
        rows.append(
            {
                "shift_type": st,
                "mean_MVG_recall": round(float(d.mean()), 6) if len(d) else np.nan,
                "median_MVG_recall": round(float(d.median()), 6) if len(d) else np.nan,
                "positive_fraction": round(float((d > 0).mean()), 4) if len(d) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_statistics(vul, slope, class_control, hidden, structured, delayed, obs_classes, config) -> Tuple[pd.DataFrame, dict]:
    cfg = config["statistics"]
    observable = {ds for ds, c in obs_classes.items() if c != "floor_risk"}
    rows = []
    p_values = []

    def add_row(comparison, mechanism, severity, a, b, cohort):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        diff = a - b
        ci = bootstrap_ci(diff, cfg["bootstrap_iterations"], cfg["confidence_level"], seed=len(rows))
        w = wilcoxon_paired(a, b)
        es = rank_biserial(a, b)
        p_values.append((w["p_value"], len(rows)))
        rows.append(
            {
                "comparison": comparison,
                "mechanism": mechanism,
                "severity": severity,
                "cohort": cohort,
                "n": int(a.size),
                "mean_effect": round(float(diff.mean()), 6),
                "median_effect": round(float(np.median(diff)), 6),
                "CI_low": round(ci["CI_low"], 6),
                "CI_high": round(ci["CI_high"], 6),
                "wilcoxon_p": w["p_value"],
                "effect_size": round(es, 6),
            }
        )

    # Core: per shift_type x severity for observable datasets.
    core = vul[vul["dataset"].isin(observable)].dropna(subset=["MRD_recall", "MaRD_recall"])
    for st in ["random", "global_importance", "minority_specific", "majority_specific"]:
        for sev in sorted(core["severity"].unique()):
            d = core[(core["shift_type"].eq(st)) & (core["severity"].eq(sev))].dropna(subset=["MRD_recall", "MaRD_recall"])
            if d.empty:
                continue
            add_row("MRD_vs_MaRD", st, sev, d["MRD_recall"].values, d["MaRD_recall"].values, "observable")
    # Pooled core (observable).
    add_row("MRD_vs_MaRD", "pooled_core", "all", core["MRD_recall"].values, core["MaRD_recall"].values, "observable")
    # Pooled MVG/f1 CI over observable core.
    core_mvg = core["MVG_recall"].dropna()
    mvg_ci = bootstrap_ci(core_mvg.values, cfg["bootstrap_iterations"], cfg["confidence_level"], seed=1)
    rows.append(
        {
            "comparison": "MVG_recall",
            "mechanism": "pooled_core",
            "severity": "all",
            "cohort": "observable",
            "n": int(core.shape[0]),
            "mean_effect": round(float(core_mvg.mean()), 6),
            "median_effect": round(float(core_mvg.median()), 6),
            "CI_low": round(mvg_ci["CI_low"], 6),
            "CI_high": round(mvg_ci["CI_high"], 6),
            "wilcoxon_p": np.nan,
            "effect_size": np.nan,
        }
    )
    core_f1 = core["MVG_f1"].dropna()
    add_row("MVG_f1", "pooled_core", "all", core_f1.values, np.zeros(len(core_f1)), "observable")
    # SlopeGap, CSRC, HFS.
    add_row("SlopeGap", "pooled", "all", slope[slope["dataset"].isin(observable)]["slope_gap"].values, np.zeros(len(slope[slope["dataset"].isin(observable)])), "observable")
    add_row("CSRC", "pooled", "all", class_control[class_control["dataset"].isin(observable)]["DeltaMVG_class"].values, np.zeros(len(class_control[class_control["dataset"].isin(observable)])), "observable")
    hfs = hidden[hidden["dataset"].isin(observable)]["HFS"].values
    add_row("HFS", "pooled", "all", hfs, np.zeros(len(hfs)), "observable")
    # Structured per mechanism.
    for mech in ["group_loss", "correlated_loss", "matched_random"]:
        d = structured[structured["mechanism"].eq(mech)].dropna(subset=["MRD_minority", "MVG"])
        if d.empty:
            continue
        a = d["MRD_minority"].values
        b = (d["MRD_minority"] - d["MVG"]).values  # MaRD derived
        add_row("MRD_vs_MaRD", f"structured_{mech}", "all", a, b, "all")
    # Delayed recovery gap.
    if not delayed.empty:
        add_row("RecoveryGap", "delayed", "all", delayed["RecoveryGap"].values, np.zeros(len(delayed)), "all")

    stats = pd.DataFrame(rows)
    # FDR across all wilcoxon p-values (aligned by row index to preserve order).
    p_idx = [i for i, r in enumerate(rows) if not pd.isna(r["wilcoxon_p"])]
    pvals = [rows[i]["wilcoxon_p"] for i in p_idx]
    bh = bh_correction(pvals, cfg["fdr_alpha"])
    fdr_vals = [np.nan] * len(rows)
    for pos, i in enumerate(p_idx):
        fdr_vals[i] = bh["adjusted"][pos]
    stats["FDR_p"] = fdr_vals
    stats["significant"] = [bool(v < cfg["fdr_alpha"]) for v in stats["FDR_p"]]

    # Key stats for the paper gate.
    pooled_row = stats[(stats["comparison"] == "MRD_vs_MaRD") & (stats["mechanism"] == "pooled_core")]
    mvg_row = stats[(stats["comparison"] == "MVG_recall") & (stats["mechanism"] == "pooled_core")]
    key = {
        "wilcoxon_p": float(pooled_row["wilcoxon_p"].iloc[0]) if not pooled_row.empty else np.nan,
        "wilcoxon_fdr_p": float(pooled_row["FDR_p"].iloc[0]) if not pooled_row.empty else np.nan,
        "effect_size": float(pooled_row["effect_size"].iloc[0]) if not pooled_row.empty else np.nan,
        "mvg_mean": float(mvg_row["mean_effect"].iloc[0]) if not mvg_row.empty else np.nan,
        "mvg_ci_low": float(mvg_row["CI_low"].iloc[0]) if not mvg_row.empty else np.nan,
    }
    return stats, key


def build_gate_summary(ds_sum, model_sum, mechanism_sum, key, obs_classes) -> dict:
    dataset_mvg = {
        r["dataset"]: {
            "mean_MVG_recall": r["mean_MVG_recall"],
            "observability_class": r["observability_class"],
            "imbalance_ratio": np.nan,
            "BMO": np.nan,
            "slope_gap": r["slope_gap"],
            "hidden_failure_count": r["hidden_failure_count"],
            "DeltaMVG_class": r["DeltaMVG_class"],
            "group_penalty": r["group_penalty"],
            "corr_penalty": r["corr_penalty"],
        }
        for _, r in ds_sum.iterrows()
    }
    model_mvg = {r["model"]: r["mean_MVG"] for _, r in model_sum.iterrows()}
    return {
        "dataset_mvg": dataset_mvg,
        "model_mvg": model_mvg,
        "wilcoxon": {"p_value": key["wilcoxon_p"]},
        "wilcoxon_fdr_p": key["wilcoxon_fdr_p"],
        "effect_size": key["effect_size"],
        "mvg_ci_low": key["mvg_ci_low"],
    }


def _style():
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams["figure.dpi"] = 110


def make_figures(perf, vul, slope, class_control, hidden, structured, delayed, imp, meta, out_dir: Path) -> List[str]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []

    # Fig 1/2: minority/majority recall vs removal rate (per dataset)
    for metric, fname in [("minority_recall", "figure1_minority_recall_vs_rate.png"), ("majority_recall", "figure2_majority_recall_vs_rate.png")]:
        ag = perf.groupby(["dataset", "shift_type", "removal_fraction"], as_index=False)[metric].mean()
        ag = ag[ag["shift_type"].isin(["random", "global_importance", "minority_specific", "majority_specific"])]
        datasets = ag["dataset"].unique()
        cols = min(4, len(datasets))
        rows = int(np.ceil(len(datasets) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
        for ax, ds in zip(axes.ravel(), datasets):
            sub = ag[ag["dataset"].eq(ds)]
            for st in ["random", "global_importance", "minority_specific", "majority_specific"]:
                s = sub[sub["shift_type"].eq(st)].sort_values("removal_fraction")
                ax.plot(s["removal_fraction"] * 100, s[metric], marker="o", label=ENV_LABELS.get(st, st))
            ax.set_title(ds)
            ax.set_xlabel("Removal rate (%)")
            ax.set_ylabel(f"Mean {metric}")
        for ax in axes.ravel()[len(datasets):]:
            ax.axis("off")
        fig.suptitle(f"Figure 1/2: {metric} vs removal rate")
        fig.tight_layout()
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        names.append(fname)

    # Fig 3: minority vs majority sensitivity slope
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ag = slope.groupby(["dataset", "model"], as_index=False)[["minority_slope", "majority_slope"]].mean()
    ax.scatter(ag["majority_slope"], ag["minority_slope"], s=45, c="#4C72B0")
    lims = [min(ag["majority_slope"].min(), ag["minority_slope"].min()), max(ag["majority_slope"].max(), ag["minority_slope"].max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Majority slope (abs)")
    ax.set_ylabel("Minority slope (abs)")
    ax.set_title("Figure 3: minority vs majority sensitivity slope")
    fig.tight_layout(); fig.savefig(out_dir / "figure3_slope_scatter.png", bbox_inches="tight"); plt.close(fig); names.append("figure3_slope_scatter.png")

    # Fig 4: MVG vs removal rate
    fig, ax = plt.subplots(figsize=(7, 5))
    ag = vul.groupby(["dataset", "severity"], as_index=False)["MVG_recall"].mean()
    for ds in ag["dataset"].unique():
        s = ag[ag["dataset"].eq(ds)].sort_values("severity")
        ax.plot(s["severity"] * 100, s["MVG_recall"], marker="o", label=ds)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("Removal rate (%)"); ax.set_ylabel("Mean MVG_recall")
    ax.set_title("Figure 4: MVG vs removal rate"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / "figure4_mvg_vs_rate.png", bbox_inches="tight"); plt.close(fig); names.append("figure4_mvg_vs_rate.png")

    # Fig 5: MVG heatmap (dataset x shift_type)
    hm = vul.groupby(["dataset", "shift_type"], as_index=False)["MVG_recall"].mean().pivot(index="dataset", columns="shift_type", values="MVG_recall")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(hm, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Figure 5: MVG heatmap"); fig.tight_layout(); fig.savefig(out_dir / "figure5_heatmap.png", bbox_inches="tight"); plt.close(fig); names.append("figure5_heatmap.png")

    # Fig 6: hidden failure
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(hidden["AUROC_drop"], hidden["minority_recall_drop"], s=25, alpha=0.7)
    ax.axvline(0.03, color="crimson", ls="--"); ax.axhline(0.10, color="crimson", ls="--")
    ymax = max(float(hidden["minority_recall_drop"].max()), 0.15)
    ax.fill_betweenx([0.10, ymax], -0.2, 0.03, color="red", alpha=0.08)
    ax.set_xlabel("AUROC drop"); ax.set_ylabel("Minority recall drop"); ax.set_title("Figure 6: hidden minority failure")
    fig.tight_layout(); fig.savefig(out_dir / "figure6_hidden_failure.png", bbox_inches="tight"); plt.close(fig); names.append("figure6_hidden_failure.png")

    # Fig 7: minority vs majority specific DeltaMVG_class
    fig, ax = plt.subplots(figsize=(7, 5))
    ag = class_control.groupby(["dataset", "severity"], as_index=False)["DeltaMVG_class"].mean()
    for ds in ag["dataset"].unique():
        s = ag[ag["dataset"].eq(ds)].sort_values("severity")
        ax.plot(s["severity"] * 100, s["DeltaMVG_class"], marker="o", label=ds)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("Removal rate (%)"); ax.set_ylabel("DeltaMVG_class")
    ax.set_title("Figure 7: minority-specific vs majority-specific removal"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / "figure7_class_control.png", bbox_inches="tight"); plt.close(fig); names.append("figure7_class_control.png")

    # Fig 8: global vs minority importance
    ag = imp.groupby(["dataset", "feature"], as_index=False)[["global_importance", "minority_importance", "MSI"]].mean()
    datasets = ag["dataset"].unique(); cols = min(4, len(datasets)); rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for ax, ds in zip(axes.ravel(), datasets):
        sub = ag[ag["dataset"].eq(ds)]
        hi = sub.nlargest(3, "MSI")
        ax.scatter(sub["global_importance"], sub["minority_importance"], s=25, alpha=0.7)
        ax.scatter(hi["global_importance"], hi["minority_importance"], s=60, c="crimson", edgecolor="k")
        ax.axline((0, 0), slope=1, color="grey", ls="--", lw=0.8)
        ax.set_title(ds); ax.set_xlabel("Global importance"); ax.set_ylabel("Minority importance")
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    fig.suptitle("Figure 8: global vs minority importance"); fig.tight_layout(); fig.savefig(out_dir / "figure8_importance.png", bbox_inches="tight"); plt.close(fig); names.append("figure8_importance.png")

    # Fig 9: structured vs matched random
    fig, ax = plt.subplots(figsize=(7, 5))
    pen = structured[structured["mechanism"].isin(["group_loss_penalty", "correlated_loss_penalty"])]
    if not pen.empty:
        sns.barplot(data=pen, x="dataset", y="StructuredPenalty", hue="mechanism", ax=ax)
        ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Figure 9: structured loss penalty vs matched random"); fig.tight_layout(); fig.savefig(out_dir / "figure9_structured.png", bbox_inches="tight"); plt.close(fig); names.append("figure9_structured.png")

    # Fig 10: delayed recovery curves
    fig, ax = plt.subplots(figsize=(7, 5))
    if not delayed.empty:
        for ds in delayed["dataset"].unique():
            s = delayed[delayed["dataset"].eq(ds)].sort_values("availability_fraction")
            ax.plot(s["availability_fraction"] * 100, s["minority_recall"], marker="o", label=f"{ds} min")
    ax.set_xlabel("Feature availability (%)"); ax.set_ylabel("Minority recall"); ax.set_title("Figure 10: delayed recovery curves"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / "figure10_delayed.png", bbox_inches="tight"); plt.close(fig); names.append("figure10_delayed.png")

    # Fig 11: imbalance vs MVG, Fig 12: BMO vs MVG (uses real metadata).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes = np.atleast_1d(axes)
    ds_mvg = vul.groupby("dataset", as_index=False)["MVG_recall"].mean()
    imp_vals = [meta.get("imbalance_ratio", {}).get(ds, np.nan) for ds in ds_mvg["dataset"]]
    bmo_vals = [meta.get("BMO", {}).get(ds, np.nan) for ds in ds_mvg["dataset"]]
    axes[0].scatter(np.log(np.asarray(imp_vals, dtype=float)), ds_mvg["MVG_recall"], s=60, c="#4C72B0")
    axes[0].set_xlabel("log10 imbalance ratio"); axes[0].set_ylabel("Mean MVG"); axes[0].set_title("Figure 11: imbalance ratio vs MVG")
    axes[1].scatter(np.asarray(bmo_vals, dtype=float), ds_mvg["MVG_recall"], s=60, c="#DD8452")
    axes[1].set_xlabel("Baseline Minority Observability (BMO)"); axes[1].set_ylabel("Mean MVG"); axes[1].set_title("Figure 12: BMO vs MVG")
    fig.tight_layout(); fig.savefig(out_dir / "figure11_12_imbalance_bmo.png", bbox_inches="tight"); plt.close(fig); names.append("figure11_12_imbalance_bmo.png")
    return names
