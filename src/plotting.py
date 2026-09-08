"""Generate the five Gate A figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .gate_decision import SHIFT_ENVS


ENV_LABELS = {
    "full": "Full",
    "random": "Random",
    "global_importance": "Global-important",
    "minority_specific": "Minority-specific",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams["figure.dpi"] = 110


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / name, bbox_inches="tight")
    plt.close(fig)


def figure1_minority_recall(perf: pd.DataFrame, out_dir: Path) -> None:
    _style()
    ag = perf.groupby(["dataset", "model", "environment"], as_index=False)["minority_recall"].mean()
    ag["env_label"] = ag["environment"].map(ENV_LABELS)
    datasets = ag["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = ag[ag["dataset"].eq(ds)]
        sns.barplot(
            data=sub, x="env_label", y="minority_recall", hue="model", ax=ax,
            order=list(ENV_LABELS.values()), palette=["#4C72B0", "#DD8452"],
        )
        ax.set_title(ds)
        ax.set_xlabel("Feature availability environment")
        ax.set_ylabel("Minority recall (mean over seeds)")
        ax.legend(title="model")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 1: Minority recall under feature availability shift", y=1.03)
    _save(fig, out_dir, "figure1_minority_recall.png")


def figure2_mvg(evidence: pd.DataFrame, out_dir: Path) -> None:
    _style()
    long = evidence.melt(
        id_vars=["dataset", "model"],
        value_vars=["mean_MVG_recall", "mean_MVG_f1"],
        var_name="metric",
        value_name="MVG",
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=long, x="dataset", y="MVG", hue="metric", ax=ax, palette=["#4C72B0", "#94C47D"])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.10, color="crimson", ls="--", lw=1)
    ax.axhline(0.05, color="orange", ls=":", lw=1)
    ax.set_title("Figure 2: Minority Vulnerability Gap by dataset/model")
    ax.set_ylabel("MVG (mean over seeds)")
    _save(fig, out_dir, "figure2_mvg.png")


def figure3_importance(imp: pd.DataFrame, out_dir: Path) -> None:
    _style()
    ag = imp.groupby(["dataset", "feature"], as_index=False)[
        ["global_importance", "minority_importance", "MSI"]
    ].mean()
    datasets = ag["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = ag[ag["dataset"].eq(ds)].copy()
        hi = sub.nlargest(5, "MSI")
        scatter = ax.scatter(
            sub["global_importance"], sub["minority_importance"], s=45, c="#4C72B0", alpha=0.8
        )
        ax.scatter(
            hi["global_importance"], hi["minority_importance"], s=70, c="crimson",
            edgecolor="black", label="top-5 MSI",
        )
        for _, r in hi.iterrows():
            ax.annotate(r["feature"], (r["global_importance"], r["minority_importance"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axline((0, 0), slope=1, color="grey", ls="--", lw=0.8)
        ax.set_title(ds)
        ax.set_xlabel("Global permutation importance (AUPRC)")
        ax.set_ylabel("Minority-specific importance")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 3: Global vs minority-specific feature importance", y=1.03)
    _save(fig, out_dir, "figure3_importance.png")


def figure4_hidden_failure(perf: pd.DataFrame, out_dir: Path) -> None:
    _style()
    rows = []
    for (ds, model, seed), g in perf.groupby(["dataset", "model", "seed"]):
        full = g[g["environment"].eq("full")].iloc[0]
        for env in SHIFT_ENVS:
            s = g[g["environment"].eq(env)]
            if s.empty:
                continue
            s = s.iloc[0]
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "seed": seed,
                    "environment": env,
                    "auroc_drop": full["AUROC"] - s["AUROC"],
                    "recall_drop": full["minority_recall"] - s["minority_recall"],
                }
            )
    scatter = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        scatter["auroc_drop"], scatter["recall_drop"], s=28,
        c=scatter["seed"].map({42: "#4C72B0", 52: "#55A868", 62: "#C44E52"}), alpha=0.8,
    )
    ax.set_xlabel("AUROC drop (full - shifted)")
    ax.set_ylabel("Minority recall drop (full - shifted)")
    ax.set_title("Figure 4: AUROC drop vs minority recall drop")
    # Hidden failure region: auroc_drop <= 0.03 and recall_drop >= 0.10
    ax.axvline(0.03, color="crimson", ls="--", lw=1)
    ax.axhline(0.10, color="crimson", ls="--", lw=1)
    ymax = max(float(scatter["recall_drop"].max()), 0.15)
    ax.fill_betweenx([0.10, ymax], -0.2, 0.03, color="red", alpha=0.08)
    ax.text(0.0, 0.12, "hidden minority failure region", color="crimson", fontsize=8, ha="center")
    _save(fig, out_dir, "figure4_hidden_failure.png")


def figure5_rankings(imp: pd.DataFrame, out_dir: Path) -> None:
    _style()
    ag = imp.groupby(["dataset", "feature"], as_index=False)["MSI"].mean()
    datasets = ag["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = ag[ag["dataset"].eq(ds)].copy()
        sub["msi_rank"] = sub["MSI"].rank(ascending=False, method="min").astype(int)
        top = sub.sort_values("msi_rank").head(10)
        ax.barh(top["feature"], top["msi_rank"], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_title(ds)
        ax.set_xlabel("MSI rank (1 = highest)")
    fig.suptitle("Figure 5: Top-10 features by Minority Specificity Index", y=1.03)
    _save(fig, out_dir, "figure5_ranking.png")


def make_all_figures(perf: pd.DataFrame, imp: pd.DataFrame, evidence: pd.DataFrame, out_dir: Path) -> List[str]:
    figure1_minority_recall(perf, out_dir)
    figure2_mvg(evidence, out_dir)
    figure3_importance(imp, out_dir)
    figure4_hidden_failure(perf, out_dir)
    figure5_rankings(imp, out_dir)
    return [
        "figure1_minority_recall.png",
        "figure2_mvg.png",
        "figure3_importance.png",
        "figure4_hidden_failure.png",
        "figure5_ranking.png",
    ]
