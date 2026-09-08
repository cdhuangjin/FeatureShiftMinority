"""Pre-submission Reviewer-Proof checks (Phase C0 supplement).

All checks here are recomputed from the frozen Phase C1/C2 artifacts
``results/phase_c12/*.csv``.  No model is retrained and no Phase C1/C2 artifact is
modified.  Outputs are written only under ``results/reviewer_checks/``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .statistical_analysis import bootstrap_ci, wilcoxon_paired, rank_biserial
from .sensitivity_analysis import sensitivity_summary


EPS = 1.0e-12
CORE_SHIFTS = ["random", "global_importance", "minority_specific", "majority_specific"]
METRICS = [
    "minority_recall",
    "minority_f1",
    "minority_precision",
    "balanced_accuracy",
    "MCC",
    "Gmean",
    "AUPRC",
    "AUROC",
]


def relative_drop(full: float, shift: float, eps: float = EPS) -> float:
    """(full - shift) / max(abs(full), eps). Positive means performance dropped."""
    return float((full - shift) / max(abs(full), eps))


def observable_datasets(bmo: pd.DataFrame) -> set:
    return set(bmo[bmo["observability_class"].ne("floor_risk")]["dataset"])


def full_by_cell(raw: pd.DataFrame) -> pd.DataFrame:
    """Full-feature (shift_type == 'full') metrics indexed by (dataset, model, seed)."""
    f = raw[raw["shift_type"].eq("full")]
    return f.set_index(["dataset", "model", "seed"])


def compute_metric_sensitivity(
    raw: pd.DataFrame, observable: Sequence[str] | None = None, eps: float = EPS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """RC1: per-metric degradation against the full-feature baseline (no retraining).

    Relative degradation is only meaningful where the full-feature minority baseline is
    non-trivial, so observable datasets are used (consistent with the frozen C1/C2 MVG
    scope). Floor-risk datasets have near-zero baselines and are left out.
    """
    if observable is not None:
        raw = raw[raw["dataset"].isin(observable)]
    full = full_by_cell(raw)
    shifted = raw[~raw["shift_type"].eq("full")].copy()
    rows = []
    for _, r in shifted.iterrows():
        fidx = (r["dataset"], r["model"], r["seed"])
        if fidx not in full.index:
            continue
        fm = full.loc[fidx]
        if isinstance(fm, pd.DataFrame):  # defensive; should be a single row
            fm = fm.iloc[0]
        for metric in METRICS:
            full_v = float(fm[metric])
            shift_v = float(r[metric])
            rows.append(
                {
                    "dataset": r["dataset"],
                    "model": r["model"],
                    "seed": r["seed"],
                    "mechanism": r["shift_type"],
                    "severity": r["severity"],
                    "metric": metric,
                    "full_value": round(full_v, 6),
                    "shift_value": round(shift_v, 6),
                    "absolute_drop": round(full_v - shift_v, 6),
                    "relative_drop": round(relative_drop(full_v, shift_v, eps), 6),
                }
            )
    sens = pd.DataFrame(rows)
    summary = _metric_summary(sens)
    return sens, summary


def _metric_summary(sens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, g in sens.groupby("metric"):
        vals = g["relative_drop"].dropna().values
        ci = bootstrap_ci(vals, 2000, 0.95, seed=0)
        rows.append(
            {
                "metric": metric,
                "mean_relative_drop": round(float(vals.mean()), 6) if vals.size else np.nan,
                "median_relative_drop": round(float(np.median(vals)), 6) if vals.size else np.nan,
                "CI_low": round(ci["CI_low"], 6),
                "CI_high": round(ci["CI_high"], 6),
                "positive_rate": round(float((vals > 0).mean()), 4) if vals.size else np.nan,
                "n": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def hidden_definitions() -> Dict[str, Dict[str, float]]:
    """Definition name -> thresholds. 'Recall/F1' field selects the minority metric drop."""
    return {
        "Primary": {"gtype": "AUROC", "gmax": 0.03, "mtype": "minority_recall", "mmin": 0.10},
        "A": {"gtype": "AUROC", "gmax": 0.01, "mtype": "minority_recall", "mmin": 0.20},
        "B": {"gtype": "AUROC", "gmax": 0.02, "mtype": "minority_recall", "mmin": 0.20},
        "C": {"gtype": "AUROC", "gmax": 0.01, "mtype": "minority_recall", "mmin": 0.30},
        "D": {"gtype": "AUPRC", "gmax": 0.02, "mtype": "minority_recall", "mmin": 0.20},
        "E": {"gtype": "AUROC", "gmax": 0.02, "mtype": "minority_f1", "mmin": 0.20},
    }


def _drops(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-shifted-row absolute drops and a sorted score for ranking."""
    full = full_by_cell(raw)
    shifted = raw[~raw["shift_type"].eq("full")].copy()
    recs = []
    for _, r in shifted.iterrows():
        fidx = (r["dataset"], r["model"], r["seed"])
        if fidx not in full.index:
            continue
        fm = full.loc[fidx]
        if isinstance(fm, pd.DataFrame):
            fm = fm.iloc[0]
        auroc_drop = float(fm["AUROC"] - r["AUROC"])
        auprc_drop = float(fm["AUPRC"] - r["AUPRC"])
        recall_drop = float(fm["minority_recall"] - r["minority_recall"])
        f1_drop = float(fm["minority_f1"] - r["minority_f1"])
        recs.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "seed": r["seed"],
                "mechanism": r["shift_type"],
                "severity": r["severity"],
                "AUROC_drop": round(auroc_drop, 6),
                "AUPRC_drop": round(auprc_drop, 6),
                "minority_recall_drop": round(recall_drop, 6),
                "minority_f1_drop": round(f1_drop, 6),
                "score": round(recall_drop - auroc_drop, 6),
            }
        )
    return pd.DataFrame(recs)


def compute_hidden_threshold_sensitivity(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """RC2: count hidden failures under multiple pre-registered definitions."""
    drops = _drops(raw)
    defs = hidden_definitions()
    rows = []
    examples = []
    for name, d in defs.items():
        if d["gtype"] == "AUROC":
            gmask = drops["AUROC_drop"] <= d["gmax"]
        else:
            gmask = drops["AUPRC_drop"] <= d["gmax"]
        mmask = drops[f'{d["mtype"]}_drop'] >= d["mmin"]
        sub = drops[gmask & mmask]
        rows.append(
            {
                "definition": name,
                "n_hidden_cases": int(len(sub)),
                "n_datasets": int(sub["dataset"].nunique()),
                "n_models": int(sub["model"].nunique()) if not sub.empty else 0,
                "n_mechanisms": int(sub["mechanism"].nunique()) if not sub.empty else 0,
                "n_severity_levels": int(sub["severity"].nunique()) if not sub.empty else 0,
            }
        )
        top = sub.sort_values("score", ascending=False).head(10)
        for _, e in top.iterrows():
            examples.append({**{"definition": name}, **e.to_dict()})
    return pd.DataFrame(rows), drops, pd.DataFrame(examples)


def leave_one_dataset_out(vul: pd.DataFrame, cc: pd.DataFrame, observable: set) -> pd.DataFrame:
    """RC3: pooled influence with each observable dataset removed (no retraining)."""
    obs = vul[vul["dataset"].isin(observable)].copy()
    cc_obs = cc[cc["dataset"].isin(observable)].copy()
    rows = []
    for d in sorted(observable):
        sub = obs[obs["dataset"].ne(d)]
        cc_sub = cc_obs[cc_obs["dataset"].ne(d)]
        mvg = sub["MVG_recall"].dropna().values
        ci = bootstrap_ci(mvg, 2000, 0.95, seed=len(rows))
        rows.append(
            {
                "excluded_dataset": d,
                "pooled_mvg": round(float(mvg.mean()), 6) if mvg.size else np.nan,
                "pooled_mvg_CI_low": round(ci["CI_low"], 6),
                "pooled_mvg_CI_high": round(ci["CI_high"], 6),
                "delta_mvg_class": round(float(cc_sub["DeltaMVG_class"].mean()), 6) if not cc_sub.empty else np.nan,
                "minority_specific_mvg": round(float(sub[sub["shift_type"].eq("minority_specific")]["MVG_recall"].dropna().mean()), 6) if not sub.empty else np.nan,
                "majority_specific_mvg": round(float(sub[sub["shift_type"].eq("majority_specific")]["MVG_recall"].dropna().mean()), 6) if not sub.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def leave_one_model_out(
    vul: pd.DataFrame, cc: pd.DataFrame, observable: set, models: Sequence[str]
) -> pd.DataFrame:
    """RC4: pooled influence with each model removed."""
    obs = vul[vul["dataset"].isin(observable)].copy()
    cc_obs = cc[cc["dataset"].isin(observable)].copy()
    rows = []
    for m in models:
        sub = obs[obs["model"].ne(m)]
        cc_sub = cc_obs[cc_obs["model"].ne(m)]
        mech = {st: float(sub[sub["shift_type"].eq(st)]["MVG_recall"].dropna().mean()) for st in CORE_SHIFTS}
        rows.append(
            {
                "excluded_model": m,
                "pooled_mvg": round(float(sub["MVG_recall"].dropna().mean()), 6),
                "delta_mvg_class": round(float(cc_sub["DeltaMVG_class"].mean()), 6) if not cc_sub.empty else np.nan,
                "minority_specific_mvg": round(mech["minority_specific"], 6),
                "majority_specific_mvg": round(mech["majority_specific"], 6),
                "global_importance_mvg": round(mech["global_importance"], 6),
                "random_mvg": round(mech["random"], 6),
                "asymmetry_preserved": bool(mech["minority_specific"] > mech["majority_specific"]),
            }
        )
    return pd.DataFrame(rows)


def seed_sensitivity(vul: pd.DataFrame, cc: pd.DataFrame, observable: set, seeds: Sequence[int]) -> pd.DataFrame:
    """RC5: per-seed and per-seed-subset aggregates (no retraining)."""
    obs = vul[vul["dataset"].isin(observable)].copy()
    cc_obs = cc[cc["dataset"].isin(observable)].copy()
    rows = []

    def _aggregate(sub, cc_sub, label):
        mech = {st: float(sub[sub["shift_type"].eq(st)]["MVG_recall"].dropna().mean()) for st in CORE_SHIFTS}
        mvg = sub["MVG_recall"].dropna().mean()
        delta = float(cc_sub["DeltaMVG_class"].mean()) if not cc_sub.empty else np.nan
        rows.append(
            {
                "seed_set": label,
                "mean_mvg": round(float(mvg), 6),
                "minority_specific_mvg": round(mech["minority_specific"], 6),
                "majority_specific_mvg": round(mech["majority_specific"], 6),
                "delta_mvg_class": round(delta, 6),
                "asymmetry_preserved": bool(mech["minority_specific"] > mech["majority_specific"]),
            }
        )

    for s in seeds:
        _aggregate(obs[obs["seed"].eq(s)], cc_obs[cc_obs["seed"].eq(s)], str(s))
    for combo in [(42, 52), (42, 62), (52, 62)]:
        _aggregate(obs[obs["seed"].isin(combo)], cc_obs[cc_obs["seed"].isin(combo)], "+".join(map(str, combo)))
    _aggregate(obs, cc_obs, "42+52+62")
    return pd.DataFrame(rows)


def floor_risk_absolute(
    raw: pd.DataFrame, bmo: pd.DataFrame, observable: set
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """RC6: absolute drops for all 10 datasets (no denominator -> floor-risk still uses absolute)."""
    full = full_by_cell(raw)
    shifted = raw[~raw["shift_type"].eq("full")].copy()
    rows = []
    for _, r in shifted.iterrows():
        fidx = (r["dataset"], r["model"], r["seed"])
        if fidx not in full.index:
            continue
        fm = full.loc[fidx]
        if isinstance(fm, pd.DataFrame):
            fm = fm.iloc[0]
        cls = bmo.set_index("dataset").loc[r["dataset"]]["observability_class"]
        min_drop = float(fm["minority_recall"] - r["minority_recall"])
        maj_drop = float(fm["majority_recall"] - r["majority_recall"])
        rows.append(
            {
                "dataset": r["dataset"],
                "observability_class": cls,
                "mechanism": r["shift_type"],
                "severity": r["severity"],
                "model": r["model"],
                "seed": r["seed"],
                "minority_abs_drop": round(min_drop, 6),
                "majority_abs_drop": round(maj_drop, 6),
                "abs_gap": round(min_drop - maj_drop, 6),
            }
        )
    long = pd.DataFrame(rows)
    groups = []
    for label, sub in [
        ("observable", long[long["observability_class"].ne("floor_risk")]),
        ("floor_risk", long[long["observability_class"].eq("floor_risk")]),
        ("all_datasets", long),
    ]:
        groups.append(
            {
                "group": label,
                "n": int(len(sub)),
                "mean_minority_abs_drop": round(float(sub["minority_abs_drop"].mean()), 6) if not sub.empty else np.nan,
                "mean_majority_abs_drop": round(float(sub["majority_abs_drop"].mean()), 6) if not sub.empty else np.nan,
                "mean_abs_gap": round(float(sub["abs_gap"].mean()), 6) if not sub.empty else np.nan,
                "positive_abs_gap_rate": round(float((sub["abs_gap"] > 0).mean()), 4) if not sub.empty else np.nan,
            }
        )
    return long, pd.DataFrame(groups)


def severity_sensitivity(
    raw: pd.DataFrame, observable: Sequence[str] | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """RC7: slope recomputed under severity-subsets; full point always included.

    Slopes are scoped to observable datasets (same as the frozen C1/C2 slope gap).
    Floor-risk datasets have a zero minority baseline, so their slope gap is
    meaningless and would otherwise drag the pooled gap downward.
    """
    if observable is not None:
        raw = raw[raw["dataset"].isin(observable)]
    full = full_by_cell(raw)
    shifted = raw[~raw["shift_type"].eq("full")].copy()
    severity_sets = {
        "all": [0.05, 0.10, 0.20, 0.30, 0.40],
        "exclude_0.40": [0.05, 0.10, 0.20, 0.30],
        "exclude_0.30_and_0.40": [0.05, 0.10, 0.20],
    }
    rows = []
    for label, sevs in severity_sets.items():
        for (ds, model, seed, shift_type), g in shifted[shifted["severity"].isin(sevs)].groupby(
            ["dataset", "model", "seed", "shift_type"]
        ):
            fidx = (ds, model, seed)
            if fidx not in full.index:
                continue
            fm = full.loc[fidx]
            if isinstance(fm, pd.DataFrame):
                fm = fm.iloc[0]
            g = g.sort_values("removal_fraction")
            rates = [0.0] + g["removal_fraction"].tolist()
            mins = [float(fm["minority_recall"])] + g["minority_recall"].tolist()
            majs = [float(fm["majority_recall"])] + g["majority_recall"].tolist()
            ss = sensitivity_summary(rates, mins, majs, EPS)
            rows.append(
                {
                    "severity_set": label,
                    "dataset": ds,
                    "model": model,
                    "mechanism": shift_type,
                    "minority_slope": ss["minority_slope"],
                    "majority_slope": ss["majority_slope"],
                    "slope_gap": ss["slope_gap"],
                }
            )
    long = pd.DataFrame(rows)
    summary = []
    for label, g in long.groupby("severity_set"):
        summary.append(
            {
                "severity_set": label,
                "mean_minority_slope": round(float(g["minority_slope"].mean()), 6),
                "mean_majority_slope": round(float(g["majority_slope"].mean()), 6),
                "mean_slope_gap": round(float(g["slope_gap"].mean()), 6),
                "positive_gap_rate": round(float((g["slope_gap"] > 0).mean()), 4),
            }
        )
    return long, pd.DataFrame(summary)


def mechanism_pairwise(vul: pd.DataFrame, observable: set) -> pd.DataFrame:
    """RC8: per-dataset class-asymmetry gaps (minority-specific vs majority-specific primary)."""
    obs = vul[vul["dataset"].isin(observable)].copy()
    # Per-dataset mean MVG per shift_type across (model, seed, severity).
    per_ds = obs.groupby(["dataset", "shift_type"])["MVG_recall"].mean().unstack("shift_type")
    contrasts = {
        "minority_vs_majority": ("minority_specific", "majority_specific"),
        "global_vs_majority": ("global_importance", "majority_specific"),
        "random_vs_majority": ("random", "majority_specific"),
    }
    rows = []
    for name, (a, b) in contrasts.items():
        if a not in per_ds.columns or b not in per_ds.columns:
            continue
        av = per_ds[a].values.astype(float)
        bv = per_ds[b].values.astype(float)
        gap = av - bv
        ci = bootstrap_ci(gap, 2000, 0.95, seed=len(rows))
        w = wilcoxon_paired(av, bv)
        es = rank_biserial(av, bv)
        rows.append(
            {
                "contrast": name,
                "n_datasets": int(per_ds.shape[0]),
                "n_positive_gap": int((gap > 0).sum()),
                "mean_gap": round(float(gap.mean()), 6),
                "median_gap": round(float(np.median(gap)), 6),
                "CI_low": round(ci["CI_low"], 6),
                "CI_high": round(ci["CI_high"], 6),
                "wilcoxon_p": w["p_value"],
                "effect_size": round(es, 6),
            }
        )
    return pd.DataFrame(rows)
