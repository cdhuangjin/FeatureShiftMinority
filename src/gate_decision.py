"""Aggregate evidence and apply the Gate A GO / HOLD / STOP rules."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SHIFT_ENVS = ("random", "global_importance", "minority_specific")


def _frac_positive(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float((arr > 0).mean())


def build_evidence(perf: pd.DataFrame, vul: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return one row per (dataset, model) summarising vulnerability evidence."""
    gate = config["gate"]
    seeds = config["seeds"]
    rows = []
    for (ds, model), vg in vul[vul["environment"].isin(SHIFT_ENVS)].groupby(["dataset", "model"]):
        pg = perf[perf["dataset"].eq(ds) & perf["model"].eq(model)]
        full = pg[pg["environment"].eq("full")].set_index("seed")
        msi = pg[pg["environment"].eq("minority_specific")].set_index("seed")
        glo = pg[pg["environment"].eq("global_importance")].set_index("seed")

        seed_mvg_recall = vg.groupby("seed")["MVG_recall"].mean()
        seed_mvg_f1 = vg.groupby("seed")["MVG_f1"].mean()
        cons_recall = _frac_positive(seed_mvg_recall.tolist())
        cons_f1 = _frac_positive(seed_mvg_f1.tolist())

        msi_mrd = vg[vg["environment"].eq("minority_specific")]["minority_RD_recall"]
        rand_mrd = vg[vg["environment"].eq("random")]["minority_RD_recall"]
        glo_mrd = vg[vg["environment"].eq("global_importance")]["minority_RD_recall"]
        msi_vs_random_gap = float(msi_mrd.mean() - rand_mrd.mean())
        msi_vs_global_gap = float(msi_mrd.mean() - glo_mrd.mean())

        per_seed_msi_vs_random = vg[
            vg["environment"].eq("minority_specific")
        ]["minority_RD_recall"].to_numpy() - vg[
            vg["environment"].eq("random")
        ]["minority_RD_recall"].to_numpy()
        cons_msi = _frac_positive(per_seed_msi_vs_random.tolist())

        # Absolute minority-recall drop from full to each shift environment.
        def rec_drop(env_df: pd.DataFrame) -> float:
            if env_df.empty:
                return 0.0
            return float((full["minority_recall"] - env_df["minority_recall"]).mean())

        msi_drop = rec_drop(msi)
        glo_drop = rec_drop(glo)
        msi_minus_global_recall_drop_gap = msi_drop - glo_drop

        # Hidden minority failure count within this (dataset, model).
        hf = 0
        for seed in seeds:
            for env in SHIFT_ENVS:
                f = full.loc[seed]
                s = pg[pg["environment"].eq(env) & pg["seed"].eq(seed)]
                if s.empty:
                    continue
                s = s.iloc[0]
                auroc_drop = f["AUROC"] - s["AUROC"]
                rec_drop_v = f["minority_recall"] - s["minority_recall"]
                if (
                    auroc_drop <= gate["hidden_auroc_drop_max"]
                    and rec_drop_v >= gate["hidden_minority_recall_drop_min"]
                ):
                    hf += 1

        rows.append(
            {
                "dataset": ds,
                "model": model,
                "mean_MVG_recall": round(float(seed_mvg_recall.mean()), 6),
                "mean_MVG_f1": round(float(seed_mvg_f1.mean()), 6),
                "MSI_vs_random_gap": round(msi_vs_random_gap, 6),
                "MSI_vs_global_gap": round(msi_vs_global_gap, 6),
                "MSI_minus_global_recall_drop_gap": round(msi_minus_global_recall_drop_gap, 6),
                "hidden_failure_count": hf,
                "seed_consistency_recall": round(cons_recall, 4),
                "seed_consistency_f1": round(cons_f1, 4),
                "seed_consistency_msi": round(cons_msi, 4),
            }
        )
    return pd.DataFrame(rows)


def decide_gate(evidence: pd.DataFrame, config: dict, imbalance_ratios: Dict[str, float]) -> Tuple[str, Dict]:
    """Apply GO-A/B/C/D, then HOLD/STOP. Returns (decision, evidence dict)."""
    gate = config["gate"]
    datasets = sorted(evidence["dataset"].unique().tolist())
    models = sorted(evidence["model"].unique().tolist())
    n_seeds = len(config["seeds"])
    min_cons = gate["min_seed_consistency"] / n_seeds
    n_min_go = gate["min_datasets_for_go"]

    def ds_mean(col: str) -> Dict[str, float]:
        return {ds: float(evidence[evidence["dataset"].eq(ds)][col].mean()) for ds in datasets}

    mvg_recall_ds = ds_mean("mean_MVG_recall")
    mvg_f1_ds = ds_mean("mean_MVG_f1")
    msi_vs_rand_ds = ds_mean("MSI_vs_random_gap")
    msi_vs_glob_ds = ds_mean("MSI_vs_global_gap")
    msi_glob_drop_ds = ds_mean("MSI_minus_global_recall_drop_gap")
    cons_recall_ds = ds_mean("seed_consistency_recall")
    cons_msi_ds = ds_mean("seed_consistency_msi")

    strong_mvg = gate["strong_mvg"]
    mod_mvg = gate["moderate_mvg"]
    spec_gap = gate["strong_specificity_gap"]

    reasons: List[str] = []
    go_flags: List[str] = []

    # ---- GO-A: minority vulnerability gap ----
    strong_datasets = [
        ds for ds in datasets if mvg_recall_ds[ds] >= strong_mvg or mvg_f1_ds[ds] >= strong_mvg
    ]
    if len(strong_datasets) >= n_min_go:
        consistent = all(
            cons_recall_ds[ds] >= min_cons or cons_f1_ds[ds] >= min_cons
            for ds in strong_datasets
        )
        if consistent:
            go_flags.append("GO-A")
            reasons.append(
                f"GO-A: >= {n_min_go} datasets ({strong_datasets}) show mean MVG_recall or "
                f"MVG_f1 >= {strong_mvg} with >= {gate['min_seed_consistency']}/{n_seeds} seed consistency."
            )

    # ---- GO-B: minority-specific removal clearly worse than random ----
    b_datasets = [ds for ds in datasets if msi_vs_rand_ds[ds] >= spec_gap]
    if len(b_datasets) >= n_min_go:
        consistent = all(cons_msi_ds[ds] >= min_cons for ds in b_datasets)
        if consistent:
            go_flags.append("GO-B")
            reasons.append(
                f"GO-B: >= {n_min_go} datasets ({b_datasets}) show MSI removal minus random "
                f"removal MRD >= {spec_gap}."
            )

    # ---- GO-C: global importance misses minority-critical features ----
    c_datasets = [ds for ds in datasets if msi_glob_drop_ds[ds] >= spec_gap]
    for ds in c_datasets:
        ds_ev = evidence[evidence["dataset"].eq(ds)]
        both_models_direction = all(
            ds_ev[ds_ev["model"].eq(m)]["MSI_minus_global_recall_drop_gap"].max() > 0
            for m in models
        )
        cons = cons_recall_ds[ds]
        if both_models_direction and cons >= min_cons:
            go_flags.append("GO-C")
            reasons.append(
                f"GO-C: dataset {ds} MSI-removal minority drop exceeds global-importance "
                f"removal drop by >= {spec_gap}, both models agree, seed consistency {cons:.2f}."
            )
            break

    # ---- GO-D: hidden minority failure ----
    stable_cells = 0
    for _, ev in evidence.iterrows():
        if ev["hidden_failure_count"] >= gate["min_seed_consistency"]:
            stable_cells += 1
    if stable_cells >= n_min_go:
        go_flags.append("GO-D")
        reasons.append(
            f"GO-D: >= {n_min_go} stable (dataset, model) cells meet AUROC drop <= "
            f"{gate['hidden_auroc_drop_max']} with minority recall drop >= "
            f"{gate['hidden_minority_recall_drop_min']}."
        )

    decision = "GO" if go_flags else None

    if decision is None:
        # ---- HOLD heuristics ----
        moderate_datasets = [
            ds
            for ds in datasets
            if mod_mvg <= mvg_recall_ds[ds] < strong_mvg or mod_mvg <= mvg_f1_ds[ds] < strong_mvg
        ]
        extreme_active = [ds for ds in datasets if mvg_recall_ds[ds] >= mod_mvg]
        only_extreme = bool(extreme_active) and all(
            imbalance_ratios.get(ds, 1) >= 100 for ds in extreme_active
        )
        single_model = (
            sum(
                1
                for m in models
                if evidence[evidence["model"].eq(m)]["mean_MVG_recall"].max() >= strong_mvg
            )
            == 1
        )
        msi_gap_low = any(0.01 < msi_vs_rand_ds[ds] < spec_gap for ds in datasets) and not any(
            msi_vs_rand_ds[ds] >= spec_gap for ds in datasets
        )
        consistent_but_weak = any(
            (cons_recall_ds[ds] >= min_cons)
            and (0.01 < mvg_recall_ds[ds] < strong_mvg)
            for ds in datasets
        )

        if (
            moderate_datasets
            or only_extreme
            or single_model
            or msi_gap_low
            or consistent_but_weak
        ):
            decision = "HOLD"
            hold_reasons = []
            if moderate_datasets:
                hold_reasons.append(
                    f"only moderate MVG ({mod_mvg}-{strong_mvg}) in {moderate_datasets}"
                )
            if only_extreme:
                hold_reasons.append("effect confined to extreme-imbalance dataset(s)")
            if single_model:
                hold_reasons.append("only one model shows a strong effect")
            if msi_gap_low:
                hold_reasons.append("MSI removal only slightly worse than random (<0.10)")
            if consistent_but_weak:
                hold_reasons.append("seed-consistent but effect below strong threshold")
            reasons.append("HOLD: " + "; ".join(hold_reasons) + ".")
        else:
            decision = "STOP/PIVOT"
            reasons.append(
                "STOP/PIVOT: minority and majority degradations are broadly symmetric, "
                "global importance adequately explains minority drops, and effects do not "
                "replicate across dataset/model/seed in a decisive way."
            )

    gate_evidence = {
        "GO_flags": go_flags,
        "datasets": datasets,
        "models": models,
        "n_seeds": n_seeds,
        "thresholds": {
            "strong_mvg": strong_mvg,
            "moderate_mvg": mod_mvg,
            "strong_specificity_gap": spec_gap,
            "hidden_auroc_drop_max": gate["hidden_auroc_drop_max"],
            "hidden_minority_recall_drop_min": gate["hidden_minority_recall_drop_min"],
            "min_seed_consistency": gate["min_seed_consistency"],
            "min_datasets_for_go": n_min_go,
        },
        "per_dataset": {
            ds: {
                "mean_MVG_recall": mvg_recall_ds[ds],
                "mean_MVG_f1": mvg_f1_ds[ds],
                "MSI_vs_random_gap": msi_vs_rand_ds[ds],
                "MSI_vs_global_gap": msi_vs_glob_ds[ds],
                "MSI_minus_global_recall_drop_gap": msi_glob_drop_ds[ds],
                "seed_consistency_recall": cons_recall_ds[ds],
                "seed_consistency_msi": cons_msi_ds[ds],
                "imbalance_ratio": imbalance_ratios.get(ds, None),
            }
            for ds in datasets
        },
        "reasons": reasons,
    }
    return decision, gate_evidence


def summarize_report_data(perf: pd.DataFrame, vul: pd.DataFrame, evidence: pd.DataFrame) -> Dict:
    """Extract the strongest minority-vulnerability and MSI-removal cases for reporting."""
    shift_vul = vul[vul["environment"].isin(SHIFT_ENVS)]
    strongest = shift_vul.sort_values(["MVG_recall", "MVG_f1"], ascending=False).iloc[0]
    msi_vul = shift_vul[shift_vul["environment"].eq("minority_specific")]
    strongest_msi = msi_vul.sort_values("minority_RD_recall", ascending=False).iloc[0]
    worst_row = perf[perf["model"].eq(strongest["model"]) & perf["dataset"].eq(strongest["dataset"])]
    full_w = worst_row[worst_row["environment"].eq("full")].iloc[0]
    shift_w = worst_row[worst_row["environment"].eq(strongest["environment"])].iloc[0]

    # Hidden failure strongest case (largest recall drop while AUROC stable).
    hidden_candidates = []
    for _, v in shift_vul.iterrows():
        row = perf[
            perf["dataset"].eq(v["dataset"])
            & perf["model"].eq(v["model"])
            & perf["seed"].eq(v["seed"])
        ]
        full = row[row["environment"].eq("full")].iloc[0]
        shift = row[row["environment"].eq(v["environment"])].iloc[0]
        auroc_drop = full["AUROC"] - shift["AUROC"]
        rec_drop = full["minority_recall"] - shift["minority_recall"]
        if auroc_drop <= 0.03 and rec_drop >= 0.10:
            hidden_candidates.append((rec_drop, v["dataset"], v["model"], v["environment"], full, shift))
    hidden_candidates.sort(reverse=True, key=lambda x: x[0])

    return {
        "strongest": {
            "dataset": strongest["dataset"],
            "model": strongest["model"],
            "environment": "minority_specific",
            "seed": int(strongest["seed"]),
            "min_recall_full": full_w["minority_recall"],
            "min_recall_shifted": shift_w["minority_recall"],
            "maj_recall_full": full_w["majority_recall"],
            "maj_recall_shifted": shift_w["majority_recall"],
            "MVG_recall": strongest["MVG_recall"],
        },
        "strongest_msi": {
            "dataset": strongest_msi["dataset"],
            "model": strongest_msi["model"],
            "seed": int(strongest_msi["seed"]),
            "minority_RD_recall": strongest_msi["minority_RD_recall"],
            "majority_RD_recall": strongest_msi["majority_RD_recall"],
            "MVG_recall": strongest_msi["MVG_recall"],
        },
        "hidden": [
            {
                "dataset": c[1],
                "model": c[2],
                "environment": c[3],
                "auroc_drop": round(float(c[4]["AUROC"] - c[5]["AUROC"]), 4),
                "minority_recall_drop": round(float(c[4]["minority_recall"] - c[5]["minority_recall"]), 4),
            }
            for c in hidden_candidates[:3]
        ],
        "evidence": evidence.to_dict(orient="records"),
    }
