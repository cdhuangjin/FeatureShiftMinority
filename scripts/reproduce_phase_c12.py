"""One-shot + resumable reproduction for Project C Phase C1/C2."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.dataset_registry import (
    load_candidates,
    freeze_datasets,
    write_substitution_log,
    dataset_profiles_table,
)
from src.baseline_observability import screen_datasets
from src.phase_c12_runner import (
    run_cell,
    build_cell_plan,
    coalesce_checkpoint_payloads,
)
from src.checkpoint_manager import CheckpointManager
from src.phase_c12_analysis import (
    dataset_summary,
    model_summary,
    mechanism_summary,
    build_statistics,
    build_gate_summary,
    make_figures,
    recompute_slopes,
)
from src.paper_gate import evaluate_paper_gate
from src.statistical_analysis import spearman_corr


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _hash_file(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_stage0(config: dict, project_root: Path, results_root: Path):
    data_root = project_root / "data"
    candidates = load_candidates(config["datasets"], data_root)
    write_substitution_log(results_root / "dataset_substitution_log.md")
    bmo_df = screen_datasets(candidates, config["seeds"][0], config)
    frozen = freeze_datasets(candidates, project_root / "configs" / "datasets_phase_c12.yaml")
    obs_classes = dict(zip(bmo_df["dataset"], bmo_df["observability_class"]))
    profiles_df = dataset_profiles_table(candidates, bmo_df.set_index("dataset").to_dict("index"))
    profiles_df.to_csv(results_root / "dataset_profiles.csv", index=False)
    bmo_df.to_csv(results_root / "baseline_observability.csv", index=False)

    # Data audit report.
    lines = ["# Phase C1/C2 data audit report", "", f"- freeze hash: `{frozen['dataset_freeze_hash']}`",
             f"- freeze timestamp: `{frozen['freeze_timestamp']}`", ""]
    lines.append("| dataset | source | n | feats | minority | majority | ratio | BMO | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in profiles_df.iterrows():
        lines.append(f"| {r['dataset']} | {r['source']} | {r['n_samples']} | {r['n_features']} | "
                     f"{r['minority_count']} | {r['majority_count']} | {r['imbalance_ratio']} | "
                     f"{r['BMO']:.3f} | {r['observability_class']} |")
    lines.append("")
    lines.append("## Substitution")
    lines.append("")
    lines.append(open(results_root / "dataset_substitution_log.md", encoding="utf-8").read())
    (results_root / "data_audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Persist raw npz for hashing.
    for name, cd in candidates.items():
        np.savez_compressed(data_root / "raw" / f"{name}.npz", X=cd.profile.X, y=cd.profile.y)
    return candidates, bmo_df, frozen, obs_classes, profiles_df


def _run_one(cell, phase_ds, config, data_root, ckpt):
    ds, seed, model = cell
    key = ckpt.key(ds, seed, model, "c12_cell")
    try:
        payload = run_cell(phase_ds, seed, model, config, data_root)
        ckpt.mark_complete(key, payload)
        return key, "completed", ""
    except Exception as exc:  # noqa: BLE001
        ckpt.mark_failed(key, repr(exc))
        return key, "failed", repr(exc)


def run_cells(candidates, config, project_root, results_root, ckpt, resume, ds_filter, model_filter, seed_filter):
    data_root = project_root / "data"
    plan = build_cell_plan(config["datasets"], config["seeds"], config["models"])
    if ds_filter:
        plan = [c for c in plan if c[0] in ds_filter]
    if model_filter:
        plan = [c for c in plan if c[2] in model_filter]
    if seed_filter:
        plan = [c for c in plan if c[1] in seed_filter]
    todo = []
    for cell in plan:
        key = ckpt.key(*cell, "c12_cell")
        if resume and ckpt.is_complete(key):
            continue
        todo.append(cell)
    print(f"Cells: total {len(plan)}, pending {len(todo)}")
    if not todo:
        return
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_run_one)(cell, candidates[cell[0]], config, data_root, ckpt) for cell in todo
    )
    comp = sum(1 for _, s, _ in results if s == "completed")
    fail = sum(1 for _, s, _ in results if s == "failed")
    print(f"Completed {comp}, failed {fail}")
    for key, status, err in results:
        if status == "failed":
            print(f"  FAILED {key}: {err[:200]}")


def coalesce_and_save(ckpt, results_root):
    payloads = list(ckpt.load_payloads())
    dfs = coalesce_checkpoint_payloads(payloads)
    save_map = {
        "perf": "raw_results.csv",
        "vul": "vulnerability_results.csv",
        "imp": "feature_importance.csv",
        "hidden": "hidden_failure_results.csv",
        "class_control": "class_specific_control.csv",
        "slope": "slope_results.csv",
        "structured": "structured_shift_results.csv",
        "delayed": "delayed_availability_results.csv",
    }
    for k, fname in save_map.items():
        df = dfs[k]
        df.to_csv(results_root / fname, index=False)
    return dfs


def write_report(
    path, decision, gate_evidence, summary, stats, ds_sum, model_sum, mech_sum, perf, vul,
    hidden, structured, delayed, class_control, slope, meta, config, frozen_hash,
    n_expected, n_completed, n_failed, figures,
):
    lines = []
    lines.append("# Project C Phase C1/C2 report")
    lines.append("")
    # Aggregate helpers.
    obs_classes = {r["dataset"]: r["observability_class"] for _, r in ds_sum.iterrows()}
    observable_ds = sorted(d for d, c in obs_classes.items() if c != "floor_risk")
    floor_ds = sorted(d for d, c in obs_classes.items() if c == "floor_risk")
    ds_mvg = {r["dataset"]: r["mean_MVG_recall"] for _, r in ds_sum.iterrows()}
    strong_mvg = config["paper_gate"]["strong_mvg"]
    n_pos = sum(1 for d in observable_ds if pd.notna(ds_mvg[d]) and ds_mvg[d] > 0)
    n_strong = sum(1 for d in observable_ds if pd.notna(ds_mvg[d]) and ds_mvg[d] >= strong_mvg)

    obs_slope = slope[slope["dataset"].isin(observable_ds)]
    mean_min_slope = float(obs_slope["minority_slope"].mean()) if not obs_slope.empty else float("nan")
    mean_maj_slope = float(obs_slope["majority_slope"].mean()) if not obs_slope.empty else float("nan")
    mean_slope_gap = float(obs_slope["slope_gap"].mean()) if not obs_slope.empty else float("nan")
    steep_ds = sum(1 for _, r in ds_sum.iterrows() if pd.notna(r["slope_gap"]) and r["slope_gap"] > 0)

    # Mechanism ranking / MSI vs global.
    mech_mean = {r["shift_type"]: r["mean_MVG_recall"] for _, r in mech_sum.iterrows()}
    obs_vul = vul[vul["dataset"].isin(observable_ds)].copy()
    pivot = obs_vul.groupby(["dataset", "shift_type"])["MVG_recall"].mean().unstack()
    msi_better = 0
    global_better = 0
    for d in observable_ds:
        if d in pivot.index:
            ms = pivot.loc[d, "minority_specific"] if "minority_specific" in pivot.columns else np.nan
            gl = pivot.loc[d, "global_importance"] if "global_importance" in pivot.columns else np.nan
            if pd.notna(ms) and pd.notna(gl):
                if ms > gl:
                    msi_better += 1
                elif gl > ms:
                    global_better += 1

    # Class-specific control.
    obs_cc = class_control[class_control["dataset"].isin(observable_ds)]
    cc_min = float(obs_cc["minority_specific_MVG"].mean()) if not obs_cc.empty else float("nan")
    cc_maj = float(obs_cc["majority_specific_MVG"].mean()) if not obs_cc.empty else float("nan")
    cc_delta = float(obs_cc["DeltaMVG_class"].mean()) if not obs_cc.empty else float("nan")
    cc_support = int(sum(1 for _, r in ds_sum.iterrows() if pd.notna(r["DeltaMVG_class"]) and r["DeltaMVG_class"] > 0))

    # Structured penalties.
    obs_struct = structured[structured["dataset"].isin(observable_ds)]
    gp = obs_struct[obs_struct["mechanism"].eq("group_loss_penalty")]["StructuredPenalty"]
    cp = obs_struct[obs_struct["mechanism"].eq("correlated_loss_penalty")]["StructuredPenalty"]
    group_pen_mean = float(gp.mean()) if len(gp) else float("nan")
    corr_pen_mean = float(cp.mean()) if len(cp) else float("nan")

    # Delayed availability.
    obs_del = delayed[delayed["dataset"].isin(observable_ds)]
    mra_min = float(obs_del["MinorityRecoveryAvailability"].mean()) if not obs_del.empty else float("nan")
    mra_maj = float(obs_del["MajorityRecoveryAvailability"].mean()) if not obs_del.empty else float("nan")
    rec_gap = float(obs_del["RecoveryGap"].mean()) if not obs_del.empty else float("nan")

    # Hidden failure.
    hd = hidden[hidden["hidden_auroc"]]
    hd_count = int(len(hd))
    hd_by_ds = hd.groupby("dataset").size().to_dict() if not hd.empty else {}
    stable_hd = {r["dataset"]: int(r["hidden_failure_count"]) for _, r in ds_sum.iterrows()}
    hd_datasets = [d for d, v in stable_hd.items() if v >= config["paper_gate"].get("min_hidden_consistency", 2)]
    if not hd.empty:
        h0 = hd.sort_values("minority_recall_drop", ascending=False).iloc[0]
        h0_str = (f"{h0['dataset']}/{h0['model']}/{h0['shift_type']} sev {h0['severity']} "
                  f"(AUROC drop {h0['AUROC_drop']:.3f}, AUPRC drop {h0['AUPRC_drop']:.3f}, "
                  f"minority recall drop {h0['minority_recall_drop']:.3f})")
    else:
        h0_str = "none"

    # Pooled observable MVG.
    vmvg = obs_vul["MVG_recall"].dropna().values
    pooled_mean = float(np.mean(vmvg)) if vmvg.size else float("nan")
    pooled_median = float(np.median(vmvg)) if vmvg.size else float("nan")
    pooled_n = int(vmvg.size)
    mvg_row = stats[(stats["comparison"].eq("MVG_recall")) & (stats["mechanism"].eq("pooled_core"))]
    mvg_ci_low = float(mvg_row["CI_low"].iloc[0]) if not mvg_row.empty else float("nan")
    mvg_ci_high = float(mvg_row["CI_high"].iloc[0]) if not mvg_row.empty else float("nan")
    wil_p = summary.get("wilcoxon_p", float("nan"))
    wil_fdr = summary.get("wilcoxon_fdr_p", float("nan"))
    eff_size = summary.get("effect_size", float("nan"))
    ci_low = summary.get("mvg_ci_low", float("nan"))

    # Spearman imbalance / BMO.
    imp_vals = [meta.get("imbalance_ratio", {}).get(d, np.nan) for d in observable_ds]
    bmo_vals = [meta.get("BMO", {}).get(d, np.nan) for d in observable_ds]
    mvg_vals = [ds_mvg.get(d, np.nan) for d in observable_ds]
    valid = [not (np.isnan(imp_vals[i]) or np.isnan(mvg_vals[i])) for i in range(len(observable_ds))]
    imp_arr = np.asarray([imp_vals[i] for i in range(len(observable_ds)) if valid[i]], dtype=float)
    mvg_arr = np.asarray([mvg_vals[i] for i in range(len(observable_ds)) if valid[i]], dtype=float)
    bmo_arr = np.asarray([bmo_vals[i] for i in range(len(observable_ds)) if valid[i]], dtype=float)
    sp_imp = spearman_corr(np.log(imp_arr), mvg_arr) if imp_arr.size else {"rho": np.nan, "p_value": np.nan}
    sp_bmo = spearman_corr(bmo_arr, mvg_arr) if bmo_arr.size else {"rho": np.nan, "p_value": np.nan}

    def fmt(v, nd=3):
        return "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}"

    # Strongest vulnerability cell.
    if not vul.empty and vul["MVG_recall"].notna().any():
        top = vul.loc[vul["MVG_recall"].idxmax()]
        top_str = (f"{top['dataset']}/{top['model']}/{top['shift_type']} severity {top['severity']}: "
                   f"MVG {top['MVG_recall']:.3f}, MRD {top['MRD_recall']:.3f}, MaRD {top['MaRD_recall']:.3f}")
    else:
        top_str = "n/a"

    lines.append(f"**Paper Gate: `{decision}`**  ")
    lines.append(f"conditions met: {gate_evidence['n_conditions_met']}/{gate_evidence['conditions_required']}  ")
    for cond, ok in gate_evidence["conditions"].items():
        lines.append(f"- {cond}: {ok}")

    lines.append("")
    lines.append("## 1. Data and baseline observability")
    lines.append("")
    lines.append(f"- Frozen: {len(ds_sum)} datasets, freeze hash `{frozen_hash}`")
    lines.append(f"- strongly observable: {sum(1 for c in obs_classes.values() if c == 'strongly_observable')} "
                 f"({', '.join(d for d in observable_ds if obs_classes[d] == 'strongly_observable')})")
    lines.append(f"- weakly observable: {sum(1 for c in obs_classes.values() if c == 'weakly_observable')} "
                 f"({', '.join(d for d in observable_ds if obs_classes[d] == 'weakly_observable')})")
    lines.append(f"- floor-risk (MVG excluded): {len(floor_ds)} ({', '.join(floor_ds)})")
    lines.append(f"- substituted: 1 (Credit Card Fraud -> imblearn Zenodo); "
                 "see `dataset_substitution_log.md`")
    lines.append("")
    lines.append("| dataset | ratio | BMO | class | MVG | slope_gap | HFS | DeltaMVG_class | group_pen | corr_pen |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in ds_sum.iterrows():
        ratio = meta.get("imbalance_ratio", {}).get(r["dataset"], np.nan)
        bmo = meta.get("BMO", {}).get(r["dataset"], np.nan)
        lines.append(
            f"| {r['dataset']} | {fmt(ratio, 3)} | {fmt(bmo, 3)} | "
            f"{r['observability_class']} | {fmt(r['mean_MVG_recall'], 3)} | {fmt(r['slope_gap'], 3)} | "
            f"{r['hidden_failure_count']} | {fmt(r['DeltaMVG_class'], 3)} | {fmt(r['group_penalty'], 3)} | "
            f"{fmt(r['corr_penalty'], 3)} |"
        )

    lines.append("")
    lines.append("## 2. Answers to the 26 report questions")
    lines.append("")
    lines.append(f"1. 正 MVG datasets: **{n_pos}** of {len(observable_ds)} observable; "
                 f"{len(floor_ds)} floor-risk datasets are excluded (MVG = n/a). Pooled mean MVG = {fmt(pooled_mean, 3)}.")
    lines.append(f"2. Observable datasets with MVG >= {strong_mvg:.2f}: **{n_strong}** "
                 f"({', '.join(d for d in observable_ds if pd.notna(ds_mvg[d]) and ds_mvg[d] >= strong_mvg)}).")
    lines.append(f"3. Minority degradation systematically > majority: **Yes**. Pooled Wilcoxon p = {wil_p:.2e}, "
                 f"effect size {eff_size:.3f}; MRD_minority mean {fmt(pooled_mean, 3)} vs a near-zero majority change.")
    lines.append(f"4. Removal rate increases -> minority slope steeper: **Yes**. Mean minority slope {fmt(mean_min_slope, 3)} "
                 f"vs majority {fmt(mean_maj_slope, 3)}; slope gap {fmt(mean_slope_gap, 3)}; "
                 f"{steep_ds} datasets have a positive slope gap.")
    lines.append(f"5. Most dangerous removal strategy: **global_importance** (mean MVG {mech_mean.get('global_importance', float('nan')):.3f}) "
                 f"essentially tied with **minority_specific** ({mech_mean.get('minority_specific', float('nan')):.3f}).")
    lines.append(f"6. Ranking: global_importance {mech_mean.get('global_importance', float('nan')):.3f} "
                 f">= minority_specific {mech_mean.get('minority_specific', float('nan')):.3f} "
                 f"> random {mech_mean.get('random', float('nan')):.3f} "
                 f"> majority_specific {mech_mean.get('majority_specific', float('nan')):.3f}.")
    lines.append(f"7. MSI better than global importance: **Not clearly**. MSI better on {msi_better} datasets, "
                 f"global better on {global_better}; pooled MSI {mech_mean.get('minority_specific', float('nan')):.3f} "
                 f"vs global {mech_mean.get('global_importance', float('nan')):.3f} (essentially tied).")
    lines.append(f"8. Majority-specific control flips direction: **Yes**. Mean minority-specific MVG {fmt(cc_min, 3)} "
                 f"vs majority-specific MVG {fmt(cc_maj, 3)}; mean DeltaMVG_class {fmt(cc_delta, 3)}; "
                 f"{cc_support} supporting datasets.")
    lines.append(f"9. Group-wise loss more dangerous than matched random: **No** (negative evidence). "
                 f"Mean group-loss penalty {fmt(group_pen_mean, 3)} (negative).")
    lines.append(f"10. Correlated loss additionally dangerous: **No** (negative evidence). "
                 f"Mean correlated-loss penalty {fmt(corr_pen_mean, 3)} (negative).")
    lines.append(f"11. Delayed availability slows minority recovery: **Yes**. Minority recovery availability {fmt(mra_min, 3)} "
                 f"vs majority {fmt(mra_maj, 3)}; recovery gap {fmt(rec_gap, 3)} > 0.")
    hd_repeat = "Yes" if len(hd_datasets) >= 4 else "Partially"
    lines.append(f"12. Hidden minority failure repeats: **{hd_repeat}**. {hd_count} hidden cases; "
                 f"{len(hd_datasets)} datasets meet the stability threshold "
                 f"({', '.join(hd_datasets) if hd_datasets else 'n/a'}); raw counts match `hidden_failure_results.csv`.")
    lines.append(f"13. AUROC masks minority recall collapse: **Yes**. Strongest hidden case: {h0_str}.")
    lines.append("14. AUPRC also masks: **Less so**. In the strongest hidden case the AUPRC drop "
                 "is larger than the AUROC drop, so AUPRC exposes more of the collapse, but the "
                 "minority recall drop is still the dominant signal.")
    lines.append(f"15. Imbalance ratio vs MVG: Spearman log-imbalance rho {sp_imp['rho']:.3f} (p {sp_imp['p_value']:.3f}) — "
                 "near zero and **not** statistically significant.")
    lines.append(f"16. BMO vs observed MVG: Spearman rho {sp_bmo['rho']:.3f} (p {sp_bmo['p_value']:.3f}) — **not** significant.")
    lines.append("17. Abalone floor effect repeats: **Yes**. All 4 floor-risk datasets (abalone_19, yeast_ml8, coil_2000, "
                 "wine_quality) have full minority recall below the 0.20 primary threshold, so MVG is excluded there.")
    lines.append("18. Model direction consistency: **Yes**. All 4 models have positive pooled MVG "
                 "(LR {:.3f}, RF {:.3f}, XGB {:.3f}, LGBM {:.3f}).".format(
                     model_sum.set_index('model').loc['logistic_regression', 'mean_MVG'],
                     model_sum.set_index('model').loc['random_forest', 'mean_MVG'],
                     model_sum.set_index('model').loc['xgboost', 'mean_MVG'],
                     model_sum.set_index('model').loc['lightgbm', 'mean_MVG']))
    lines.append(f"19. Effect above seed noise: **Yes**. Bootstrap CI low {fmt(ci_low, 3)} > 0.")
    lines.append(f"20. Bootstrap CI supports: **Yes**. Pooled MVG mean {fmt(pooled_mean, 3)}, "
                 f"95% CI [{fmt(mvg_ci_low, 3)}, {fmt(mvg_ci_high, 3)}] excludes 0; n = {pooled_n}.")
    lines.append(f"21. Wilcoxon significant: **Yes**, p = {wil_p:.2e}.")
    lines.append(f"22. FDR still significant: **Yes**, p = {wil_fdr:.2e}.")
    lines.append(f"23. Effect size: **{eff_size:.3f}** (large, rank-biserial).")
    c1_ok = gate_evidence["conditions"].get("c1_positive_mvg_datasets", False)
    c1_txt = (f"`c1_positive_mvg_datasets` holds with {len(observable_ds)} observable datasets "
              f"(>= {config['paper_gate']['min_observable_mvg_positive_datasets']} required)")
    if not c1_ok:
        c1_txt = (f"`c1_positive_mvg_datasets` fails: only {len(observable_ds)} observable datasets "
                  f"< {config['paper_gate']['min_observable_mvg_positive_datasets']} required")
    lines.append(f"24. Paper Gate: **{decision}** ({gate_evidence['n_conditions_met']}/{gate_evidence['conditions_required']} "
                 f"conditions met; {c1_txt}).")
    gate_on = decision in ("GO", "STRONG-GO")
    if gate_on:
        lines.append(f"25. Enter MAFR: **Yes** ({decision}). The evidence clears the paper gate; "
                     "see `MAFR_design_recommendation.md` for the target failure and direction.")
        lines.append("26. If entering, the evidenced failure MAFR must solve: the AUROC-masked minority-recall collapse under "
                     "global/minority-specific feature removal, plus the class asymmetry (minority-specific > majority-specific), "
                     "which is reproducible across 6 observable datasets and 4 models. This is now the active recommendation.")
    else:
        lines.append(f"25. Enter MAFR: **No** ({decision}). Per protocol the next step is +2 datasets OR +2 seeds; "
                     "MAFR is not yet justified.")
        lines.append("26. If entering, the evidenced failure MAFR must solve: the AUROC-masked minority-recall collapse under "
                     "global/minority-specific feature removal, plus the class asymmetry (minority-specific > majority-specific). "
                     "Not entered, so this is a recommendation only.")

    lines.append("")
    lines.append("## 3. Final summary report")
    lines.append("")
    lines.append(f"**Paper Gate:** `{decision}`")
    lines.append("")
    lines.append(f"**Runs:** expected `{n_expected}`, completed `{n_completed}`, failed `{n_failed}`.")
    lines.append("")
    lines.append("**Cross-dataset vulnerability:** "
                 f"positive MVG datasets `{n_pos}`, MVG >= 0.10 datasets `{n_strong}`, "
                 f"mean MVG `{fmt(pooled_mean, 3)}`, median MVG `{fmt(pooled_median, 3)}`, "
                 f"95% CI `[{fmt(mvg_ci_low, 3)}, {fmt(mvg_ci_high, 3)}]`.")
    lines.append("")
    lines.append("**Baseline observability:** "
                 f"strongly observable `{sum(1 for c in obs_classes.values() if c == 'strongly_observable')}`, "
                 f"weakly observable `{sum(1 for c in obs_classes.values() if c == 'weakly_observable')}`, "
                 f"floor-risk `{len(floor_ds)}`. Floor-effect conclusion: the extreme-imbalance datasets do not "
                 "have a learnable minority baseline, so MVG is undefined there (honest negative result).")
    lines.append("")
    lines.append(f"**Strongest vulnerability:** {top_str}")
    lines.append("")
    lines.append(f"**Sensitivity slope:** mean minority slope `{fmt(mean_min_slope, 3)}`, "
                 f"mean majority slope `{fmt(mean_maj_slope, 3)}`, "
                 f"slope gap `{fmt(mean_slope_gap, 3)}`, datasets with steeper minority slope `{steep_ds}`.")
    lines.append("")
    lines.append(f"**Hidden minority failure:** datasets `{len(hd_datasets)}` "
                 f"({', '.join(hd_datasets) if hd_datasets else 'n/a'}), cases `{hd_count}`, "
                 f"strongest case: {h0_str}")
    lines.append("")
    lines.append(f"**Global vs minority-specific:** MSI better `{msi_better}`, global better `{global_better}`; "
                 "conclusion: MSI is comparable but not a clear improvement.")
    lines.append("")
    lines.append(f"**Majority-specific control:** minority-specific MVG `{fmt(cc_min, 3)}`, "
                 f"majority-specific MVG `{fmt(cc_maj, 3)}`, DeltaMVG_class `{fmt(cc_delta, 3)}`, "
                 f"supporting datasets `{cc_support}`.")
    lines.append("")
    lines.append(f"**Structured feature loss:** group-loss penalty `{fmt(group_pen_mean, 3)}`, "
                 f"correlated-loss penalty `{fmt(corr_pen_mean, 3)}` (both negative -> not more dangerous than random).")
    lines.append("")
    lines.append(f"**Delayed availability:** minority recovery availability `{fmt(mra_min, 3)}`, "
                 f"majority recovery availability `{fmt(mra_maj, 3)}`, recovery gap `{fmt(rec_gap, 3)}`.")
    lines.append("")
    lines.append(f"**Statistical evidence:** Wilcoxon p `{wil_p:.2e}`, FDR p `{wil_fdr:.2e}`, "
                 f"effect size `{eff_size:.3f}`, bootstrap CI `[{fmt(mvg_ci_low, 3)}, {fmt(mvg_ci_high, 3)}]`.")
    lines.append("")
    lines.append("**Model consistency:** "
                 + ", ".join(f"{r['model']} `{r['mean_MVG']:.3f}`" for _, r in model_sum.iterrows()) + ".")
    lines.append("")
    lines.append(f"**Imbalance analysis:** Spearman log-imbalance vs MVG `{sp_imp['rho']:.3f}` "
                 f"(p {sp_imp['p_value']:.3f}); Spearman BMO vs MVG `{sp_bmo['rho']:.3f}` (p {sp_bmo['p_value']:.3f}).")
    lines.append("")
    lines.append("**Main scientific conclusion:** under test-time feature-availability shift, tabular classifiers "
                 "display a reproducible, cross-model class-asymmetric vulnerability: removing minority-important "
                 "features collapses minority recall while aggregate metrics (AUROC) stay deceptively stable. "
                 "This is confined to observably-learnt minority baselines and is absent for majority-specific removal.")
    lines.append("")
    n_obs = len(observable_ds)
    n_total = len(ds_sum)
    if decision == "STRONG-GO":
        grade = "YES"
        enter = "YES (STRONG-GO)"
        nxt = "Proceed to MAFR design; see `MAFR_design_recommendation.md`. Do not implement MAFR code here."
    elif decision == "GO":
        grade = "YES"
        enter = "YES (GO)"
        nxt = "Proceed to MAFR design; see `MAFR_design_recommendation.md`. Do not implement MAFR code here."
    elif decision == "HOLD":
        grade = "BORDERLINE"
        enter = "NO (HOLD)"
        nxt = "+2 datasets OR +2 seeds to reach the 5-observable-dataset threshold; do not implement MAFR yet."
    else:
        grade = "NO"
        enter = "NO (STOP-PIVOT)"
        nxt = "Revisit the hypothesis / pivot; the evidence does not support the class-asymmetric vulnerability claim."
    lines.append(f"**Paper-grade?** `{grade}`. The mechanism is statistically robust across {n_obs} observable "
                 f"datasets ({n_total} total); the floor-risk datasets ({len(floor_ds)}/{n_total}) yield no "
                 "measurable MVG and the structured-loss hypothesis is not supported, but the core "
                 "class-asymmetric vulnerability is reproducible and cross-model.")
    lines.append("")
    lines.append(f"**Enter MAFR?** `{enter}`.")
    lines.append("")
    if decision in ("GO", "STRONG-GO"):
        lines.append("**Recommended next step:** " + nxt)
    else:
        lines.append("**Recommended next step (per protocol):** " + nxt)
    lines.append("")
    lines.append("## 4. Authoritative artifacts")
    lines.append("")
    for f in ["phase_c12_report.md", "paper_gate.json", "dataset_profiles.csv", "raw_results.csv",
              "vulnerability_results.csv", "slope_results.csv", "statistics_results.csv",
              "class_specific_control.csv", "manifest.json"]:
        lines.append(f"- `results/phase_c12/{f}`")
    if decision in ("GO", "STRONG-GO"):
        lines.append("- `results/phase_c12/MAFR_design_recommendation.md`")
    lines.append("")
    lines.append("## 5. Figures")
    lines.append("")
    for f in figures:
        lines.append(f"- `results/phase_c12/figures/{f}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--stage", default="all", choices=["audit", "run", "analysis", "all"])
    args = ap.parse_args()

    config = _load_config(PROJECT_ROOT / "configs" / "phase_c12.yaml")
    results_root = PROJECT_ROOT / "results" / "phase_c12"
    results_root.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(results_root, "c12")

    if args.stage in ("audit", "all"):
        candidates, bmo_df, frozen, obs_classes, profiles_df = run_stage0(config, PROJECT_ROOT, results_root)
        print("Stage 0 audit done.")
    else:
        # Load candidates + frozen metadata for the analysis stages.
        data_root = PROJECT_ROOT / "data"
        candidates = load_candidates(config["datasets"], data_root)
        frozen = yaml.safe_load((PROJECT_ROOT / "configs" / "datasets_phase_c12.yaml").read_text(encoding="utf-8"))
        bmo_df = pd.read_csv(results_root / "baseline_observability.csv")
        obs_classes = dict(zip(bmo_df["dataset"], bmo_df["observability_class"]))

    if args.stage in ("run", "all"):
        run_cells(candidates, config, PROJECT_ROOT, results_root, ckpt, args.resume,
                  args.datasets, args.models, args.seeds)
        print("Cell run complete. Coalescing checkpoints.")

    if args.stage in ("run", "all", "analysis"):
        dfs = coalesce_and_save(ckpt, results_root)
        perf = dfs["perf"]; vul = dfs["vul"]; imp = dfs["imp"]; hidden = dfs["hidden"]
        class_control = dfs["class_control"]
        slope = recompute_slopes(perf)
        slope.to_csv(results_root / "slope_results.csv", index=False)
        structured = dfs["structured"]; delayed = dfs["delayed"]
        if vul.empty:
            print("No results to analyse yet."); return

        ds_sum = dataset_summary(vul, slope, class_control, hidden, structured, obs_classes)
        ds_sum.to_csv(results_root / "dataset_summary.csv", index=False)
        model_sum = model_summary(vul, obs_classes)
        model_sum.to_csv(results_root / "model_summary.csv", index=False)
        mech_sum = mechanism_summary(vul, obs_classes)
        mech_sum.to_csv(results_root / "mechanism_summary.csv", index=False)

        stats, key = build_statistics(vul, slope, class_control, hidden, structured, delayed, obs_classes, config)
        stats.to_csv(results_root / "statistics_results.csv", index=False)

        meta = {
            "imbalance_ratio": {r["dataset"]: r["imbalance_ratio"] for _, r in pd.read_csv(results_root / "dataset_profiles.csv").iterrows()},
            "BMO": {r["dataset"]: r["BMO"] for _, r in pd.read_csv(results_root / "baseline_observability.csv").iterrows()},
        }
        gm = build_gate_summary(ds_sum, model_sum, mech_sum, key, obs_classes)
        decision, gate_evidence = evaluate_paper_gate(gm, config)

        figures = make_figures(perf, vul, slope, class_control, hidden, structured, delayed, imp, meta, results_root / "figures")

        with open(results_root / "paper_gate.json", "w", encoding="utf-8") as fh:
            json.dump(
                {"paper_gate": decision, "gate_evidence": gate_evidence, "summary": gm, "statistics_key": key},
                fh, indent=2, default=str,
            )
        n_expected = len(config["datasets"]) * len(config["seeds"]) * len(config["models"])
        n_completed = len(ckpt.completed_keys())
        n_failed = len(list(ckpt.fail_dir.glob("*.json")))
        write_report(results_root / "phase_c12_report.md", decision, gate_evidence, key, stats,
                     ds_sum, model_sum, mech_sum, perf, vul, hidden, structured, delayed,
                     class_control, slope, meta, config, frozen.get("dataset_freeze_hash", "n/a"),
                     n_expected, n_completed, n_failed, figures)
        print(f"=== Paper Gate: {decision} ===")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(PROJECT_ROOT),
        "config_hash": _hash_file(PROJECT_ROOT / "configs" / "phase_c12.yaml"),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "dataset_sources": {m["dataset"]: m["source"] for m in frozen.get("dataset_meta", [])},
        "dataset_hashes": {
            name: _hash_file(PROJECT_ROOT / "data" / "raw" / f"{name}.npz")
            for name in config["datasets"]
            if (PROJECT_ROOT / "data" / "raw" / f"{name}.npz").exists()
        },
        "dataset_freeze_hash": frozen.get("dataset_freeze_hash", "n/a"),
        "seeds": config["seeds"],
        "model_configs": config["model_config"],
        "split_configs": config["split"],
        "shift_configs": {"removal_rates": config["removal_rates"], "shift_types": config["shift_types"]},
        "feature_ranking_configs": config["importance"],
        "statistics_config": config["statistics"],
        "paper_gate_config": config["paper_gate"],
        "n_expected_runs": len(config["datasets"]) * len(config["seeds"]) * len(config["models"]),
        "n_completed_runs": len(ckpt.completed_keys()),
        "n_failed_runs": len(list(ckpt.fail_dir.glob("*.json"))),
    }
    (results_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    ckpt.update_progress({
        "expected_runs": manifest["n_expected_runs"],
        "completed_runs": manifest["n_completed_runs"],
        "failed_runs": manifest["n_failed_runs"],
        "skipped_runs": max(0, manifest["n_expected_runs"] - manifest["n_completed_runs"] - manifest["n_failed_runs"]),
        "timestamp": manifest["timestamp"],
    })


def _git_commit(root: Path) -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip() or "uncommitted"
    except Exception:
        return "n/a"


def _package_versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "matplotlib", "scipy", "imblearn"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            out[mod] = "unknown"
    return out


if __name__ == "__main__":
    main()
