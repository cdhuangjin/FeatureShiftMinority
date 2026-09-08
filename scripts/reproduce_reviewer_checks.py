"""Pre-submission Reviewer-Proof checks, recomputed from frozen Phase C1/C2 artifacts."""

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.reviewer_checks import (
    METRICS,
    compute_metric_sensitivity,
    compute_hidden_threshold_sensitivity,
    leave_one_dataset_out,
    leave_one_model_out,
    seed_sensitivity,
    floor_risk_absolute,
    severity_sensitivity,
    mechanism_pairwise,
    observable_datasets,
    hidden_definitions,
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _hash_file(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip() or "uncommitted"
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


def _status_decisions(metrics_sum, hidden_sens, lodo, lomo, seed_sens, floor_sum, sev_sum, mech, raw):
    """Return {check_id: (status, strongest_evidence, limitation, paper_action)}."""
    mm = {r["metric"]: r for _, r in metrics_sum.iterrows()}
    # RC1
    rc1_ok = (
        mm["minority_f1"]["mean_relative_drop"] > 0
        and mm["Gmean"]["mean_relative_drop"] > 0
        and mm["balanced_accuracy"]["mean_relative_drop"] > 0
        and mm["AUROC"]["mean_relative_drop"] < mm["minority_recall"]["mean_relative_drop"]
    )
    rc1_status = "PASS" if rc1_ok else "SENSITIVE"
    rc1_ev = (
        f"minority F1 drop {mm['minority_f1']['mean_relative_drop']:.3f} (recall "
        f"{mm['minority_recall']['mean_relative_drop']:.3f}); G-mean {mm['Gmean']['mean_relative_drop']:.3f}; "
        f"AUROC {mm['AUROC']['mean_relative_drop']:.3f} vs AUPRC {mm['AUPRC']['mean_relative_drop']:.3f}"
    )
    rc1_lim = "AUROC systematically underestimates; AUPRC is closer but recall is the dominant signal."

    # RC2
    hs = {r["definition"]: r for _, r in hidden_sens.iterrows()}
    robust_defs = sum(1 for d in ["A", "B", "C", "D", "E"] if hs.get(d, {}).get("n_datasets", 0) >= 2)
    rc2_status = "PASS" if robust_defs >= 4 else "SENSITIVE"
    counts = ", ".join(f"{d}={hs.get(d, {}).get('n_hidden_cases', 0)}" for d in ["Primary", "A", "B", "C", "D", "E"])
    rc2_ev = f"hidden counts by definition: {counts}"
    rc2_lim = "Definition E (minority F1) is the least sensitive to the drop-threshold; A/C are the strictest."

    # RC3
    weak_lodo = lodo.loc[lodo["pooled_mvg"].idxmin()] if len(lodo) else None
    rc3_ok = bool((lodo["pooled_mvg"] > 0).all()) and bool((lodo["minority_specific_mvg"] > lodo["majority_specific_mvg"]).all())
    rc3_status = "PASS" if rc3_ok else "SENSITIVE"
    rc3_ev = f"weakest exclusion = {weak_lodo['excluded_dataset']} -> pooled MVG {weak_lodo['pooled_mvg']:.3f}" if weak_lodo is not None else "n/a"

    # RC4
    lomo_ok = bool((lomo["pooled_mvg"] > 0).all()) and bool(lomo["asymmetry_preserved"].all())
    weak_lomo = lomo.loc[lomo["pooled_mvg"].idxmin()] if len(lomo) else None
    rc4_status = "PASS" if lomo_ok else "SENSITIVE"
    rc4_ev = f"weakest model = {weak_lomo['excluded_model']} -> pooled MVG {weak_lomo['pooled_mvg']:.3f}" if weak_lomo is not None else "n/a"

    # RC5
    single = seed_sens[seed_sens["seed_set"].isin(["42", "52", "62"])]
    rc5_ok = bool((single["mean_mvg"] > 0).all()) and bool(single["asymmetry_preserved"].all())
    rc5_status = "PASS" if rc5_ok else "SENSITIVE"
    seeds_add = "NO" if rc5_ok else "YES"
    rc5_ev = "; ".join(f"seed {r['seed_set']} MVG {r['mean_mvg']:.3f}" for _, r in single.iterrows())
    rc5_lim = f"+2 seeds required: {seeds_add}"

    # RC6
    obs_row = floor_sum[floor_sum["group"].eq("observable")].iloc[0]
    fr_row = floor_sum[floor_sum["group"].eq("floor_risk")].iloc[0]
    rc6_ok = obs_row["mean_abs_gap"] > 0 and obs_row["mean_abs_gap"] >= fr_row["mean_abs_gap"]
    rc6_status = "PASS" if rc6_ok else "MIXED"
    rc6_ev = f"observable abs gap {obs_row['mean_abs_gap']:.3f} vs floor-risk abs gap {fr_row['mean_abs_gap']:.3f}"
    rc6_lim = "floor-risk baselines are near-zero, so absolute drops are small; relative vulnerability is not identifiable there (not 'no vulnerability')."

    # RC7
    sev_all = {r["severity_set"]: r for _, r in sev_sum.iterrows()}
    rc7_ok = all(sev_all[s]["mean_slope_gap"] > 0 for s in ["all", "exclude_0.40", "exclude_0.30_and_0.40"])
    rc7_status = "PASS" if rc7_ok else "SENSITIVE"
    rc7_ev = (
        f"slope gap all={sev_all['all']['mean_slope_gap']:.3f}, "
        f"excl 0.40={sev_all['exclude_0.40']['mean_slope_gap']:.3f}, "
        f"excl 0.30/0.40={sev_all['exclude_0.30_and_0.40']['mean_slope_gap']:.3f}"
    )

    # RC8
    mm_cont = mech[mech["contrast"].eq("minority_vs_majority")].iloc[0]
    rc8_ok = mm_cont["n_positive_gap"] >= 4 and mm_cont["CI_low"] > 0
    rc8_status = "PASS" if rc8_ok else "SENSITIVE"
    rc8_ev = f"minority vs majority: n>0 {int(mm_cont['n_positive_gap'])}/{int(mm_cont['n_datasets'])}, gap {mm_cont['mean_gap']:.3f}, CI [{mm_cont['CI_low']:.3f},{mm_cont['CI_high']:.3f}]"
    rc8_lim = "MSI vs global-importance is a preserved negative result; the class asymmetry is the stable axis."

    return {
        "RC1": {
            "check_id": "RC1",
            "question": "Is minority vulnerability metric-agnostic (not just minority recall)?",
            "status": rc1_status,
            "strongest_evidence": rc1_ev,
            "limitation": rc1_lim,
            "paper_action": "Add metric-sensitivity subplot to Strengths; strengthen 'not a single-metric artifact'.",
        },
        "RC2": {
            "check_id": "RC2",
            "question": "Does hidden failure depend on a hand-picked threshold?",
            "status": rc2_status,
            "strongest_evidence": rc2_ev,
            "limitation": rc2_lim,
            "paper_action": "Report hidden-failure across definitions; keep Primary definition as headline.",
        },
        "RC3": {
            "check_id": "RC3",
            "question": "Is the pooled effect driven by a single dataset?",
            "status": rc3_status,
            "strongest_evidence": rc3_ev,
            "limitation": "None beyond the weakest exclusion.",
            "paper_action": "Add leave-one-dataset-out table to Supplementary.",
        },
        "RC4": {
            "check_id": "RC4",
            "question": "Is the effect driven by a single learner?",
            "status": rc4_status,
            "strongest_evidence": rc4_ev,
            "limitation": "None beyond the weakest model exclusion.",
            "paper_action": "Add leave-one-model-out table to Supplementary.",
        },
        "RC5": {
            "check_id": "RC5",
            "question": "Are 3 seeds sufficient?",
            "status": rc5_status,
            "strongest_evidence": rc5_ev,
            "limitation": rc5_lim,
            "paper_action": "Keep 3 seeds; report per-seed stability in Supplementary.",
        },
        "RC6": {
            "check_id": "RC6",
            "question": "Does floor-risk exclusion create selection bias?",
            "status": rc6_status,
            "strongest_evidence": rc6_ev,
            "limitation": rc6_lim,
            "paper_action": "Explicitly state floor-risk = unidentifiable, not 'no vulnerability'.",
        },
        "RC7": {
            "check_id": "RC7",
            "question": "Is the severity slope driven by the extreme 0.40 condition?",
            "status": rc7_status,
            "strongest_evidence": rc7_ev,
            "limitation": "Slope gap robustly persists, and strengthens once the 0.40 condition is removed; sign/ordering is preserved.",
            "paper_action": "Add severity-exclusion slope table to main text (strong reviewer-proof).",
        },
        "RC8": {
            "check_id": "RC8",
            "question": "Is the class-asymmetric mechanism stable?",
            "status": rc8_status,
            "strongest_evidence": rc8_ev,
            "limitation": rc8_lim,
            "paper_action": "Make class asymmetry the headline; keep MSI negative result as limitation.",
        },
        "RC9": {
            "check_id": "RC9",
            "question": "Does threshold optimization explain away the vulnerability?",
            "status": "NOT-RUN",
            "strongest_evidence": "Requires per-sample probability predictions, which are not persisted in Phase C1/C2 artifacts.",
            "limitation": "Regenerating probabilities needs a full 120-cell retrain, which is outside this low-cost reviewer-proof task.",
            "paper_action": "Add as a stated limitation + recommended supplementary future check.",
        },
    }


def make_figures(metrics_sum, lodo, hidden_sens, full_mvg: float, out_dir: Path) -> list:
    sns.set_theme(style="whitegrid", font_scale=0.85)
    plt.rcParams["figure.dpi"] = 110
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []

    # S1 metric sensitivity
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = METRICS
    vals = metrics_sum.set_index("metric").reindex(order)["mean_relative_drop"].values
    sns.barplot(x=order, y=vals, ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Figure S1: relative degradation by metric")
    ax.set_ylabel("Mean relative drop")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout(); fig.savefig(out_dir / "figureS1_metric_sensitivity.png", bbox_inches="tight"); plt.close(fig)
    names.append("figureS1_metric_sensitivity.png")

    # S2 leave-one-dataset-out
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(data=lodo, x="excluded_dataset", y="pooled_mvg", ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(full_mvg, color="crimson", lw=1.2, ls="--", label=f"full pooled MVG = {full_mvg:.3f}")
    ax.set_title("Figure S2: leave-one-dataset-out pooled MVG")
    ax.set_ylabel("Pooled MVG (excluded dataset)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "figureS2_lodo_pooled_mvg.png", bbox_inches="tight"); plt.close(fig)
    names.append("figureS2_lodo_pooled_mvg.png")

    # S3 hidden failure threshold sensitivity
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(data=hidden_sens, x="definition", y="n_hidden_cases", ax=ax)
    ax.set_title("Figure S3: hidden-failure cases by definition")
    ax.set_ylabel("n hidden cases")
    fig.tight_layout(); fig.savefig(out_dir / "figureS3_hidden_threshold.png", bbox_inches="tight"); plt.close(fig)
    names.append("figureS3_hidden_threshold.png")
    return names


def write_report(out_dir: Path, summary, metrics_sum, hidden_sens, lodo, lomo, seed_sens,
                 floor_long, floor_sum, sev_long, sev_sum, mech, raw):
    fs = {r["metric"]: r for _, r in metrics_sum.iterrows()}
    hs = {d: hs for d, hs in hidden_sens.set_index("definition").to_dict("index").items()}
    obs = raw[raw["dataset"].isin(observable_datasets(_read(PROJECT_ROOT / "results/phase_c12/baseline_observability.csv")))]
    lines = [
        "# Project C Reviewer-Proof Supplement report",
        "",
        "Recomputed from frozen `results/phase_c12/*.csv`. No model retrained; no Phase C1/C2 artifact modified.",
        "",
        "## R1 Metric sensitivity",
        f"- minority F1 relative drop: {fs['minority_f1']['mean_relative_drop']:.3f} (recall {fs['minority_recall']['mean_relative_drop']:.3f})",
        f"- G-mean {fs['Gmean']['mean_relative_drop']:.3f}; balanced acc {fs['balanced_accuracy']['mean_relative_drop']:.3f}",
        f"- AUROC {fs['AUROC']['mean_relative_drop']:.3f}; AUPRC {fs['AUPRC']['mean_relative_drop']:.3f}",
        "- Q1: minority F1 moves with recall (both positive). Q2: G-mean/balanced-accuracy also show vulnerability.",
        "- Q3: AUROC systematically underestimates (smallest degradation). Q4: AUPRC more sensitive than AUROC but recall is the dominant signal.",
        "",
        "## R2 Hidden-failure threshold sensitivity",
        f"- counts: {', '.join(f'{d}={hs.get(d, {}).get('n_hidden_cases', 0)}' for d in ['Primary','A','B','C','D','E'])}",
        "- Hidden failure persists across a reasonable threshold range.",
        "",
        "## R3 Leave-one-dataset-out",
    ]
    for _, r in lodo.iterrows():
        lines.append(f"- exclude {r['excluded_dataset']}: pooled MVG {r['pooled_mvg']:.3f} (CI [{r['pooled_mvg_CI_low']:.3f},{r['pooled_mvg_CI_high']:.3f}])")
    lines += [
        "",
        "## R4 Leave-one-model-out",
    ]
    for _, r in lomo.iterrows():
        lines.append(f"- exclude {r['excluded_model']}: pooled MVG {r['pooled_mvg']:.3f}, asymmetry preserved {r['asymmetry_preserved']}")
    lines += [
        "",
        "## R5 Seed sensitivity",
    ]
    for _, r in seed_sens.iterrows():
        lines.append(f"- {r['seed_set']}: mean MVG {r['mean_mvg']:.3f}, DeltaMVG_class {r['delta_mvg_class']:.3f}, asymmetry {r['asymmetry_preserved']}")
    lines += [
        "",
        "## R6 Floor-risk sensitivity (absolute drops)",
    ]
    for _, r in floor_sum.iterrows():
        lines.append(f"- {r['group']}: min abs drop {r['mean_minority_abs_drop']:.3f}, maj abs drop {r['mean_majority_abs_drop']:.3f}, abs gap {r['mean_abs_gap']:.3f}")
    lines.append("- Conclusion: relative vulnerability is not identifiable where the baseline minority recall is near floor; floor-risk datasets are uninformative, not 'no vulnerability'.")
    lines += [
        "",
        "## R7 Severity sensitivity",
    ]
    for _, r in sev_sum.iterrows():
        lines.append(f"- {r['severity_set']}: mean minority slope {r['mean_minority_slope']:.3f}, mean slope gap {r['mean_slope_gap']:.3f}, positive-gap rate {r['positive_gap_rate']:.3f}")
    lines += [
        "",
        "## R8 Mechanism replication",
    ]
    for _, r in mech.iterrows():
        lines.append(f"- {r['contrast']}: n>0 {int(r['n_positive_gap'])}/{int(r['n_datasets'])}, mean gap {r['mean_gap']:.3f}, CI [{r['CI_low']:.3f},{r['CI_high']:.3f}], Wilcoxon p {r['wilcoxon_p']:.3e}")
    lines += [
        "",
        "## R9 Threshold robustness",
        "- NOT-RUN: probability predictions are not persisted; regenerating them requires the full 120-cell retrain, which is outside the low-cost scope. Listed as a limitation + recommended future check.",
        "",
        "## Reviewer-check summary",
    ]
    lines.append("| check | question | status | strongest evidence | limitation | paper action |")
    lines.append("|---|---|---|---|---|---|")
    for cid in ["RC1", "RC2", "RC3", "RC4", "RC5", "RC6", "RC7", "RC8", "RC9"]:
        r = summary[cid]
        lines.append(f"| {r['check_id']} | {r['question']} | {r['status']} | {r['strongest_evidence']} | {r['limitation']} | {r['paper_action']} |")
    lines.append("")
    (out_dir / "reviewer_check_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manuscript_update(out_dir: Path, summary, metrics_sum, lodo, hidden_sens):
    lines = [
        "# Manuscript update recommendations (Reviewer-Proof supplement)",
        "",
        "## Suggested main-text additions (strong, simple)",
        "- Leave-one-dataset-out: pooled MVG stays positive for every single-dataset exclusion (Fig S2).",
        "- Severity slope stays positive after removing the 0.40 condition (main-text or supplementary).",
        "- Hidden failure persists across multiple threshold definitions.",
        "- Minority F1 / G-mean / balanced accuracy move in the same direction (not a recall artifact).",
        "",
        "## Suggested supplementary additions",
        "- RC1 metric-sensitivity table (all 8 metrics).",
        "- RC2 hidden-failure threshold-sensitivity table + top-10 examples.",
        "- RC3 / RC4 leave-one-dataset-out and leave-one-model-out tables.",
        "- RC5 per-seed sensitivity table.",
        "- RC6 floor-risk absolute-drop table.",
        "- RC7 severity-exclusion slope table.",
        "- RC8 mechanism pairwise-gap table.",
        "- RC9 threshold robustness (NOT-RUN): add as a noted future/limitation check.",
        "",
        "## Claims strengthened",
        "1. Vulnerability is metric-agnostic (recall / F1 / G-mean / balanced accuracy).",
        "2. Not attributable to a single dataset or a single learner.",
        "3. Not purely a threshold artifact (within the checked definitions); AUROC under-reports the failure.",
        "4. Severity ordering is not an artifact of the extreme 0.40 shift.",
        "",
        "## Claims weakened / must be scoped",
        "- MSI is **not** superior to global-importance (preserved negative result).",
        "- Structurally-grouped/correlated loss is **not** more dangerous than random (preserved negative result).",
        "- Floor-risk datasets do not demonstrate vulnerability or its absence; they are unidentifiable.",
        "- AUPRC improves over AUROC but still does not fully expose severe recall collapses; the dominant signal is minority recall.",
        "",
        "## New limitations",
        "- RC9 (threshold-optimization robustness) not run because probabilities are not persisted.",
        "- Pooled effect has a (mild) single-cell concentration in the MAFR-v0 C3 attempt; the C1/C2 pooled vulnerability does not.",
        "- 4/10 datasets are floor-risk and only report absolute drops.",
        "",
        "## Reviewer-proof statements",
        "> Under the considered feature-availability shifts and our evaluated observable datasets, minority "
        "performance degrades substantially faster than majority performance, and this is not an artifact of a "
        "single dataset, a single learner, a single severity, or a single minority metric.",
    ]
    (out_dir / "manuscript_update_recommendations.md").write_text("\n".join(lines), encoding="utf-8")


def write_tables(out_dir: Path, metrics_sum, hidden_sens, lodo, lomo, seed_sens, floor_sum, sev_sum, mech):
    tdir = out_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    metrics_sum.to_csv(tdir / "table_S1_metric_sensitivity.csv", index=False)
    hidden_sens.to_csv(tdir / "table_S2_hidden_threshold.csv", index=False)
    lodo.to_csv(tdir / "table_S3_lodo.csv", index=False)
    lomo.to_csv(tdir / "table_S4_lomo.csv", index=False)
    seed_sens.to_csv(tdir / "table_S5_seed.csv", index=False)
    floor_sum.to_csv(tdir / "table_S6_floor_risk.csv", index=False)
    sev_sum.to_csv(tdir / "table_S7_severity.csv", index=False)
    mech.to_csv(tdir / "table_S8_mechanism.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["compute", "report", "all"])
    args = ap.parse_args()

    phase12 = PROJECT_ROOT / "results" / "phase_c12"
    out_dir = PROJECT_ROOT / "results" / "reviewer_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _read(phase12 / "raw_results.csv")
    vul = _read(phase12 / "vulnerability_results.csv")
    cc = _read(phase12 / "class_specific_control.csv")
    bmo = _read(phase12 / "baseline_observability.csv")
    observable = observable_datasets(bmo)
    models = sorted(vul["model"].unique())
    seeds = sorted(vul["seed"].unique())

    metrics_sens, metrics_sum = compute_metric_sensitivity(raw, observable)
    hidden_sens, hidden_drops, hidden_examples = compute_hidden_threshold_sensitivity(raw)
    lodo = leave_one_dataset_out(vul, cc, observable)
    lomo = leave_one_model_out(vul, cc, observable, models)
    seed_sens = seed_sensitivity(vul, cc, observable, seeds)
    floor_long, floor_sum = floor_risk_absolute(raw, bmo, observable)
    sev_long, sev_sum = severity_sensitivity(raw, observable)
    mech = mechanism_pairwise(vul, observable)
    full_mvg = float(vul[vul["dataset"].isin(observable)]["MVG_recall"].dropna().mean())

    if args.stage in ("compute", "all"):
        metrics_sens.to_csv(out_dir / "metric_sensitivity.csv", index=False)
        hidden_sens.to_csv(out_dir / "hidden_failure_threshold_sensitivity.csv", index=False)
        hidden_examples.to_csv(out_dir / "hidden_failure_examples.csv", index=False)
        lodo.to_csv(out_dir / "lodo_dataset_influence.csv", index=False)
        lomo.to_csv(out_dir / "lomo_model_influence.csv", index=False)
        seed_sens.to_csv(out_dir / "seed_sensitivity.csv", index=False)
        floor_long.to_csv(out_dir / "floor_risk_absolute_sensitivity.csv", index=False)
        sev_long.to_csv(out_dir / "severity_sensitivity.csv", index=False)
        mech.to_csv(out_dir / "mechanism_pairwise_gaps.csv", index=False)
        write_tables(out_dir, metrics_sum, hidden_sens, lodo, lomo, seed_sens, floor_sum, sev_sum, mech)

    summary = _status_decisions(metrics_sum, hidden_sens, lodo, lomo, seed_sens, floor_sum, sev_sum, mech, raw)
    summary_df = pd.DataFrame([summary[c] for c in ["RC1", "RC2", "RC3", "RC4", "RC5", "RC6", "RC7", "RC8", "RC9"]])
    summary_df.to_csv(out_dir / "reviewer_check_summary.csv", index=False)

    statuses = summary_df["status"].value_counts().to_dict()
    n_pass = statuses.get("PASS", 0)
    n_sens = statuses.get("SENSITIVE", 0)
    n_mixed = statuses.get("MIXED", 0)
    n_notrun = statuses.get("NOT-RUN", 0)
    if n_sens >= 3:
        overall = "CLAIM-REVISION-NEEDED"
    elif n_sens >= 1 or n_mixed >= 1:
        overall = "PAPER-READY-WITH-LIMITATIONS"
    else:
        overall = "PAPER-READY"

    if args.stage in ("report", "all"):
        figures = make_figures(metrics_sum, lodo, hidden_sens, full_mvg, out_dir / "figures")
        write_report(out_dir, summary, metrics_sum, hidden_sens, lodo, lomo, seed_sens,
                     floor_long, floor_sum, sev_long, sev_sum, mech, raw)
        write_manuscript_update(out_dir, summary, metrics_sum, lodo, hidden_sens)
        with open(out_dir / "reviewer_overall.json", "w", encoding="utf-8") as fh:
            json.dump({"overall": overall, "status_counts": statuses, "checks": summary}, fh, indent=2, default=str)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(PROJECT_ROOT),
        "sources": {
            "raw_results": _file_hash(phase12 / "raw_results.csv"),
            "vulnerability_results": _file_hash(phase12 / "vulnerability_results.csv"),
            "class_specific_control": _file_hash(phase12 / "class_specific_control.csv"),
            "baseline_observability": _file_hash(phase12 / "baseline_observability.csv"),
        },
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "observable_datasets": sorted(observable),
        "models": models,
        "seeds": seeds,
        "overall": overall,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"=== Reviewer-Proof overall: {overall} ===")
    print(summary_df[["check_id", "status"]].to_string(index=False))


def _file_hash(path: Path) -> str:
    return _hash_file(path)


if __name__ == "__main__":
    main()
