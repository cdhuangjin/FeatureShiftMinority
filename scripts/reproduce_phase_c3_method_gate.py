"""One-shot + resumable reproduction for Project C Phase C3 MAFR Method Gate."""

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

from src.dataset_registry import load_candidates
from src.checkpoint_manager import CheckpointManager
from src.mafr_selection import run_cell, build_cell_plan, coalesce_checkpoint_payloads
from src.method_gate_analysis import (
    cell_aggregates,
    model_mvg_reduction,
    hidden_failure_summary,
    block_statistics,
    evaluate_method_gate,
    make_figures,
    SHIFTED_ENVS,
)


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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


def write_pre_registration(results_root: Path, config: dict) -> None:
    mafr = config["mafr"]
    mg = config["method_gate"]
    lines = [
        "# Phase C3 MAFR Method Gate pre-registration",
        "",
        "This file is written before any run. The Method Gate decision is produced from the",
        "pre-registered conditions below; results are not used to move the gate.",
        "",
        "## Fixed experiment",
        f"- datasets: {config['datasets']}",
        f"- models: {config['models']}",
        f"- seeds: {config['seeds']}",
        f"- split: {config['split']}",
        f"- test shifts: {config['test_shifts']}",
        "",
        "## MAFR-v0 definition",
        f"- weights variants: {json.dumps(mafr['weights_variants'])}",
        f"- retention ratios: {mafr['retention_ratios']} (default {mafr['default_retention']}, min {mafr['min_features']} features)",
        f"- validation stress: {json.dumps(mafr['validation_stress'])}",
        "- score = w1*norm(I_minority) + w2*norm(class_specificity) + w3*norm(redundancy) + w4*norm(availability_robustness)",
        "- variant/retention selected by source-validation stress AUPRC only; frozen before test evaluation.",
        "",
        "## Method Gate conditions",
        f"- MG1 minority robustness: >= {mg['mg1_min_cells']}/6 cells mean MVG_Reduction >= {mg['mg1_min_reduction']}",
        f"- MG2 minority recall: >= {mg['mg2_min_cells']}/6 cells with MR_Recovery >= {mg['mg2_min_recovery']} in a global/minority shift",
        f"- MG3 hidden failure: HFR reduction >= {mg['mg3_hfr_reduction_min']}",
        f"- MG4 IID AUPRC: mean drop <= {mg['mg4_max_mean_auprc_drop']}, each cell <= {mg['mg4_max_cell_auprc_drop']}",
        f"- MG5 IID AUROC: mean drop <= {mg['mg5_max_mean_auroc_drop']}",
        f"- MG6 strong baseline: >= {mg['mg6_min_cells']}/6 cells worst-case MAFR beats best of B1/B2/B3",
        "- MG7 cross-model: LR and XGB mean MVG_Reduction both > 0",
        f"- MG8 statistics: paired bootstrap CI supports positive improvement and effect size >= {mg['mg8_min_effect_size']}",
        f"- decision scale: STRONG-GO / GO / HOLD / STOP-METHOD; STRONG-GO requires MG1-MG8, "
        f"GO requires core {mg['strong_go_conditions_required']} conditions with a partial MG3/MG8.",
        "",
        "## Anti-pseudo-replication",
        "Formal block = dataset x model x seed. Shift environments are aggregated within a block before",
        "any paired comparison. Fine-grained rows are descriptive/sensitivity only.",
    ]
    (results_root / "pre_registration.md").write_text("\n".join(lines), encoding="utf-8")


def write_method_spec(results_root: Path, config: dict) -> None:
    mafr = config["mafr"]
    lines = [
        "# MAFR-v0 method specification",
        "",
        "## Feature score components (source train/validation only)",
        "1. global permutation importance (validation, AUPRC)",
        "2. minority-specific permutation importance (minority recall)",
        "3. majority-specific permutation importance (majority recall)",
        "4. class specificity = minority importance - majority importance",
        "5. train-only redundancy / substitutability (max absolute Spearman correlation)",
        "6. validation availability-stress robustness (leave-one-out minority-recall retention)",
        "",
        "## Score",
        "`MAFRScore = w1*norm(I_minority) + w2*norm(class_specificity) + w3*norm(redundancy) + w4*norm(availability_robustness)`",
        "",
        "Allowed weights:",
        f"- MAFR-A = {mafr['weights_variants']['MAFR-A']}",
        f"- MAFR-B = {mafr['weights_variants']['MAFR-B']}",
        f"- MAFR-C = {mafr['weights_variants']['MAFR-C']}",
        "",
        "## Selection",
        "- Variant and retention are chosen by the pre-registered source-validation stress AUPRC.",
        "- Retention options 70/80/90% (default 80%); at least 3 original features kept.",
        "- Output = robust feature subset + fallback priority list (score-ranked).",
        "",
        "## Evaluation",
        "- Test-time shifts are defined on original feature identities; effective affected-feature count is reported.",
        "- Baselines: B0 full, B1 global importance, B2 minority-only, B3 redundancy-aware global, B4 random (sanity).",
        "- All methods share the same budget, split, preprocessing, model config, seed and threshold 0.5.",
        "- No MAFR-only HPO / class_weight / threshold tuning.",
    ]
    (results_root / "method_spec.md").write_text("\n".join(lines), encoding="utf-8")


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
        key = ckpt.key(*cell, "c3_cell")
        if resume and ckpt.is_complete(key):
            continue
        todo.append(cell)
    print(f"Cells: total {len(plan)}, pending {len(todo)}")
    if not todo:
        return

    def _run_one(cell):
        ds, seed, model = cell
        key = ckpt.key(ds, seed, model, "c3_cell")
        try:
            payload = run_cell(candidates[ds], seed, model, config, data_root)
            ckpt.mark_complete(key, payload)
            return key, "completed", ""
        except Exception as exc:  # noqa: BLE001
            ckpt.mark_failed(key, repr(exc))
            return key, "failed", repr(exc)

    results = Parallel(n_jobs=-1, verbose=1)(delayed(_run_one)(cell) for cell in todo)
    comp = sum(1 for _, s, _ in results if s == "completed")
    fail = sum(1 for _, s, _ in results if s == "failed")
    print(f"Completed {comp}, failed {fail}")
    for key, status, err in results:
        if status == "failed":
            print(f"  FAILED {key}: {err[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--stage", default="all", choices=["setup", "run", "analysis", "all"])
    args = ap.parse_args()

    config = _load_config(PROJECT_ROOT / "configs" / "phase_c3_method_gate.yaml")
    results_root = PROJECT_ROOT / "results" / "phase_c3_method_gate"
    results_root.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(results_root, "c3")
    data_root = PROJECT_ROOT / "data"

    if args.stage in ("setup", "all"):
        candidates = load_candidates(config["datasets"], data_root)
        write_pre_registration(results_root, config)
        write_method_spec(results_root, config)
        print("Setup done (pre-registration + method spec written, datasets loaded).")
    else:
        candidates = load_candidates(config["datasets"], data_root)

    if args.stage in ("run", "all"):
        run_cells(candidates, config, PROJECT_ROOT, results_root, ckpt, args.resume,
                  args.datasets, args.models, args.seeds)

    if args.stage in ("run", "all", "analysis"):
        dfs = coalesce_checkpoint_payloads(list(ckpt.load_payloads()))
        results = dfs["results"]
        feature_scores = dfs["feature_scores"]
        selection = dfs["selection"]
        if results.empty:
            print("No results to analyse yet. Run the cells first.")
            return

        (results_root / "raw_results.csv").parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_root / "raw_results.csv", index=False)
        feature_scores.to_csv(results_root / "feature_scores.csv", index=False)
        selection.to_csv(results_root / "selected_features.csv", index=False)

        # selected_mafr_variant.json
        sel_records = selection.to_dict("records")
        (results_root / "selected_mafr_variant.json").write_text(
            json.dumps(
                {
                    "inherited_paper_gate": "STRONG-GO",
                    "selection_rule": "source-validation stress AUPRC (pre-registered)",
                    "cells": sel_records,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        sh = results[results["shift_env"].ne("full")].copy()
        full = results[results["shift_env"].eq("full")].copy()
        b0 = sh[sh["method"].eq("B0")][["dataset", "model", "seed", "shift_env", "MVG_recall", "minority_recall"]]
        b0 = b0.rename(columns={"MVG_recall": "MVG_B0", "minority_recall": "recall_B0"})
        rob = sh.merge(b0, on=["dataset", "model", "seed", "shift_env"], how="left")
        rob["MVG_Reduction"] = rob["MVG_B0"] - rob["MVG_recall"]
        rob["MR_Recovery"] = rob["minority_recall"] - rob["recall_B0"]
        rob.to_csv(results_root / "robustness_results.csv", index=False)

        cells = cell_aggregates(results, config["method_gate"]["mg2_min_recovery"])
        cells.to_csv(results_root / "worst_case_results.csv", index=False)

        hfr = hidden_failure_summary(results, config)
        hfr_rows = [
            {"method": m, "hidden_failure_count": hfr["counts"][m], "hidden_failure_rate": round(hfr["rates"][m], 6),
             "HFR_reduction_vs_B0": round(hfr["hfr_reduction"], 6) if m == "M1" else np.nan}
            for m in ["B0", "B1", "B2", "B3", "B4", "M1"]
        ]
        pd.DataFrame(hfr_rows).to_csv(results_root / "hidden_failure_comparison.csv", index=False)

        iid_rows = []
        for (ds, model), g in full.groupby(["dataset", "model"]):
            b0g = g[g["method"].eq("B0")]
            m1g = g[g["method"].eq("M1")]
            iid_rows.append({
                "dataset": ds, "model": model,
                "AUPRC_B0": round(float(b0g["AUPRC"].mean()), 6),
                "AUPRC_M1": round(float(m1g["AUPRC"].mean()), 6),
                "AUROC_B0": round(float(b0g["AUROC"].mean()), 6),
                "AUROC_M1": round(float(m1g["AUROC"].mean()), 6),
                "minority_recall_B0": round(float(b0g["minority_recall"].mean()), 6),
                "minority_recall_M1": round(float(m1g["minority_recall"].mean()), 6),
                "IID_AUPRC_Delta": round(float(m1g["AUPRC"].mean() - b0g["AUPRC"].mean()), 6),
                "IID_AUROC_Delta": round(float(m1g["AUROC"].mean() - b0g["AUROC"].mean()), 6),
            })
        pd.DataFrame(iid_rows).to_csv(results_root / "iid_preservation.csv", index=False)

        model_red = model_mvg_reduction(cells, config)
        stats, strongest_row = block_statistics(results, config)
        stats.to_csv(results_root / "statistics_results.csv", index=False)
        decision, conds, evidence = evaluate_method_gate(cells, model_red, hfr, strongest_row, config)

        gate_rows = [{"condition": k, "passed": bool(v)} for k, v in conds.items()]
        gate_rows.append({"condition": "n_cells", "passed": evidence["n_cells"]})
        gate_rows.append({"condition": "mean_MVG_Reduction", "passed": evidence["mean_MVG_Reduction"]})
        pd.DataFrame(gate_rows).to_csv(results_root / "gate_conditions.csv", index=False)

        figures = make_figures(results, cells, hfr, config, results_root / "figures")

        summary = {
            "decision": decision,
            "conditions": conds,
            "evidence": evidence,
            "model_mvg_reduction": model_red,
            "hfr": hfr,
            "strongest_row": strongest_row,
        }
        (results_root / "method_gate.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        write_report(results_root, summary, cells, config, figures, len(ckpt.completed_keys()),
                     len(list(ckpt.fail_dir.glob("*.json"))))
        print(f"=== Method Gate: {decision} ===")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(PROJECT_ROOT),
        "config_hash": _hash_file(PROJECT_ROOT / "configs" / "phase_c3_method_gate.yaml"),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "datasets": config["datasets"],
        "dataset_hashes": {
            name: _hash_file(PROJECT_ROOT / "data" / "raw" / f"{name}.npz")
            for name in config["datasets"]
            if (PROJECT_ROOT / "data" / "raw" / f"{name}.npz").exists()
        },
        "seeds": config["seeds"],
        "model_configs": config["model_config"],
        "split_configs": config["split"],
        "mafr_configs": {
            "weights_variants": config["mafr"]["weights_variants"],
            "retention_ratios": config["mafr"]["retention_ratios"],
            "default_retention": config["mafr"]["default_retention"],
            "validation_stress": config["mafr"]["validation_stress"],
        },
        "test_shifts": config["test_shifts"],
        "statistics_config": config["statistics"],
        "method_gate_config": config["method_gate"],
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


def write_report(results_root, summary, cells, config, figures, n_completed, n_failed):
    decision = summary["decision"]
    conds = summary["conditions"]
    ev = summary["evidence"]
    model_red = summary["model_mvg_reduction"]
    strongest = summary["strongest_row"]
    hfr = summary["hfr"]

    def fmt(v, nd=3):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "n/a"
        return f"{v:.{nd}f}"

    lines = [
        "# Project C Phase C3 MAFR Method Gate report",
        "",
        f"**Method Gate: `{decision}`**",
        f"  runs expected: {len(config['datasets'])*len(config['seeds'])*len(config['models'])}, "
        f"completed: {n_completed}, failed: {n_failed}",
        "",
        "## Conditions (MG1-MG8)",
    ]
    for k, v in conds.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## MAFR definition",
        f"- variant weights: {json.dumps(config['mafr']['weights_variants'])}",
        f"- retention ratios: {config['mafr']['retention_ratios']}",
        f"- selection rule: source-validation stress AUPRC (pre-registered)",
        "",
        "## Per-cell mean MVG reduction",
    ]
    lines.append("| dataset | model | reduction | B0 MVG | M1 MVG | MG1 | MG2 | worst M1 | worst B1/B2/B3 best | iid AUPRC cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in cells.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['model']} | {fmt(r['mean_MVG_Reduction'], 3)} | {fmt(r['mean_MVG_B0'], 3)} | "
            f"{fmt(r['mean_MVG_M1'], 3)} | {'Y' if r['mean_MVG_Reduction'] >= config['method_gate']['mg1_min_reduction'] else 'N'} | "
            f"{'Y' if r['mvg_present'] else 'N'} | {fmt(r['worst_min_M1'], 3)} | {fmt(r['worst_B_best'], 3)} | "
            f"{fmt(r['iid_AUPRC_cost'], 3)} |"
        )
    lines += [
        "",
        "## Model consistency",
        f"- LR mean MVG reduction: {fmt(model_red.get('logistic_regression'), 3)}",
        f"- XGB mean MVG reduction: {fmt(model_red.get('xgboost'), 3)}",
        "",
        "## Hidden failure",
        f"- B0 HFR: {fmt(hfr['rates']['B0'], 3)}, M1 HFR: {fmt(hfr['rates']['M1'], 3)}, reduction: {fmt(hfr['hfr_reduction'], 3)}",
        "",
        "## Statistics (block = dataset x model x seed)",
        f"- strongest non-MAFR baseline: {strongest['baseline']}",
        f"- mean MVG baseline: {fmt(strongest['mean_MVG_baseline'], 3)}, mean MVG M1: {fmt(strongest['mean_MVG_M1'], 3)}",
        f"- mean effect: {fmt(strongest['mean_effect'], 3)}, bootstrap 95% CI: [{fmt(strongest['CI_low'], 3)}, {fmt(strongest['CI_high'], 3)}]",
        f"- effect size: {fmt(strongest['effect_size'], 3)}, Wilcoxon p: {strongest['wilcoxon_p']:.3e}, n blocks: {strongest['n_blocks']}",
        "",
        "## Evidence",
        f"- cells meeting MG1 threshold: {ev['cells_mg1']}/{ev['n_cells']}",
        f"- cells meeting MG2: {ev['cells_mg2']}/{ev['n_cells']}",
        f"- cells improving (reduction > 0): {ev['cells_improve']}/{ev['n_cells']}",
        f"- mean MVG reduction: {fmt(ev['mean_MVG_Reduction'], 3)}",
        f"- mean / max IID AUPRC cost: {fmt(ev['mean_iid_AUPRC_cost'], 3)} / {fmt(ev['max_iid_AUPRC_cost'], 3)}",
        f"- mean IID AUROC cost: {fmt(ev['mean_iid_AUROC_cost'], 3)}",
        f"- Hidden failure reduction: {fmt(ev['hfr_reduction'], 3)}",
        "",
        "## Authoritative artifacts",
    ]
    for f in ["pre_registration.md", "method_spec.md", "selected_mafr_variant.json", "feature_scores.csv",
              "selected_features.csv", "raw_results.csv", "robustness_results.csv", "hidden_failure_comparison.csv",
              "iid_preservation.csv", "worst_case_results.csv", "statistics_results.csv", "gate_conditions.csv",
              "method_gate.json", "manifest.json"]:
        lines.append(f"- `results/phase_c3_method_gate/{f}`")
    lines.append("")
    lines.append("## Figures")
    for f in figures:
        lines.append(f"- `results/phase_c3_method_gate/figures/{f}`")
    lines.append("")
    (results_root / "method_gate_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
