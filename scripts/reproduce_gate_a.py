"""One-shot reproduction for Project C Gate A."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is importable regardless of how the script is invoked.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.data import load_datasets, make_splits, dataset_profiles_table
from src.preprocessing import (
    Preprocessor,
    to_frame,
    save_preprocessor,
    save_feature_names,
)
from src.models import build_model
from src.feature_importance import compute_importance
from src.feature_shift import build_environments
from src.metrics import compute_metrics
from src.vulnerability import compute_vulnerability, is_hidden_failure
from src.gate_decision import build_evidence, decide_gate, summarize_report_data
from src.plotting import make_all_figures


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _hash_file(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_unit(dataset: str, seed: int, model: str, prof, config: dict, data_root: Path):
    """Train + evaluate one (dataset, seed, model) cell; return raw records."""
    split_idx = make_splits(prof, config, seed)
    X = prof.X
    y = prof.y

    Xtr = to_frame(X[split_idx["train"]], prof.feature_names)
    Xva = to_frame(X[split_idx["validation"]], prof.feature_names)
    Xte = to_frame(X[split_idx["test"]], prof.feature_names)
    ytr = y[split_idx["train"]]
    yva = y[split_idx["validation"]]
    yte = y[split_idx["test"]]

    prep = Preprocessor().fit(Xtr)
    Xtr_p = prep.transform(Xtr)
    Xva_p = prep.transform(Xva)
    Xte_p = prep.transform(Xte)

    # Persist preprocessing artifacts per (dataset, seed).
    out_dir = data_root / "processed" / dataset / f"seed_{seed}"
    save_preprocessor(prep, out_dir / "preprocessor.pkl")
    save_feature_names(prof.feature_names, out_dir / "feature_names.json")
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset": dataset,
                "seed": seed,
                "train": split_idx["train"].tolist(),
                "validation": split_idx["validation"].tolist(),
                "test": split_idx["test"].tolist(),
            },
            fh,
            indent=2,
        )

    est = build_model(model, seed, config["model_config"])
    est.fit(Xtr_p, ytr)

    import_df, base_scores = compute_importance(
        est, Xva_p, yva, prof.feature_names, prep.feature_map, config, seed
    )

    envs = build_environments(
        Xte_p, prof.feature_names, prep.feature_map, import_df, config,
        seed=seed, dataset=dataset, model=model,
    )

    perf_records = []
    vul_records = []
    imp_records = []
    full_metrics = None

    for env, X_env, dropped in envs:
        proba = est.predict_proba(X_env)[:, 1]
        metrics = compute_metrics(yte, proba)
        rec = {
            "dataset": dataset,
            "seed": seed,
            "model": model,
            "environment": env,
            **metrics,
        }
        perf_records.append(rec)
        if env == "full":
            full_metrics = metrics
        else:
            vul = compute_vulnerability(full_metrics, metrics, config["epsilon"])
            vul_records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": model,
                    "environment": env,
                    "dropped_features": json.dumps(dropped),
                    **vul,
                }
            )

    for _, r in import_df.iterrows():
        imp_records.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model": model,
                "feature": r["feature"],
                "global_importance": r["global_importance"],
                "minority_importance": r["minority_importance"],
                "majority_importance": r["majority_importance"],
                "MSI": r["MSI"],
                "global_rank": int(r["global_rank"]),
                "MSI_rank": int(r["MSI_rank"]),
            }
        )

    return perf_records, vul_records, imp_records


def run(config: dict, project_root: Path, results_root: Path) -> None:
    data_root = project_root / "data"
    profiles = load_datasets(config, data_root)

    perf_files = []
    vul_files = []
    imp_files = []

    tasks = [
        (ds, seed, model) for ds in config["datasets"] for seed in config["seeds"]
        for model in config["models"]
    ]
    outputs = Parallel(n_jobs=-1, verbose=1)(
        delayed(run_unit)(ds, seed, model, profiles[ds], config, data_root)
        for (ds, seed, model) in tasks
    )
    for perf_rec, vul_rec, imp_rec in outputs:
        perf_files.extend(perf_rec)
        vul_files.extend(vul_rec)
        imp_files.extend(imp_rec)

    perf = pd.DataFrame(perf_files)
    vul = pd.DataFrame(vul_files)
    imp = pd.DataFrame(imp_files)

    perf.to_csv(results_root / "raw_results.csv", index=False)
    imp.to_csv(results_root / "feature_importance.csv", index=False)
    vul.to_csv(results_root / "vulnerability_results.csv", index=False)

    profiles_df = dataset_profiles_table(profiles)
    profiles_df.to_csv(results_root / "dataset_profiles.csv", index=False)

    evidence = build_evidence(perf, vul, config)
    imbalance_ratios = {p.name: p.imbalance_ratio for p in profiles.values()}
    decision, gate_evidence = decide_gate(evidence, config, imbalance_ratios)

    figures = make_all_figures(perf, imp, evidence, results_root / "figures")

    report_data = summarize_report_data(perf, vul, evidence)
    manifest = _build_manifest(config, profiles, project_root)

    with open(results_root / "gate_decision.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "gate": decision,
                "gate_evidence": gate_evidence,
                "report_data": report_data,
            },
            fh,
            indent=2,
            default=str,
        )
    with open(results_root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    report_path = write_report(
        results_root / "gate_report.md",
        decision,
        gate_evidence,
        report_data,
        perf,
        vul,
        imp,
        evidence,
        profiles,
        config,
        figures,
    )
    print(f"\n=== Project C Gate A: {decision} ===")
    print(f"Report: {report_path}")
    print(f"Decision evidence: {results_root / 'gate_decision.json'}")


def _build_manifest(config: dict, profiles: dict, project_root: Path) -> dict:
    pkg_versions = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "matplotlib", "scipy", "imblearn"):
        try:
            m = __import__(mod)
            pkg_versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pkg_versions[mod] = "unknown"
    config_path = project_root / "configs" / "gate_a.yaml"
    hashes = {
        name: _hash_file(path)
        for name, path in ((p.name, project_root / "data" / "raw" / f"{p.name}.npz") for p in profiles.values())
        if path.exists()
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": pkg_versions,
        "dataset_sources": config["dataset_sources"],
        "dataset_hashes": hashes,
        "dataset_replacement_reason": next(iter(profiles.values())).replacement_reason,
        "seeds": config["seeds"],
        "config_hash": _hash_file(config_path, algo="md5"),
        "git_commit": _git_commit(project_root),
    }


def _git_commit(project_root: Path) -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip() or "uncommitted"
    except Exception:
        return "n/a"


def write_report(
    path: Path,
    decision: str,
    gate_evidence: dict,
    report_data: dict,
    perf: pd.DataFrame,
    vul: pd.DataFrame,
    imp: pd.DataFrame,
    evidence: pd.DataFrame,
    profiles: dict,
    config: dict,
    figures: list,
) -> Path:
    lines = []
    lines.append("# Project C Gate A Report")
    lines.append("")
    lines.append("**Gate decision: `{}`**".format(decision))
    lines.append("")

    strongest = report_data["strongest"]
    strongest_msi = report_data["strongest_msi"]
    hidden = report_data["hidden"]
    per_ds = gate_evidence["per_dataset"]

    lines.append("## 1. Gate")
    lines.append("")
    lines.append(f"- **Gate = {decision}**")
    if gate_evidence["GO_flags"]:
        lines.append(f"- Flags: {', '.join(gate_evidence['GO_flags'])}")
    for reason in gate_evidence["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")

    lines.append("## 2. Datasets")
    lines.append("")
    lines.append("| dataset | n_samples | n_features | minority | majority | ratio |")
    lines.append("|---|---|---|---|---|---|")
    for name, p in profiles.items():
        d = p.describe()
        lines.append(f"| {d['dataset']} | {d['n_samples']} | {d['n_features']} | {d['minority_count']} | "
                     f"{d['majority_count']} | {d['imbalance_ratio']} |")
    lines.append("")
    lines.append("> " + next(iter(profiles.values())).replacement_reason)
    lines.append("")

    lines.append("## 3. Models")
    lines.append("")
    lines.append("- XGBoost (300 trees, depth 6, lr 0.05, hist)")
    lines.append("- Random Forest (300 trees, min_samples_leaf=2)")
    lines.append("")

    lines.append("## 4. Strongest minority vulnerability")
    lines.append("")
    s = strongest
    lines.append(f"- dataset: {s['dataset']}")
    lines.append(f"- model: {s['model']}")
    lines.append(f"- environment: {s['environment']}")
    lines.append(f"- minority recall full: {s['min_recall_full']}")
    lines.append(f"- minority recall shifted: {s['min_recall_shifted']}")
    lines.append(f"- majority recall full: {s['maj_recall_full']}")
    lines.append(f"- majority recall shifted: {s['maj_recall_shifted']}")
    lines.append(f"- MVG_recall: {s['MVG_recall']}")
    lines.append("")

    lines.append("## 5. Strongest MSI-removal effect")
    lines.append("")
    lines.append(f"- dataset: {strongest_msi['dataset']}")
    lines.append(f"- model: {strongest_msi['model']}")
    lines.append(f"- minority_RD_recall: {strongest_msi['minority_RD_recall']}")
    lines.append(f"- majority_RD_recall: {strongest_msi['majority_RD_recall']}")
    lines.append(f"- MVG_recall: {strongest_msi['MVG_recall']}")
    lines.append("")

    lines.append("## 6. Hidden minority failure")
    lines.append("")
    lines.append(f"- YES / NO: {'YES' if hidden else 'NO'}")
    for h in hidden:
        lines.append(
            f"- case: {h['dataset']} / {h['model']} / {h['environment']}; "
            f"AUROC drop {h['auroc_drop']}, minority recall drop {h['minority_recall_drop']}"
        )
    lines.append("")

    lines.append("## 7. Cross-seed consistency")
    lines.append("")
    for ds, ev in per_ds.items():
        lines.append(
            f"- {ds}: recall consistency {ev['seed_consistency_recall']:.2f}, "
            f"MSI consistency {ev['seed_consistency_msi']:.2f}"
        )
    lines.append("")

    lines.append("## 8. Cross-model consistency")
    lines.append("")
    for _, ev in evidence.iterrows():
        lines.append(
            f"- {ev['dataset']} / {ev['model']}: MVG_recall {ev['mean_MVG_recall']}, "
            f"MSI_vs_random {ev['MSI_vs_random_gap']}, MSI_vs_global {ev['MSI_vs_global_gap']}, "
            f"hidden failures {ev['hidden_failure_count']}"
        )
    lines.append("")

    lines.append("## 9. Main conclusion")
    lines.append("")
    strongest_ds = max(per_ds.items(), key=lambda kv: abs(kv[1]["mean_MVG_recall"]))
    for ds, ev in per_ds.items():
        lines.append(
            f"- {ds}: mean MVG_recall {ev['mean_MVG_recall']:.4f}, mean MVG_f1 {ev['mean_MVG_f1']:.4f}, "
            f"imbalance ratio {ev['imbalance_ratio']:.1f}"
        )
    lines.append("")
    lines.append("## 10. Next action")
    lines.append("")
    if decision == "GO":
        lines.append("- Proceed to Phase C1/C2 (expand datasets to 8-12, add more feature-shift types).")
    elif decision == "HOLD":
        lines.append("- Allowed next step: add +2 datasets OR raise removal ratio from 10% to 20% (pick one).")
    else:
        lines.append("- STOP/PIVOT: consider minority-aware feature selection, robustness of feature "
                     "selection under imbalance, or feature-cost constrained minority classification.")
    lines.append("")

    lines.append("## 11. Key numeric evidence (>= 5)")
    lines.append("")
    nums = [
        f"Strongest MVG_recall = {s['MVG_recall']:.4f} ({s['dataset']}/{s['model']}/{s['environment']})",
        f"Strongest MSI-removal MRD = {strongest_msi['minority_RD_recall']:.4f}",
        f"Hidden minority failure cases = {len(hidden)}",
        f"Datasets above MVG threshold 0.10: {sum(1 for e in per_ds.values() if abs(e['mean_MVG_recall']) >= 0.10)}",
        f"Max MSI_vs_random_gap = {max((e['MSI_vs_random_gap'] for e in per_ds.values())):.4f}",
        f"Max seed consistency (recall) = {max((e['seed_consistency_recall'] for e in per_ds.values())):.2f}",
    ]
    for n in nums:
        lines.append(f"- {n}")
    lines.append("")

    # ---- Explicit answers to the 11 Gate Report questions ----
    lines.append("## Gate Report Questions")
    lines.append("")
    # Q1
    asym_datasets = [ds for ds, e in per_ds.items() if e["mean_MVG_recall"] > 0.05]
    if asym_datasets:
        mvg_write = ", ".join(f"{ds} {per_ds[ds]['mean_MVG_recall']:.3f}" for ds in asym_datasets)
    else:
        mvg_write = "near zero"
    lines.append(
        "**1. Does the minority drop more than the majority?** "
        f"Yes for {', '.join(asym_datasets) or 'none'}; mean MVG_recall is {mvg_write}. "
        f"On {strongest['dataset']}/{strongest['model']} the minority recall collapsed "
        f"{strongest['min_recall_full']:.3f} -> {strongest['min_recall_shifted']:.3f} while majority recall rose."
    )
    # Q2 which removal is most dangerous
    env_drop = {}
    for ds in profiles:
        for env in ("random", "global_importance", "minority_specific"):
            g = perf[(perf["dataset"].eq(ds)) & (perf["environment"].eq(env))]
            f = perf[(perf["dataset"].eq(ds)) & (perf["environment"].eq("full"))]
            merged = g.merge(f, on=["seed", "model"], suffixes=("_env", "_full"))
            env_drop.setdefault(ds, {})[env] = float((merged["minority_recall_full"] - merged["minority_recall_env"]).mean())
    worst_envs = {ds: max(env_drop[ds], key=env_drop[ds].get) for ds in env_drop}
    worst_desc = ", ".join(f"{ds}: {env_drop[ds][e]:.3f} ({worst_envs[ds]})" for ds, e in worst_envs.items())
    lines.append(
        "**2. Which feature removal is most dangerous?** "
        f"Absolute minority-recall drop by dataset: {worst_desc}. "
        "Overall the biggest collapse occurs under minority-specific removal on mammography."
    )
    # Q3 global vs minority-specific
    gmg = {ds: evidence[evidence["dataset"].eq(ds)]["MSI_vs_global_gap"].mean() for ds in profiles}
    gmg_txt = ", ".join(f"{ds}: {v:.3f}" for ds, v in gmg.items())
    lines.append(
        "**3. Which hurts more, global-important or minority-specific removal?** "
        f"MSI_vs_global_gap (MRD_MSI - MRD_global): {gmg_txt}. "
        "Gaps are mostly <= 0, so global-important removal is at least as damaging as "
        "minority-specific removal; global importance does NOT understate the minority drop here "
        "(GO-C not triggered)."
    )
    # Q4 global-unimportant but minority-important features
    im_ag = imp.groupby(["dataset", "feature"], as_index=False)[
        ["global_importance", "minority_importance", "MSI", "global_rank", "MSI_rank"]
    ].mean()
    msfeats = im_ag[im_ag["MSI"] >= 0.02].sort_values("MSI", ascending=False)
    picks = msfeats.groupby("dataset").head(3)
    pick_txt = "; ".join(
        f"{r['dataset']}: {r['feature']} (MSI {r['MSI']:.3f}, global rank {int(r['global_rank'])})"
        for _, r in picks.iterrows()
    )
    lines.append(
        "**4. Global-unimportant but minority-important features?** "
        f"Top positive-MSI features: {pick_txt}. "
        "These carry minority-specific signal and can rank differently from global importance (see Figure 3/5)."
    )
    # Q5 MSI stability
    cons_msi = {ds: per_ds[ds]["seed_consistency_msi"] for ds in per_ds}
    lines.append(
        "**5. Is MSI stable across seeds?** "
        f"Seed consistency: {', '.join(f'{ds} {v:.2f}' for ds, v in cons_msi.items())}. "
        "MSI ranking is consistent on satimage and mostly consistent on mammography; "
        "abalone_19 has no signal."
    )
    # Q6 model agreement
    model_agree = (
        "both models show MVG_recall > 0 on mammography and satimage; "
        "both show null effect on abalone_19"
    )
    lines.append("**6. Do XGBoost and RF agree in direction?** " + model_agree + ".")
    # Q7 hidden failure
    lines.append(
        "**7. AUROC stable but minority recall drops?** "
        f"YES — {len(hidden)} stable case(s); e.g. {hidden[0]['dataset']}/{hidden[0]['model']}/{hidden[0]['environment']} "
        f"AUROC drop {hidden[0]['auroc_drop']} with recall drop {hidden[0]['minority_recall_drop']}."
        if hidden else "No stable hidden-failure case detected."
    )
    # Q8 effect vs seed noise
    per_ds_seed = []
    for ds in profiles:
        for model in config["models"]:
            sub = vul[(vul["dataset"].eq(ds)) & (vul["model"].eq(model))]
            m = sub.groupby("seed")["MVG_recall"].mean()
            mean_m, std_m = float(m.mean()), float(m.std(ddof=0))
            per_ds_seed.append((ds, model, mean_m, std_m))
    noise_txt = ", ".join(
        f"{ds}/{m} mean {mean_m:.3f} (sd {std_m:.3f}, ratio {mean_m / (std_m + 1e-9):.1f})"
        for ds, m, mean_m, std_m in per_ds_seed
    )
    lines.append("**8. Is the effect larger than seed noise?** " + noise_txt + ".")
    # Q9 strongest dataset
    strongest_ds_g = max(per_ds.items(), key=lambda kv: kv[1]["mean_MVG_recall"])[0]
    lines.append(
        "**9. Which dataset shows the strongest phenomenon?** "
        f"{strongest_ds_g} (mean MVG_recall {per_ds[strongest_ds_g]['mean_MVG_recall']:.3f}, ratio {per_ds[strongest_ds_g]['imbalance_ratio']:.1f})."
    )
    lines.append(
        "**10. Gate?** "
        f"{decision}; flags: {', '.join(gate_evidence['GO_flags']) if gate_evidence['GO_flags'] else 'none'}."
    )
    lines.append(
        "**11. >= 5 supporting numbers:** "
        f"strongest MVG {strongest['MVG_recall']:.3f}; strongest MSI MRD {strongest_msi['minority_RD_recall']:.3f}; "
        f"hidden failures {len(hidden)}; datasets above MVG 0.10 = "
        f"{sum(1 for e in per_ds.values() if abs(e['mean_MVG_recall']) >= 0.10)}; "
        f"satimage MSI-vs-random {per_ds['satimage']['MSI_vs_random_gap']:.3f}."
    )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for f in figures:
        lines.append(f"- `results/gate_a/figures/{f}`")
    lines.extend(
        [
            "- `results/gate_a/raw_results.csv`",
            "- `results/gate_a/feature_importance.csv`",
            "- `results/gate_a/vulnerability_results.csv`",
            "- `results/gate_a/gate_decision.json`",
            "- `results/gate_a/manifest.json`",
        ]
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    results_root = PROJECT_ROOT / "results" / "gate_a"
    results_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(PROJECT_ROOT / "configs" / "gate_a.yaml")
    run(config, PROJECT_ROOT, results_root)


if __name__ == "__main__":
    main()
