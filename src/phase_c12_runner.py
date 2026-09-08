"""Phase C1/C2 cell runner: one train, then core/structured/delayed evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .data import make_splits
from .preprocessing import Preprocessor, to_frame
from .models import build_model
from .feature_importance import compute_importance
from .feature_shift import apply_mask, _seed_from
from .metrics import compute_metrics
from .vulnerability import compute_vulnerability, relative_degradation
from .feature_grouping import cluster_features
from .structured_shift import build_structured_envs
from .delayed_availability import build_delayed_envs, recovery_metrics
from .sensitivity_analysis import sensitivity_summary
from .dataset_registry import PhaseDataset


CORE_SHIFT_TYPES = ["random", "global_importance", "minority_specific", "majority_specific"]
TIER2_MODELS = ["xgboost", "logistic_regression"]
RECORD_KEYS = ["perf", "vul", "imp", "hidden", "class_control", "slope", "structured", "delayed"]


def build_cell_plan(datasets, seeds, models) -> List[Tuple[str, int, str]]:
    return [(ds, seed, model) for ds in datasets for seed in seeds for model in models]


def coalesce_checkpoint_payloads(payloads: Iterable[dict]) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for k in RECORD_KEYS:
        recs = []
        for p in payloads:
            recs.extend(p.get(k, []))
        dfs[k] = pd.DataFrame(recs)
    return dfs


def _removed_for(shift_type: str, importance, feature_names, n_removed, seed, dataset, model):
    if shift_type == "random":
        rng = np.random.default_rng(_seed_from(f"{dataset}|{seed}|{model}|random|{n_removed}"))
        return sorted(rng.choice(feature_names, size=n_removed, replace=False).tolist())
    if shift_type == "global_importance":
        ranked = importance.sort_values("global_importance", ascending=False)["feature"].tolist()
        return sorted(ranked[:n_removed])
    if shift_type == "minority_specific":
        ranked = importance.sort_values("MSI", ascending=False)["feature"].tolist()
        return sorted(ranked[:n_removed])
    if shift_type == "majority_specific":
        ranked = importance.sort_values("MSI", ascending=True)["feature"].tolist()
        return sorted(ranked[:n_removed])
    raise ValueError(f"Unknown shift type {shift_type}")


def _n_removed(severity: float, n_features: int) -> int:
    return max(1, min(n_features, int(round(severity * n_features))))


def run_cell(
    phase_ds: PhaseDataset,
    seed: int,
    model: str,
    config: dict,
    data_root: Path,
) -> Dict[str, list]:
    prof = phase_ds.profile
    split_idx = make_splits(prof, config, seed)
    X, y = prof.X, prof.y
    Xtr = to_frame(X[split_idx["train"]], prof.feature_names)
    Xva = to_frame(X[split_idx["validation"]], prof.feature_names)
    Xte = to_frame(X[split_idx["test"]], prof.feature_names)
    ytr = y[split_idx["train"]]
    yte = y[split_idx["test"]]

    prep = Preprocessor().fit(Xtr)
    Xtr_p = prep.transform(Xtr)
    Xva_p = prep.transform(Xva)
    Xte_p = prep.transform(Xte)
    est = build_model(model, seed, config["model_config"])
    est.fit(Xtr_p, ytr)
    importance, bases = compute_importance(
        est, Xva_p, yva := y[split_idx["validation"]], prof.feature_names, prep.feature_map,
        config, seed,
    )

    feature_names = prof.feature_names
    feature_map = prep.feature_map
    n_features = len(feature_names)

    perf_rows: List[dict] = []
    vul_rows: List[dict] = []
    hidden_rows: List[dict] = []
    class_rows: List[dict] = []
    slope_rows: List[dict] = []
    imp_rows: List[dict] = []
    for _, r in importance.iterrows():
        imp_rows.append(
            {
                "dataset": prof.name,
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

    def evals(shift_type: str, severity: float, removed: List[str]) -> Dict[str, float]:
        Xm = apply_mask(Xte_p, removed, feature_map)
        proba = est.predict_proba(Xm)[:, 1]
        return compute_metrics(yte, proba)

    full_metrics = evals("full", 0.0, [])
    feat_rows = [full_metrics, ("full", 0.0, 0, 0.0, [])]
    # Relative degradation is unstable when the full-feature minority baseline is
    # essentially zero (floor-risk).  We keep absolute drops but mark relative
    # metrics NaN so they are excluded from the primary degradation counts.
    min_primary = config["observability"]["minimum_primary_recall"]
    stable_baseline = full_metrics["minority_recall"] >= min_primary

    def _vul(full, m):
        v = compute_vulnerability(full, m, config["epsilon"])
        if not stable_baseline:
            for k in ("minority_RD_recall", "minority_RD_f1", "MVG_recall", "MVG_f1"):
                v[k] = np.nan
        return v

    # severity curves per shift type
    slopes_data: Dict[str, Tuple[List[float], List[float], List[float]]] = {
        st: ([0.0], [full_metrics["minority_recall"]], [full_metrics["majority_recall"]])
        for st in CORE_SHIFT_TYPES
    }
    for shift_type in CORE_SHIFT_TYPES:
        for severity in config["removal_rates"]:
            n = _n_removed(severity, n_features)
            removed = _removed_for(shift_type, importance, feature_names, n, seed, prof.name, model)
            frac = n / max(n_features, 1)
            m = evals(shift_type, severity, removed)
            perf_rows.append(
                {
                    "dataset": prof.name,
                    "model": model,
                    "seed": seed,
                    "shift_type": shift_type,
                    "severity": severity,
                    "n_removed": n,
                    "removal_fraction": round(frac, 4),
                    **m,
                }
            )
            vul = _vul(full_metrics, m)
            arb_min = full_metrics["minority_recall"] - m["minority_recall"]
            arb_maj = full_metrics["majority_recall"] - m["majority_recall"]
            vul_rows.append(
                {
                    "dataset": prof.name,
                    "model": model,
                    "seed": seed,
                    "shift_type": shift_type,
                    "severity": severity,
                    "MRD_recall": vul["minority_RD_recall"],
                    "MaRD_recall": vul["majority_RD_recall"],
                    "MVG_recall": vul["MVG_recall"],
                    "ARD_minority": round(arb_min, 6),
                    "ARD_majority": round(arb_maj, 6),
                    "MRD_f1": vul["minority_RD_f1"],
                    "MaRD_f1": vul["majority_RD_f1"],
                    "MVG_f1": vul["MVG_f1"],
                }
            )
            auroc_drop = full_metrics["AUROC"] - m["AUROC"]
            auprc_drop = full_metrics["AUPRC"] - m["AUPRC"]
            rec_drop = full_metrics["minority_recall"] - m["minority_recall"]
            hidden_rows.append(
                {
                    "dataset": prof.name,
                    "model": model,
                    "seed": seed,
                    "shift_type": shift_type,
                    "severity": severity,
                    "AUROC_drop": round(auroc_drop, 6),
                    "AUPRC_drop": round(auprc_drop, 6),
                    "minority_recall_drop": round(rec_drop, 6),
                    "hidden_auroc": bool(auroc_drop <= config["paper_gate"]["hidden_auroc_max"] and rec_drop >= config["paper_gate"]["hidden_recall_drop_min"]),
                    "hidden_auprc": bool(auprc_drop <= config["paper_gate"]["hidden_auprc_max"] and rec_drop >= config["paper_gate"]["hidden_recall_drop_min"]),
                    "HFS": round(rec_drop - auroc_drop, 6),
                    "HFS_auprc": round(rec_drop - auprc_drop, 6),
                }
            )
            slopes_data[shift_type][0].append(frac)
            slopes_data[shift_type][1].append(m["minority_recall"])
            slopes_data[shift_type][2].append(m["majority_recall"])

    # Class-specific control contrasts minority-specific vs majority-specific per severity.
    for severity in config["removal_rates"]:
        def find_mvg(st):
            rows = [r for r in vul_rows if r["shift_type"] == st and r["severity"] == severity]
            return rows[0]["MVG_recall"], rows[0]["ARD_minority"]
        m_mvg, m_drop = find_mvg("minority_specific")
        maj_mvg, maj_drop = find_mvg("majority_specific")
        class_rows.append(
            {
                "dataset": prof.name,
                "model": model,
                "seed": seed,
                "severity": severity,
                "minority_specific_MVG": m_mvg,
                "majority_specific_MVG": maj_mvg,
                "DeltaMVG_class": round(m_mvg - maj_mvg, 6),
                "minority_specific_minority_drop": m_drop,
                "majority_specific_minority_drop": maj_drop,
            }
        )

    # Sensitivity slope per shift type (x = actual removal fraction).
    for shift_type in CORE_SHIFT_TYPES:
        rates, mins, majs = slopes_data[shift_type]
        ss = sensitivity_summary(rates, mins, majs, config["epsilon"])
        slope_rows.append({"dataset": prof.name, "model": model, "seed": seed, "shift_type": shift_type, **ss})

    # Full-feature row.
    perf_rows.append(
        {
            "dataset": prof.name,
            "model": model,
            "seed": seed,
            "shift_type": "full",
            "severity": 0.0,
            "n_removed": 0,
            "removal_fraction": 0.0,
            **full_metrics,
        }
    )

    structured_rows: List[dict] = []
    delayed_rows: List[dict] = []
    if model in TIER2_MODELS:
        # Feature groups (train-only).
        group_map = cluster_features(Xtr_p, feature_names, config["structured"]["correlation_threshold"])
        envs = build_structured_envs(feature_names, group_map, config, seed, prof.name, model)
        for env in envs:
            m = evals(env["mechanism"], env["severity"], env["removed_features"])
            vul = _vul(full_metrics, m)
            structured_rows.append(
                {
                    "dataset": prof.name,
                    "model": model,
                    "seed": seed,
                    "mechanism": env["mechanism"],
                    "pair_id": env.get("pair_id", ""),
                    "severity": env["severity"],
                    "removed_features": json.dumps(env["removed_features"]),
                    "n_removed": env["n_removed"],
                    "removal_fraction": round(env["removal_fraction"], 4),
                    "MRD_minority": vul["minority_RD_recall"],
                    "MVG": vul["MVG_recall"],
                }
            )
        # Pair structured with its matched random control (same pair_id).
        by_pair: Dict[str, Dict[str, dict]] = {}
        for r in structured_rows:
            by_pair.setdefault(r["pair_id"], {})[r["mechanism"]] = r
        for _pid, mapping in by_pair.items():
            if "matched_random" in mapping:
                for mech in ("group_loss", "correlated_loss"):
                    if mech in mapping:
                        structured_rows.append(
                            {
                                "dataset": prof.name,
                                "model": model,
                                "seed": seed,
                                "mechanism": f"{mech}_penalty",
                                "pair_id": _pid,
                                "severity": mapping[mech]["severity"],
                                "removed_features": "",
                                "n_removed": mapping[mech]["n_removed"],
                                "removal_fraction": mapping[mech]["removal_fraction"],
                                "MRD_minority": mapping[mech]["MRD_minority"],
                                "MVG": mapping[mech]["MVG"],
                                "matched_random_MRD": mapping["matched_random"]["MRD_minority"],
                                "StructuredPenalty": round(mapping[mech]["MRD_minority"] - mapping["matched_random"]["MRD_minority"], 6),
                            }
                        )
        # Delayed availability.
        import pandas as pd
        imp_df = importance
        del_envs = build_delayed_envs(feature_names, imp_df, config["delayed"]["availability"])
        min_points = []
        maj_points = []
        for de in del_envs:
            m = evals("delayed", de["availability_fraction"], de["unavailable_features"])
            min_ret = (
                m["minority_recall"] / (full_metrics["minority_recall"] + config["epsilon"])
                if stable_baseline
                else np.nan
            )
            maj_ret = m["majority_recall"] / (full_metrics["majority_recall"] + config["epsilon"])
            min_points.append((de["availability_fraction"], m["minority_recall"]))
            maj_points.append((de["availability_fraction"], m["majority_recall"]))
            delayed_rows.append(
                {
                    "dataset": prof.name,
                    "model": model,
                    "seed": seed,
                    "availability_fraction": de["availability_fraction"],
                    "minority_recall": m["minority_recall"],
                    "majority_recall": m["majority_recall"],
                    "minority_retention": round(min_ret, 6),
                    "majority_retention": round(maj_ret, 6),
                }
            )
        rm = recovery_metrics(full_metrics["minority_recall"], full_metrics["majority_recall"], min_points, maj_points)
        for row in delayed_rows:
            row["MinorityRecoveryAvailability"] = rm["MinorityRecoveryAvailability"]
            row["MajorityRecoveryAvailability"] = rm["MajorityRecoveryAvailability"]
            row["RecoveryGap"] = rm["RecoveryGap"]

    return {
        "perf": perf_rows,
        "vul": vul_rows,
        "imp": imp_rows,
        "hidden": hidden_rows,
        "class_control": class_rows,
        "slope": slope_rows,
        "structured": structured_rows,
        "delayed": delayed_rows,
    }
