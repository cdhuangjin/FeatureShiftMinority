"""Baseline Minority Observability (BMO) screen and floor-risk labelling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .data import make_splits
from .preprocessing import Preprocessor, to_frame
from .metrics import compute_metrics
from .dataset_registry import PhaseDataset


def observability_label(bmo: float, config: dict) -> str:
    obs = config["observability"]
    if bmo >= obs["strong_minority_recall"]:
        return "strongly_observable"
    if bmo >= obs["minimum_primary_recall"]:
        return "weakly_observable"
    return "floor_risk"


def compute_bmo(phase_ds: PhaseDataset, seed: int, config: dict) -> Dict[str, float]:
    """Quick BMO screen: XGBoost on 60% train, minority recall on 20% test."""
    prof = phase_ds.profile
    split_idx = make_splits(prof, config, seed)
    X = prof.X
    y = prof.y
    Xtr = to_frame(X[split_idx["train"]], prof.feature_names)
    Xte = to_frame(X[split_idx["test"]], prof.feature_names)
    yte = y[split_idx["test"]]
    prep = Preprocessor().fit(Xtr)
    Xtr_p = prep.transform(Xtr)
    Xte_p = prep.transform(Xte)
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(Xtr_p, y[split_idx["train"]])
    proba = model.predict_proba(Xte_p)[:, 1]
    m = compute_metrics(yte, proba)
    return {
        "BMO": m["minority_recall"],
        "AUPRC": m["AUPRC"],
        "AUROC": m["AUROC"],
        "minority_f1": m["minority_f1"],
        "majority_recall": m["majority_recall"],
    }


def screen_datasets(
    candidates: Dict[str, PhaseDataset], seed: int, config: dict
) -> pd.DataFrame:
    rows = []
    for name, phase_ds in candidates.items():
        m = compute_bmo(phase_ds, seed, config)
        rows.append(
            {
                "dataset": name,
                "BMO": round(m["BMO"], 6),
                "AUPRC": m["AUPRC"],
                "AUROC": m["AUROC"],
                "minority_f1": m["minority_f1"],
                "majority_recall": m["majority_recall"],
                "observability_class": observability_label(m["BMO"], config),
            }
        )
    return pd.DataFrame(rows)

