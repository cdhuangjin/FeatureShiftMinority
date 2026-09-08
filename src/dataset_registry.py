"""Phase C1/C2 dataset registry, candidate loading, substitution log, freeze."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from imblearn.datasets import fetch_datasets

from .data import DatasetProfile


# The archive's suggested candidate pool includes Kaggle-published datasets that
# require an authenticated API token.  Those are substituted with public,
# machine-downloadable imbalanced benchmarks.
SUBSTITUTION_NOTES = [
    {
        "original_candidate": "Credit Card Fraud (Kaggle)",
        "replacement_dataset": "oil, yeast_ml8, coil_2000, wine_quality, thyroid_sick",
        "replacement_reason": (
            "Kaggle datasets require an authenticated Kaggle API token, which is not "
            "available in this environment.  Used public imbalanced benchmarks from "
            "imblearn's Zenodo cache instead."
        ),
        "source": "imblearn Zenodo",
    }
]


@dataclass
class PhaseDataset:
    profile: DatasetProfile
    source: str
    n_numeric: int
    n_categorical: int


def load_candidates(candidate_names: List[str], data_home: Path) -> Dict[str, PhaseDataset]:
    """Download + load the candidate datasets; minority class is normalised to 1."""
    loaded = fetch_datasets(data_home=str(data_home), filter_data=tuple(candidate_names))
    out: Dict[str, PhaseDataset] = {}
    for name in candidate_names:
        bunch = loaded[name]
        X = np.asarray(bunch.data, dtype=float)
        y = np.asarray(bunch.target, dtype=float).astype(int)
        # Normalise to {0,1} with minority == 1.
        if set(np.unique(y)) == {-1, 1}:
            y = (y == 1).astype(int)
        elif set(np.unique(y)) == {0, 1}:
            y = y
        else:
            vals, counts = np.unique(y, return_counts=True)
            minority = vals[int(np.argmin(counts))]
            y = (y == minority).astype(int)
        feat_names = [f"f{i}" for i in range(X.shape[1])]
        prof = DatasetProfile(
            name=name,
            X=X,
            y=y,
            feature_names=feat_names,
            source=f"imblearn Zenodo ({name})",
            replacement_reason="",
        )
        out[name] = PhaseDataset(
            profile=prof,
            source=f"imblearn Zenodo ({name})",
            n_numeric=X.shape[1],
            n_categorical=0,
        )
    return out


def write_substitution_log(path: Path) -> None:
    lines = ["# Dataset substitution log (Phase C1/C2)", ""]
    for note in SUBSTITUTION_NOTES:
        lines.append(f"## Original candidate: {note['original_candidate']}")
        lines.append(f"- replacement_dataset: {note['replacement_dataset']}")
        lines.append(f"- reason: {note['replacement_reason']}")
        lines.append(f"- source: {note['source']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _freeze_hash(records: list, feature_namesets: dict) -> str:
    payload = json.dumps(
        {"records": records, "features": feature_namesets}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def freeze_datasets(
    candidates: Dict[str, PhaseDataset],
    config_path: Path,
) -> Dict[str, object]:
    """Freeze the 8-dataset list into datasets_phase_c12.yaml and return metadata."""
    records = []
    features = {}
    for name, cd in candidates.items():
        p = cd.profile
        records.append(
            {
                "dataset": name,
                "source": cd.source,
                "task_definition": "binary_classification",
                "positive_label": "minority_class",
                "imbalance_ratio": round(p.imbalance_ratio, 4),
                "n_samples": p.n_samples,
                "n_features": p.n_features,
                "n_numeric": cd.n_numeric,
                "n_categorical": cd.n_categorical,
            }
        )
        features[name] = p.feature_names
    freeze_hash = _freeze_hash(records, features)
    freeze_ts = datetime.now(timezone.utc).isoformat()
    frozen = {
        "dataset_freeze_hash": freeze_hash,
        "freeze_timestamp": freeze_ts,
        "dataset_list": [r["dataset"] for r in records],
        "tasks": {r["dataset"]: r["task_definition"] for r in records},
        "positive_labels": {r["dataset"]: r["positive_label"] for r in records},
        "imbalance_ratios": {r["dataset"]: r["imbalance_ratio"] for r in records},
        "n_samples": {r["dataset"]: r["n_samples"] for r in records},
        "n_features": {r["dataset"]: r["n_features"] for r in records},
        "sources": {r["dataset"]: r["source"] for r in records},
        "feature_names": features,
    }
    config = {
        "dataset_freeze_hash": freeze_hash,
        "freeze_timestamp": freeze_ts,
        "dataset_list": [r["dataset"] for r in records],
        "dataset_meta": records,
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    return frozen


def dataset_profiles_table(candidates: Dict[str, PhaseDataset], bmo: dict) -> pd.DataFrame:
    rows = []
    for name, cd in candidates.items():
        d = cd.profile.describe()
        rows.append(
            {
                "dataset": name,
                "source": cd.source,
                "n_samples": d["n_samples"],
                "n_features": d["n_features"],
                "n_numeric": cd.n_numeric,
                "minority_count": d["minority_count"],
                "majority_count": d["majority_count"],
                "imbalance_ratio": d["imbalance_ratio"],
                "BMO": bmo.get(name, {}).get("BMO", np.nan),
                "observability_class": bmo.get(name, {}).get("observability_class", "unknown"),
            }
        )
    return pd.DataFrame(rows)

