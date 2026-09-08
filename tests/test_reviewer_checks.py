"""Unit tests for the pre-submission Reviewer-Proof checks."""

from __future__ import annotations

import math

import pandas as pd

from src.reviewer_checks import (
    compute_hidden_threshold_sensitivity,
    compute_metric_sensitivity,
    hidden_definitions,
    relative_drop,
    severity_sensitivity,
    leave_one_dataset_out,
)


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


def _full_row(dataset, minority=0.80, majority=0.90):
    row = {
        "dataset": dataset,
        "model": "lr",
        "seed": 42,
        "shift_type": "full",
        "severity": 0.0,
        "removal_fraction": 0.0,
        "minority_recall": minority,
        "minority_f1": 0.75,
        "minority_precision": 0.70,
        "balanced_accuracy": 0.85,
        "MCC": 0.70,
        "Gmean": 0.85,
        "AUPRC": 0.90,
        "AUROC": 0.95,
        "majority_recall": majority,
        "majority_f1": 0.95,
    }
    return row


def _shift_rows(dataset, shift_type, minority, auroc):
    rows = []
    for sev, fr in [(0.05, 0.05), (0.10, 0.10), (0.20, 0.20)]:
        rows.append(
            {
                "dataset": dataset,
                "model": "lr",
                "seed": 42,
                "shift_type": shift_type,
                "severity": sev,
                "removal_fraction": fr,
                "minority_recall": minority,
                "minority_f1": 0.60,
                "minority_precision": 0.55,
                "balanced_accuracy": 0.78,
                "MCC": 0.55,
                "Gmean": 0.74,
                "AUPRC": 0.70,
                "AUROC": auroc,
                "majority_recall": 0.89,
                "majority_f1": 0.94,
            }
        )
    return rows


def _raw():
    rows = [_full_row("obs")] + _shift_rows("obs", "minority_specific", 0.60, 0.94)
    # Random shift degrades minority less (drop 0.02) so it does not qualify as a
    # hidden failure under the Primary definition (recall drop >= 0.10).
    rows += _shift_rows("obs", "random", 0.78, 0.955)
    # Floor-risk dataset: near-zero minority baseline.
    rows.append(_full_row("fr", minority=0.001))
    rows += _shift_rows("fr", "minority_specific", 0.0005, 0.90)
    return pd.DataFrame(rows)


def test_relative_drop_formula():
    assert math.isclose(relative_drop(0.80, 0.60), 0.25, rel_tol=1e-9)
    assert math.isclose(relative_drop(0.00, 0.00), 0.0, abs_tol=1e-12)


def test_hidden_definitions_keys():
    defs = hidden_definitions()
    assert set(defs) == {"Primary", "A", "B", "C", "D", "E"}
    assert defs["Primary"]["gtype"] == "AUROC"
    assert defs["D"]["gtype"] == "AUPRC"
    assert defs["E"]["mtype"] == "minority_f1"


def test_metric_sensitivity_scopes_to_observable():
    raw = _raw()
    sens, summary = compute_metric_sensitivity(raw, observable=["obs"])
    # Floor-risk 'fr' is excluded -> only the observable rows survive.
    assert set(sens["dataset"]) == {"obs"}
    # 8 metrics x (2 mechanisms x 3 severities) = 48 rows.
    assert len(sens) == 8 * 2 * 3
    # Hand-checked minority_recall relative drop for obs/minority_specific at sev 0.05:
    # (0.80 - 0.60)/0.80 = 0.25.
    sub = sens[(sens["mechanism"] == "minority_specific") & (sens["severity"] == 0.05)
               & (sens["metric"] == "minority_recall")]
    assert math.isclose(float(sub["relative_drop"].iloc[0]), 0.25, rel_tol=1e-6)
    # All relative drops are bounded (no denominator explosion from near-zero baselines).
    assert sens["relative_drop"].abs().max() <= 1.0 + 1e-6


def test_hidden_threshold_counts_drop_mapping():
    raw = _raw()
    hidden_sens, drops, examples = compute_hidden_threshold_sensitivity(raw)
    # The hidden-definition logic maps minority_recall -> minority_recall_drop column,
    # so every definition resolves to an integer count (no KeyError).
    assert set(hidden_sens["definition"]) == {"Primary", "A", "B", "C", "D", "E"}
    assert (hidden_sens["n_hidden_cases"] >= 0).all()
    # Primary uses 0.03 AUROC cap and 0.10 recall drop; only the minority_specific rows
    # (AUROC drop 0.01, recall drop 0.20) qualify -> 3 rows.
    primary = hidden_sens[hidden_sens["definition"] == "Primary"].iloc[0]
    assert primary["n_hidden_cases"] == 3
    assert not examples.empty


def test_severity_sensitivity_slope_refit():
    raw = _raw()
    sev_long, sev_sum = severity_sensitivity(raw, observable=["obs"])
    assert not sev_long.empty
    # Slope gap = |minority_slope| - |majority_slope|; minority degrades so gap > 0.
    assert (sev_long["slope_gap"] > 0).all()
    assert {"all", "exclude_0.40", "exclude_0.30_and_0.40"} == set(sev_sum["severity_set"])


def test_leave_one_dataset_out_positive():
    vul = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d2", "d2"],
            "model": ["lr"] * 4,
            "seed": [42] * 4,
            "shift_type": ["minority_specific", "majority_specific"] * 2,
            "MVG_recall": [0.60, -0.10, 0.55, -0.05],
        }
    )
    cc = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d2", "d2"],
            "model": ["lr"] * 4,
            "seed": [42] * 4,
            "DeltaMVG_class": [0.60, 0.60, 0.55, 0.55],
        }
    )
    out = leave_one_dataset_out(vul, cc, {"d1", "d2"})
    assert set(out["excluded_dataset"]) == {"d1", "d2"}
    assert (out["pooled_mvg"] > 0).all()
    assert (out["minority_specific_mvg"] > out["majority_specific_mvg"]).all()
