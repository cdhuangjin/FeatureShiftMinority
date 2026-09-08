"""Synthetic Gate rule cases: GO / HOLD / STOP."""

import pandas as pd

from src.gate_decision import decide_gate


def _cfg():
    return {
        "seeds": [42, 52, 62],
        "gate": {
            "strong_mvg": 0.10,
            "moderate_mvg": 0.05,
            "strong_specificity_gap": 0.10,
            "hidden_auroc_drop_max": 0.03,
            "hidden_minority_recall_drop_min": 0.10,
            "min_seed_consistency": 2,
            "min_datasets_for_go": 2,
        },
    }


def _cols():
    return [
        "dataset", "model", "mean_MVG_recall", "mean_MVG_f1", "MSI_vs_random_gap",
        "MSI_vs_global_gap", "MSI_minus_global_recall_drop_gap", "hidden_failure_count",
        "seed_consistency_recall", "seed_consistency_f1", "seed_consistency_msi",
    ]


def test_go_case():
    rows = []
    for ds in ["d1", "d2"]:
        for m in ["xgboost", "random_forest"]:
            rows.append(dict(zip(_cols(), [ds, m, 0.25, 0.20, 0.18, 0.12, 0.15, 2, 1.0, 1.0, 1.0])))
    decision, _ = decide_gate(pd.DataFrame(rows), _cfg(), {"d1": 10.0, "d2": 40.0})
    assert decision == "GO"


def test_hold_case():
    rows = [
        dict(zip(_cols(), ["d1", "xgboost", 0.07, 0.06, 0.04, 0.02, 0.03, 1, 0.8, 0.8, 0.8])),
        dict(zip(_cols(), ["d1", "random_forest", 0.08, 0.05, 0.05, 0.03, 0.02, 1, 0.8, 0.8, 0.8])),
    ]
    decision, _ = decide_gate(pd.DataFrame(rows), _cfg(), {"d1": 10.0})
    assert decision == "HOLD"


def test_stop_case():
    rows = []
    for ds in ["d1", "d2", "d3"]:
        for m in ["xgboost", "random_forest"]:
            rows.append(dict(zip(_cols(), [ds, m, 0.005, 0.004, 0.001, 0.001, 0.002, 0, 0.33, 0.33, 0.33])))
    decision, _ = decide_gate(pd.DataFrame(rows), _cfg(), {"d1": 10.0, "d2": 40.0, "d3": 120.0})
    assert decision == "STOP/PIVOT"

