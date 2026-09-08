"""Synthetic Paper Gate cases: STRONG-GO / GO / HOLD / STOP."""

from src.paper_gate import evaluate_paper_gate


def _cfg():
    return {
        "paper_gate": {
            "strong_mvg": 0.10,
            "min_observable_mvg_positive_datasets": 5,
            "min_observable_mvg_strong_datasets": 4,
            "min_models_positive": 3,
            "min_hidden_failure_datasets": 3,
            "min_hidden_consistency": 2,
            "min_structured_positive_datasets": 3,
            "min_class_control_datasets": 4,
            "strong_go_conditions_required": 7,
        }
    }


def _entry(mvg=0.0, slope=0.0, hidden=0, delta=0.0, gp=0.0, cp=0.0, obs="strongly_observable"):
    return {
        "mean_MVG_recall": mvg,
        "observability_class": obs,
        "slope_gap": slope,
        "hidden_failure_count": hidden,
        "DeltaMVG_class": delta,
        "group_penalty": gp,
        "corr_penalty": cp,
    }


def _summary(datasets, models, wilcox_p=0.001, fdr_p=0.001, es=0.6, ci_low=0.1):
    return {
        "dataset_mvg": datasets,
        "model_mvg": models,
        "wilcoxon": {"p_value": wilcox_p},
        "wilcoxon_fdr_p": fdr_p,
        "effect_size": es,
        "mvg_ci_low": ci_low,
    }


def test_strong_go():
    datasets = {
        f"d{i}": _entry(mvg=0.20, slope=0.1, hidden=3, delta=0.1, gp=0.1, cp=0.1)
        for i in range(1, 7)
    }
    models = {m: 0.20 for m in ["lr", "rf", "xgb", "lgbm"]}
    dec, ev = evaluate_paper_gate(_summary(datasets, models), _cfg())
    assert dec == "STRONG-GO"
    assert ev["n_conditions_met"] > _cfg()["paper_gate"]["strong_go_conditions_required"]


def test_go():
    datasets = {
        f"d{i}": _entry(mvg=0.20, slope=0.0, hidden=0, delta=0.0, gp=0.0, cp=0.0)
        for i in range(1, 6)
    }
    models = {m: 0.20 for m in ["lr", "rf", "xgb", "lgbm"]}
    dec, ev = evaluate_paper_gate(_summary(datasets, models), _cfg())
    assert dec == "GO"


def test_hold():
    datasets = {
        f"d{i}": _entry(mvg=0.15, slope=0.0, hidden=0, delta=0.0, gp=0.0, cp=0.0)
        for i in range(1, 5)
    }
    datasets["d5"] = _entry(mvg=0.0, obs="floor_risk")
    models = {m: 0.15 for m in ["lr", "rf", "xgb", "lgbm"]}
    dec, ev = evaluate_paper_gate(_summary(datasets, models), _cfg())
    assert dec == "HOLD"


def test_stop():
    datasets = {
        f"d{i}": _entry(mvg=0.0, slope=0.0, hidden=0, delta=0.0, gp=0.0, cp=0.0)
        for i in range(1, 7)
    }
    models = {m: 0.0 for m in ["lr", "rf", "xgb", "lgbm"]}
    dec, ev = evaluate_paper_gate(
        _summary(datasets, models, wilcox_p=0.9, fdr_p=0.9, es=0.0, ci_low=-0.1), _cfg()
    )
    assert dec == "STOP-PIVOT"

