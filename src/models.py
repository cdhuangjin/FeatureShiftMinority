"""Fixed model factory (XGBoost, Random Forest, Logistic Regression, LightGBM)."""

from __future__ import annotations

from typing import Any, Dict

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_model(name: str, seed: int, model_config: Dict[str, dict]) -> Any:
    """Return an unfitted classifier for the given model name."""
    cfg = model_config[name]
    if name == "xgboost":
        params = {
            "n_estimators": cfg.get("n_estimators", 300),
            "max_depth": cfg.get("max_depth", 6),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "subsample": cfg.get("subsample", 0.9),
            "colsample_bytree": cfg.get("colsample_bytree", 0.9),
            "objective": cfg.get("objective", "binary:logistic"),
            "eval_metric": cfg.get("eval_metric", "logloss"),
            "tree_method": cfg.get("tree_method", "hist"),
            "n_jobs": cfg.get("n_jobs", -1),
            "random_state": seed,
        }
        if "n_jobs" not in params or params["n_jobs"] is None:
            params.pop("n_jobs", None)
        return XGBClassifier(**params)
    if name == "random_forest":
        params = {
            "n_estimators": cfg.get("n_estimators", 300),
            "max_depth": cfg.get("max_depth"),
            "min_samples_leaf": cfg.get("min_samples_leaf", 2),
            "class_weight": cfg.get("class_weight"),
            "n_jobs": cfg.get("n_jobs", -1),
            "random_state": seed,
        }
        params = {k: v for k, v in params.items() if v is not None}
        return RandomForestClassifier(**params)
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=cfg.get("max_iter", 2000),
            C=cfg.get("C", 1.0),
            class_weight=cfg.get("class_weight"),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=seed,
        )
    if name == "lightgbm":
        params = {
            "n_estimators": cfg.get("n_estimators", 300),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "max_depth": cfg.get("max_depth", -1),
            "num_leaves": cfg.get("num_leaves", 31),
            "subsample": cfg.get("subsample", 0.9),
            "colsample_bytree": cfg.get("colsample_bytree", 0.9),
            "verbose": -1,
            "n_jobs": cfg.get("n_jobs", -1),
            "random_state": seed,
        }
        params = {k: v for k, v in params.items() if v is not None}
        return LGBMClassifier(**params)
    raise ValueError(f"Unknown model: {name}")
