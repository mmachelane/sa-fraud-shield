"""
SIM swap detector — LightGBM classifier with SHAP explainability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

# Columns excluded from model features (metadata only)
NON_FEATURE_COLS = {
    "transaction_id",
    "sender_account_id",
    "timestamp",
    "label",
    "date",
}

# LightGBM hyperparameters (tuned for class imbalance ~2%)
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": 40,  # approx 1/fraud_rate to handle imbalance
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
}


class SimSwapDetector:
    """LightGBM-based SIM swap fraud detector."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or DEFAULT_PARAMS.copy()
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []
        self.explainer: shap.TreeExplainer | None = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in NON_FEATURE_COLS]

    def fit(
        self,
        X_train: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,
        X_val: pd.DataFrame,  # noqa: N803
        y_val: pd.Series,
    ) -> dict[str, float]:
        """Train on train split, early-stop on val split. Returns val metrics."""
        self.feature_names = list(X_train.columns)

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
        )

        from sklearn.metrics import average_precision_score, roc_auc_score

        val_proba = self.model.predict_proba(X_val)[:, 1]
        return {
            "auc_roc": roc_auc_score(y_val, val_proba),
            "avg_precision": average_precision_score(y_val, val_proba),
            "best_iteration": self.model.best_iteration_,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        assert self.model is not None, "Model not trained"
        return self.model.predict_proba(X)[:, 1]

    def explain(self, X: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:  # noqa: N803
        """Return SHAP values for X (up to max_rows rows)."""
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
        subset = X.head(max_rows)
        shap_values = self.explainer.shap_values(subset)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary: index 1 = positive class
        return pd.DataFrame(shap_values, columns=self.feature_names, index=subset.index)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance sorted descending."""
        assert self.model is not None
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path | str) -> None:
        """Save model to disk using LightGBM native format."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        self.model.booster_.save_model(str(path / "model.txt"))
        import json

        (path / "feature_names.json").write_text(json.dumps(self.feature_names, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> SimSwapDetector:
        import json

        path = Path(path)
        detector = cls()
        booster = lgb.Booster(model_file=str(path / "model.txt"))
        detector.model = lgb.LGBMClassifier()
        detector.model._Booster = booster
        detector.model._n_features = booster.num_feature()
        detector.feature_names = json.loads((path / "feature_names.json").read_text())
        return detector
