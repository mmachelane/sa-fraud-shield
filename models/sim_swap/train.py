"""
SIM swap model training with temporal cross-validation.

Usage:
    python -m models.sim_swap.train \
        --data-path data/processed/sim_swap_features.parquet \
        --experiment-name sim-swap-detector

Temporal CV:
    Fold 1: train months 1-4, validate month 5
    Fold 2: train months 1-5, validate month 6
    Holdout: month 7 (final evaluation, not used for tuning)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from models.sim_swap.model import NON_FEATURE_COLS, SimSwapDetector

logger = logging.getLogger(__name__)


def _temporal_folds(
    df: pd.DataFrame,
    n_train_months: list[int],
    holdout_month: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build temporal CV folds.
    Each fold: train on months [1..k], validate on month k+1.
    """
    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M")
    months = sorted(df["month"].unique())

    if len(months) < 2:
        raise ValueError(f"Need at least 2 months of data, got {len(months)}")

    folds = []
    for k in n_train_months:
        train_months = months[:k]
        val_months = months[k : k + 1]
        if not val_months:
            break
        train = df[df["month"].isin(train_months)]
        val = df[df["month"].isin(val_months)]
        folds.append((train, val))

    return folds


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c != "month"]
    X = df[feature_cols].fillna(0).astype(float)  # noqa: N806
    y = df["label"].astype(int)
    return X, y


def train(
    data_path: Path,
    experiment_name: str,
    mlflow_uri: str = "http://localhost:5001",
    model_output_dir: Path = Path("models/sim_swap/artifacts"),
    build_features_if_missing: bool = True,
) -> None:
    # Build features if not yet generated
    if not data_path.exists() and build_features_if_missing:
        logger.info("Features not found — running feature engineering first...")
        from models.sim_swap.features import build_features

        data_dir = data_path.parent.parent
        build_features(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {data_path}\n"
            f"Run: python -m models.sim_swap.features --data-dir data/"
        )

    logger.info(f"Loading features from {data_path}...")
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fraud_rate = df["label"].mean()
    logger.info(
        f"Dataset: {len(df):,} rows | {df['label'].sum():,} positive ({fraud_rate:.3%} fraud rate)"
    )

    # ── Temporal split ────────────────────────────────────────────────────────
    df["month"] = df["timestamp"].dt.to_period("M")
    months = sorted(df["month"].unique())
    n_months = len(months)

    if n_months < 3:
        raise ValueError(f"Need at least 3 months of data for temporal CV, got {n_months}")

    # Reserve last month as holdout; CV on the rest
    holdout_month = months[-1]
    cv_months = months[:-1]

    holdout_df = df[df["month"] == holdout_month]
    cv_df = df[df["month"].isin(cv_months)]

    # Folds: train on first k months, validate on month k+1
    # (at least 2 train, 1 val)
    n_cv = len(cv_months)
    fold_splits = list(range(max(2, n_cv - 2), n_cv))  # e.g. [3, 4] for 5 cv months

    folds = _temporal_folds(cv_df, fold_splits, holdout_month)
    logger.info(f"Running {len(folds)} temporal CV folds over {n_cv} months")

    # ── MLflow tracking ───────────────────────────────────────────────────────
    def _mlflow_reachable(uri: str) -> bool:
        import socket
        import urllib.parse

        parsed = urllib.parse.urlparse(uri)
        host, port = parsed.hostname, parsed.port or 80
        try:
            socket.create_connection((host, port), timeout=2).close()
            return True
        except OSError:
            return False

    if _mlflow_reachable(mlflow_uri):
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        logger.warning(f"MLflow server at {mlflow_uri} unreachable — using local ./mlruns")
        mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="temporal-cv"):
        fold_aucs = []

        for fold_idx, (train_df, val_df) in enumerate(folds):
            logger.info(
                f"Fold {fold_idx + 1}/{len(folds)}: "
                f"train={len(train_df):,}  val={len(val_df):,}  "
                f"val_fraud={val_df['label'].sum():,}"
            )

            X_train, y_train = _prepare_xy(train_df)  # noqa: N806
            X_val, y_val = _prepare_xy(val_df)  # noqa: N806

            detector = SimSwapDetector()
            metrics = detector.fit(X_train, y_train, X_val, y_val)
            fold_aucs.append(metrics["auc_roc"])

            logger.info(
                f"  Fold {fold_idx + 1}: AUC={metrics['auc_roc']:.4f}  "
                f"AvgPrecision={metrics['avg_precision']:.4f}  "
                f"BestIter={metrics['best_iteration']}"
            )
            mlflow.log_metrics(
                {
                    f"fold_{fold_idx + 1}_auc": metrics["auc_roc"],
                    f"fold_{fold_idx + 1}_avg_precision": metrics["avg_precision"],
                }
            )

        mean_cv_auc = np.mean(fold_aucs)
        logger.info(f"CV AUC: {mean_cv_auc:.4f} ± {np.std(fold_aucs):.4f}")
        mlflow.log_metrics(
            {
                "cv_auc_mean": mean_cv_auc,
                "cv_auc_std": float(np.std(fold_aucs)),
            }
        )

        # ── Final model: train on all CV data, evaluate on holdout ──
        logger.info("Training final model on all CV data...")
        X_cv, y_cv = _prepare_xy(cv_df)  # noqa: N806
        X_hold, y_hold = _prepare_xy(holdout_df)  # noqa: N806

        # Use last fold's val as early-stopping monitor
        last_train, last_val = folds[-1]
        X_last_val, y_last_val = _prepare_xy(last_val)  # noqa: N806

        final_detector = SimSwapDetector()
        final_detector.fit(X_cv, y_cv, X_last_val, y_last_val)

        hold_proba = final_detector.predict_proba(X_hold)
        holdout_auc = roc_auc_score(y_hold, hold_proba)
        holdout_ap = average_precision_score(y_hold, hold_proba)

        logger.info(f"Holdout AUC: {holdout_auc:.4f}  AvgPrecision: {holdout_ap:.4f}")
        mlflow.log_metrics({"holdout_auc": holdout_auc, "holdout_avg_precision": holdout_ap})

        # ── Feature importance ──
        importance_df = final_detector.feature_importance()
        logger.info(f"Top 10 features:\n{importance_df.head(10).to_string(index=False)}")

        # ── SHAP explanation on holdout sample ──
        sample_size = min(500, len(X_hold))
        shap_df = final_detector.explain(X_hold.sample(sample_size, random_state=42))
        shap_importance = shap_df.abs().mean().sort_values(ascending=False)
        logger.info(f"Top 10 SHAP features:\n{shap_importance.head(10).to_string()}")

        # ── Save model ──
        final_detector.save(model_output_dir)
        try:
            mlflow.log_artifact(str(model_output_dir / "model.txt"), artifact_path="model")
            mlflow.log_artifact(str(model_output_dir / "feature_names.json"), artifact_path="model")
        except Exception as e:
            logger.warning(f"MLflow artifact upload skipped: {e}")

        logger.info(f"Model saved to {model_output_dir}")
        mlflow.log_param("n_cv_folds", len(folds))
        mlflow.log_param("n_train_months", len(cv_months))
        mlflow.log_param("holdout_month", str(holdout_month))
        mlflow.log_param("fraud_rate", f"{fraud_rate:.4f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train SIM swap detector")
    parser.add_argument("--data-path", default="data/processed/sim_swap_features.parquet")
    parser.add_argument("--experiment-name", default="sim-swap-detector")
    parser.add_argument("--mlflow-uri", default="http://localhost:5001")
    parser.add_argument("--model-output-dir", default="models/sim_swap/artifacts")
    args = parser.parse_args()

    train(
        data_path=Path(args.data_path),
        experiment_name=args.experiment_name,
        mlflow_uri=args.mlflow_uri,
        model_output_dir=Path(args.model_output_dir),
    )


if __name__ == "__main__":
    main()
