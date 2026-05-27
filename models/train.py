"""
models/train.py
Training script — trains autoencoder + Isolation Forest and logs
everything to MLflow.

Usage:
  python -m models.train
  python -m models.train --epochs 50 --dataset data/paysim_sample.csv
"""

from __future__ import annotations
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    classification_report,
)

from models.preprocessor    import FraudPreprocessor
from models.autoencoder     import AutoencoderTrainer
from models.isolation_forest import FraudIsolationForest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("train")

MODEL_DIR = Path("model_artifacts")
MODEL_DIR.mkdir(exist_ok=True)


def evaluate_combined(
    ae_scores: np.ndarray,
    if_scores: np.ndarray,
    y_test:    np.ndarray,
    ae_weight: float = 0.6,
    if_weight: float = 0.4,
) -> dict:
    """
    Combine autoencoder + isolation forest scores with weighted average.
    The autoencoder gets higher weight as it's more principled for this task.
    """
    combined = ae_weight * ae_scores + if_weight * if_scores

    # Find optimal F1 threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1, best_threshold = 0.0, 0.5
    for t in thresholds:
        preds = (combined >= t).astype(int)
        f1    = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1:
            best_f1       = f1
            best_threshold = t

    final_preds = (combined >= best_threshold).astype(int)

    metrics = {
        "combined_auc_pr":    round(float(average_precision_score(y_test, combined)), 4),
        "combined_auc_roc":   round(float(roc_auc_score(y_test, combined)),           4),
        "combined_f1":        round(float(best_f1),                                   4),
        "optimal_threshold":  round(float(best_threshold),                            4),
        "ae_weight":          ae_weight,
        "if_weight":          if_weight,
    }

    logger.info(
        "Combined model | AUC-PR=%.4f | AUC-ROC=%.4f | F1=%.4f @ threshold=%.2f",
        metrics["combined_auc_pr"],
        metrics["combined_auc_roc"],
        metrics["combined_f1"],
        metrics["optimal_threshold"],
    )

    print("\n=== Classification Report (combined model) ===")
    print(classification_report(y_test, final_preds, target_names=["Legitimate", "Fraud"]))

    return metrics


def train(
    dataset_path: Path,
    epochs:       int   = 30,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    ae_weight:    float = 0.6,
    experiment:   str   = "fraud_detection",
):
    # ── MLflow setup ──────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(experiment)
        use_mlflow = True
    except Exception as e:
        logger.warning("MLflow not available (%s) — training without tracking", e)
        use_mlflow = False

    with (mlflow.start_run() if use_mlflow else _null_context()) as run:

        # ── Log parameters ─────────────────────────────────────────────────
        params = {
            "dataset":    str(dataset_path),
            "epochs":     epochs,
            "batch_size": batch_size,
            "lr":         lr,
            "ae_weight":  ae_weight,
            "if_weight":  1 - ae_weight,
        }
        if use_mlflow:
            mlflow.log_params(params)
        logger.info("Training params: %s", params)

        # ── Preprocess ─────────────────────────────────────────────────────
        preprocessor = FraudPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_paysim(dataset_path)

        # ── Train Autoencoder ──────────────────────────────────────────────
        logger.info("=== Training Autoencoder ===")
        ae_trainer = AutoencoderTrainer(
            input_dim  = X_train.shape[1],
            epochs     = epochs,
            batch_size = batch_size,
            lr         = lr,
        )
        ae_metrics = ae_trainer.train(X_train, y_train, X_test, y_test)

        # Evaluate autoencoder alone
        ae_scores  = ae_trainer.score(X_test)
        ae_auc_pr  = average_precision_score(y_test, ae_scores)
        ae_auc_roc = roc_auc_score(y_test, ae_scores)
        logger.info("Autoencoder | AUC-PR=%.4f | AUC-ROC=%.4f", ae_auc_pr, ae_auc_roc)

        # ── Train Isolation Forest ─────────────────────────────────────────
        logger.info("=== Training Isolation Forest ===")
        fraud_rate = float(y_train.mean())
        iso_forest = FraudIsolationForest(contamination=max(fraud_rate, 0.01))
        iso_forest.fit(X_train)
        if_metrics = iso_forest.evaluate(X_test, y_test)
        if_scores  = iso_forest.score(X_test)

        # ── Combined evaluation ────────────────────────────────────────────
        logger.info("=== Combined Model Evaluation ===")
        combined_metrics = evaluate_combined(
            ae_scores, if_scores, y_test, ae_weight=ae_weight
        )

        # ── Save artifacts ─────────────────────────────────────────────────
        preprocessor.save(MODEL_DIR / "preprocessor.joblib")
        ae_trainer.save(MODEL_DIR  / "autoencoder.pt")
        iso_forest.save(MODEL_DIR  / "isolation_forest.joblib")

        # Save combined threshold config
        config = {
            "ae_weight":         ae_weight,
            "if_weight":         1 - ae_weight,
            "fraud_threshold":   combined_metrics["optimal_threshold"],
            "input_dim":         X_train.shape[1],
            "feature_cols":      preprocessor.feature_cols,
        }
        config_path = MODEL_DIR / "model_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        logger.info("Model config saved to %s", config_path)

        # ── Log to MLflow ──────────────────────────────────────────────────
        if use_mlflow:
            all_metrics = {
                "ae_auc_pr":   ae_auc_pr,
                "ae_auc_roc":  ae_auc_roc,
                "if_auc_pr":   if_metrics["auc_pr"],
                "if_auc_roc":  if_metrics["auc_roc"],
                **combined_metrics,
            }
            mlflow.log_metrics(all_metrics)
            mlflow.log_artifacts(str(MODEL_DIR))
            run_id = run.info.run_id
            logger.info("MLflow run: http://localhost:5000/#/experiments/1/runs/%s", run_id)

        logger.info("=== Training complete ===")
        logger.info("Models saved to %s/", MODEL_DIR)
        return combined_metrics


class _null_context:
    """Fallback context manager when MLflow is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    info = type("info", (), {"run_id": "no-mlflow"})()


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--dataset",    type=str,   default="data/paysim_sample.csv")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--ae-weight",  type=float, default=0.6)
    parser.add_argument("--experiment", type=str,   default="fraud_detection")
    args = parser.parse_args()

    train(
        dataset_path = Path(args.dataset),
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        ae_weight    = args.ae_weight,
        experiment   = args.experiment,
    )


if __name__ == "__main__":
    main()
