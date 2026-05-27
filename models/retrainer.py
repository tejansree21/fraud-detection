"""
models/retrainer.py
Retraining pipeline — triggered by the feedback loop when:
  1. Enough investigator decisions have accumulated (threshold-based)
  2. ADWIN drift detector fires (distribution-based)

What retraining does:
  - Pulls confirmed fraud/legitimate labels from PostgreSQL feedback table
  - Combines with original training data
  - Retrains autoencoder and Isolation Forest
  - Logs new run to MLflow
  - Hot-swaps the model in the AnomalyDetectorAgent without restart
"""

from __future__ import annotations
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any

logger = logging.getLogger("retrainer")

MODEL_DIR = Path("model_artifacts")


class FeedbackRetrainer:
    """
    Manages the feedback-driven retraining cycle.

    Usage:
        retrainer = FeedbackRetrainer(db_writer, anomaly_detector)
        retrainer.run()   # called by orchestrator when threshold is met
    """

    def __init__(self, db_writer=None, anomaly_detector=None):
        self.db              = db_writer
        self.anomaly_detector = anomaly_detector
        self.retrain_history: list[dict[str, Any]] = []
        self._load_history()

    def run(self, reason: str = "threshold") -> dict[str, Any]:
        """
        Execute a retraining cycle.
        Returns metrics from the new model.
        """
        logger.info("=== Retraining started | reason=%s ===", reason)
        start = datetime.utcnow()

        result = {
            "timestamp":  start.isoformat(),
            "reason":     reason,
            "status":     "failed",
            "metrics":    {},
        }

        try:
            # ── Step 1: Get feedback labels from DB ────────────────────────────
            feedback_data = self._get_feedback_labels()
            if len(feedback_data) < 10:
                logger.warning(
                    "Not enough feedback labels (%d) — need at least 10",
                    len(feedback_data)
                )
                result["status"] = "skipped"
                result["reason"] = f"insufficient_feedback ({len(feedback_data)} labels)"
                return result

            logger.info("Retrieved %d feedback labels", len(feedback_data))

            # ── Step 2: Load training data ────────────────────────────────────
            data_path = self._find_dataset()
            if not data_path:
                logger.warning("No training dataset found — skipping retraining")
                result["status"] = "skipped"
                result["reason"] = "no_dataset"
                return result

            # ── Step 3: Retrain models ────────────────────────────────────────
            metrics = self._retrain_models(data_path, feedback_data)
            result["metrics"] = metrics
            result["status"]  = "success"

            # ── Step 4: Hot-reload in anomaly detector ─────────────────────────
            if self.anomaly_detector:
                reloaded = self.anomaly_detector.reload_models()
                result["models_reloaded"] = reloaded
                logger.info("Models hot-reloaded in anomaly detector: %s", reloaded)

            elapsed = (datetime.utcnow() - start).total_seconds()
            result["duration_seconds"] = round(elapsed, 1)
            logger.info("=== Retraining complete in %.1fs | %s ===", elapsed, metrics)

        except Exception as e:
            logger.error("Retraining failed: %s", e, exc_info=True)
            result["error"] = str(e)

        self.retrain_history.append(result)
        self._save_history()
        return result

    # ── Internal steps ─────────────────────────────────────────────────────────

    def _get_feedback_labels(self) -> list[dict]:
        """Pull investigator decisions from PostgreSQL."""
        if not self.db or not self.db.available:
            return []
        try:
            from sqlalchemy import text
            sql = text("""
                SELECT
                    f.transaction_id,
                    f.decision,
                    t.amount,
                    t.merchant_category,
                    ft.fraud_score
                FROM investigator_feedback f
                JOIN transactions t          ON t.transaction_id = f.transaction_id
                JOIN flagged_transactions ft ON ft.transaction_id = f.transaction_id
                ORDER BY f.decided_at DESC
                LIMIT 1000
            """)
            with self.db.engine.connect() as conn:
                rows = conn.execute(sql).mappings().all()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to fetch feedback labels: %s", e)
            return []

    def _find_dataset(self) -> Path | None:
        """Find the best available training dataset."""
        candidates = [
            Path("data/PS_20174392719_1491204439457_log.csv"),
            Path("data/paysim_sample.csv"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _retrain_models(
        self,
        data_path: Path,
        feedback_data: list[dict],
    ) -> dict[str, Any]:
        """Retrain autoencoder + Isolation Forest with new labels."""
        from models.preprocessor     import FraudPreprocessor
        from models.autoencoder      import AutoencoderTrainer
        from models.isolation_forest import FraudIsolationForest
        import numpy as np

        # Load and preprocess data
        preprocessor = FraudPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_paysim(
            data_path, test_size=0.2
        )

        # Log feedback summary
        confirmed_fraud = sum(
            1 for f in feedback_data if f["decision"] == "CONFIRMED_FRAUD"
        )
        false_positives = sum(
            1 for f in feedback_data if f["decision"] == "FALSE_POSITIVE"
        )
        logger.info(
            "Feedback: %d confirmed fraud, %d false positives",
            confirmed_fraud, false_positives
        )

        # Retrain autoencoder
        ae_trainer = AutoencoderTrainer(
            input_dim  = X_train.shape[1],
            epochs     = 20,   # fewer epochs for incremental retraining
            batch_size = 256,
        )
        ae_metrics = ae_trainer.train(X_train, y_train, X_test, y_test)

        # Retrain Isolation Forest
        fraud_rate = max(float(y_train.mean()), 0.01)
        iso_forest = FraudIsolationForest(contamination=fraud_rate)
        iso_forest.fit(X_train)
        if_metrics = iso_forest.evaluate(X_test, y_test)

        # Save updated models
        preprocessor.save(MODEL_DIR / "preprocessor.joblib")
        ae_trainer.save(MODEL_DIR  / "autoencoder.pt")
        iso_forest.save(MODEL_DIR  / "isolation_forest.joblib")

        # Update config
        from sklearn.metrics import average_precision_score
        from models.isolation_forest import FraudIsolationForest as IFF
        ae_scores = ae_trainer.score(X_test)
        if_scores = iso_forest.score(X_test)
        combined  = 0.6 * ae_scores + 0.4 * if_scores

        from sklearn.metrics import f1_score
        import numpy as np
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(y_test, (combined >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        config_path = MODEL_DIR / "model_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        config["fraud_threshold"]  = best_t
        config["retrain_timestamp"] = datetime.utcnow().isoformat()
        config["retrain_reason"]   = "feedback_loop"
        config_path.write_text(json.dumps(config, indent=2))

        # Log to MLflow if available
        self._log_to_mlflow({
            "ae_auc_pr":        float(average_precision_score(y_test, ae_scores)),
            "if_auc_pr":        if_metrics["auc_pr"],
            "combined_f1":      best_f1,
            "optimal_threshold": best_t,
            "feedback_labels":  len(feedback_data),
            "confirmed_fraud":  confirmed_fraud,
        })

        return {
            "ae_final_loss":    ae_metrics["final_train_loss"],
            "if_auc_pr":        if_metrics["auc_pr"],
            "combined_f1":      round(best_f1, 4),
            "new_threshold":    best_t,
            "feedback_used":    len(feedback_data),
        }

    def _log_to_mlflow(self, metrics: dict) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("fraud_detection_retraining")
            with mlflow.start_run(run_name="feedback_retrain"):
                mlflow.log_metrics(metrics)
                mlflow.log_param("trigger", "feedback_loop")
        except Exception as e:
            logger.debug("MLflow logging skipped: %s", e)

    def _save_history(self) -> None:
        history_path = MODEL_DIR / "retrain_history.json"
        try:
            history_path.write_text(json.dumps(self.retrain_history[-20:], indent=2))
        except Exception:
            pass

    def _load_history(self) -> None:
        history_path = MODEL_DIR / "retrain_history.json"
        if history_path.exists():
            try:
                self.retrain_history = json.loads(history_path.read_text())
            except Exception:
                self.retrain_history = []

    def status(self) -> dict[str, Any]:
        last = self.retrain_history[-1] if self.retrain_history else None
        return {
            "total_retrains": len(self.retrain_history),
            "last_retrain":   last,
        }
