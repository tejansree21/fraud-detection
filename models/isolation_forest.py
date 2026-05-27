"""
models/isolation_forest.py
Isolation Forest for novelty/anomaly detection.

Complements the autoencoder:
- Autoencoder: learns reconstruction of normal patterns
- Isolation Forest: detects outliers based on feature space isolation

The two scores are combined by the AnomalyDetectorAgent.
"""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)
import joblib

logger = logging.getLogger("isolation_forest")


class FraudIsolationForest:
    """
    Isolation Forest wrapper with fraud-specific configuration.

    Key choices:
    - contamination: set to approximate fraud rate in training data
    - n_estimators: 200 trees for stable scores
    - Trained on ALL training data (unlike autoencoder which uses legit only)
      because IsolationForest is inherently unsupervised
    """

    def __init__(
        self,
        contamination: float = 0.013,  # ~PaySim fraud rate
        n_estimators:  int   = 200,
        max_samples:   str   = "auto",
        random_state:  int   = 42,
        n_jobs:        int   = -1,
    ):
        self.model = IsolationForest(
            contamination = contamination,
            n_estimators  = n_estimators,
            max_samples   = max_samples,
            random_state  = random_state,
            n_jobs        = n_jobs,
        )
        self.is_fitted = False
        logger.info(
            "IsolationForest | contamination=%.3f | n_estimators=%d",
            contamination, n_estimators
        )

    def fit(self, X_train: np.ndarray) -> None:
        """Train on the full training set (unsupervised)."""
        logger.info("Fitting IsolationForest on %d samples...", len(X_train))
        self.model.fit(X_train)
        self.is_fitted = True
        logger.info("IsolationForest fitted.")

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return fraud scores in [0, 1].
        IsolationForest's decision_function returns negative scores for outliers.
        We invert and normalise to [0, 1] where 1 = most anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        raw_scores = self.model.decision_function(X)
        # decision_function: more negative = more anomalous
        # Invert and normalise to [0, 1]
        inverted = -raw_scores
        min_s, max_s = inverted.min(), inverted.max()
        if max_s - min_s < 1e-8:
            return np.full(len(X), 0.5, dtype=np.float32)
        normalised = (inverted - min_s) / (max_s - min_s)
        return normalised.astype(np.float32)

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions: 1=anomaly, 0=normal."""
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute evaluation metrics on test set."""
        scores  = self.score(X_test)
        preds   = self.predict_binary(X_test)

        auc_pr  = average_precision_score(y_test, scores)
        auc_roc = roc_auc_score(y_test, scores)
        f1      = f1_score(y_test, preds)

        precision, recall, thresholds = precision_recall_curve(y_test, scores)

        metrics = {
            "auc_pr":       round(float(auc_pr),  4),
            "auc_roc":      round(float(auc_roc), 4),
            "f1_score":     round(float(f1),       4),
            "n_test":       len(y_test),
            "n_fraud_test": int(y_test.sum()),
        }

        logger.info(
            "IsolationForest eval | AUC-PR=%.4f | AUC-ROC=%.4f | F1=%.4f",
            auc_pr, auc_roc, f1
        )
        return metrics

    def save(self, path: Path) -> None:
        joblib.dump({"model": self.model, "is_fitted": self.is_fitted}, path)
        logger.info("IsolationForest saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FraudIsolationForest":
        data = joblib.load(path)
        inst = cls()
        inst.model     = data["model"]
        inst.is_fitted = data["is_fitted"]
        logger.info("IsolationForest loaded from %s", path)
        return inst
