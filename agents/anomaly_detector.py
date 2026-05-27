"""
agents/anomaly_detector.py
Anomaly Detector Agent — Stage 1 of the fraud detection pipeline.

Phase 2: Real ML models (PyTorch autoencoder + Isolation Forest).
Falls back to stub scoring if models are not yet trained.
"""

from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path

from pipeline.schemas import Transaction, AnomalyResult
from agents.base_agent import BaseAgent

logger = logging.getLogger("agent.anomaly_detector")

MODEL_DIR   = Path("model_artifacts")
CONFIG_PATH = MODEL_DIR / "model_config.json"


class AnomalyDetectorAgent(BaseAgent):
    """
    Monitors transaction stream and assigns a fraud probability score.

    When models are trained (Phase 2):
      - Loads autoencoder + isolation forest from model_artifacts/
      - Combines scores with configured weights
      - Uses learned threshold for flagging

    When models are NOT trained (stub mode):
      - Falls back to heuristic scoring
      - Logs a warning to remind you to train

    Inputs:  Transaction
    Outputs: AnomalyResult
    SLA:     <200ms
    """

    def __init__(self, fraud_score_threshold: float = 0.5):
        super().__init__(name="anomaly_detector", sla_ms=200.0)
        self.threshold    = fraud_score_threshold
        self.preprocessor = None
        self.ae_trainer   = None
        self.iso_forest   = None
        self.config       = {}
        self.model_loaded = False

        self._try_load_models()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _try_load_models(self) -> None:
        """Attempt to load trained models. Falls back to stub if not found."""
        try:
            if not CONFIG_PATH.exists():
                self.logger.info("No trained models found — running in STUB mode")
                self.logger.info("Run: python -m models.train  to train models")
                return

            from models.preprocessor     import FraudPreprocessor
            from models.autoencoder      import AutoencoderTrainer
            from models.isolation_forest import FraudIsolationForest

            self.config       = json.loads(CONFIG_PATH.read_text())
            self.preprocessor = FraudPreprocessor.load(MODEL_DIR / "preprocessor.joblib")
            self.ae_trainer   = AutoencoderTrainer.load(MODEL_DIR / "autoencoder.pt")
            self.iso_forest   = FraudIsolationForest.load(MODEL_DIR / "isolation_forest.joblib")

            # Use the learned optimal threshold
            if "fraud_threshold" in self.config:
                self.threshold = self.config["fraud_threshold"]

            self.model_loaded = True
            self.logger.info(
                "Models loaded | threshold=%.3f | ae_weight=%.1f | if_weight=%.1f",
                self.threshold,
                self.config.get("ae_weight", 0.6),
                self.config.get("if_weight", 0.4),
            )

        except Exception as exc:
            self.logger.warning("Failed to load models (%s) — using stub", exc)
            self.model_loaded = False

    # ── BaseAgent interface ────────────────────────────────────────────────────

    def process(self, payload: Transaction) -> AnomalyResult:
        if self.model_loaded:
            fraud_score, anomaly_vector = self._score_real(payload)
        else:
            fraud_score, anomaly_vector = self._score_stub(payload)

        is_flagged = fraud_score >= self.threshold

        return AnomalyResult(
            transaction_id   = payload.transaction_id,
            fraud_score      = round(float(fraud_score), 4),
            is_flagged       = is_flagged,
            anomaly_vector   = anomaly_vector,
            detector_version = "v1.0-ml" if self.model_loaded else "v0.1-stub",
        )

    def health_check(self) -> bool:
        return True

    def reload_models(self) -> bool:
        """Hot-reload models after retraining — called by orchestrator."""
        self.model_loaded = False
        self._try_load_models()
        return self.model_loaded

    # ── Real scoring (Phase 2) ─────────────────────────────────────────────────

    def _score_real(self, tx: Transaction) -> tuple[float, dict]:
        """Score using trained autoencoder + isolation forest."""
        try:
            tx_dict = {
                "amount":         tx.amount,
                "type":           self._infer_tx_type(tx),
                "oldbalanceOrg":  0.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }

            X = self.preprocessor.transform_transaction(tx_dict)

            ae_weight = self.config.get("ae_weight", 0.6)
            if_weight = self.config.get("if_weight", 0.4)

            ae_score = float(self.ae_trainer.score(X)[0])
            if_score = float(self.iso_forest.score(X)[0])

            combined = ae_weight * ae_score + if_weight * if_score
            combined = float(np.clip(combined, 0.0, 1.0))

            anomaly_vector = {
                "autoencoder_score":    round(ae_score, 4),
                "isolation_forest_score": round(if_score, 4),
                "combined_score":       round(combined,  4),
                "ae_weight":            ae_weight,
                "if_weight":            if_weight,
                "threshold":            self.threshold,
                "model_version":        "v1.0-ml",
            }

            return combined, anomaly_vector

        except Exception as exc:
            self.logger.warning("Real scoring failed (%s) — falling back to stub", exc)
            return self._score_stub(tx)

    def _infer_tx_type(self, tx: Transaction) -> str:
        """Map merchant category back to PaySim transaction type."""
        mapping = {
            "retail":        "PAYMENT",
            "bank_transfer": "TRANSFER",
            "atm_cash":      "CASH_OUT",
            "bank_deposit":  "CASH_IN",
        }
        return mapping.get(tx.merchant_category, "PAYMENT")

    # ── Stub scoring (fallback) ────────────────────────────────────────────────

    def _score_stub(self, tx: Transaction) -> tuple[float, dict]:
        import random
        base_score = random.uniform(0.0, 0.3)
        if tx.amount > 1000:       base_score += random.uniform(0.1, 0.4)
        if tx.merchant_category == "unknown": base_score += 0.15
        if tx.location == "unknown":          base_score += 0.1
        fraud_score = min(round(base_score, 4), 1.0)

        return fraud_score, {
            "reconstruction_error": round(random.uniform(0, 1), 4),
            "isolation_score":      round(random.uniform(-0.5, 0.5), 4),
            "note": "STUB — run: python -m models.train",
        }
