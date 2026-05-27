"""
agents/context_enricher.py
Context Enricher Agent — Stage 2 of the fraud detection pipeline.

Phase 3: Real Redis feature store lookups replacing all stubs.

Enrichment signals:
  1. Merchant category risk score     (Redis lookup)
  2. User behaviour deviation score   (compared to historical profile)
  3. Geolocation consistency score    (vs user's location history)
  4. Device fingerprint match         (known device registry)
  5. Velocity features                (tx/hour, tx/day sliding windows)
"""

from __future__ import annotations
import logging
import os
from pipeline.schemas import Transaction, AnomalyResult, EnrichedTransaction
from agents.base_agent import BaseAgent

logger = logging.getLogger("agent.context_enricher")


class ContextEnricherAgent(BaseAgent):
    """
    Augments flagged transactions with contextual risk signals via Redis.

    Falls back to stub values gracefully if Redis is unavailable.

    Inputs:  (Transaction, AnomalyResult)
    Outputs: EnrichedTransaction
    SLA:     <500ms
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        super().__init__(name="context_enricher", sla_ms=500.0)
        self.store = None
        self._try_connect(redis_host, redis_port)

    def _try_connect(self, host: str, port: int) -> None:
        try:
            from feature_store.redis_store import RedisFeatureStore
            self.store = RedisFeatureStore(host=host, port=port)
            if self.store.available:
                self.logger.info(
                    "ContextEnricherAgent connected to Redis feature store"
                )
            else:
                self.logger.warning(
                    "Redis unavailable — ContextEnricher running in fallback mode"
                )
        except Exception as e:
            self.logger.warning("Could not connect to Redis (%s) — fallback mode", e)
            self.store = None

    # ── BaseAgent interface ────────────────────────────────────────────────────

    def process(self, payload: tuple[Transaction, AnomalyResult]) -> EnrichedTransaction:
        tx, anomaly = payload

        if self.store and self.store.available:
            return self._enrich_real(tx, anomaly)
        else:
            return self._enrich_fallback(tx, anomaly)

    def health_check(self) -> bool:
        if self.store:
            return self.store._check_connection()
        return False

    # ── Real enrichment (Phase 3) ──────────────────────────────────────────────

    def _enrich_real(self, tx: Transaction, anomaly: AnomalyResult) -> EnrichedTransaction:
        """Live Redis feature store lookups."""

        # 1. Merchant risk
        merchant_risk = self.store.get_merchant_risk(tx.merchant_category)

        # 2. Geo consistency
        geo_score = self.store.get_geo_consistency(tx.user_id, tx.location)

        # 3. Device match
        device_match = self.store.is_known_device(tx.user_id, tx.device_id)
        device_count = self.store.get_device_count(tx.user_id)

        # 4. Velocity
        velocity_hour = self.store.get_velocity(tx.user_id, window_seconds=3600)
        velocity_day  = self.store.get_velocity(tx.user_id, window_seconds=86400)

        # 5. User behaviour deviation
        behaviour_score = self.store.compute_behaviour_score(tx.user_id, tx.amount)

        # 6. Update store with this transaction (for future lookups)
        self.store.record_transaction(tx.user_id, tx.transaction_id)
        self.store.update_location(tx.user_id, tx.location)
        if tx.device_id != "unknown":
            self.store.register_device(tx.user_id, tx.device_id)

        # 7. Get user profile for report context
        profile = self.store.get_user_profile(tx.user_id)

        enriched_payload = {
            "merchant_category":     tx.merchant_category,
            "known_device_count":    device_count,
            "avg_tx_amount_30d":     profile.get("avg_amount", 0.0),
            "user_tx_count":         profile.get("tx_count", 0),
            "user_max_amount":       profile.get("max_amount", 0.0),
            "geo_consistency_detail": {
                "current":  tx.location,
                "is_known": geo_score > 0.5,
            },
            "source": "redis_feature_store",
        }

        return EnrichedTransaction(
            transaction_id        = tx.transaction_id,
            fraud_score           = anomaly.fraud_score,
            merchant_risk_score   = round(merchant_risk,   4),
            geo_consistency_score = round(geo_score,       4),
            device_match          = device_match,
            velocity_hour         = velocity_hour,
            velocity_day          = velocity_day,
            user_behaviour_score  = round(behaviour_score, 4),
            enriched_payload      = enriched_payload,
        )

    # ── Fallback (stub) when Redis is unavailable ──────────────────────────────

    def _enrich_fallback(self, tx: Transaction, anomaly: AnomalyResult) -> EnrichedTransaction:
        """Heuristic fallback when Redis is not available."""
        import random
        suspicion = anomaly.fraud_score

        category_risk = {
            "atm_cash": 0.65, "bank_transfer": 0.5, "unknown": 0.75,
            "retail": 0.15, "food_and_drink": 0.1, "bank_deposit": 0.15,
        }
        merchant_risk = category_risk.get(tx.merchant_category, 0.4)
        geo_score     = 0.3 if tx.location == "unknown" else 0.8
        device_match  = tx.device_id != "unknown"

        return EnrichedTransaction(
            transaction_id        = tx.transaction_id,
            fraud_score           = anomaly.fraud_score,
            merchant_risk_score   = round(min(merchant_risk + suspicion * 0.2, 1.0), 4),
            geo_consistency_score = round(geo_score, 4),
            device_match          = device_match,
            velocity_hour         = random.randint(0, 5),
            velocity_day          = random.randint(0, 20),
            user_behaviour_score  = round(min(suspicion * 0.7, 1.0), 4),
            enriched_payload      = {"source": "fallback_heuristic"},
        )
