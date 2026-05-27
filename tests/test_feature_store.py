"""
tests/test_feature_store.py
Tests for Redis feature store and real Context Enricher.
Uses fakeredis for unit tests (no real Redis needed).
"""

import pytest
import os
os.environ.setdefault("GROQ_API_KEY", "test_key")

from unittest.mock import patch, MagicMock
from pipeline.schemas import Transaction, AnomalyResult, EnrichedTransaction
from datetime import datetime


# ── Feature store tests (using fakeredis) ─────────────────────────────────────

class TestRedisFeatureStore:

    @pytest.fixture
    def store(self):
        """Create a feature store backed by fakeredis."""
        import fakeredis
        from feature_store.redis_store import RedisFeatureStore
        s = RedisFeatureStore.__new__(RedisFeatureStore)
        s.client = fakeredis.FakeRedis(decode_responses=True)
        s._available = True
        return s

    def test_merchant_risk_default(self, store):
        score = store.get_merchant_risk("unknown")
        assert score == 0.7

    def test_merchant_risk_set_get(self, store):
        store.set_merchant_risk("crypto", 0.9)
        assert store.get_merchant_risk("crypto") == 0.9

    def test_merchant_risk_bulk_seed(self, store):
        risks = {"retail": 0.1, "atm_cash": 0.7, "unknown": 0.8}
        count = store.seed_merchant_risks(risks)
        assert count == 3
        assert store.get_merchant_risk("retail") == 0.1

    def test_user_profile_empty(self, store):
        profile = store.get_user_profile("new_user")
        assert profile == {}

    def test_user_profile_update_and_get(self, store):
        store.update_user_profile("user_1", 100.0)
        store.update_user_profile("user_1", 200.0)
        profile = store.get_user_profile("user_1")
        assert profile["tx_count"] == 2
        assert profile["total_amount"] == 300.0
        assert profile["avg_amount"] == 150.0
        assert profile["max_amount"] == 200.0

    def test_behaviour_score_normal_transaction(self, store):
        # Build history with avg ~100
        for _ in range(5):
            store.update_user_profile("user_2", 100.0)
        score = store.compute_behaviour_score("user_2", 110.0)
        assert score < 0.3  # normal transaction

    def test_behaviour_score_anomalous_transaction(self, store):
        for _ in range(5):
            store.update_user_profile("user_3", 100.0)
        score = store.compute_behaviour_score("user_3", 5000.0)
        assert score >= 0.8  # 50x normal = very anomalous

    def test_device_registration(self, store):
        assert not store.is_known_device("user_4", "dev_abc")
        store.register_device("user_4", "dev_abc")
        assert store.is_known_device("user_4", "dev_abc")

    def test_unknown_device_always_false(self, store):
        assert not store.is_known_device("user_5", "unknown")

    def test_device_count(self, store):
        store.register_device("user_6", "dev_001")
        store.register_device("user_6", "dev_002")
        assert store.get_device_count("user_6") == 2

    def test_geo_consistency_no_history(self, store):
        score = store.get_geo_consistency("new_user", "Dublin, IE")
        assert score == 0.7  # neutral for new user

    def test_geo_consistency_known_location(self, store):
        store.update_location("user_7", "Dublin, IE")
        score = store.get_geo_consistency("user_7", "Dublin, IE")
        assert score == 1.0  # most recent location

    def test_geo_consistency_unknown_location(self, store):
        store.update_location("user_8", "Dublin, IE")
        score = store.get_geo_consistency("user_8", "Lagos, NG")
        assert score == 0.2  # never seen

    def test_geo_unknown_location_string(self, store):
        score = store.get_geo_consistency("user_9", "unknown")
        assert score == 0.3

    def test_velocity_empty(self, store):
        count = store.get_velocity("new_user", 3600)
        assert count == 0

    def test_velocity_increments(self, store):
        store.record_transaction("user_10", "tx_1")
        store.record_transaction("user_10", "tx_2")
        store.record_transaction("user_10", "tx_3")
        count = store.get_velocity("user_10", 3600)
        assert count == 3

    def test_flush_user_data(self, store):
        store.update_user_profile("user_11", 100.0)
        store.register_device("user_11", "dev_xyz")
        store.flush_user_data("user_11")
        assert store.get_user_profile("user_11") == {}
        assert not store.is_known_device("user_11", "dev_xyz")


# ── Real Context Enricher tests ────────────────────────────────────────────────

class TestRealContextEnricher:

    @pytest.fixture
    def enricher_with_store(self):
        """Context enricher with a fakeredis-backed store."""
        import fakeredis
        from agents.context_enricher import ContextEnricherAgent
        from feature_store.redis_store import RedisFeatureStore

        agent = ContextEnricherAgent.__new__(ContextEnricherAgent)
        agent.name    = "context_enricher"
        agent.sla_ms  = 500.0
        agent.logger  = __import__("logging").getLogger("test_enricher")
        from agents.base_agent import AgentMetrics
        agent.metrics = AgentMetrics()

        store = RedisFeatureStore.__new__(RedisFeatureStore)
        store.client     = fakeredis.FakeRedis(decode_responses=True)
        store._available = True
        agent.store      = store

        # Seed some data
        store.seed_merchant_risks({"atm_cash": 0.65, "retail": 0.15, "unknown": 0.75})
        store.register_device("user_42", "dev_known")
        store.update_location("user_42", "Dublin, IE")
        for _ in range(5):
            store.update_user_profile("user_42", 100.0)

        return agent

    @pytest.fixture
    def sample_tx(self):
        return Transaction(
            transaction_id    = "tx_test_real",
            user_id           = "user_42",
            amount            = 150.0,
            merchant          = "SuperMart",
            merchant_category = "retail",
            location          = "Dublin, IE",
            device_id         = "dev_known",
            ip_address        = "1.2.3.4",
        )

    @pytest.fixture
    def sample_anomaly(self, sample_tx):
        return AnomalyResult(
            transaction_id = sample_tx.transaction_id,
            fraud_score    = 0.3,
            is_flagged     = True,
            anomaly_vector = {},
        )

    def test_enrichment_returns_correct_type(self, enricher_with_store, sample_tx, sample_anomaly):
        result = enricher_with_store._enrich_real(sample_tx, sample_anomaly)
        assert isinstance(result, EnrichedTransaction)

    def test_known_device_detected(self, enricher_with_store, sample_tx, sample_anomaly):
        result = enricher_with_store._enrich_real(sample_tx, sample_anomaly)
        assert result.device_match is True

    def test_unknown_device_flagged(self, enricher_with_store, sample_anomaly):
        tx = Transaction(
            user_id="user_42", amount=500.0, merchant="Shop",
            device_id="dev_NEW_UNKNOWN", merchant_category="retail",
        )
        result = enricher_with_store._enrich_real(tx, sample_anomaly)
        assert result.device_match is False

    def test_known_location_high_score(self, enricher_with_store, sample_tx, sample_anomaly):
        result = enricher_with_store._enrich_real(sample_tx, sample_anomaly)
        assert result.geo_consistency_score >= 0.8

    def test_merchant_risk_populated(self, enricher_with_store, sample_tx, sample_anomaly):
        result = enricher_with_store._enrich_real(sample_tx, sample_anomaly)
        assert result.merchant_risk_score == 0.15  # retail

    def test_high_amount_behaviour_score(self, enricher_with_store, sample_anomaly):
        tx = Transaction(
            user_id="user_42", amount=50000.0, merchant="CryptoExchange",
            merchant_category="unknown", device_id="dev_known",
        )
        result = enricher_with_store._enrich_real(tx, sample_anomaly)
        assert result.user_behaviour_score >= 0.8

    def test_velocity_tracked(self, enricher_with_store, sample_tx, sample_anomaly):
        # Process 3 transactions
        for i in range(3):
            tx = Transaction(
                transaction_id=f"tx_{i}", user_id="user_vel",
                amount=100.0, merchant="Shop", device_id="dev_x",
            )
            anomaly = AnomalyResult(
                transaction_id=f"tx_{i}", fraud_score=0.3,
                is_flagged=True, anomaly_vector={},
            )
            enricher_with_store._enrich_real(tx, anomaly)
        # 4th transaction should see velocity >= 3
        tx4 = Transaction(
            transaction_id="tx_4", user_id="user_vel",
            amount=100.0, merchant="Shop", device_id="dev_x",
        )
        result = enricher_with_store._enrich_real(tx4, sample_anomaly)
        assert result.velocity_hour >= 3

    def test_enriched_payload_has_source(self, enricher_with_store, sample_tx, sample_anomaly):
        result = enricher_with_store._enrich_real(sample_tx, sample_anomaly)
        assert result.enriched_payload.get("source") == "redis_feature_store"

    def test_fallback_when_redis_unavailable(self):
        from agents.context_enricher import ContextEnricherAgent
        from agents.base_agent import AgentMetrics
        import logging

        agent = ContextEnricherAgent.__new__(ContextEnricherAgent)
        agent.name    = "context_enricher"
        agent.sla_ms  = 500.0
        agent.logger  = logging.getLogger("test")
        agent.metrics = AgentMetrics()
        agent.store   = None  # simulate unavailable Redis

        tx = Transaction(user_id="u1", amount=100.0, merchant="Shop")
        anomaly = AnomalyResult(
            transaction_id=tx.transaction_id, fraud_score=0.4,
            is_flagged=True, anomaly_vector={},
        )
        result = agent._enrich_fallback(tx, anomaly)
        assert isinstance(result, EnrichedTransaction)
        assert result.enriched_payload.get("source") == "fallback_heuristic"
