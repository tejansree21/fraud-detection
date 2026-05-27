"""
tests/test_pipeline.py
End-to-end pipeline tests — verifies the full flow with mock/stub agents.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import patch
import os

# Ensure test env vars are set before importing settings
os.environ.setdefault("GROQ_API_KEY", "test_key_not_real")
os.environ.setdefault("POSTGRES_PASSWORD", "fraud_pass")

from pipeline.schemas import (
    Transaction, AnomalyResult, EnrichedTransaction,
    CaseReport, InvestigatorFeedback,
    RecommendedAction, InvestigatorDecision
)
from agents.anomaly_detector import AnomalyDetectorAgent
from agents.context_enricher import ContextEnricherAgent
from orchestrator.orchestrator import FraudOrchestrator


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_transaction():
    return Transaction(
        transaction_id    = "tx_test_001",
        user_id           = "user_42",
        amount            = 1500.00,        # high amount → likely flagged by stub
        merchant          = "CryptoExchange",
        merchant_category = "unknown",
        location          = "unknown",
        device_id         = "dev_abc",
        ip_address        = "192.168.1.1",
    )

@pytest.fixture
def legit_transaction():
    return Transaction(
        transaction_id    = "tx_test_002",
        user_id           = "user_43",
        amount            = 12.50,
        merchant          = "Starbucks",
        merchant_category = "food_and_drink",
        location          = "Dublin, IE",
        device_id         = "dev_known_123",
        ip_address        = "10.0.0.1",
    )

@pytest.fixture
def mock_case_reporter():
    """Returns a CaseReporterAgent with Groq calls mocked out."""
    from agents.case_reporter import CaseReporterAgent
    with patch.object(CaseReporterAgent, "_call_groq") as mock_groq:
        mock_groq.return_value = """{
            "report_text": "Test fraud report: transaction shows multiple high-risk signals.",
            "recommended_action": "BLOCK",
            "confidence_score": 0.87,
            "key_risk_factors": ["high amount", "unknown merchant", "unknown location"],
            "timeline_summary": "Single large transaction on unknown merchant."
        }"""
        reporter = CaseReporterAgent(groq_api_key="test_key")
        yield reporter

@pytest.fixture
def orchestrator(mock_case_reporter):
    return FraudOrchestrator(
        anomaly_detector  = AnomalyDetectorAgent(fraud_score_threshold=0.3),
        context_enricher  = ContextEnricherAgent(),
        case_reporter     = mock_case_reporter,
        fraud_threshold   = 0.3,
        retraining_threshold = 3,  # low for testing
    )


# ── Schema tests ───────────────────────────────────────────────────────────────

class TestSchemas:
    def test_transaction_defaults(self):
        tx = Transaction(user_id="u1", amount=100.0, merchant="Shop")
        assert tx.transaction_id is not None
        assert tx.merchant_category == "unknown"
        assert isinstance(tx.timestamp, datetime)

    def test_anomaly_result_flagged(self):
        result = AnomalyResult(
            transaction_id="tx1",
            fraud_score=0.75,
            is_flagged=True,
            anomaly_vector={"feature": 0.5},
        )
        assert result.is_flagged is True
        assert result.fraud_score == 0.75

    def test_recommended_action_enum(self):
        assert RecommendedAction.BLOCK == "BLOCK"
        assert RecommendedAction("REVIEW") == RecommendedAction.REVIEW


# ── Agent unit tests ───────────────────────────────────────────────────────────

class TestAnomalyDetector:
    def test_process_returns_anomaly_result(self, sample_transaction):
        agent = AnomalyDetectorAgent(fraud_score_threshold=0.5)
        result = agent.run(sample_transaction)
        assert isinstance(result, AnomalyResult)
        assert 0.0 <= result.fraud_score <= 1.0
        assert result.transaction_id == sample_transaction.transaction_id

    def test_health_check(self):
        agent = AnomalyDetectorAgent()
        assert agent.health_check() is True

    def test_metrics_tracked(self, sample_transaction):
        agent = AnomalyDetectorAgent()
        for _ in range(5):
            agent.run(sample_transaction)
        assert agent.metrics.total_processed == 5
        assert agent.metrics.avg_latency_ms >= 0

    def test_status_dict(self, sample_transaction):
        agent = AnomalyDetectorAgent()
        agent.run(sample_transaction)
        status = agent.status()
        assert status["agent"] == "anomaly_detector"
        assert "avg_latency_ms" in status


class TestContextEnricher:
    def test_process_returns_enriched(self, sample_transaction):
        detector = AnomalyDetectorAgent(fraud_score_threshold=0.0)  # always flag
        anomaly  = detector.run(sample_transaction)
        enricher = ContextEnricherAgent()
        result   = enricher.run((sample_transaction, anomaly))
        assert isinstance(result, EnrichedTransaction)
        assert result.transaction_id == sample_transaction.transaction_id
        assert 0.0 <= result.merchant_risk_score <= 1.0

    def test_fraud_score_carried_forward(self, sample_transaction):
        detector = AnomalyDetectorAgent(fraud_score_threshold=0.0)
        anomaly  = detector.run(sample_transaction)
        enricher = ContextEnricherAgent()
        result   = enricher.run((sample_transaction, anomaly))
        assert result.fraud_score == anomaly.fraud_score


# ── Orchestrator integration tests ────────────────────────────────────────────

class TestOrchestrator:
    def test_high_risk_transaction_gets_report(self, orchestrator, sample_transaction):
        """A high-risk transaction should flow through all 3 stages."""
        # Force anomaly detector to always flag by using threshold=0
        orchestrator.anomaly_detector.threshold = 0.0
        result = orchestrator.process_transaction(sample_transaction)
        assert result.was_flagged is True
        assert result.enriched is not None
        assert result.report is not None
        assert result.report.recommended_action in list(RecommendedAction)

    def test_low_risk_transaction_stops_at_stage1(self, orchestrator):
        """A low-risk transaction should stop after anomaly detection (no enrichment)."""
        orchestrator.anomaly_detector.threshold = 1.1  # impossible to flag
        tx = Transaction(user_id="u99", amount=5.0, merchant="Coffee")
        result = orchestrator.process_transaction(tx)
        assert result.was_flagged is False
        assert result.enriched is None
        assert result.report is None

    def test_pipeline_latency_tracked(self, orchestrator, sample_transaction):
        orchestrator.anomaly_detector.threshold = 0.0
        result = orchestrator.process_transaction(sample_transaction)
        assert result.total_latency_ms > 0

    def test_stats_increment(self, orchestrator, sample_transaction):
        orchestrator.anomaly_detector.threshold = 0.0
        for _ in range(3):
            orchestrator.process_transaction(sample_transaction)
        stats = orchestrator.stats()
        assert stats["total_processed"] == 3
        assert stats["total_flagged"] == 3

    def test_feedback_triggers_retraining(self, orchestrator, sample_transaction, caplog):
        """After retraining_threshold feedbacks, retraining should be triggered."""
        import logging
        with caplog.at_level(logging.INFO, logger="orchestrator"):
            for i in range(3):  # threshold=3 in fixture
                fb = InvestigatorFeedback(
                    transaction_id  = f"tx_{i}",
                    investigator_id = "analyst_1",
                    decision        = InvestigatorDecision.CONFIRMED_FRAUD,
                )
                orchestrator.record_feedback(fb)
        assert "RETRAINING TRIGGERED" in caplog.text

    def test_error_in_agent_doesnt_crash_orchestrator(self, orchestrator):
        """If an agent throws, the orchestrator should catch it and set result.error."""
        from unittest.mock import patch
        with patch.object(orchestrator.anomaly_detector, 'process', side_effect=RuntimeError("model error")):
            tx = Transaction(user_id="u1", amount=100.0, merchant="Shop")
            result = orchestrator.process_transaction(tx)
        assert result.error is not None
        assert "model error" in result.error
