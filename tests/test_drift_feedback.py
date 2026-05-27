"""
tests/test_drift_feedback.py
Tests for ADWIN drift detector and feedback loop retraining trigger.
"""

import pytest
import os
os.environ.setdefault("GROQ_API_KEY", "test_key")


class TestADWINDriftDetector:

    @pytest.fixture
    def detector(self):
        from models.drift_detector import ADWINDriftDetector
        return ADWINDriftDetector(delta=0.002, min_window=10)

    def test_starts_clean(self, detector):
        assert detector.window_size == 0
        assert detector.drift_detected is False
        assert detector.n_total == 0

    def test_update_increments_count(self, detector):
        detector.update(0.5)
        detector.update(0.6)
        assert detector.n_total == 2

    def test_no_drift_on_stable_stream(self, detector):
        """Constant stream should not trigger drift."""
        fired = False
        for _ in range(100):
            if detector.update(0.3):
                fired = True
        assert not fired

    def test_drift_detected_on_sudden_shift(self, detector):
        """Sudden mean shift should trigger ADWIN."""
        # Build stable window
        for _ in range(60):
            detector.update(0.1)
        # Inject sudden high values
        fired = False
        for _ in range(60):
            if detector.update(0.9):
                fired = True
        assert fired

    def test_drift_events_recorded(self, detector):
        for _ in range(60):
            detector.update(0.1)
        for _ in range(60):
            detector.update(0.9)
        assert len(detector.drift_events) >= 1

    def test_drift_event_has_correct_fields(self, detector):
        for _ in range(60):
            detector.update(0.1)
        for _ in range(60):
            detector.update(0.9)
        if detector.drift_events:
            event = detector.drift_events[0]
            assert hasattr(event, "mean_before")
            assert hasattr(event, "mean_after")
            assert hasattr(event, "drift_magnitude")
            assert event.drift_magnitude > 0

    def test_mean_calculation(self, detector):
        for v in [0.2, 0.4, 0.6]:
            detector.update(v)
        assert abs(detector.mean - 0.4) < 0.01

    def test_reset_clears_state(self, detector):
        for _ in range(20):
            detector.update(0.5)
        detector.reset()
        assert detector.window_size == 0
        assert detector.drift_detected is False

    def test_status_dict(self, detector):
        detector.update(0.5)
        status = detector.status()
        assert "window_size"    in status
        assert "current_mean"   in status
        assert "drift_detected" in status
        assert "drift_events"   in status

    def test_below_min_window_no_drift(self, detector):
        """Should never fire before min_window observations."""
        for i in range(5):  # less than min_window=10
            fired = detector.update(0.9 if i % 2 == 0 else 0.1)
            assert not fired


class TestFeedbackLoop:

    @pytest.fixture
    def orchestrator(self):
        from unittest.mock import patch, MagicMock
        with patch("pipeline.consumer.FraudOrchestrator.from_env"):
            from orchestrator.orchestrator import FraudOrchestrator
            from agents.anomaly_detector import AnomalyDetectorAgent
            from agents.context_enricher import ContextEnricherAgent
            from unittest.mock import MagicMock

            mock_reporter = MagicMock()
            mock_reporter.run.return_value = MagicMock(
                recommended_action=MagicMock(value="BLOCK"),
                confidence_score=0.9,
                similar_cases=[],
            )
            mock_reporter.status.return_value = {}

            return FraudOrchestrator(
                anomaly_detector     = AnomalyDetectorAgent(fraud_score_threshold=0.0),
                context_enricher     = ContextEnricherAgent(),
                case_reporter        = mock_reporter,
                retraining_threshold = 5,
            )

    def test_feedback_increments_counter(self, orchestrator):
        from pipeline.schemas import InvestigatorFeedback, InvestigatorDecision
        fb = InvestigatorFeedback(
            transaction_id  = "tx_1",
            investigator_id = "analyst_1",
            decision        = InvestigatorDecision.CONFIRMED_FRAUD,
        )
        orchestrator.record_feedback(fb)
        assert orchestrator._feedback_since_retrain == 1

    def test_feedback_threshold_triggers_retrain(self, orchestrator, caplog):
        from pipeline.schemas import InvestigatorFeedback, InvestigatorDecision
        import logging
        with caplog.at_level(logging.INFO, logger="orchestrator"):
            for i in range(5):
                fb = InvestigatorFeedback(
                    transaction_id  = f"tx_{i}",
                    investigator_id = "analyst_1",
                    decision        = InvestigatorDecision.CONFIRMED_FRAUD,
                )
                orchestrator.record_feedback(fb)
        assert "RETRAINING TRIGGERED" in caplog.text

    def test_feedback_counter_resets_after_retrain(self, orchestrator):
        from pipeline.schemas import InvestigatorFeedback, InvestigatorDecision
        for i in range(5):
            orchestrator.record_feedback(InvestigatorFeedback(
                transaction_id  = f"tx_{i}",
                investigator_id = "analyst_1",
                decision        = InvestigatorDecision.CONFIRMED_FRAUD,
            ))
        assert orchestrator._feedback_since_retrain == 0

    def test_drift_detector_in_health_check(self, orchestrator):
        health = orchestrator.health_check()
        assert "drift" in health
        assert "score_drift" in health["drift"]
        assert "flag_rate_drift" in health["drift"]

    def test_drift_detector_in_stats(self, orchestrator):
        stats = orchestrator.stats()
        assert "drift" in stats
        assert "score" in stats["drift"]
