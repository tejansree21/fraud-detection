"""
orchestrator/orchestrator.py
Phase 6: Full feedback loop with ADWIN drift detection and auto-retraining.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Any

from pipeline.schemas import (
    Transaction, AnomalyResult, EnrichedTransaction,
    CaseReport, InvestigatorFeedback, RecommendedAction
)
from agents.anomaly_detector import AnomalyDetectorAgent
from agents.context_enricher import ContextEnricherAgent
from agents.case_reporter    import CaseReporterAgent
from models.drift_detector   import ADWINDriftDetector

logger = logging.getLogger("orchestrator")


@dataclass
class PipelineResult:
    transaction:      Transaction
    anomaly:          AnomalyResult
    enriched:         EnrichedTransaction | None = None
    report:           CaseReport | None          = None
    total_latency_ms: float                      = 0.0
    error:            str | None                 = None

    @property
    def was_flagged(self) -> bool:
        return self.anomaly.is_flagged

    @property
    def final_action(self) -> RecommendedAction | None:
        return self.report.recommended_action if self.report else None


class FraudOrchestrator:

    def __init__(
        self,
        anomaly_detector:     AnomalyDetectorAgent,
        context_enricher:     ContextEnricherAgent,
        case_reporter:        CaseReporterAgent,
        fraud_threshold:      float = 0.5,
        retraining_threshold: int   = 100,
        db_writer=None,
    ):
        self.anomaly_detector     = anomaly_detector
        self.context_enricher     = context_enricher
        self.case_reporter        = case_reporter
        self.fraud_threshold      = fraud_threshold
        self.retraining_threshold = retraining_threshold
        self.db                   = db_writer

        # Drift detectors
        self.score_drift_detector    = ADWINDriftDetector(delta=0.002, min_window=50)
        self.flag_rate_drift_detector = ADWINDriftDetector(delta=0.002, min_window=100)

        # Retrainer (lazy init to avoid circular imports)
        self._retrainer = None

        # Counters
        self._processed_count        = 0
        self._flagged_count          = 0
        self._feedback_since_retrain = 0
        self._feedback_log: list[InvestigatorFeedback] = []

        logger.info(
            "FraudOrchestrator ready | threshold=%.2f | retrain_after=%d | db=%s",
            fraud_threshold, retraining_threshold,
            "connected" if db_writer and db_writer.available else "disabled",
        )

    @classmethod
    def from_env(cls) -> "FraudOrchestrator":
        from config.settings import settings
        from pipeline.db_writer import DatabaseWriter

        db = DatabaseWriter(settings.postgres_dsn)

        return cls(
            anomaly_detector  = AnomalyDetectorAgent(
                fraud_score_threshold=settings.fraud_score_threshold
            ),
            context_enricher  = ContextEnricherAgent(),
            case_reporter     = CaseReporterAgent(
                groq_api_key=settings.groq_api_key,
                model=settings.groq_model,
            ),
            fraud_threshold         = settings.fraud_score_threshold,
            retraining_threshold    = settings.retraining_feedback_threshold,
            db_writer               = db,
        )

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def process_transaction(self, tx: Transaction) -> PipelineResult:
        pipeline_start = time.perf_counter()
        result = PipelineResult(transaction=tx, anomaly=None)  # type: ignore

        try:
            # ── Stage 1: Anomaly Detection ─────────────────────────────────────
            anomaly: AnomalyResult = self.anomaly_detector.run(tx)
            result.anomaly = anomaly
            self._processed_count += 1

            # Feed score to drift detector
            drift_fired = self.score_drift_detector.update(anomaly.fraud_score)
            flag_val    = 1.0 if anomaly.is_flagged else 0.0
            self.flag_rate_drift_detector.update(flag_val)

            if drift_fired:
                logger.warning(
                    "SCORE DRIFT detected after %d observations — consider retraining",
                    self.score_drift_detector.n_total
                )
                self._trigger_retraining(reason="drift_detected")

            # Persist
            if self.db:
                self.db.write_transaction(tx)
                if anomaly.is_flagged:
                    self.db.write_anomaly_result(tx.transaction_id, anomaly)

            if not anomaly.is_flagged:
                result.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
                return result

            # ── Stage 2: Context Enrichment ────────────────────────────────────
            self._flagged_count += 1
            logger.info("Stage 2: Enriching | tx=%s score=%.4f", tx.transaction_id, anomaly.fraud_score)
            enriched: EnrichedTransaction = self.context_enricher.run((tx, anomaly))
            result.enriched = enriched

            if self.db:
                self.db.write_enriched_transaction(tx.transaction_id, enriched)

            # ── Stage 3: Case Report (high-risk only) ──────────────────────────
            if anomaly.fraud_score >= 0.7:
                try:
                    logger.info("Stage 3: Generating report | tx=%s", tx.transaction_id)
                    report: CaseReport = self.case_reporter.run((tx, enriched))
                    result.report = report

                    if self.db:
                        self.db.write_case_report(tx.transaction_id, report)

                    logger.info(
                        "Pipeline complete | tx=%s action=%s confidence=%.2f",
                        tx.transaction_id,
                        report.recommended_action.value,
                        report.confidence_score,
                    )
                except Exception as llm_exc:
                    if "429" in str(llm_exc) or "rate_limit" in str(llm_exc):
                        logger.warning("Groq rate limited — skipping report | tx=%s", tx.transaction_id)
                    else:
                        raise
            else:
                logger.info(
                    "Pipeline complete (no report) | tx=%s score=%.4f",
                    tx.transaction_id, anomaly.fraud_score,
                )

        except Exception as exc:
            result.error = str(exc)
            logger.error("Pipeline error | tx=%s | %s", tx.transaction_id, exc, exc_info=True)

        finally:
            result.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000

        return result

    # ── Feedback loop ──────────────────────────────────────────────────────────

    def record_feedback(self, feedback: InvestigatorFeedback) -> None:
        self._feedback_log.append(feedback)
        self._feedback_since_retrain += 1

        if self.db:
            self.db.write_feedback(feedback)

        logger.info(
            "Feedback recorded | tx=%s decision=%s | total_since_retrain=%d",
            feedback.transaction_id,
            feedback.decision.value,
            self._feedback_since_retrain,
        )

        if self._feedback_since_retrain >= self.retraining_threshold:
            self._trigger_retraining(reason="feedback_threshold")

    def _trigger_retraining(self, reason: str = "manual") -> None:
        logger.info("RETRAINING TRIGGERED | reason=%s | feedback=%d",
                    reason, self._feedback_since_retrain)
        self._feedback_since_retrain = 0

        # Run retraining in background (non-blocking)
        try:
            if self._retrainer is None:
                from models.retrainer import FeedbackRetrainer
                self._retrainer = FeedbackRetrainer(
                    db_writer        = self.db,
                    anomaly_detector = self.anomaly_detector,
                )
            import threading
            t = threading.Thread(
                target=self._retrainer.run,
                kwargs={"reason": reason},
                daemon=True,
            )
            t.start()
            logger.info("Retraining started in background thread")
        except Exception as e:
            logger.error("Failed to start retraining: %s", e)

    # ── Health & status ────────────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        return {
            "orchestrator": "ok",
            "db":           "connected" if self.db and self.db.available else "disabled",
            "drift": {
                "score_drift":    self.score_drift_detector.status(),
                "flag_rate_drift": self.flag_rate_drift_detector.status(),
            },
            "agents": {
                "anomaly_detector": self.anomaly_detector.status(),
                "context_enricher": self.context_enricher.status(),
                "case_reporter":    self.case_reporter.status(),
            },
        }

    def stats(self) -> dict[str, Any]:
        flag_rate = (
            self._flagged_count / self._processed_count
            if self._processed_count > 0 else 0.0
        )
        base = {
            "total_processed":        self._processed_count,
            "total_flagged":          self._flagged_count,
            "flag_rate":              round(flag_rate, 4),
            "feedback_since_retrain": self._feedback_since_retrain,
            "retraining_threshold":   self.retraining_threshold,
            "drift": {
                "score":     self.score_drift_detector.status(),
                "flag_rate": self.flag_rate_drift_detector.status(),
            },
        }
        if self.db and self.db.available:
            base["db"] = self.db.get_pipeline_stats()
        if self._retrainer:
            base["retrainer"] = self._retrainer.status()
        return base
