"""
pipeline/db_writer.py
PostgreSQL persistence layer for the fraud detection pipeline.

Writes every pipeline stage result to the database:
  - transactions          → raw transaction data
  - flagged_transactions  → anomaly detector output
  - enriched_transactions → context enricher output
  - case_reports          → LLM-generated reports
  - investigator_feedback → human decisions (feedback loop)

Uses SQLAlchemy Core for fast batch inserts with connection pooling.
Falls back gracefully if DB is unavailable.
"""

from __future__ import annotations
import logging
import json
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from pipeline.schemas import (
    Transaction, AnomalyResult, EnrichedTransaction,
    CaseReport, InvestigatorFeedback
)

logger = logging.getLogger("db_writer")


class DatabaseWriter:
    """
    Writes fraud detection pipeline results to PostgreSQL.
    All writes are fire-and-forget — errors are logged but never
    propagate to the pipeline (DB failure must not stop processing).
    """

    def __init__(self, dsn: str):
        try:
            self.engine = create_engine(
                dsn,
                poolclass        = QueuePool,
                pool_size        = 5,
                max_overflow     = 10,
                pool_pre_ping    = True,
                pool_recycle     = 3600,
            )
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._available = True
            logger.info("DatabaseWriter connected to PostgreSQL")
        except Exception as e:
            logger.warning("PostgreSQL unavailable (%s) — writes disabled", e)
            self._available = False
            self.engine = None

    @property
    def available(self) -> bool:
        return self._available

    # ── Write methods ──────────────────────────────────────────────────────────

    def write_transaction(self, tx: Transaction) -> bool:
        """Persist a raw transaction."""
        if not self._available:
            return False
        try:
            sql = text("""
                INSERT INTO transactions
                    (transaction_id, user_id, amount, merchant, merchant_category,
                     timestamp, location, device_id, ip_address, raw_payload)
                VALUES
                    (:transaction_id, :user_id, :amount, :merchant, :merchant_category,
                     :timestamp, :location, :device_id, :ip_address, :raw_payload)
                ON CONFLICT (transaction_id) DO NOTHING
            """)
            with self.engine.begin() as conn:
                conn.execute(sql, {
                    "transaction_id":    tx.transaction_id,
                    "user_id":           tx.user_id,
                    "amount":            float(tx.amount),
                    "merchant":          tx.merchant,
                    "merchant_category": tx.merchant_category,
                    "timestamp":         tx.timestamp,
                    "location":          tx.location,
                    "device_id":         tx.device_id,
                    "ip_address":        tx.ip_address,
                    "raw_payload":       json.dumps(tx.model_dump(), default=str),
                })
            return True
        except Exception as e:
            logger.error("Failed to write transaction %s: %s", tx.transaction_id, e)
            return False

    def write_anomaly_result(self, tx_id: str, result: AnomalyResult) -> bool:
        """Persist anomaly detector output."""
        if not self._available:
            return False
        try:
            sql = text("""
                INSERT INTO flagged_transactions
                    (transaction_id, fraud_score, anomaly_vector, detector_version, flagged_at)
                VALUES
                    (:transaction_id, :fraud_score, :anomaly_vector, :detector_version, :flagged_at)
                ON CONFLICT DO NOTHING
            """)
            with self.engine.begin() as conn:
                conn.execute(sql, {
                    "transaction_id":   tx_id,
                    "fraud_score":      float(result.fraud_score),
                    "anomaly_vector":   json.dumps(result.anomaly_vector),
                    "detector_version": result.detector_version,
                    "flagged_at":       datetime.utcnow(),
                })
            return True
        except Exception as e:
            logger.error("Failed to write anomaly result %s: %s", tx_id, e)
            return False

    def write_enriched_transaction(self, tx_id: str, enriched: EnrichedTransaction) -> bool:
        """Persist context enricher output."""
        if not self._available:
            return False
        try:
            sql = text("""
                INSERT INTO enriched_transactions
                    (transaction_id, merchant_risk_score, geo_consistency_score,
                     device_match, velocity_hour, velocity_day,
                     user_behaviour_score, enriched_payload, enriched_at)
                VALUES
                    (:transaction_id, :merchant_risk_score, :geo_consistency_score,
                     :device_match, :velocity_hour, :velocity_day,
                     :user_behaviour_score, :enriched_payload, :enriched_at)
                ON CONFLICT DO NOTHING
            """)
            with self.engine.begin() as conn:
                conn.execute(sql, {
                    "transaction_id":       tx_id,
                    "merchant_risk_score":  float(enriched.merchant_risk_score),
                    "geo_consistency_score": float(enriched.geo_consistency_score),
                    "device_match":         bool(enriched.device_match),
                    "velocity_hour":        int(enriched.velocity_hour),
                    "velocity_day":         int(enriched.velocity_day),
                    "user_behaviour_score": float(enriched.user_behaviour_score),
                    "enriched_payload":     json.dumps(enriched.enriched_payload),
                    "enriched_at":          datetime.utcnow(),
                })
            return True
        except Exception as e:
            logger.error("Failed to write enriched transaction %s: %s", tx_id, e)
            return False

    def write_case_report(self, tx_id: str, report: CaseReport) -> bool:
        """Persist LLM-generated case report."""
        if not self._available:
            return False
        try:
            sql = text("""
                INSERT INTO case_reports
                    (transaction_id, report_text, recommended_action,
                     confidence_score, similar_cases, generated_at)
                VALUES
                    (:transaction_id, :report_text, :recommended_action,
                     :confidence_score, :similar_cases, :generated_at)
                ON CONFLICT DO NOTHING
            """)
            with self.engine.begin() as conn:
                conn.execute(sql, {
                    "transaction_id":    tx_id,
                    "report_text":       report.report_text,
                    "recommended_action": report.recommended_action.value,
                    "confidence_score":  float(report.confidence_score),
                    "similar_cases":     json.dumps(report.similar_cases),
                    "generated_at":      datetime.utcnow(),
                })
            return True
        except Exception as e:
            logger.error("Failed to write case report %s: %s", tx_id, e)
            return False

    def write_feedback(self, feedback: InvestigatorFeedback) -> bool:
        """Persist investigator decision."""
        if not self._available:
            return False
        try:
            sql = text("""
                INSERT INTO investigator_feedback
                    (transaction_id, investigator_id, decision, notes, decided_at)
                VALUES
                    (:transaction_id, :investigator_id, :decision, :notes, :decided_at)
            """)
            with self.engine.begin() as conn:
                conn.execute(sql, {
                    "transaction_id":  feedback.transaction_id,
                    "investigator_id": feedback.investigator_id,
                    "decision":        feedback.decision.value,
                    "notes":           feedback.notes,
                    "decided_at":      datetime.utcnow(),
                })
            return True
        except Exception as e:
            logger.error("Failed to write feedback %s: %s", feedback.transaction_id, e)
            return False

    # ── Query methods (for dashboard) ─────────────────────────────────────────

    def get_recent_flagged(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent flagged transactions with reports for dashboard."""
        if not self._available:
            return []
        try:
            sql = text("""
                SELECT
                    t.transaction_id,
                    t.user_id,
                    t.amount,
                    t.merchant,
                    t.merchant_category,
                    t.timestamp,
                    t.location,
                    f.fraud_score,
                    f.flagged_at,
                    e.merchant_risk_score,
                    e.geo_consistency_score,
                    e.device_match,
                    e.velocity_hour,
                    e.user_behaviour_score,
                    r.recommended_action,
                    r.confidence_score,
                    r.report_text
                FROM flagged_transactions f
                JOIN transactions t ON t.transaction_id = f.transaction_id
                LEFT JOIN enriched_transactions e ON e.transaction_id = f.transaction_id
                LEFT JOIN case_reports r ON r.transaction_id = f.transaction_id
                ORDER BY f.flagged_at DESC
                LIMIT :limit
            """)
            with self.engine.connect() as conn:
                rows = conn.execute(sql, {"limit": limit}).mappings().all()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to query recent flagged: %s", e)
            return []

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get aggregate pipeline statistics for dashboard."""
        if not self._available:
            return {}
        try:
            sql = text("""
                SELECT
                    COUNT(DISTINCT t.transaction_id)                    AS total_transactions,
                    COUNT(DISTINCT f.transaction_id)                    AS total_flagged,
                    COUNT(DISTINCT r.transaction_id)                    AS total_reports,
                    AVG(f.fraud_score)                                  AS avg_fraud_score,
                    COUNT(CASE WHEN r.recommended_action = 'BLOCK'  THEN 1 END) AS blocked,
                    COUNT(CASE WHEN r.recommended_action = 'REVIEW' THEN 1 END) AS reviewed,
                    COUNT(CASE WHEN r.recommended_action = 'ALLOW'  THEN 1 END) AS allowed,
                    COUNT(DISTINCT fb.transaction_id)                   AS feedback_count
                FROM transactions t
                LEFT JOIN flagged_transactions f  ON f.transaction_id = t.transaction_id
                LEFT JOIN case_reports r          ON r.transaction_id = t.transaction_id
                LEFT JOIN investigator_feedback fb ON fb.transaction_id = t.transaction_id
            """)
            with self.engine.connect() as conn:
                row = conn.execute(sql).mappings().first()
                return dict(row) if row else {}
        except Exception as e:
            logger.error("Failed to query pipeline stats: %s", e)
            return {}

    def get_fraud_score_distribution(self) -> list[dict]:
        """Get fraud score distribution for histogram."""
        if not self._available:
            return []
        try:
            sql = text("""
                SELECT
                    ROUND(fraud_score::numeric, 1) AS score_bucket,
                    COUNT(*) AS count
                FROM flagged_transactions
                GROUP BY score_bucket
                ORDER BY score_bucket
            """)
            with self.engine.connect() as conn:
                rows = conn.execute(sql).mappings().all()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to query score distribution: %s", e)
            return []

    def get_hourly_volume(self, hours: int = 24) -> list[dict]:
        """Get transaction volume by hour for time series chart."""
        if not self._available:
            return []
        try:
            sql = text("""
                SELECT
                    DATE_TRUNC('hour', flagged_at) AS hour,
                    COUNT(*) AS flagged_count,
                    AVG(fraud_score) AS avg_score
                FROM flagged_transactions
                WHERE flagged_at >= NOW() - INTERVAL ':hours hours'
                GROUP BY hour
                ORDER BY hour
            """.replace(":hours", str(hours)))
            with self.engine.connect() as conn:
                rows = conn.execute(sql).mappings().all()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to query hourly volume: %s", e)
            return []
