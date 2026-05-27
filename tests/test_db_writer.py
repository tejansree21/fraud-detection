"""
tests/test_db_writer.py
Tests for PostgreSQL database writer.
Uses SQLite in-memory as a fast test backend.
"""

import pytest
import os
os.environ.setdefault("GROQ_API_KEY", "test_key")

from datetime import datetime
from pipeline.schemas import (
    Transaction, AnomalyResult, EnrichedTransaction,
    CaseReport, InvestigatorFeedback,
    RecommendedAction, InvestigatorDecision
)


@pytest.fixture
def db(tmp_path):
    """
    DatabaseWriter backed by SQLite for fast unit tests.
    Creates all tables from the schema.
    """
    from pipeline.db_writer import DatabaseWriter
    from sqlalchemy import create_engine, text

    sqlite_path = tmp_path / "test.db"
    dsn = f"sqlite:///{sqlite_path}"

    # Create tables (SQLite-compatible subset of our schema)
    engine = create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                user_id TEXT, amount REAL, merchant TEXT,
                merchant_category TEXT, timestamp TEXT,
                location TEXT, device_id TEXT, ip_address TEXT,
                raw_payload TEXT, created_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE flagged_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT, fraud_score REAL,
                anomaly_vector TEXT, detector_version TEXT, flagged_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE enriched_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT, merchant_risk_score REAL,
                geo_consistency_score REAL, device_match INTEGER,
                velocity_hour INTEGER, velocity_day INTEGER,
                user_behaviour_score REAL, enriched_payload TEXT, enriched_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE case_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT, report_text TEXT,
                recommended_action TEXT, confidence_score REAL,
                similar_cases TEXT, generated_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE investigator_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT, investigator_id TEXT,
                decision TEXT, notes TEXT, decided_at TEXT
            )
        """))

    writer = DatabaseWriter.__new__(DatabaseWriter)
    writer.engine      = engine
    writer._available  = True
    return writer


@pytest.fixture
def sample_tx():
    return Transaction(
        transaction_id    = "tx_db_test_001",
        user_id           = "user_42",
        amount            = 500.0,
        merchant          = "TestShop",
        merchant_category = "retail",
        location          = "Dublin, IE",
        device_id         = "dev_001",
        ip_address        = "1.2.3.4",
    )

@pytest.fixture
def sample_anomaly(sample_tx):
    return AnomalyResult(
        transaction_id   = sample_tx.transaction_id,
        fraud_score      = 0.75,
        is_flagged       = True,
        anomaly_vector   = {"score": 0.75},
        detector_version = "v1.0",
    )

@pytest.fixture
def sample_enriched(sample_tx):
    return EnrichedTransaction(
        transaction_id        = sample_tx.transaction_id,
        fraud_score           = 0.75,
        merchant_risk_score   = 0.3,
        geo_consistency_score = 0.9,
        device_match          = True,
        velocity_hour         = 2,
        velocity_day          = 8,
        user_behaviour_score  = 0.2,
        enriched_payload      = {"source": "redis"},
    )

@pytest.fixture
def sample_report(sample_tx):
    return CaseReport(
        transaction_id     = sample_tx.transaction_id,
        report_text        = "Test fraud investigation report.",
        recommended_action = RecommendedAction.REVIEW,
        confidence_score   = 0.85,
        similar_cases      = [],
    )


class TestDatabaseWriter:

    def test_write_transaction(self, db, sample_tx):
        result = db.write_transaction(sample_tx)
        assert result is True

    def test_write_transaction_idempotent(self, db, sample_tx):
        # Writing twice should not raise (ON CONFLICT DO NOTHING)
        db.write_transaction(sample_tx)
        result = db.write_transaction(sample_tx)
        assert result is True

    def test_write_anomaly_result(self, db, sample_tx, sample_anomaly):
        db.write_transaction(sample_tx)
        result = db.write_anomaly_result(sample_tx.transaction_id, sample_anomaly)
        assert result is True

    def test_write_enriched_transaction(self, db, sample_tx, sample_enriched):
        db.write_transaction(sample_tx)
        result = db.write_enriched_transaction(sample_tx.transaction_id, sample_enriched)
        assert result is True

    def test_write_case_report(self, db, sample_tx, sample_report):
        db.write_transaction(sample_tx)
        result = db.write_case_report(sample_tx.transaction_id, sample_report)
        assert result is True

    def test_write_feedback(self, db, sample_tx):
        db.write_transaction(sample_tx)
        fb = InvestigatorFeedback(
            transaction_id  = sample_tx.transaction_id,
            investigator_id = "analyst_1",
            decision        = InvestigatorDecision.CONFIRMED_FRAUD,
            notes           = "Clear fraud pattern",
        )
        result = db.write_feedback(fb)
        assert result is True

    def test_unavailable_db_returns_false(self, sample_tx):
        from pipeline.db_writer import DatabaseWriter
        writer = DatabaseWriter.__new__(DatabaseWriter)
        writer._available = False
        writer.engine     = None
        result = writer.write_transaction(sample_tx)
        assert result is False

    def test_full_pipeline_write(self, db, sample_tx, sample_anomaly, sample_enriched, sample_report):
        """Simulate writing all stages for one transaction."""
        assert db.write_transaction(sample_tx)
        assert db.write_anomaly_result(sample_tx.transaction_id, sample_anomaly)
        assert db.write_enriched_transaction(sample_tx.transaction_id, sample_enriched)
        assert db.write_case_report(sample_tx.transaction_id, sample_report)

    def test_available_property(self, db):
        assert db.available is True
