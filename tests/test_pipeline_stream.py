"""
tests/test_pipeline_stream.py
Tests for the Kafka producer/consumer and dataset loading.
Uses mocks so no real Kafka connection is needed.
"""

import pytest
import os
import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test_key")

from pipeline.producer import (
    paysim_row_to_payload,
    ccfraud_row_to_payload,
    _random_ip,
)
from pipeline.schemas import Transaction
import pandas as pd


# ── Producer unit tests ────────────────────────────────────────────────────────

class TestProducer:

    def test_paysim_row_to_payload(self):
        row = pd.Series({
            "step": 10, "type": "TRANSFER", "amount": 5000.0,
            "nameOrig": "C123456789", "nameDest": "M987654321",
            "oldbalanceOrg": 10000.0, "newbalanceOrig": 5000.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 5000.0,
            "isFraud": 1,
        })
        payload = paysim_row_to_payload(row, datetime(2024, 1, 1))

        assert payload["user_id"]           == "C123456789"
        assert payload["amount"]            == 5000.0
        assert payload["merchant_category"] == "bank_transfer"
        assert payload["_ground_truth"]     == "FRAUD"
        assert "timestamp" in payload

    def test_paysim_legit_row(self):
        row = pd.Series({
            "step": 1, "type": "PAYMENT", "amount": 25.0,
            "nameOrig": "C111111111", "nameDest": "M222222222",
            "oldbalanceOrg": 500.0, "newbalanceOrig": 475.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 25.0,
            "isFraud": 0,
        })
        payload = paysim_row_to_payload(row, datetime(2024, 1, 1))
        assert payload["_ground_truth"] == "LEGITIMATE"
        assert payload["merchant_category"] == "retail"

    def test_ccfraud_row_to_payload(self):
        row = pd.Series({
            **{f"V{i}": float(i) * 0.1 for i in range(1, 29)},
            "Time": 3600.0,
            "Amount": 150.0,
            "Class": 0,
        })
        payload = ccfraud_row_to_payload(row, datetime(2024, 1, 1))
        assert payload["amount"]        == 150.0
        assert payload["_ground_truth"] == "LEGITIMATE"
        assert "user_id" in payload
        assert "device_id" in payload

    def test_ccfraud_fraud_row(self):
        row = pd.Series({
            **{f"V{i}": float(i) * -0.5 for i in range(1, 29)},
            "Time": 7200.0,
            "Amount": 999.0,
            "Class": 1,
        })
        payload = ccfraud_row_to_payload(row, datetime(2024, 1, 1))
        assert payload["_ground_truth"] == "FRAUD"

    def test_payload_is_valid_transaction(self):
        """Payload dict should parse into a valid Transaction schema."""
        row = pd.Series({
            "step": 5, "type": "CASH_OUT", "amount": 800.0,
            "nameOrig": "C555555555", "nameDest": "M666666666",
            "oldbalanceOrg": 800.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 800.0,
            "isFraud": 1,
        })
        payload = paysim_row_to_payload(row, datetime(2024, 1, 1))
        payload.pop("_ground_truth")
        payload["timestamp"] = datetime.fromisoformat(payload["timestamp"])
        tx = Transaction(**payload)
        assert tx.amount == 800.0
        assert tx.user_id == "C555555555"

    def test_random_ip_suspicious(self):
        # With suspicious=True, should sometimes return Tor-range IPs
        ips = [_random_ip(suspicious=True) for _ in range(100)]
        tor_ips = [ip for ip in ips if ip.startswith("185.220.")]
        assert len(tor_ips) > 0  # at least some Tor IPs in 100 tries

    def test_random_ip_normal(self):
        ip = _random_ip(suspicious=False)
        parts = ip.split(".")
        assert len(parts) == 4


# ── Dataset generation tests ───────────────────────────────────────────────────

class TestDatasetGeneration:

    def test_generate_paysim_sample(self, tmp_path, monkeypatch):
        from scripts import download_datasets
        monkeypatch.setattr(download_datasets, "DATA_DIR", tmp_path)
        dest = download_datasets.generate_paysim_sample(n_rows=1000)
        assert dest.exists()
        df = pd.read_csv(dest)
        assert len(df) == 1000
        assert "isFraud" in df.columns
        assert df["isFraud"].sum() > 0  # should have some fraud

    def test_generate_cc_fraud_sample(self, tmp_path, monkeypatch):
        from scripts import download_datasets
        monkeypatch.setattr(download_datasets, "DATA_DIR", tmp_path)
        dest = download_datasets.generate_cc_fraud_sample(n_rows=1000)
        assert dest.exists()
        df = pd.read_csv(dest)
        assert len(df) == 1000
        assert "Class" in df.columns
        assert all(f"V{i}" in df.columns for i in range(1, 29))


# ── Consumer unit tests ────────────────────────────────────────────────────────

class TestConsumer:

    def test_message_processing(self):
        """Test that a Kafka message payload is correctly processed."""
        from pipeline.consumer import FraudConsumer

        with patch("pipeline.consumer.Consumer"), \
             patch("pipeline.consumer.FraudOrchestrator.from_env") as mock_orch:

            mock_result = MagicMock()
            mock_result.was_flagged = True
            mock_result.anomaly.fraud_score = 0.85
            mock_result.final_action.value = "BLOCK"
            mock_result.report.confidence_score = 0.9
            mock_orch.return_value.process_transaction.return_value = mock_result

            consumer = FraudConsumer()

            # Simulate a Kafka message
            msg = MagicMock()
            msg.value.return_value = json.dumps({
                "user_id": "C123456",
                "amount": 5000.0,
                "merchant": "TestShop",
                "merchant_category": "unknown",
                "location": "unknown",
                "device_id": "dev_001",
                "ip_address": "1.2.3.4",
                "_ground_truth": "FRAUD",
            }).encode()

            consumer._process_message(msg)
            assert consumer.stats.processed == 1
            assert consumer.stats.flagged == 1
            assert consumer.stats.blocked == 1

    def test_stats_flag_rate(self):
        from pipeline.consumer import ConsumerStats
        stats = ConsumerStats()
        stats.processed = 100
        stats.flagged   = 13
        assert abs(stats.flagged / stats.processed - 0.13) < 0.001
