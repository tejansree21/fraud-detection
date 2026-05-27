"""
pipeline/producer.py
Kafka Producer — reads fraud datasets and streams transactions
into the 'transactions' Kafka topic at a configurable rate.

Supports two dataset formats:
  - PaySim CSV  (columns: step, type, amount, nameOrig, ...)
  - CC Fraud CSV (columns: V1-V28, Time, Amount, Class)

Usage:
  python -m pipeline.producer --dataset paysim --rate 50
  python -m pipeline.producer --dataset ccfraud --rate 100 --limit 5000
"""

from __future__ import annotations
import argparse
import json
import time
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("producer")

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC             = "transactions"
DATA_DIR          = Path("data")

# Merchant category mapping for PaySim transaction types
PAYSIM_CATEGORY_MAP = {
    "PAYMENT":  "retail",
    "TRANSFER": "bank_transfer",
    "CASH_OUT": "atm_cash",
    "DEBIT":    "retail",
    "CASH_IN":  "bank_deposit",
}

LOCATIONS = [
    "Dublin, IE", "London, UK", "New York, US", "Lagos, NG",
    "unknown", "unknown", "Accra, GH", "Berlin, DE", "Paris, FR"
]


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_paysim(path: Path) -> pd.DataFrame:
    logger.info("Loading PaySim dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows (%d fraud)", len(df), df["isFraud"].sum())
    return df


def load_ccfraud(path: Path) -> pd.DataFrame:
    logger.info("Loading CC Fraud dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows (%d fraud)", len(df), df["Class"].sum())
    return df


# ── Row → Transaction payload ──────────────────────────────────────────────────

def paysim_row_to_payload(row: pd.Series, base_time: datetime) -> dict:
    """Convert a PaySim row into a Transaction-compatible dict."""
    # Simulate timestamps: each 'step' = 1 hour from base_time
    tx_time = base_time + timedelta(hours=int(row["step"]))
    is_fraud = bool(row["isFraud"])

    return {
        "user_id":           str(row["nameOrig"]),
        "amount":            float(row["amount"]),
        "merchant":          str(row["nameDest"]),
        "merchant_category": PAYSIM_CATEGORY_MAP.get(row["type"], "unknown"),
        "timestamp":         tx_time.isoformat(),
        "location":          "unknown" if is_fraud and random.random() < 0.6
                             else random.choice(LOCATIONS),
        "device_id":         f"dev_{row['nameOrig'][-6:]}",
        "ip_address":        _random_ip(suspicious=is_fraud),
        "_ground_truth":     "FRAUD" if is_fraud else "LEGITIMATE",  # for evaluation
    }


def ccfraud_row_to_payload(row: pd.Series, base_time: datetime) -> dict:
    """Convert a CC Fraud row into a Transaction-compatible dict."""
    tx_time = base_time + timedelta(seconds=float(row["Time"]))
    is_fraud = bool(row["Class"])
    user_id  = f"user_{abs(hash(row['V1']))  % 10000:05d}"

    return {
        "user_id":           user_id,
        "amount":            float(row["Amount"]),
        "merchant":          f"merchant_{abs(hash(row['V2'])) % 1000:04d}",
        "merchant_category": "unknown" if is_fraud and random.random() < 0.5
                             else random.choice(["retail", "food_and_drink", "travel", "online"]),
        "timestamp":         tx_time.isoformat(),
        "location":          "unknown" if is_fraud and random.random() < 0.4
                             else random.choice(LOCATIONS),
        "device_id":         f"dev_{user_id}_{abs(hash(row['V3'])) % 100:02d}",
        "ip_address":        _random_ip(suspicious=is_fraud),
        "_ground_truth":     "FRAUD" if is_fraud else "LEGITIMATE",
    }


def _random_ip(suspicious: bool = False) -> str:
    if suspicious and random.random() < 0.5:
        # Known Tor exit node ranges (simulated)
        return f"185.220.{random.randint(100, 107)}.{random.randint(1, 254)}"
    return f"{random.randint(10, 203)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


# ── Kafka producer ─────────────────────────────────────────────────────────────

def delivery_callback(err, msg):
    if err:
        logger.error("Delivery failed for %s: %s", msg.key(), err)


def stream_dataset(
    df: pd.DataFrame,
    row_to_payload_fn,
    rate_per_second: int = 50,
    limit: int | None    = None,
    shuffle: bool        = True,
):
    """Stream rows from a dataframe to Kafka at the given rate."""
    producer = Producer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "queue.buffering.max.messages": 100_000,
        "batch.num.messages": 1000,
        "linger.ms": 10,
    })

    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if limit:
        df = df.head(limit)

    base_time   = datetime(2024, 1, 1, 0, 0, 0)
    interval    = 1.0 / rate_per_second
    total       = len(df)
    sent        = 0
    fraud_sent  = 0
    start       = time.perf_counter()

    logger.info("Starting stream: %d transactions at %d tx/s", total, rate_per_second)

    for _, row in df.iterrows():
        payload = row_to_payload_fn(row, base_time)
        is_fraud = payload.get("_ground_truth") == "FRAUD"

        message = json.dumps(payload).encode("utf-8")
        key     = payload["user_id"].encode("utf-8")

        producer.produce(
            topic    = TOPIC,
            key      = key,
            value    = message,
            callback = delivery_callback,
        )
        producer.poll(0)  # non-blocking poll to trigger callbacks

        sent += 1
        if is_fraud:
            fraud_sent += 1

        # Progress log every 1000 messages
        if sent % 1000 == 0:
            elapsed = time.perf_counter() - start
            actual_rate = sent / elapsed
            logger.info(
                "Progress: %d/%d (%.1f%%) | fraud=%d | rate=%.0f tx/s",
                sent, total, sent/total*100, fraud_sent, actual_rate
            )

        time.sleep(interval)

    producer.flush(timeout=10)
    elapsed = time.perf_counter() - start
    logger.info(
        "Stream complete: %d transactions in %.1fs (%.0f tx/s) | fraud=%d (%.2f%%)",
        sent, elapsed, sent/elapsed, fraud_sent, fraud_sent/sent*100
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Kafka Producer")
    parser.add_argument("--dataset", choices=["paysim", "ccfraud", "both"], default="paysim")
    parser.add_argument("--rate",    type=int,   default=20,    help="Transactions per second")
    parser.add_argument("--limit",   type=int,   default=10000, help="Max transactions to send (0=all)")
    parser.add_argument("--shuffle", action="store_true", default=True)
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    if args.dataset in ("paysim", "both"):
        # Try real dataset first, fall back to synthetic
        paysim_paths = [
            DATA_DIR / "PS_20174392719_1491204439457_log.csv",
            DATA_DIR / "paysim_sample.csv",
        ]
        paysim_path = next((p for p in paysim_paths if p.exists()), None)
        if paysim_path is None:
            logger.error("PaySim data not found. Run: python scripts/download_datasets.py")
            return
        df = load_paysim(paysim_path)
        stream_dataset(df, paysim_row_to_payload, rate_per_second=args.rate, limit=limit)

    if args.dataset in ("ccfraud", "both"):
        cc_paths = [
            DATA_DIR / "creditcard.csv",
            DATA_DIR / "creditcard_sample.csv",
        ]
        cc_path = next((p for p in cc_paths if p.exists()), None)
        if cc_path is None:
            logger.error("CC Fraud data not found. Run: python scripts/download_datasets.py")
            return
        df = load_ccfraud(cc_path)
        stream_dataset(df, ccfraud_row_to_payload, rate_per_second=args.rate, limit=limit)


if __name__ == "__main__":
    main()
