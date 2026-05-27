"""
pipeline/consumer.py
Kafka Consumer — reads transactions from the 'transactions' topic
and feeds each one through the fraud detection orchestrator pipeline.

Runs as a long-lived background process alongside the FastAPI server.

Usage:
  python -m pipeline.consumer
  python -m pipeline.consumer --workers 2
"""

from __future__ import annotations
import argparse
import json
import logging
import signal
import sys
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from confluent_kafka import Consumer, KafkaError, KafkaException

os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

from pipeline.schemas import Transaction
from orchestrator.orchestrator import FraudOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("consumer")


# ── Stats tracker ──────────────────────────────────────────────────────────────

class ConsumerStats:
    def __init__(self):
        self.received   = 0
        self.processed  = 0
        self.flagged    = 0
        self.blocked    = 0
        self.errors     = 0
        self.start_time = time.perf_counter()

    def log(self):
        elapsed = time.perf_counter() - self.start_time
        rate    = self.processed / elapsed if elapsed > 0 else 0
        flag_rate = self.flagged / self.processed if self.processed > 0 else 0
        logger.info(
            "Stats | received=%d processed=%d flagged=%d blocked=%d errors=%d "
            "rate=%.1f tx/s flag_rate=%.2f%%",
            self.received, self.processed, self.flagged, self.blocked,
            self.errors, rate, flag_rate * 100
        )


# ── Consumer ───────────────────────────────────────────────────────────────────

class FraudConsumer:
    """
    Kafka consumer that feeds transactions into the fraud detection pipeline.

    Each message from the 'transactions' topic is:
      1. Deserialised into a Transaction object
      2. Passed to orchestrator.process_transaction()
      3. Results logged; flagged transactions trigger full agent pipeline
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str          = "fraud_detection_group",
        topic: str             = "transactions",
        workers: int           = 1,
    ):
        self.topic       = topic
        self.stats       = ConsumerStats()
        self.running     = False
        self.workers     = workers
        self.orchestrator = FraudOrchestrator.from_env()

        self.consumer = Consumer({
            "bootstrap.servers":        bootstrap_servers,
            "group.id":                 group_id,
            "auto.offset.reset":        "earliest",
            "enable.auto.commit":       True,
            "auto.commit.interval.ms":  1000,
            "max.poll.interval.ms":     300000,
            "session.timeout.ms":       30000,
        })

        logger.info(
            "FraudConsumer initialised | topic=%s group=%s workers=%d",
            topic, group_id, workers
        )

    def start(self):
        """Start consuming — blocks until stopped."""
        self.consumer.subscribe([self.topic])
        self.running = True

        # Graceful shutdown on SIGINT / SIGTERM
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        logger.info("Consumer started — listening on topic '%s'", self.topic)

        if self.workers > 1:
            self._start_threaded()
        else:
            self._start_single()

    def _start_single(self):
        """Single-threaded consume loop."""
        stats_interval = 100  # log every N messages
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    self._handle_error(msg)
                    continue
                self._process_message(msg)
                if self.stats.processed % stats_interval == 0:
                    self.stats.log()
        finally:
            self._close()

    def _start_threaded(self):
        """Multi-threaded consume loop — one consumer thread, N worker threads."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            try:
                while self.running:
                    msg = self.consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        self._handle_error(msg)
                        continue
                    self.stats.received += 1
                    pool.submit(self._process_message, msg)
                    if self.stats.received % 100 == 0:
                        self.stats.log()
            finally:
                self._close()

    def _process_message(self, msg):
        """Deserialise one Kafka message and run it through the pipeline."""
        try:
            raw     = json.loads(msg.value().decode("utf-8"))
            ground_truth = raw.pop("_ground_truth", None)  # remove eval field

            # Parse timestamp if present
            if "timestamp" in raw and isinstance(raw["timestamp"], str):
                raw["timestamp"] = datetime.fromisoformat(raw["timestamp"])

            tx = Transaction(**raw)
            result = self.orchestrator.process_transaction(tx)

            self.stats.processed += 1

            if result.was_flagged:
                self.stats.flagged += 1
                action = result.final_action.value if result.final_action else "N/A"

                if action == "BLOCK":
                    self.stats.blocked += 1

                logger.info(
                    "FLAGGED | tx=%-36s score=%.3f action=%-6s truth=%-10s confidence=%.2f",
                    tx.transaction_id,
                    result.anomaly.fraud_score,
                    action,
                    ground_truth or "UNKNOWN",
                    result.report.confidence_score if result.report else 0,
                )
            else:
                logger.debug(
                    "PASSED  | tx=%s score=%.3f truth=%s",
                    tx.transaction_id, result.anomaly.fraud_score, ground_truth or "UNKNOWN"
                )

        except Exception as exc:
            self.stats.errors += 1
            logger.error("Error processing message: %s", exc, exc_info=True)

    def _handle_error(self, msg):
        if msg.error().code() == KafkaError._PARTITION_EOF:
            logger.debug("End of partition: %s [%d]", msg.topic(), msg.partition())
        else:
            raise KafkaException(msg.error())

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received — stopping consumer...")
        self.running = False

    def _close(self):
        logger.info("Closing consumer...")
        self.stats.log()
        self.consumer.close()
        logger.info("Consumer closed.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Kafka Consumer")
    parser.add_argument("--workers",  type=int, default=1,           help="Number of worker threads")
    parser.add_argument("--servers",  type=str, default="localhost:9092")
    parser.add_argument("--group",    type=str, default="fraud_detection_group")
    parser.add_argument("--topic",    type=str, default="transactions")
    args = parser.parse_args()

    consumer = FraudConsumer(
        bootstrap_servers = args.servers,
        group_id          = args.group,
        topic             = args.topic,
        workers           = args.workers,
    )
    consumer.start()


if __name__ == "__main__":
    main()
