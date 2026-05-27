"""
feature_store/seeder.py
Seeds the Redis feature store with realistic data from the PaySim dataset.

Populates:
  - Merchant risk scores (from transaction type frequencies)
  - User profiles (avg amount, tx count, max amount)
  - Device registrations (simulated per user)
  - Location history (simulated per user)

Usage:
  python -m feature_store.seeder
  python -m feature_store.seeder --dataset data/paysim_sample.csv --limit 50000
"""

from __future__ import annotations
import argparse
import logging
import random
from pathlib import Path

import pandas as pd

from feature_store.redis_store import RedisFeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("seeder")

# Merchant category risk scores (domain knowledge)
MERCHANT_RISK_SCORES = {
    "atm_cash":       0.65,
    "bank_transfer":  0.50,
    "unknown":        0.75,
    "retail":         0.15,
    "food_and_drink": 0.10,
    "travel":         0.30,
    "online":         0.40,
    "bank_deposit":   0.15,
    "crypto":         0.85,
    "gambling":       0.80,
    "wire_transfer":  0.60,
    "peer_to_peer":   0.55,
}

PAYSIM_TYPE_TO_CATEGORY = {
    "PAYMENT":  "retail",
    "TRANSFER": "bank_transfer",
    "CASH_OUT": "atm_cash",
    "DEBIT":    "retail",
    "CASH_IN":  "bank_deposit",
}

LOCATIONS = [
    "Dublin, IE", "London, UK", "New York, US", "Lagos, NG",
    "Accra, GH", "Berlin, DE", "Paris, FR", "Toronto, CA"
]


def seed_from_paysim(
    store:      RedisFeatureStore,
    csv_path:   Path,
    limit:      int = 50_000,
) -> dict:
    """
    Build user profiles and device registrations from PaySim data.
    Returns summary of what was seeded.
    """
    logger.info("Loading PaySim data from %s (limit=%d)", csv_path, limit)
    df = pd.read_csv(csv_path, nrows=limit)
    logger.info("Loaded %d rows", len(df))

    # ── Seed merchant risks ────────────────────────────────────────────────────
    logger.info("Seeding merchant risk scores...")
    store.seed_merchant_risks(MERCHANT_RISK_SCORES)

    # ── Build user profiles ────────────────────────────────────────────────────
    logger.info("Building user profiles...")
    user_stats   = df.groupby("nameOrig").agg(
        avg_amount   = ("amount", "mean"),
        max_amount   = ("amount", "max"),
        total_amount = ("amount", "sum"),
        tx_count     = ("amount", "count"),
        fraud_count  = ("isFraud", "sum"),
    ).reset_index()

    pipe         = store.client.pipeline(transaction=False)
    user_count   = 0
    batch_size   = 500

    for i, row in user_stats.iterrows():
        user_id = row["nameOrig"]
        key     = f"user:profile:{user_id}"
        pipe.hset(key, mapping={
            "avg_amount":   round(float(row["avg_amount"]),   2),
            "max_amount":   round(float(row["max_amount"]),   2),
            "total_amount": round(float(row["total_amount"]), 2),
            "tx_count":     int(row["tx_count"]),
            "fraud_count":  int(row["fraud_count"]),
        })
        user_count += 1

        if user_count % batch_size == 0:
            pipe.execute()
            pipe = store.client.pipeline(transaction=False)
            logger.debug("Seeded %d user profiles...", user_count)

    pipe.execute()
    logger.info("Seeded %d user profiles", user_count)

    # ── Register devices ───────────────────────────────────────────────────────
    logger.info("Registering user devices...")
    device_count = 0
    rng          = random.Random(42)

    for user_id in user_stats["nameOrig"].unique()[:10_000]:  # cap for speed
        # Each user has 1-3 known devices
        n_devices = rng.randint(1, 3)
        for d in range(n_devices):
            device_id = f"dev_{user_id[-6:]}_{d}"
            store.register_device(user_id, device_id)
            device_count += 1

    logger.info("Registered %d device fingerprints", device_count)

    # ── Seed location history ──────────────────────────────────────────────────
    logger.info("Seeding location history...")
    loc_count = 0

    for user_id in user_stats["nameOrig"].unique()[:10_000]:
        # Each user has 2-4 known locations
        n_locs = rng.randint(2, 4)
        user_locations = rng.sample(LOCATIONS, min(n_locs, len(LOCATIONS)))
        for loc in user_locations:
            store.update_location(user_id, loc)
            loc_count += 1

    logger.info("Seeded %d location history entries", loc_count)

    summary = {
        "merchant_risks": len(MERCHANT_RISK_SCORES),
        "user_profiles":  user_count,
        "devices":        device_count,
        "locations":      loc_count,
    }
    logger.info("Seeding complete: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Seed Redis feature store")
    parser.add_argument("--dataset", type=str, default="data/paysim_sample.csv")
    parser.add_argument("--limit",   type=int, default=50_000)
    parser.add_argument("--host",    type=str, default="localhost")
    parser.add_argument("--port",    type=int, default=6379)
    args = parser.parse_args()

    store = RedisFeatureStore(host=args.host, port=args.port)
    if not store.available:
        logger.error("Redis not available — is Docker running?")
        return

    path = Path(args.dataset)
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        logger.error("Run: python scripts/download_datasets.py")
        return

    summary = seed_from_paysim(store, path, limit=args.limit)
    stats   = store.stats()
    logger.info("Feature store stats: %s", stats)
    print("\n=== Feature Store Seeded ===")
    for k, v in summary.items():
        print(f"  {k:20s}: {v:,}")
    print(f"\n  Redis keys total : {stats.get('total_keys', '?'):,}")
    print(f"  Memory used      : {stats.get('used_memory', '?')}")


if __name__ == "__main__":
    main()
