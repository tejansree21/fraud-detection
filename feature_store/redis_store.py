"""
feature_store/redis_store.py
Redis-backed feature store for real-time transaction enrichment.

Stores and retrieves:
  - Merchant category risk scores
  - User behaviour profiles (avg amount, tx frequency)
  - Known device fingerprints per user
  - User location history (last 5 locations)
  - Velocity counters (tx/hour, tx/day) using Redis sorted sets

Key schema:
  merchant:risk:{category}          → float (0-1)
  user:profile:{user_id}            → hash (avg_amount, tx_count, etc)
  user:devices:{user_id}            → set of known device_ids
  user:locations:{user_id}          → list of last 5 locations
  user:velocity:{user_id}:{window}  → sorted set (tx timestamps)
"""

from __future__ import annotations
import json
import logging
import time
from typing import Any

import redis

logger = logging.getLogger("feature_store")

# Redis key TTLs
PROFILE_TTL  = 86400 * 7   # 7 days
VELOCITY_TTL = 86400        # 1 day
DEVICE_TTL   = 86400 * 30  # 30 days

# Velocity windows in seconds
WINDOW_HOUR = 3600
WINDOW_DAY  = 86400


class RedisFeatureStore:
    """
    Real-time feature store backed by Redis.
    Used by the Context Enricher agent for transaction enrichment.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(
            host=host, port=port, db=db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self._available = False
        self._check_connection()

    def _check_connection(self) -> bool:
        try:
            self.client.ping()
            self._available = True
            logger.info("Redis feature store connected | host=%s port=%s", 
                       self.client.connection_pool.connection_kwargs.get('host'),
                       self.client.connection_pool.connection_kwargs.get('port'))
            return True
        except Exception as e:
            logger.warning("Redis not available (%s) — feature store in fallback mode", e)
            self._available = False
            return False

    @property
    def available(self) -> bool:
        return self._available

    # ── Merchant risk ──────────────────────────────────────────────────────────

    def get_merchant_risk(self, merchant_category: str) -> float:
        """Return risk score [0,1] for a merchant category."""
        if not self._available:
            return self._default_merchant_risk(merchant_category)
        try:
            key = f"merchant:risk:{merchant_category}"
            val = self.client.get(key)
            return float(val) if val else self._default_merchant_risk(merchant_category)
        except Exception:
            return self._default_merchant_risk(merchant_category)

    def set_merchant_risk(self, merchant_category: str, risk_score: float) -> None:
        if not self._available:
            return
        self.client.set(f"merchant:risk:{merchant_category}", risk_score)

    def _default_merchant_risk(self, category: str) -> float:
        defaults = {
            "atm_cash":      0.6,
            "bank_transfer": 0.5,
            "unknown":       0.7,
            "retail":        0.2,
            "food_and_drink":0.1,
            "travel":        0.3,
            "online":        0.4,
            "bank_deposit":  0.2,
        }
        return defaults.get(category, 0.4)

    # ── User profile ───────────────────────────────────────────────────────────

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """Return user's historical behaviour profile."""
        if not self._available:
            return {}
        try:
            key  = f"user:profile:{user_id}"
            data = self.client.hgetall(key)
            if not data:
                return {}
            return {
                "avg_amount":    float(data.get("avg_amount", 0)),
                "tx_count":      int(data.get("tx_count", 0)),
                "max_amount":    float(data.get("max_amount", 0)),
                "total_amount":  float(data.get("total_amount", 0)),
                "fraud_count":   int(data.get("fraud_count", 0)),
            }
        except Exception:
            return {}

    def update_user_profile(self, user_id: str, amount: float, is_fraud: bool = False) -> None:
        """Update user profile with a new transaction."""
        if not self._available:
            return
        try:
            key  = f"user:profile:{user_id}"
            pipe = self.client.pipeline()

            # Increment counters
            pipe.hincrbyfloat(key, "total_amount", amount)
            pipe.hincrby(key, "tx_count", 1)
            if is_fraud:
                pipe.hincrby(key, "fraud_count", 1)

            pipe.execute()

            # Update avg and max separately
            profile  = self.get_user_profile(user_id)
            tx_count = max(profile.get("tx_count", 1), 1)
            avg      = profile.get("total_amount", amount) / tx_count
            cur_max  = profile.get("max_amount", 0)

            pipe = self.client.pipeline()
            pipe.hset(key, "avg_amount", round(avg, 2))
            pipe.hset(key, "max_amount", max(cur_max, amount))
            pipe.expire(key, PROFILE_TTL)
            pipe.execute()

        except Exception as e:
            logger.debug("Failed to update user profile: %s", e)

    def compute_behaviour_score(self, user_id: str, amount: float) -> float:
        """
        How anomalous is this transaction relative to the user's history?
        Returns 0 (normal) to 1 (very anomalous).
        """
        profile = self.get_user_profile(user_id)
        if not profile or profile.get("tx_count", 0) < 3:
            return 0.3  # not enough history

        avg    = profile.get("avg_amount", amount)
        tx_max = profile.get("max_amount", amount)

        if avg == 0:
            return 0.3

        # Z-score style deviation
        ratio = amount / avg
        if ratio > 10:   return 1.0
        if ratio > 5:    return 0.8
        if ratio > 3:    return 0.6
        if ratio > 2:    return 0.4
        if ratio > 1.5:  return 0.2
        return 0.1

    # ── Device fingerprinting ──────────────────────────────────────────────────

    def is_known_device(self, user_id: str, device_id: str) -> bool:
        """Return True if this device has been seen before for this user."""
        if not self._available or device_id == "unknown":
            return device_id != "unknown"
        try:
            key = f"user:devices:{user_id}"
            return bool(self.client.sismember(key, device_id))
        except Exception:
            return True  # fail safe

    def register_device(self, user_id: str, device_id: str) -> None:
        """Register a device as known for a user."""
        if not self._available or device_id == "unknown":
            return
        try:
            key = f"user:devices:{user_id}"
            self.client.sadd(key, device_id)
            self.client.expire(key, DEVICE_TTL)
        except Exception:
            pass

    def get_device_count(self, user_id: str) -> int:
        """Return number of known devices for a user."""
        if not self._available:
            return 1
        try:
            return self.client.scard(f"user:devices:{user_id}") or 1
        except Exception:
            return 1

    # ── Geolocation consistency ────────────────────────────────────────────────

    def get_geo_consistency(self, user_id: str, current_location: str) -> float:
        """
        Compare current location against user's recent locations.
        Returns 1.0 (consistent) to 0.0 (never seen before).
        """
        if not self._available or current_location == "unknown":
            return 0.3 if current_location == "unknown" else 0.8
        try:
            key       = f"user:locations:{user_id}"
            locations = self.client.lrange(key, 0, 4)  # last 5
            if not locations:
                return 0.7  # no history = neutral

            if current_location in locations:
                # More recent = higher score
                idx = locations.index(current_location)
                return 1.0 - (idx * 0.1)
            return 0.2  # never seen this location

        except Exception:
            return 0.7

    def update_location(self, user_id: str, location: str) -> None:
        """Add location to user's recent location history (capped at 10)."""
        if not self._available or location == "unknown":
            return
        try:
            key = f"user:locations:{user_id}"
            pipe = self.client.pipeline()
            pipe.lpush(key, location)
            pipe.ltrim(key, 0, 9)   # keep last 10
            pipe.expire(key, PROFILE_TTL)
            pipe.execute()
        except Exception:
            pass

    # ── Velocity (sliding window) ──────────────────────────────────────────────

    def get_velocity(self, user_id: str, window_seconds: int = WINDOW_HOUR) -> int:
        """
        Count transactions for user in the last window_seconds.
        Uses Redis sorted set with timestamps as scores.
        """
        if not self._available:
            return 0
        try:
            key      = f"user:velocity:{user_id}:{window_seconds}"
            now      = time.time()
            cutoff   = now - window_seconds
            count    = self.client.zcount(key, cutoff, now)
            return int(count)
        except Exception:
            return 0

    def record_transaction(self, user_id: str, tx_id: str) -> None:
        """Record a transaction timestamp for velocity tracking."""
        if not self._available:
            return
        try:
            now  = time.time()
            for window in [WINDOW_HOUR, WINDOW_DAY]:
                key = f"user:velocity:{user_id}:{window}"
                self.client.zadd(key, {tx_id: now})
                # Remove old entries outside window
                self.client.zremrangebyscore(key, 0, now - window)
                self.client.expire(key, window + 60)
        except Exception:
            pass

    # ── Bulk operations ────────────────────────────────────────────────────────

    def seed_merchant_risks(self, risk_map: dict[str, float]) -> int:
        """Bulk-load merchant risk scores. Returns count loaded."""
        if not self._available:
            return 0
        pipe  = self.client.pipeline()
        count = 0
        for category, score in risk_map.items():
            pipe.set(f"merchant:risk:{category}", score)
            count += 1
        pipe.execute()
        logger.info("Seeded %d merchant risk scores", count)
        return count

    def flush_user_data(self, user_id: str) -> None:
        """Remove all data for a user (for testing)."""
        if not self._available:
            return
        keys = [
            f"user:profile:{user_id}",
            f"user:devices:{user_id}",
            f"user:locations:{user_id}",
            f"user:velocity:{user_id}:{WINDOW_HOUR}",
            f"user:velocity:{user_id}:{WINDOW_DAY}",
        ]
        self.client.delete(*keys)

    def stats(self) -> dict[str, Any]:
        """Return feature store statistics."""
        if not self._available:
            return {"available": False}
        try:
            info = self.client.info("memory")
            return {
                "available":    True,
                "used_memory":  info.get("used_memory_human", "unknown"),
                "total_keys":   self.client.dbsize(),
            }
        except Exception:
            return {"available": True, "error": "stats unavailable"}
