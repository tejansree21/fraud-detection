"""
feature_store/redis_store.py
Redis feature store supporting both local Redis and Upstash (production).
Auto-detects which to use based on environment variables.
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger("feature_store")

PROFILE_TTL  = 86400 * 7
VELOCITY_TTL = 86400
DEVICE_TTL   = 86400 * 30
WINDOW_HOUR  = 3600
WINDOW_DAY   = 86400


class RedisFeatureStore:
    """
    Redis feature store with Upstash HTTP support for production.
    
    Priority:
      1. Upstash (if UPSTASH_REDIS_REST_URL is set)
      2. Standard Redis (REDIS_HOST / localhost)
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client     = None
        self._available = False
        self._upstash   = False

        upstash_url   = os.environ.get("UPSTASH_REDIS_REST_URL")
        upstash_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

        if upstash_url and upstash_token:
            self._try_upstash(upstash_url, upstash_token)
        else:
            self._try_local(host, port, db)

    def _try_upstash(self, url: str, token: str) -> None:
        try:
            from upstash_redis import Redis
            self.client     = Redis(url=url, token=token)
            self.client.ping()
            self._available = True
            self._upstash   = True
            logger.info("Redis feature store connected via Upstash HTTP")
        except ImportError:
            logger.warning("upstash-redis not installed — trying local Redis")
            self._try_local("localhost", 6379, 0)
        except Exception as e:
            logger.warning("Upstash failed (%s) — trying local Redis", e)
            self._try_local("localhost", 6379, 0)

    def _try_local(self, host: str, port: int, db: int) -> None:
        try:
            import redis
            self.client = redis.Redis(
                host=host, port=port, db=db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self.client.ping()
            self._available = True
            self._upstash   = False
            logger.info("Redis feature store connected | host=%s port=%s", host, port)
        except Exception as e:
            logger.warning("Redis not available (%s) — fallback mode", e)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _check_connection(self) -> bool:
        try:
            self.client.ping()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False

    def get_merchant_risk(self, merchant_category: str) -> float:
        if not self._available:
            return self._default_merchant_risk(merchant_category)
        try:
            val = self.client.get(f"merchant:risk:{merchant_category}")
            return float(val) if val else self._default_merchant_risk(merchant_category)
        except Exception:
            return self._default_merchant_risk(merchant_category)

    def set_merchant_risk(self, merchant_category: str, risk_score: float) -> None:
        if not self._available: return
        try: self.client.set(f"merchant:risk:{merchant_category}", risk_score)
        except Exception: pass

    def _default_merchant_risk(self, category: str) -> float:
        return {"atm_cash":0.6,"bank_transfer":0.5,"unknown":0.7,"retail":0.2,
                "food_and_drink":0.1,"travel":0.3,"online":0.4,"bank_deposit":0.2}.get(category, 0.4)

    def seed_merchant_risks(self, risk_map: dict[str, float]) -> int:
        if not self._available: return 0
        count = 0
        for category, score in risk_map.items():
            try: self.client.set(f"merchant:risk:{category}", score); count += 1
            except Exception: pass
        logger.info("Seeded %d merchant risk scores", count)
        return count

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        if not self._available: return {}
        try:
            data = self.client.hgetall(f"user:profile:{user_id}")
            if not data: return {}
            return {"avg_amount": float(data.get("avg_amount",0)),
                    "tx_count":   int(data.get("tx_count",0)),
                    "max_amount": float(data.get("max_amount",0)),
                    "total_amount": float(data.get("total_amount",0)),
                    "fraud_count":  int(data.get("fraud_count",0))}
        except Exception: return {}

    def update_user_profile(self, user_id: str, amount: float, is_fraud: bool = False) -> None:
        if not self._available: return
        try:
            key = f"user:profile:{user_id}"
            self.client.hincrbyfloat(key, "total_amount", amount)
            self.client.hincrby(key, "tx_count", 1)
            if is_fraud: self.client.hincrby(key, "fraud_count", 1)
            profile  = self.get_user_profile(user_id)
            tx_count = max(profile.get("tx_count", 1), 1)
            avg      = profile.get("total_amount", amount) / tx_count
            self.client.hset(key, "avg_amount", round(avg, 2))
            self.client.hset(key, "max_amount", max(profile.get("max_amount",0), amount))
            self.client.expire(key, PROFILE_TTL)
        except Exception as e: logger.debug("Failed to update user profile: %s", e)

    def compute_behaviour_score(self, user_id: str, amount: float) -> float:
        profile = self.get_user_profile(user_id)
        if not profile or profile.get("tx_count", 0) < 3: return 0.3
        avg = profile.get("avg_amount", amount)
        if avg == 0: return 0.3
        ratio = amount / avg
        if ratio > 10: return 1.0
        if ratio > 5:  return 0.8
        if ratio > 3:  return 0.6
        if ratio > 2:  return 0.4
        if ratio > 1.5: return 0.2
        return 0.1

    def is_known_device(self, user_id: str, device_id: str) -> bool:
        if not self._available or device_id == "unknown": return device_id != "unknown"
        try: return bool(self.client.sismember(f"user:devices:{user_id}", device_id))
        except Exception: return True

    def register_device(self, user_id: str, device_id: str) -> None:
        if not self._available or device_id == "unknown": return
        try:
            key = f"user:devices:{user_id}"
            self.client.sadd(key, device_id)
            self.client.expire(key, DEVICE_TTL)
        except Exception: pass

    def get_device_count(self, user_id: str) -> int:
        if not self._available: return 1
        try: return self.client.scard(f"user:devices:{user_id}") or 1
        except Exception: return 1

    def get_geo_consistency(self, user_id: str, current_location: str) -> float:
        if not self._available or current_location == "unknown":
            return 0.3 if current_location == "unknown" else 0.8
        try:
            locations = self.client.lrange(f"user:locations:{user_id}", 0, 4)
            if not locations: return 0.7
            if current_location in locations:
                return 1.0 - (locations.index(current_location) * 0.1)
            return 0.2
        except Exception: return 0.7

    def update_location(self, user_id: str, location: str) -> None:
        if not self._available or location == "unknown": return
        try:
            key = f"user:locations:{user_id}"
            self.client.lpush(key, location)
            self.client.ltrim(key, 0, 9)
            self.client.expire(key, PROFILE_TTL)
        except Exception: pass

    def get_velocity(self, user_id: str, window_seconds: int = WINDOW_HOUR) -> int:
        if not self._available: return 0
        try:
            now = time.time()
            return int(self.client.zcount(f"user:velocity:{user_id}:{window_seconds}", now - window_seconds, now))
        except Exception: return 0

    def record_transaction(self, user_id: str, tx_id: str) -> None:
        if not self._available: return
        try:
            now = time.time()
            for window in [WINDOW_HOUR, WINDOW_DAY]:
                key = f"user:velocity:{user_id}:{window}"
                self.client.zadd(key, {tx_id: now})
                self.client.zremrangebyscore(key, 0, now - window)
                self.client.expire(key, window + 60)
        except Exception: pass

    def flush_user_data(self, user_id: str) -> None:
        if not self._available: return
        keys = [f"user:profile:{user_id}", f"user:devices:{user_id}",
                f"user:locations:{user_id}",
                f"user:velocity:{user_id}:{WINDOW_HOUR}",
                f"user:velocity:{user_id}:{WINDOW_DAY}"]
        try: self.client.delete(*keys)
        except Exception: pass

    def stats(self) -> dict[str, Any]:
        if not self._available: return {"available": False}
        try:
            return {
                "available": True,
                "backend":   "upstash" if self._upstash else "local",
                "total_keys": self.client.dbsize() if not self._upstash else "n/a",
            }
        except Exception: return {"available": True}
