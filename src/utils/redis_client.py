"""Redis client utilities."""
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisManager:
    """Redis cache manager"""

    def __init__(self, host: str = "127.0.0.1", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.client = None

    def connect(self):
        if not REDIS_AVAILABLE:
            logger.warning("redis not installed")
            return False
        self.client = redis.Redis(host=self.host, port=self.port, db=self.db, decode_responses=True)
        try:
            self.client.ping()
            logger.info(f"Redis connected: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False

    def cache_market_data(self, code: str, data: dict, ttl: int = 300):
        if not self.client:
            return
        self.client.setex(f"market:{code}", ttl, json.dumps(data, default=str))

    def get_cached_data(self, code: str) -> Optional[dict]:
        if not self.client:
            return None
        data = self.client.get(f"market:{code}")
        return json.loads(data) if data else None

    def set_strategy_state(self, strategy_id: int, state: dict):
        if not self.client:
            return
        self.client.set(f"strategy:{strategy_id}", json.dumps(state, default=str))

    def get_strategy_state(self, strategy_id: int) -> Optional[dict]:
        if not self.client:
            return None
        data = self.client.get(f"strategy:{strategy_id}")
        return json.loads(data) if data else None
