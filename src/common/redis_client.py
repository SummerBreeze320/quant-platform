import json
import time
from typing import Optional, Any
from contextlib import contextmanager
import redis
from src.common.config import get_settings
from src.common.logger import logger

settings = get_settings()

class RedisClient:
    def __init__(self):
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=5
            )
        return self._client

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return False

    def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            return bool(self.client.set(key, serialized, ex=ex))
        except Exception as e:
            logger.error(f"Error setting redis key {key}: {e}")
            return False

    def get_json(self, key: str) -> Optional[Any]:
        try:
            val = self.client.get(key)
            if val:
                return json.loads(val)
            return None
        except Exception as e:
            logger.error(f"Error getting redis key {key}: {e}")
            return None

    @contextmanager
    def distributed_lock(self, lock_name: str, expire_seconds: int = 60, timeout: int = 5):
        """Distributed lock context manager using Redis."""
        lock_key = f"lock:{lock_name}"
        identifier = str(time.time())
        acquired = False
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                if self.client.set(lock_key, identifier, nx=True, ex=expire_seconds):
                    acquired = True
                    break
                time.sleep(0.1)
            
            if not acquired:
                raise TimeoutError(f"Could not acquire distributed lock for '{lock_name}' within {timeout}s")
            
            yield acquired
        finally:
            if acquired:
                # Release lock only if held value matches
                try:
                    if self.client.get(lock_key) == identifier:
                        self.client.delete(lock_key)
                except Exception as e:
                    logger.warning(f"Error releasing lock {lock_name}: {e}")

_redis_instance = RedisClient()

def get_redis() -> RedisClient:
    return _redis_instance
