import pytest
from unittest.mock import MagicMock
from src.common.redis_client import RedisClient

def test_redis_client_json_methods(monkeypatch):
    client = RedisClient()
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.set.return_value = True
    mock_redis.get.return_value = '{"status": "ok", "count": 10}'
    
    client._client = mock_redis
    
    assert client.ping() is True
    assert client.set_json("test_key", {"status": "ok", "count": 10}) is True
    
    data = client.get_json("test_key")
    assert data == {"status": "ok", "count": 10}

def test_redis_distributed_lock(monkeypatch):
    client = RedisClient()
    mock_redis = MagicMock()
    mock_redis.set.return_value = True
    mock_redis.get.return_value = "mock_id"
    mock_redis.delete.return_value = True
    
    client._client = mock_redis
    with client.distributed_lock("sync_task", expire_seconds=10, timeout=1) as acquired:
        assert acquired is True
