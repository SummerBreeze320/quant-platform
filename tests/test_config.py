import os
import pytest
from src.common.config import Settings, get_settings

def test_settings_load_defaults(monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", "quant_test")
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    settings = Settings()
    assert settings.POSTGRES_DB == "quant_test"
    assert settings.REDIS_HOST == "127.0.0.1"
    assert "postgresql://" in settings.DATABASE_URL or "postgresql+asyncpg://" in settings.DATABASE_URL
    assert settings.QLIB_DATA_DIR.endswith("qlib_data")
