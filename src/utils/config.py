"""配置加载工具"""
import yaml
from pathlib import Path


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """加载YAML配置"""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
