"""RD-Agent配置管理"""
import yaml
from pathlib import Path
from typing import Optional


class RDAgentConfig:
    """RD-Agent配置"""

    def __init__(self, config_path: str = "config/rd_agent_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load()

    def _load(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    @property
    def llm_backend(self) -> str:
        return self.config.get("rd_agent", {}).get("llm", {}).get("backend", "deepseek")

    @property
    def llm_model(self) -> str:
        return self.config.get("rd_agent", {}).get("llm", {}).get("model", "deepseek-chat")

    @property
    def qlib_data_dir(self) -> str:
        return self.config.get("rd_agent", {}).get("qlib_data_dir", "data/qlib_bin")

    @property
    def work_dir(self) -> str:
        return self.config.get("rd_agent", {}).get("work_dir", "data/rd_agent_workspace")

    @property
    def max_iterations(self) -> int:
        return self.config.get("rd_agent", {}).get("factor_config", {}).get("max_iterations", 10)

    def to_env_dict(self) -> dict:
        """转为环境变量字典（RD-Agent读取）"""
        llm = self.config.get("rd_agent", {}).get("llm", {})
        return {
            "RAG_BACKEND": llm.get("backend", "deepseek"),
            "RAG_MODEL": llm.get("model", "deepseek-chat"),
            "RAG_API_KEY": llm.get("api_key", ""),
            "RAG_BASE_URL": llm.get("base_url", ""),
            "QLIB_DATA_DIR": self.qlib_data_dir,
        }
