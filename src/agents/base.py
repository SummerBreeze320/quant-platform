"""Agent基类

所有Agent继承此类，统一生命周期管理、状态追踪和决策日志。
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Agent运行状态"""
    name: str
    status: str = "idle"  # idle / running / completed / error
    last_run: Optional[str] = None
    last_result: Optional[Dict] = None
    error: Optional[str] = None
    run_count: int = 0
    total_duration_sec: float = 0.0


class BaseAgent(ABC):
    """Agent基类

    生命周期: initialize() → run() → finalize()
    每次run()自动记录状态、耗时、结果。

    子类只需实现run()方法，输入输出类型由contracts.py定义。
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.state = AgentState(name=name)
        self._logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Any:
        """核心执行方法 — 子类必须实现

        Args:
            input_data: 输入数据，类型由contracts.py定义
            **kwargs: 额外参数（可选，由子类决定）

        Returns:
            输出数据，类型由contracts.py定义
        """
        pass

    def initialize(self):
        """初始化Agent（可选重载）"""
        self._logger.info(f"Agent [{self.name}] initialized")

    def finalize(self):
        """清理资源（可选重载）"""
        pass

    def execute(self, input_data: Any, **kwargs) -> Any:
        """执行Agent并自动记录状态

        外部调用入口，包装run()并添加状态追踪。
        kwargs会传递给run()方法。
        """
        self.state.status = "running"
        start = datetime.now()
        self.state.last_run = start.isoformat()

        try:
            result = self.run(input_data, **kwargs)
            self.state.status = "completed"
            self.state.last_result = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)[:200]
            )
            self._logger.info(f"Agent [{self.name}] completed successfully")
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            self._logger.error(f"Agent [{self.name}] failed: {e}", exc_info=True)
            raise
        finally:
            duration = (datetime.now() - start).total_seconds()
            self.state.run_count += 1
            self.state.total_duration_sec += duration
            self._logger.debug(
                f"Agent [{self.name}] run #{self.state.run_count} took {duration:.2f}s"
            )

        return result

    def get_state(self) -> AgentState:
        """获取Agent当前状态"""
        return self.state

    def get_info(self) -> Dict:
        """获取Agent信息摘要"""
        return {
            "name": self.name,
            "status": self.state.status,
            "run_count": self.state.run_count,
            "last_run": self.state.last_run,
            "avg_duration": (
                self.state.total_duration_sec / max(self.state.run_count, 1)
            ),
        }
