"""RD-Agent模型进化运行器

调用RD-Agent的fin_model场景进行模型结构自动进化和优化。
支持模型搜索、超参数调优、模型集成策略优化。
"""
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, List

from .config import RDAgentConfig

logger = logging.getLogger(__name__)


class ModelRunner:
    """调用RD-Agent进行模型进化优化

    支持场景:
    - fin_model: 模型结构进化 (探索新的模型架构)
    - fin_quant: 量化策略优化 (端到端策略+模型联合优化)
    """

    def __init__(self, config: Optional[RDAgentConfig] = None):
        self.config = config or RDAgentConfig()
        self.work_dir = Path(self.config.work_dir) / "model_evolution"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run_model_evolution(
        self,
        max_iterations: int = None,
        scenario: str = "fin_model",
        base_model: str = "lightgbm",
        target_metric: str = "IC",
    ) -> dict:
        """运行模型进化

        Args:
            max_iterations: 最大迭代次数
            scenario: RD-Agent场景 fin_model/fin_quant
            base_model: 基线模型名称
            target_metric: 优化目标指标 (IC/ICIR/Sharpe)

        Returns:
            进化结果，包含迭代历史和最佳模型配置
        """
        max_iter = max_iterations or self.config.max_iterations

        cmd = [
            "python", "-m", "rdagent.app.qlib_rd_loop",
            "--scenario", scenario,
            "--max_iterations", str(max_iter),
            "--work_dir", str(self.work_dir),
        ]

        env = self.config.to_env_dict()
        env["RD_AGENT_BASE_MODEL"] = base_model
        env["RD_AGENT_TARGET_METRIC"] = target_metric

        logger.info(
            f"Starting RD-Agent model evolution: scenario={scenario}, "
            f"base={base_model}, target={target_metric}, iter={max_iter}"
        )

        try:
            result = subprocess.run(
                cmd,
                env={**env},
                capture_output=True,
                text=True,
                timeout=7200,
            )

            if result.returncode != 0:
                logger.error(f"RD-Agent model evolution failed: {result.stderr[:500]}")
                return {"success": False, "error": result.stderr[:1000]}

            return self._parse_model_results(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("RD-Agent model evolution timed out (7200s)")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            logger.error(f"RD-Agent model evolution error: {e}")
            return {"success": False, "error": str(e)}

    def _parse_model_results(self, output: str) -> dict:
        """解析RD-Agent模型进化输出"""
        results = {
            "success": True,
            "iterations": [],
            "best_model": None,
            "best_metric": 0.0,
            "total_iterations": 0,
        }

        for line in output.strip().split("\n"):
            try:
                data = json.loads(line)
                if "model" in data or "model_config" in data:
                    iteration = {
                        "iteration": data.get("iteration", 0),
                        "model_name": data.get("model_name", ""),
                        "model_config": data.get("model_config", {}),
                        "ic": data.get("ic", 0),
                        "icir": data.get("icir", 0),
                        "sharpe": data.get("sharpe", 0),
                        "improvement": data.get("improvement", ""),
                    }
                    results["iterations"].append(iteration)
                    results["total_iterations"] += 1

                    metric_value = iteration.get("ic", 0)
                    if abs(metric_value) > abs(results["best_metric"]):
                        results["best_model"] = iteration
                        results["best_metric"] = metric_value

            except json.JSONDecodeError:
                continue

        logger.info(
            f"RD-Agent model evolution: {results['total_iterations']} iterations, "
            f"best IC={results['best_metric']:.4f}"
        )
        return results

    def get_model_config(self, results: dict) -> Optional[dict]:
        """提取最佳模型配置，可直接用于model_zoo.create_model()"""
        best = results.get("best_model")
        if not best:
            return None
        return {
            "model_name": best.get("model_name", "lightgbm"),
            "override_params": best.get("model_config", {}),
        }

    def run_quant_optimization(
        self,
        max_iterations: int = 5,
        factor_list: Optional[List[str]] = None,
    ) -> dict:
        """运行端到端量化策略优化(因子+模型联合)

        Args:
            max_iterations: 迭代次数
            factor_list: 初始因子列表(None则用Alpha158)

        Returns:
            优化结果
        """
        env = self.config.to_env_dict()
        if factor_list:
            env["RD_AGENT_FACTORS"] = json.dumps(factor_list)

        cmd = [
            "python", "-m", "rdagent.app.qlib_rd_loop",
            "--scenario", "fin_quant",
            "--max_iterations", str(max_iterations),
            "--work_dir", str(self.work_dir / "quant_opt"),
        ]

        logger.info(f"Starting RD-Agent quant optimization: {max_iterations} iterations")

        try:
            result = subprocess.run(
                cmd,
                env={**env},
                capture_output=True,
                text=True,
                timeout=10800,
            )

            if result.returncode != 0:
                logger.error(f"RD-Agent quant optimization failed: {result.stderr[:500]}")
                return {"success": False, "error": result.stderr[:1000]}

            return self._parse_quant_results(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("RD-Agent quant optimization timed out")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            logger.error(f"RD-Agent quant optimization error: {e}")
            return {"success": False, "error": str(e)}

    def _parse_quant_results(self, output: str) -> dict:
        """解析量化策略优化结果"""
        results = {
            "success": True,
            "strategies": [],
            "best_strategy": None,
            "best_sharpe": 0.0,
        }

        for line in output.strip().split("\n"):
            try:
                data = json.loads(line)
                if "strategy" in data or "sharpe" in data:
                    strategy = {
                        "name": data.get("strategy_name", ""),
                        "factors": data.get("factors", []),
                        "model": data.get("model", ""),
                        "sharpe": data.get("sharpe", 0),
                        "max_drawdown": data.get("max_drawdown", 0),
                        "annual_return": data.get("annual_return", 0),
                        "ic": data.get("ic", 0),
                    }
                    results["strategies"].append(strategy)
                    if strategy["sharpe"] > results["best_sharpe"]:
                        results["best_strategy"] = strategy
                        results["best_sharpe"] = strategy["sharpe"]
            except json.JSONDecodeError:
                continue

        logger.info(
            f"RD-Agent quant optimization: {len(results['strategies'])} strategies, "
            f"best Sharpe={results['best_sharpe']:.2f}"
        )
        return results
