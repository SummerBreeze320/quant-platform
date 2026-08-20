"""RD-Agent深度集成协调器

统一管理因子挖掘和模型进化的完整循环:
1. RD-Agent因子挖掘 → 新因子表达式
2. 因子IC验证 → 筛选有效因子
3. RD-Agent模型进化 → 新模型配置
4. Qlib回测验证 → 最优策略组合
5. 工作空间持久化 → 迭代历史记录

支持定时任务和手动触发。
"""
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

from .config import RDAgentConfig
from .factor_runner import FactorRunner
from .model_runner import ModelRunner

logger = logging.getLogger(__name__)


class RDAgentCoordinator:
    """RD-Agent深度集成协调器

    编排因子挖掘和模型进化的完整循环，管理工作空间，
    追踪迭代历史，并提供最优配置输出。

    Usage:
        coordinator = RDAgentCoordinator()
        result = coordinator.run_evolution_loop(max_rounds=3)
        best_config = coordinator.get_best_config()
    """

    def __init__(self, config: Optional[RDAgentConfig] = None):
        self.config = config or RDAgentConfig()
        self.work_dir = Path(self.config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.factor_runner = FactorRunner(self.config)
        self.model_runner = ModelRunner(self.config)

        self.history_file = self.work_dir / "evolution_history.json"
        self.workspace: Dict = self._load_workspace()

    def run_evolution_loop(
        self,
        max_rounds: int = 3,
        factor_iterations: int = 5,
        model_iterations: int = 5,
        min_factor_ic: float = 0.03,
        auto_apply: bool = True,
    ) -> Dict:
        """运行完整的进化循环

        流程:
        Round 1: 因子挖掘 → IC验证 → 模型进化 → 回测验证
        Round 2: 基于上轮结果调整因子集 → 模型进化 → 回测验证
        Round N: ...

        Args:
            max_rounds: 进化轮次
            factor_iterations: 每轮因子挖掘迭代次数
            model_iterations: 每轮模型进化迭代次数
            min_factor_ic: 因子IC筛选阈值
            auto_apply: 是否自动应用最优配置

        Returns:
            进化结果摘要
        """
        result = {
            "start_time": datetime.now().isoformat(),
            "rounds": [],
            "best_config": None,
            "best_metric": 0.0,
            "total_factors_discovered": 0,
            "total_models_evolved": 0,
        }

        current_factors: List[Dict] = []
        current_model: str = "lightgbm"

        for round_idx in range(max_rounds):
            round_num = round_idx + 1
            logger.info(f"=== Evolution Round {round_num}/{max_rounds} ===")

            round_result = {
                "round": round_num,
                "factors": [],
                "models": [],
                "best_ic": 0,
                "best_sharpe": 0,
            }

            # Step 1: Factor mining
            logger.info(f"[Round {round_num}] Factor mining ({factor_iterations} iterations)...")
            factor_result = self.factor_runner.run_factor_mining(
                max_iterations=factor_iterations,
                scenario="fin_factor",
            )

            if factor_result.get("success"):
                new_factors = self.factor_runner.get_best_factors(
                    factor_result, min_ic=min_factor_ic
                )
                round_result["factors"] = new_factors
                current_factors.extend(new_factors)
                result["total_factors_discovered"] += len(new_factors)
                logger.info(f"[Round {round_num}] Discovered {len(new_factors)} effective factors")
            else:
                logger.warning(f"[Round {round_num}] Factor mining failed: {factor_result.get('error')}")

            # Step 2: Model evolution
            logger.info(f"[Round {round_num}] Model evolution ({model_iterations} iterations)...")
            model_result = self.model_runner.run_model_evolution(
                max_iterations=model_iterations,
                scenario="fin_model",
                base_model=current_model,
            )

            if model_result.get("success"):
                round_result["models"] = model_result.get("iterations", [])
                result["total_models_evolved"] += len(model_result.get("iterations", []))

                best_model = model_result.get("best_model")
                if best_model:
                    round_result["best_ic"] = best_model.get("ic", 0)
                    model_config = self.model_runner.get_model_config(model_result)
                    if model_config:
                        current_model = model_config.get("model_name", current_model)
                        if abs(best_model.get("ic", 0)) > abs(result["best_metric"]):
                            result["best_config"] = {
                                "round": round_num,
                                "factors": current_factors[-10:],
                                "model": model_config,
                                "ic": best_model.get("ic", 0),
                                "icir": best_model.get("icir", 0),
                            }
                            result["best_metric"] = best_model.get("ic", 0)

                logger.info(f"[Round {round_num}] Best model IC: {round_result['best_ic']:.4f}")
            else:
                logger.warning(f"[Round {round_num}] Model evolution failed: {model_result.get('error')}")

            result["rounds"].append(round_result)

            # Save workspace state
            self._save_round(round_result)

        result["end_time"] = datetime.now().isoformat()

        if auto_apply and result["best_config"]:
            self._apply_config(result["best_config"])

        logger.info(
            f"Evolution complete: {result['total_factors_discovered']} factors, "
            f"{result['total_models_evolved']} models, best IC={result['best_metric']:.4f}"
        )

        # Persist full history
        self._save_history(result)

        return result

    def get_best_config(self) -> Optional[Dict]:
        """获取历史最优配置"""
        if self.workspace.get("best_config"):
            return self.workspace["best_config"]
        return None

    def get_factor_history(self) -> List[Dict]:
        """获取因子挖掘历史"""
        return self.workspace.get("factors", [])

    def get_model_history(self) -> List[Dict]:
        """获取模型进化历史"""
        return self.workspace.get("models", [])

    def run_factor_only(self, max_iterations: int = 10) -> Dict:
        """仅运行因子挖掘"""
        logger.info(f"Factor-only mining: {max_iterations} iterations")
        return self.factor_runner.run_factor_mining(max_iterations=max_iterations)

    def run_model_only(
        self,
        max_iterations: int = 5,
        base_model: str = "lightgbm",
    ) -> Dict:
        """仅运行模型进化"""
        logger.info(f"Model-only evolution: {max_iterations} iterations, base={base_model}")
        return self.model_runner.run_model_evolution(
            max_iterations=max_iterations,
            base_model=base_model,
        )

    def _load_workspace(self) -> Dict:
        """加载工作空间状态"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load workspace: {e}")
        return {"rounds": [], "factors": [], "models": [], "best_config": None}

    def _save_round(self, round_result: Dict):
        """保存单轮结果到工作空间"""
        self.workspace["rounds"].append({
            "round": round_result["round"],
            "timestamp": datetime.now().isoformat(),
            "factor_count": len(round_result.get("factors", [])),
            "model_count": len(round_result.get("models", [])),
            "best_ic": round_result.get("best_ic", 0),
        })

        self.workspace["factors"].extend(round_result.get("factors", []))
        self.workspace["models"].extend(round_result.get("models", []))

    def _save_history(self, full_result: Dict):
        """保存完整历史"""
        self.workspace["best_config"] = full_result.get("best_config")
        self.workspace["last_run"] = full_result.get("end_time")
        self.workspace["total_factors"] = full_result.get("total_factors_discovered", 0)
        self.workspace["total_models"] = full_result.get("total_models_evolved", 0)

        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.workspace, f, ensure_ascii=False, indent=2)
            logger.info(f"Evolution history saved: {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def _apply_config(self, config: Dict):
        """应用最优配置到平台(更新config文件)"""
        config_path = Path("config/qlib_config.yaml")
        if not config_path.exists():
            return

        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                qlib_config = yaml.safe_load(f)

            model_config = config.get("model", {})
            if model_config:
                model_name = model_config.get("model_name", "lightgbm")
                params = model_config.get("override_params", {})
                if model_name in qlib_config.get("qlib", {}).get("model", {}):
                    qlib_config["qlib"]["model"][model_name].update(params)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(qlib_config, f, allow_unicode=True, default_flow_style=False)

            logger.info(f"Applied best config to qlib_config.yaml (model={model_config.get('model_name')})")
        except Exception as e:
            logger.warning(f"Failed to apply config: {e}")
