"""RD-Agent因子挖掘运行器"""
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional
from .config import RDAgentConfig

logger = logging.getLogger(__name__)


class FactorRunner:
    """调用RD-Agent进行自动因子挖掘"""

    def __init__(self, config: Optional[RDAgentConfig] = None):
        self.config = config or RDAgentConfig()
        self.work_dir = Path(self.config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run_factor_mining(
        self,
        max_iterations: int = None,
        scenario: str = "fin_factor",
    ) -> dict:
        """运行因子挖掘

        Args:
            max_iterations: 最大迭代次数
            scenario: RD-Agent场景 fin_factor/fin_model/fin_quant

        Returns:
            运行结果
        """
        max_iter = max_iterations or self.config.max_iterations

        cmd = [
            "python", "-m", "rdagent.app.qlib_rd_loop",
            "--scenario", scenario,
            "--max_iterations", str(max_iter),
            "--work_dir", str(self.work_dir),
        ]

        env = self.config.to_env_dict()

        logger.info(f"Starting RD-Agent factor mining: {scenario}, {max_iter} iterations")

        try:
            result = subprocess.run(
                cmd,
                env={**env},
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if result.returncode != 0:
                logger.error(f"RD-Agent failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

            return self._parse_results(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("RD-Agent timed out")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            logger.error(f"RD-Agent error: {e}")
            return {"success": False, "error": str(e)}

    def _parse_results(self, output: str) -> dict:
        """解析RD-Agent输出"""
        results = {
            "success": True,
            "factors": [],
            "iterations": 0,
        }

        for line in output.strip().split("\n"):
            try:
                data = json.loads(line)
                if "factor" in data:
                    results["factors"].append({
                        "name": data.get("factor_name", ""),
                        "expression": data.get("expression", ""),
                        "ic": data.get("ic", 0),
                        "icir": data.get("icir", 0),
                    })
                    results["iterations"] += 1
            except json.JSONDecodeError:
                continue

        logger.info(f"RD-Agent produced {len(results['factors'])} factors")
        return results

    def get_best_factors(self, results: dict, min_ic: float = 0.03) -> list:
        """筛选有效因子"""
        factors = results.get("factors", [])
        valid = [f for f in factors if abs(f.get("ic", 0)) >= min_ic]
        valid.sort(key=lambda x: abs(x.get("ic", 0)), reverse=True)
        return valid
