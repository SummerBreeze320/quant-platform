from typing import Dict, Any, Tuple, Optional
from src.qlib_engine.factor_handler import FactorHandler
from src.common.logger import logger

class SandboxEvaluator:
    """Evaluates candidate factor expressions in an isolated sandbox against quality thresholds."""

    def __init__(
        self,
        min_abs_ic: float = 0.035,
        min_icir: float = 0.5,
        max_correlation: float = 0.6
    ):
        self.min_abs_ic = min_abs_ic
        self.min_icir = min_icir
        self.max_correlation = max_correlation

    def check_quality_gate(
        self,
        metrics: Dict[str, Any],
        correlation_with_existing: float = 0.0
    ) -> Tuple[bool, str]:
        """Validates factor metrics against institutional quality gates."""
        rank_ic = abs(metrics.get("rank_ic", 0.0))
        icir = metrics.get("icir", 0.0)

        if rank_ic < self.min_abs_ic:
            return False, f"Absolute Rank IC ({rank_ic:.4f}) is below threshold ({self.min_abs_ic})"

        if icir < self.min_icir:
            return False, f"ICIR ({icir:.4f}) is below threshold ({self.min_icir})"

        if correlation_with_existing >= self.max_correlation:
            return False, f"Correlation with existing factors ({correlation_with_existing:.2f}) is too high (>= {self.max_correlation})"

        return True, "Passed all quality criteria."

    def evaluate_expression(
        self,
        expression: str,
        instruments: Optional[list] = None,
        start_time: str = "2024-01-01",
        end_time: str = "2026-09-01"
    ) -> Dict[str, Any]:
        """
        Executes the expression on Qlib data and returns statistical performance.
        Falls back to safe default metrics if dataset is uninitialized.
        """
        try:
            from qlib.data import D
            inst = instruments or ["SZ000001", "SH600000"]
            label_expr = "Ref($close, -1) / $close - 1"
            
            df = D.features(inst, [expression, label_expr], start_time=start_time, end_time=end_time)
            df = df.reset_index().rename(columns={"instrument": "symbol", "datetime": "date"})
            
            cols = [c for c in df.columns if c not in ["symbol", "date"]]
            f_col = cols[0]
            l_col = cols[1]

            return FactorHandler.evaluate_factor_performance(df, factor_col=f_col, label_col=l_col)
        except Exception as e:
            logger.warning(f"Qlib sandbox evaluation error: {e}")
            return {
                "rank_ic": 0.0,
                "normal_ic": 0.0,
                "icir": 0.0,
                "t_stat": 0.0,
                "positive_ic_ratio": 0.0,
                "error": str(e)
            }
