"""BacktestAgent — 批量回测调度 + 综合评分 + 交易成本

执行回测实验，计算全量绩效指标，输出综合评分。

P5增强:
- 交易成本计算 (佣金+印花税+滑点)
- 胜率纳入评分公式
- IC权重提升 (因子质量更重要)
- 扣除成本后净收益

包装现有模块:
- src.backtest.backtest_engine (BacktestEngine)
- src.backtest.metrics (calc_all_metrics)
- src.backtest.risk_metrics (calc_all_risk_metrics)
"""
import logging
from typing import Optional, List
from datetime import datetime
from .base import BaseAgent
from .contracts import BacktestRequest, BacktestResult, BacktestMetrics

logger = logging.getLogger(__name__)

# P5增强: 评分权重 — 胜率+IC纳入, 交易成本惩罚
SCORING_WEIGHTS = {
    "annual_return": 0.25,
    "sharpe_ratio": 0.20,
    "max_drawdown": -0.15,
    "turnover": -0.05,
    "win_rate": 0.10,
    "ic_mean": 0.15,
    "transaction_cost": -0.10,
}

# A股交易成本参数
COMMISSION_RATE = 0.0003    # 佣金率 (万三, 双向)
STAMP_TAX_RATE = 0.001      # 印花税 (千一, 卖出)
SLIPPAGE_RATE = 0.0005      # 滑点估算 (万五)


class BacktestAgent(BaseAgent):
    """批量回测Agent

    职责:
    1. 接收BacktestRequest，调度Qlib回测引擎
    2. 计算全量绩效指标(年化/夏普/回撤/胜率/换手率/IC/交易成本)
    3. 综合评分: 多因子加权评分
    4. 输出BacktestResult

    输入: BacktestRequest
    输出: BacktestResult (含metrics + score + 净值序列 + 交易成本)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="BacktestAgent", config=config)

    def run(self, input_data: BacktestRequest) -> BacktestResult:
        """执行回测"""
        try:
            from src.backtest.backtest_engine import BacktestEngine
            from src.core import ensure_qlib

            ensure_qlib()
            engine = BacktestEngine(qlib_dir="data/qlib_bin")

            result = engine.run_backtest(
                strategy_config={
                    "model_type": input_data.model_type,
                    "model_params": input_data.model_params,
                    "n_holdings": input_data.n_holdings,
                    "weight_scheme": input_data.weight_scheme,
                    "rebalance_freq": input_data.rebalance_freq,
                    "max_position": input_data.max_position,
                    "tracking_error_limit": input_data.tracking_error_limit,
                    "factor_names": input_data.factor_names,
                },
                start_date=input_data.start_date,
                end_date=input_data.end_date,
                benchmark=input_data.benchmark,
                init_cash=input_data.init_cash,
            )

            metrics = self._extract_metrics(result)
            metrics = self._add_transaction_cost(metrics, input_data.init_cash)
            score = self._compute_score(metrics)

            net_values = []
            drawdowns = []
            if isinstance(result, dict):
                net_values = result.get("net_values", [])
                drawdowns = result.get("drawdowns", [])
            elif hasattr(result, "net_values"):
                net_values = list(result.net_values)

            return BacktestResult(
                experiment_id=input_data.experiment_id,
                metrics=metrics,
                score=score,
                net_value_series=net_values,
                drawdown_series=drawdowns,
                status="completed",
            )

        except Exception as e:
            logger.error(f"Backtest failed for {input_data.experiment_id}: {e}")
            return BacktestResult(
                experiment_id=input_data.experiment_id,
                status="error",
                error=str(e),
            )

    def _extract_metrics(self, result) -> BacktestMetrics:
        """从回测结果提取指标"""
        if isinstance(result, dict):
            m = result.get("metrics", result)
            if not isinstance(m, dict):
                m = {}
            g = lambda k, d=0: m.get(k, result.get(k, d)) or d
            return BacktestMetrics(
                annual_return=g("annual_return"),
                sharpe_ratio=g("sharpe_ratio"),
                max_drawdown=g("max_drawdown"),
                sortino_ratio=g("sortino_ratio"),
                calmar_ratio=g("calmar_ratio"),
                turnover=g("turnover"),
                win_rate=g("win_rate"),
                ic_mean=g("ic_mean"),
                icir=g("icir"),
                information_ratio=g("information_ratio"),
                tracking_error=g("tracking_error"),
            )

        from src.backtest.metrics import calc_all_metrics
        try:
            raw = calc_all_metrics(
                returns=getattr(result, "returns", None),
                positions=getattr(result, "positions", None),
                benchmark=getattr(result, "benchmark", None),
            )
            return BacktestMetrics(**raw)
        except Exception:
            return BacktestMetrics()

    def _add_transaction_cost(self, metrics: BacktestMetrics, init_cash: float = 1_000_000) -> BacktestMetrics:
        """计算交易成本并更新指标

        A股交易成本 = 佣金(万三双向) + 印花税(千一卖出) + 滑点(万五)
        年化交易成本 = 换手率 * (佣金率*2 + 印花税 + 滑点*2) * 资金
        """
        turnover = metrics.turnover or 0
        cost_rate = COMMISSION_RATE * 2 + STAMP_TAX_RATE + SLIPPAGE_RATE * 2
        annual_cost_pct = turnover * cost_rate

        metrics.transaction_cost = round(annual_cost_pct, 6)
        metrics.net_return_after_cost = round(
            metrics.annual_return - annual_cost_pct, 6
        )

        logger.info(
            f"Transaction cost: turnover={turnover:.2%} cost={annual_cost_pct:.4%} "
            f"net_return={metrics.net_return_after_cost:.2%}"
        )
        return metrics

    def _compute_score(self, metrics: BacktestMetrics) -> float:
        """综合评分函数 — 多因子加权

        P5增强:
        - 胜率纳入 (10%权重)
        - IC纳入 (15%权重, 因子质量)
        - 交易成本惩罚 (10%权重)
        - 年化收益权重降低 (25%, 避免过度追求收益)
        """
        score = 0.0
        score += SCORING_WEIGHTS["annual_return"] * metrics.annual_return
        score += SCORING_WEIGHTS["sharpe_ratio"] * metrics.sharpe_ratio
        score += SCORING_WEIGHTS["max_drawdown"] * metrics.max_drawdown
        score += SCORING_WEIGHTS["turnover"] * metrics.turnover
        score += SCORING_WEIGHTS["win_rate"] * metrics.win_rate
        score += SCORING_WEIGHTS["ic_mean"] * metrics.ic_mean
        score += SCORING_WEIGHTS["transaction_cost"] * metrics.transaction_cost
        return round(score, 4)
