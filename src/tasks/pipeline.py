from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from src.common.logger import logger
from src.common.redis_client import get_redis
from src.execution_engine.models import AlgoType
from src.qlib_engine.optimizer import ConvexOptimizer
from src.service.runtime import ServiceRuntime


class PremarketRebalancePipeline:
    """
    盘前自动调仓全流程流水线：
    1. 提取预测分数：从 Redis 或模型推理引擎提取最新 Alpha 预测排行；
    2. 筛选标的池：选取 Top-K 标的并构建预期 Alpha 向量；
    3. 凸优化求解：基于单票权重上限与风险暴露求解最优目标持仓比例；
    4. 执行调仓下单：移交 ExecutionCoordinator，支持即时或算法切片执行。
    """

    @classmethod
    def run(
        cls,
        strategy_id: str = "hft_stream_01",
        trade_date: Optional[str] = None,
        top_k: int = 10,
        max_stock_weight: float = 0.08,
        algo_type: str = "DIRECT",
        execution_mode: str = "SYNC",
        interval_seconds: float = 0.0,
        runtime: Optional[ServiceRuntime] = None,
        current_prices_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        target_date = trade_date or datetime.now().strftime("%Y-%m-%d")
        redis_client = get_redis()

        logger.info(f"[PremarketRebalance] Starting pre-market rebalance for {strategy_id} on {target_date}...")

        # 1. 提取 Alpha 预测分数
        pred_payload = None
        try:
            pred_payload = redis_client.get_json(f"predictions:{target_date}")
            if not pred_payload:
                pred_payload = redis_client.get_json("predictions:latest")
        except Exception as e:
            logger.warning(f"[PremarketRebalance] Redis query failed: {e}")

        predictions = []
        if pred_payload and isinstance(pred_payload, dict):
            predictions = pred_payload.get("predictions", [])

        # 备选回退：若无模型预测缓存，使用基准股票集模拟 Alpha 分数
        if not predictions:
            logger.info("[PremarketRebalance] No cached predictions; using default candidate pool.")
            candidates = ["600000.SH", "000001.SZ", "600036.SH", "000858.SZ", "601318.SH"]
            predictions = [
                {"symbol": s, "score": 1.0 - i * 0.1, "rank": i + 1}
                for i, s in enumerate(candidates[:top_k])
            ]

        # 2. 选取 Top-K 标的
        selected = predictions[:top_k]
        symbols = [p["symbol"] for p in selected]
        raw_scores = pd.Series([p["score"] for p in selected], index=symbols)

        # 3. 组合权重优化计算
        target_weights: Dict[str, float] = {}
        n_stocks = len(symbols)
        if n_stocks > 0 and n_stocks * max_stock_weight >= 1.0:
            try:
                exposures = pd.DataFrame(np.eye(n_stocks), index=symbols, columns=[f"fac_{i}" for i in range(n_stocks)])
                factor_cov = pd.DataFrame(np.eye(n_stocks) * 0.0004, index=exposures.columns, columns=exposures.columns)
                specific_var = pd.Series(0.0005, index=symbols)

                optimizer = ConvexOptimizer(
                    risk_aversion=1.0,
                    max_stock_weight=max_stock_weight,
                    industry_tolerance=None,
                    style_tolerance=None,
                )
                opt_res = optimizer.optimize(
                    alpha=raw_scores,
                    exposures=exposures,
                    factor_cov=factor_cov,
                    specific_var=specific_var,
                )
                w_dict = opt_res.get("weights", {})
                target_weights = {
                    k: round(float(min(v, max_stock_weight)), 4) for k, v in w_dict.items() if v > 1e-4
                }
            except Exception as opt_err:
                logger.warning(f"[PremarketRebalance] ConvexOptimizer failed ({opt_err}); applying score weighting.")

        if not target_weights and n_stocks > 0:
            # 无论标的数量多少，严格遵守 max_stock_weight 单票持仓上限（其余权益留存现金缓冲区）
            pos_scores = np.maximum(raw_scores.values, 0.01)
            max_s = np.max(pos_scores) if np.max(pos_scores) > 0 else 1.0
            norm_scores = pos_scores / max_s
            target_weights = {
                sym: round(float(min(score * max_stock_weight, max_stock_weight)), 4)
                for sym, score in zip(symbols, norm_scores)
            }

        # 4. 获取标的当前估值参考价
        current_prices: Dict[str, float] = {}
        acc = runtime.broker.get_account(strategy_id) if runtime else None

        for s in target_weights.keys():
            if current_prices_override and s in current_prices_override:
                current_prices[s] = float(current_prices_override[s])
            elif acc and s in acc.positions and acc.positions[s].last_price > 0:
                current_prices[s] = float(acc.positions[s].last_price)
            elif acc and s in acc.positions and acc.positions[s].avg_cost > 0:
                current_prices[s] = float(acc.positions[s].avg_cost)
            else:
                current_prices[s] = 100.0  # 默认参考基准价

        # 5. 调用执行引擎协调器进行调仓执行
        rebalance_result = None
        if runtime is not None:
            algo = (
                AlgoType(algo_type.upper())
                if algo_type.upper() in AlgoType._value2member_map_
                else AlgoType.DIRECT
            )
            with runtime.lock:
                rebalance_result = runtime.coordinator.execute_rebalance(
                    account_id=strategy_id,
                    target_weights=target_weights,
                    current_prices=current_prices,
                    algo_type=algo,
                    execution_mode=execution_mode,
                    interval_seconds=interval_seconds,
                )

        logger.info(f"[PremarketRebalance] Rebalance planning complete: {len(target_weights)} positions targeted.")
        return {
            "status": "SUCCESS",
            "strategy_id": strategy_id,
            "trade_date": target_date,
            "target_weights": target_weights,
            "current_prices": current_prices,
            "rebalance_result": rebalance_result,
        }


class DailySettlementPipeline:
    """
    盘后日终交收清算流水线：
    1. 解冻账户内 T+1 可用股份与交易资金；
    2. 同步并结转 PMS 组合层各策略持仓净值与收益率；
    3. 记录清算交收日志。
    """

    @classmethod
    def run(
        cls,
        account_id: Optional[str] = None,
        trade_date: Optional[str] = None,
        runtime: Optional[ServiceRuntime] = None,
    ) -> Dict[str, Any]:
        target_date = trade_date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"[DailySettlement] Starting daily settlement for date {target_date} (account: {account_id or 'ALL'})...")

        settled_accounts: List[str] = []
        if runtime is not None:
            with runtime.lock:
                # 1. 触发 Broker T+1 股份解冻
                runtime.broker.settle_overnight(account_id=account_id)

                # 2. 同步 PMS 账户组合层并记录日终净值快照
                runtime.pms_manager.sync_from_broker()
                if account_id:
                    runtime.pms_manager.record_daily_nav(account_id, date=target_date)
                    settled_accounts.append(account_id)
                else:
                    runtime.pms_manager.record_all_daily_nav(date=target_date)
                    settled_accounts = list(runtime.broker.accounts.keys())

        logger.info(f"[DailySettlement] Completed settlement for {len(settled_accounts)} accounts.")
        return {
            "status": "SUCCESS",
            "trade_date": target_date,
            "settled_accounts": settled_accounts,
            "message": f"Daily settlement completed for {len(settled_accounts)} accounts.",
        }
