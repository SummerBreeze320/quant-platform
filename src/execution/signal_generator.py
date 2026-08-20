"""信号生成层

使用Qlib表达式引擎计算因子，多因子合成 → 选股 → 组合优化 → 目标持仓。

七层架构中的第六层第一环：
模型/因子 → 信号生成 → 组合调度 → 交易网关 → 执行算法 → 风控 → 日志

工作流程：
1. 加载最新行情数据（通过Qlib D.features）
2. 计算因子值（Qlib表达式引擎，Alpha158风格）
3. 因子标准化（CSZScoreNorm截面标准化）
4. 多因子合成 → 综合得分
5. 截面选股 → Top N
6. 组合优化 → 目标权重
7. 输出信号（目标持仓）

信号在收盘后生成，次日开盘执行（T+1）。
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
import logging

from src.core import get_field, get_ohlcv, list_instruments, ensure_qlib, get_features
from src.research.factor_library import QLIB_EXPRESSION_FACTORS, FACTOR_CATEGORIES
from src.research.enhanced_optimizer import EnhancedIndexingOptimizer

logger = logging.getLogger(__name__)


class SignalGenerator:
    """信号生成器

    使用Qlib表达式引擎计算因子，多因子等权/IC加权合成，
    Top-N选股，EnhancedIndexing组合优化，输出目标持仓权重。

    可定时调用（如每日收盘后），生成次日交易信号。
    """

    def __init__(self,
                 qlib_dir: str = "data/qlib_bin",
                 combiner_method: str = "ic_weighted",
                 top_n: int = 5,
                 max_position: float = 0.15,
                 lookback: int = 252,
                 forward_period: int = 5,
                 factor_names: Optional[List[str]] = None):
        ensure_qlib(qlib_dir)
        self.combiner_method = combiner_method
        self.top_n = top_n
        self.max_position = max_position
        self.lookback = lookback
        self.forward_period = forward_period

        # 默认使用动量+波动率+成交量因子
        if factor_names is None:
            self.factor_names = FACTOR_CATEGORIES["momentum"] + FACTOR_CATEGORIES["volatility"][:3]
        else:
            self.factor_names = factor_names

        self.optimizer = EnhancedIndexingOptimizer(
            weights_limit=max_position,
            tracking_error_limit=0.10,
        )
        self._factor_weights: Optional[Dict[str, float]] = None
        self._last_signal: Optional[dict] = None

    def _load_price_data(self, codes: List[str],
                          end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """加载最新行情数据"""
        price_data = {}
        for code in codes:
            try:
                df = get_ohlcv(code)
                if df is None or df.empty:
                    continue
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]
                if len(df) > 20:
                    price_data[code] = df
            except Exception:
                pass
        return price_data

    def _calc_factors(self, codes: List[str],
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Qlib表达式引擎计算因子值

        Returns:
            MultiIndex DataFrame (instrument, datetime) with factor columns
        """
        expressions = []
        valid_names = []
        for name in self.factor_names:
            if name in QLIB_EXPRESSION_FACTORS:
                expressions.append(QLIB_EXPRESSION_FACTORS[name])
                valid_names.append(name)

        if not expressions:
            return pd.DataFrame()

        df = get_features(codes, expressions, end_time=end_date)
        if df is not None and not df.empty:
            df.columns = valid_names
        return df

    def _normalize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """截面标准化（CSZScoreNorm的简化版）"""
        if factor_df.empty:
            return factor_df

        def cross_sectional_norm(s):
            mean = s.mean()
            std = s.std()
            if std > 0:
                return (s - mean) / std
            return s - mean

        return factor_df.groupby(level="datetime").transform(cross_sectional_norm)

    def _calc_forward_returns(self, codes: List[str], period: int) -> pd.DataFrame:
        """计算前瞻收益（用于IC计算和因子权重）"""
        returns = {}
        for code in codes:
            df = get_ohlcv(code)
            if df is not None and not df.empty and len(df) > period:
                ret = df["close"].pct_change(period).shift(-period)
                returns[code] = ret
        return pd.DataFrame(returns)

    def _combine_factors(self, factor_df: pd.DataFrame,
                          fwd_returns: pd.DataFrame) -> pd.Series:
        """多因子合成"""
        if factor_df.empty:
            return pd.Series(dtype=float)

        if self.combiner_method == "equal_weight":
            # 等权合成
            combined = factor_df.mean(axis=1)
            combined = combined.groupby(level="datetime").transform(
                lambda x: x.rank(pct=True) - 0.5
            )
            return combined

        elif self.combiner_method == "ic_weighted":
            # IC加权合成
            ic_weights = {}
            for col in factor_df.columns:
                # 计算滚动IC
                aligned = factor_df[col].dropna()
                ic_values = []
                for date in aligned.index.get_level_values("datetime").unique():
                    cs = aligned.xs(date, level="datetime")
                    if len(cs) < 3:
                        continue
                    fwd_cs = fwd_returns.reindex(cs.index).iloc[:, 0] if not fwd_returns.empty else None
                    if fwd_cs is not None and len(fwd_cs.dropna()) > 3:
                        common = cs.index.intersection(fwd_cs.dropna().index)
                        if len(common) > 3:
                            ic = cs.loc[common].corr(fwd_cs.loc[common])
                            ic_values.append(ic)
                avg_ic = np.mean(ic_values) if ic_values else 0
                ic_weights[col] = avg_ic

            # 归一化权重
            total_ic = sum(abs(v) for v in ic_weights.values())
            if total_ic > 0:
                normalized = {k: v / total_ic for k, v in ic_weights.items()}
            else:
                normalized = {k: 1.0 / len(ic_weights) for k in ic_weights}

            self._factor_weights = normalized
            combined = sum(factor_df[col] * normalized.get(col, 0) for col in factor_df.columns)
            combined = combined.groupby(level="datetime").transform(
                lambda x: x.rank(pct=True) - 0.5
            )
            return combined

        else:
            combined = factor_df.mean(axis=1)
            return combined

    def generate(self,
                 codes: Optional[List[str]] = None,
                 as_of_date: Optional[str] = None,
                 objective: str = "equal_weight") -> dict:
        """生成交易信号

        Args:
            codes: 股票池（None则自动选取）
            as_of_date: 信号生成日期（None则使用最新日期）
            objective: 权重优化目标

        Returns:
            {signal_date, target_weights, factor_scores, factor_weights, metadata}
        """
        # 1. 确定股票池
        if codes is None:
            all_inst = list_instruments()
            codes = [c for c in all_inst if c.startswith("SH6") or c.startswith("SZ0")][:15]

        logger.info(f"信号生成: {len(codes)}只股票, 日期={as_of_date or 'latest'}")

        # 2. 加载数据
        price_data = self._load_price_data(codes, as_of_date)
        if not price_data:
            return {"error": "无可用数据"}
        logger.info(f"加载{len(price_data)}只股票数据")

        # 3. 计算因子（Qlib表达式引擎）
        factor_df = self._calc_factors(list(price_data.keys()), as_of_date)
        if factor_df is None or factor_df.empty:
            return {"error": "因子计算失败"}
        logger.info(f"计算{factor_df.shape[1]}个因子")

        # 4. 因子标准化
        normalized = self._normalize_factors(factor_df)

        # 5. 前瞻收益（用于IC加权）
        fwd_returns = self._calc_forward_returns(list(price_data.keys()), self.forward_period)

        # 6. 多因子合成
        scores = self._combine_factors(normalized, fwd_returns)
        if scores.empty:
            return {"error": "因子合成失败"}

        # 7. 获取最新截面得分
        if as_of_date:
            latest_date = pd.Timestamp(as_of_date)
        else:
            latest_date = scores.index.get_level_values("datetime").max()

        if latest_date not in scores.index.get_level_values("datetime"):
            dates = scores.index.get_level_values("datetime")
            latest_date = dates[dates <= latest_date].max()

        latest_scores = scores.xs(latest_date, level="datetime") if isinstance(
            scores.index, pd.MultiIndex
        ) else scores.tail(1)

        if len(latest_scores) < 3:
            return {"error": "有效得分不足"}

        # 8. 选股
        top_stocks = latest_scores.nlargest(min(self.top_n, len(latest_scores)))

        # 9. 组合优化
        if objective == "equal_weight":
            weights = pd.Series(
                1.0 / len(top_stocks),
                index=top_stocks.index,
            )
        elif objective == "score_weighted":
            positive = top_stocks.clip(lower=0)
            if positive.sum() > 0:
                weights = positive / positive.sum()
            else:
                weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
        elif objective == "enhanced_indexing":
            # 使用EnhancedIndexing优化器
            returns_data = {}
            for code in top_stocks.index:
                df = get_ohlcv(str(code))
                if df is not None and not df.empty:
                    returns_data[str(code)] = df["close"].pct_change()
            returns_df = pd.DataFrame(returns_data).tail(60)

            if not returns_df.empty:
                cov_matrix = returns_df.cov()
                factor_matrix = pd.DataFrame(
                    {col: factor_df[col].xs(latest_date, level="datetime")
                     for col in factor_df.columns
                     if latest_date in factor_df.index.get_level_values("datetime")}
                ).reindex(top_stocks.index)
                weights = self.optimizer.optimize(
                    pred_scores=top_stocks,
                    factor_matrix=factor_matrix,
                    cov_matrix=cov_matrix,
                )
            else:
                weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
        else:
            weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)

        weights = weights[weights > 1e-6]

        # 10. 信号输出
        signal = {
            "signal_date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, 'strftime') else str(latest_date),
            "generated_at": datetime.now().isoformat(),
            "target_weights": {str(k): float(v) for k, v in weights.items()},
            "factor_scores": {str(k): float(v) for k, v in latest_scores.items()},
            "factor_weights": {k: float(v) for k, v in (self._factor_weights or {}).items()},
            "n_selected": len(weights),
            "n_universe": len(price_data),
            "combiner_method": self.combiner_method,
            "objective": objective,
            "factor_names": self.factor_names,
        }

        self._last_signal = signal
        logger.info(f"信号生成完成: {signal['n_selected']}只目标持仓, "
                     f"日期={signal['signal_date']}")
        return signal

    @property
    def last_signal(self) -> Optional[dict]:
        return self._last_signal
