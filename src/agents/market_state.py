"""市场状态识别器

自动识别市场环境: Bull / Bear / Sideways / High_Vol / Low_Vol

识别逻辑:
- 趋势: 20日均线 vs 60日均线 (金叉/死叉)
- 波动率: 20日收益率标准差
- 成交量: 当日量 vs 20日均量
- 综合: 趋势+波动率+成交量 → 市场状态分类
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from src.core import ensure_qlib, get_ohlcv
from .contracts import MarketState

logger = logging.getLogger(__name__)


class MarketStateRecognizer:
    """市场状态自动识别

    使用SH000300(沪深300)作为市场基准，通过趋势、波动率、成交量
    三维度综合判断当前市场环境。

    输出 → RiskAgent 用于动态仓位决策:
        Bull      → 100% 仓位
        Sideways  → 50% 仓位
        Bear/HighVol → 20% 仓位
    """

    BENCHMARK = "SH000300"

    def __init__(self, qlib_dir: str = "data/qlib_bin"):
        ensure_qlib(qlib_dir)

    def recognize(
        self, as_of_date: Optional[str] = None, lookback: int = 60
    ) -> MarketState:
        """识别当前市场状态

        Args:
            as_of_date: 截止日期，None则用最新数据
            lookback: 回看天数(需≥60)

        Returns:
            MarketState enum
        """
        try:
            df = get_ohlcv(self.BENCHMARK, start_time=None, end_time=as_of_date)
            if df is None or len(df) < lookback:
                logger.warning(f"Insufficient data for {self.BENCHMARK}, using default Sideways")
                return MarketState.SIDEWAYS

            df = df.tail(lookback)
            close = df["close"].values

            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:]) if len(close) >= 60 else np.mean(close)
            recent_close = close[-1]

            vol_20d = np.std(np.diff(np.log(close[-21:]))) * np.sqrt(252)
            vol_ratio = df["volume"].iloc[-1] / max(np.mean(df["volume"].iloc[-20:].values), 1)

            trend = 0.0
            if ma20 > ma60 * 1.02:
                trend = 1.0
            elif ma20 < ma60 * 0.98:
                trend = -1.0

            logger.info(
                f"Market: close={recent_close:.0f} MA20={ma20:.0f} MA60={ma60:.0f} "
                f"vol_20d={vol_20d:.4f} vol_ratio={vol_ratio:.2f} trend={trend}"
            )

            if trend > 0 and vol_20d < 0.25:
                return MarketState.BULL
            elif trend < 0 or vol_20d > 0.35:
                return MarketState.BEAR if trend < 0 else MarketState.HIGH_VOL
            elif vol_20d < 0.12:
                return MarketState.LOW_VOL
            else:
                return MarketState.SIDEWAYS

        except Exception as e:
            logger.error(f"Market state recognition failed: {e}")
            return MarketState.SIDEWAYS

    def get_signals(self, as_of_date: Optional[str] = None) -> dict:
        """获取市场信号详情(用于RiskAgent决策)"""
        try:
            df = get_ohlcv(self.BENCHMARK, start_time=None, end_time=as_of_date)
            if df is None or len(df) < 60:
                return {"trend": 0, "volatility": 0, "volume_ratio": 1, "state": "sideways"}

            close = df["close"].values
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:])
            trend = (ma20 - ma60) / ma60 if ma60 > 0 else 0
            vol_20d = float(np.std(np.diff(np.log(close[-21:]))) * np.sqrt(252))
            vol_ratio = float(
                df["volume"].iloc[-1] / max(np.mean(df["volume"].iloc[-20:].values), 1)
            )
            state = self.recognize(as_of_date)

            return {
                "trend": float(trend),
                "volatility": vol_20d,
                "volume_ratio": vol_ratio,
                "ma20": float(ma20),
                "ma60": float(ma60),
                "state": state.value,
            }
        except Exception as e:
            logger.error(f"Failed to get market signals: {e}")
            return {"trend": 0, "volatility": 0, "volume_ratio": 1, "state": "sideways"}
