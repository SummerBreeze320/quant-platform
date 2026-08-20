"""宏观经济数据处理器

从Wind获取宏观经济指标，包括：
1. GDP/CPI/PPI/PMI 等核心宏观指标
2. 货币供应量(M0/M1/M2)
3. 利率（Shibor/LPR/国债收益率）
4. 汇率、大宗商品价格

数据流：Wind EDB → Parquet存储 → 宏观因子计算
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/alternative/macro")

MACRO_INDICATORS = {
    "gdp_yoy": "M0000752",
    "cpi_yoy": "M0000612",
    "ppi_yoy": "M0001226",
    "pmi_mfg": "M0017126",
    "pmi_non_mfg": "M0017132",
    "m2_yoy": "M0041718",
    "m1_yoy": "M0041712",
    "shibor_3m": "M0056725",
    "lpr_1y": "M0061720",
    "cny_usd": "M0066765",
    "brent_oil": "S0069858",
    "gold": "S0058096",
    "copper": "S0027807",
    "rebar": "S5033669",
    "treasury_10y": "G0000170",
}


class MacroHandler:
    """宏观经济数据获取与因子计算

    Wind API:
        w.edb("M0000612,M0001226", "2020-01-01", "2024-12-31")
        获取CPI/PPI等宏观指标时间序列

    Usage:
        handler = MacroHandler()
        handler.fetch_macro(indicators=["cpi_yoy", "ppi_yoy"], start="2020-01-01")
        factor = handler.compute_macro_factor(end_date="2024-06-01")
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_macro(
        self,
        indicators: List[str] = None,
        start: str = "2015-01-01",
        end: str = None,
    ) -> pd.DataFrame:
        """从Wind EDB获取宏观经济指标

        Args:
            indicators: 指标名称列表，None则获取全部
            start: 起始日期
            end: 结束日期
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        if indicators is None:
            indicators = list(MACRO_INDICATORS.keys())

        wind_codes = [MACRO_INDICATORS.get(name, name) for name in indicators]

        try:
            from WindPy import w
            if not w.isconnected():
                w.start()

            result = w.edb(
                ",".join(wind_codes),
                start,
                end,
                "Fill=Previous",
            )
            if result.ErrorCode != 0:
                logger.error(f"Wind EDB error: {result.ErrorCode}")
                return pd.DataFrame()

            df = pd.DataFrame(
                {name: result.Data[i] for i, name in enumerate(indicators)},
                index=pd.to_datetime(result.Times),
            )
            df.index.name = "date"

            self._save(df, "macro_raw")
            logger.info(f"Fetched {len(indicators)} macro indicators, {len(df)} rows")
            return df

        except ImportError:
            logger.warning("WindPy not installed — run mock generator for offline dev")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Macro fetch error: {e}")
            return pd.DataFrame()

    def compute_macro_factor(
        self,
        end_date: str,
        lookback: int = 3,
    ) -> Dict[str, float]:
        """计算宏观趋势因子

        因子逻辑：
        - 指标同比变化 → macro_yoy
        - 指标环比变化 → macro_mom
        - 指标趋势（连续上行/下行） → macro_trend

        Returns:
            Dict[str, float]: 因子名 → 因子值
        """
        df = self._load("macro_raw")
        if df is None or df.empty:
            return {}

        end_dt = pd.Timestamp(end_date)
        recent = df[df.index <= end_dt].tail(lookback + 1)

        if len(recent) < 2:
            return {}

        factors = {}
        for col in df.columns:
            series = recent[col].dropna()
            if len(series) < 2:
                continue

            latest = series.iloc[-1]
            prev = series.iloc[-2]
            if pd.isna(latest) or pd.isna(prev) or prev == 0:
                continue

            mom = (latest - prev) / abs(prev) if prev != 0 else 0

            if len(series) >= 3:
                first = series.iloc[0]
                trend = (latest - first) / abs(first) if first != 0 else 0
            else:
                trend = mom

            factors[f"macro_{col}_mom"] = float(mom)
            factors[f"macro_{col}_trend"] = float(trend)

        return factors

    def compute_macro_surprise_factor(
        self,
        end_date: str,
        forecast_col: str = "cpi_yoy",
    ) -> float:
        """计算宏观预期差因子（实际值vs预期值）"""
        df = self._load("macro_raw")
        if df is None or df.empty:
            return 0.0

        if forecast_col not in df.columns:
            return 0.0

        end_dt = pd.Timestamp(end_date)
        recent = df[df.index <= end_dt][forecast_col].dropna().tail(2)
        if len(recent) < 2:
            return 0.0

        surprise = (recent.iloc[-1] - recent.iloc[-2]) / abs(recent.iloc[-2]) if recent.iloc[-2] != 0 else 0
        return float(surprise)

    def list_indicators(self) -> Dict[str, str]:
        """列出所有可用宏观指标"""
        return MACRO_INDICATORS.copy()

    def _save(self, df: pd.DataFrame, name: str):
        df.to_parquet(self.data_dir / f"{name}.parquet")

    def _load(self, name: str) -> Optional[pd.DataFrame]:
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)
