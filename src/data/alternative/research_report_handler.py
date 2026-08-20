"""研报评级与一致预期数据处理器

从Wind获取券商研报数据，包括：
1. 个股评级（买入/增持/中性/减持）
2. 目标价
3. 一致预期EPS/Revenue（卖方共识预测）

数据流：Wind WSS/WSET → Parquet存储 → 一致预期因子计算
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/alternative/research_reports")

RATING_MAP = {
    "买入": 2, "增持": 1, "推荐": 1, "中性": 0,
    "持有": 0, "减持": -1, "卖出": -2, "回避": -2,
}

WIND_RATING_FIELDS = [
    "rating_latest",
    "rating_change",
    "target_price",
    "west_avgeps_fy1",
    "west_avgrev_fy1",
    "west_avg_rating",
    "rating_count",
]


class ResearchReportHandler:
    """研报评级数据获取与一致预期因子计算

    Wind API:
        w.wss(codes, "rating_latest,target_price,west_avgeps_fy1,west_avg_rating")
        获取最新评级、目标价、一致预期EPS、卖方一致评级

    Usage:
        handler = ResearchReportHandler()
        handler.fetch_ratings(codes=["SH600000"])
        factor = handler.compute_consensus_factor(codes)
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ratings(self, codes: List[str], trade_date: str = None) -> pd.DataFrame:
        """从Wind获取研报评级/一致预期数据

        使用WSS截面数据函数，一次获取所有标的的评级信息
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        try:
            from WindPy import w
            if not w.isconnected():
                w.start()

            wind_codes = [self._to_wind_code(c) for c in codes]
            fields = ";".join(WIND_RATING_FIELDS)

            result = w.wss(
                wind_codes,
                fields,
                f"tradeDate={trade_date}",
            )
            if result.ErrorCode != 0:
                logger.error(f"Wind WSS error: {result.ErrorCode}")
                return pd.DataFrame()

            df = pd.DataFrame(result.Data, index=result.Fields).T
            df.columns = result.Fields
            df["code"] = codes
            df["trade_date"] = trade_date

            self._save(df, "ratings_raw")
            logger.info(f"Fetched ratings for {len(codes)} codes")
            return df

        except ImportError:
            logger.warning("WindPy not installed — run mock generator for offline dev")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Rating fetch error: {e}")
            return pd.DataFrame()

    def compute_consensus_factor(self, codes: List[str]) -> pd.Series:
        """计算一致预期评级因子

        因子逻辑：
        - 将文字评级映射为数值 → consensus_rating [-2, 2]
        - 评级变化 → rating_momentum
        - 评级覆盖数 → analyst_attention
        """
        df = self._load("ratings_raw")
        if df is None or df.empty:
            return pd.Series(dtype=float)

        results = {}
        for _, row in df.iterrows():
            code = row.get("code")
            if code not in codes:
                continue

            rating_col = self._find_rating_col(df)
            if rating_col is None:
                results[code] = 0.0
                continue

            rating_text = str(row.get(rating_col, ""))
            rating_score = RATING_MAP.get(rating_text, 0)
            results[code] = float(rating_score)

        return pd.Series(results, name="consensus_rating")

    def compute_target_price_factor(
        self, codes: List[str], current_prices: Dict[str, float] = None
    ) -> pd.Series:
        """计算目标价上行空间因子

        factor = (target_price - current_price) / current_price
        """
        df = self._load("ratings_raw")
        if df is None or df.empty:
            return pd.Series(dtype=float)

        results = {}
        tp_col = None
        for col in ["target_price", "west_target_price", "targetprice"]:
            if col in df.columns:
                tp_col = col
                break

        if tp_col is None:
            return pd.Series(dtype=float)

        for _, row in df.iterrows():
            code = row.get("code")
            if code not in codes:
                continue
            tp = row.get(tp_col)
            if pd.isna(tp) or tp is None or float(tp) <= 0:
                results[code] = 0.0
                continue
            cp = (current_prices or {}).get(code, 0)
            if cp > 0:
                results[code] = (float(tp) - cp) / cp
            else:
                results[code] = 0.0

        return pd.Series(results, name="target_price_upside")

    def compute_analyst_attention_factor(self, codes: List[str]) -> pd.Series:
        """计算分析师关注度因子（覆盖机构数）"""
        df = self._load("ratings_raw")
        if df is None or df.empty:
            return pd.Series(dtype=float)

        count_col = None
        for col in ["rating_count", "west_rating_cnt", "analyst_count"]:
            if col in df.columns:
                count_col = col
                break

        if count_col is None:
            return pd.Series(dtype=float)

        results = {}
        for _, row in df.iterrows():
            code = row.get("code")
            val = row.get(count_col)
            results[code] = np.log1p(float(val)) if val and not pd.isna(val) else 0.0

        series = pd.Series(results, name="analyst_attention")
        if series.max() > 0:
            series = series / series.max()
        return series

    def _find_rating_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["rating_latest", "west_avg_rating", "rating"]:
            if col in df.columns:
                return col
        return None

    def _to_wind_code(self, code: str) -> str:
        if code.startswith("SH"):
            return code[2:] + ".SH"
        elif code.startswith("SZ"):
            return code[2:] + ".SZ"
        return code

    def _save(self, df: pd.DataFrame, name: str):
        df.to_parquet(self.data_dir / f"{name}.parquet", index=False)

    def _load(self, name: str) -> Optional[pd.DataFrame]:
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)
