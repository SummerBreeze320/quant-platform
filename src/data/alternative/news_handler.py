"""新闻舆情数据处理器

从Wind获取新闻舆情数据，包括：
1. 新闻列表（来源、标题、时间、关联标的）
2. AI情感分析评分（Wind Client API内置）

数据流：Wind WSET/WSS → Parquet存储 → 情感因子计算
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/alternative/news")


class NewsHandler:
    """新闻舆情数据获取与因子计算

    Wind API支持：
    - 实时新闻列表（180+财经媒体，200+行业网站）
    - AI情感分析评分（正向/负向/中性）

    Usage:
        handler = NewsHandler()
        handler.fetch_news(codes=["SH600000"], start="2024-01-01")
        factor = handler.compute_sentiment_factor(codes, end_date="2024-06-01")
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_news(self, codes: List[str], start: str, end: str = None) -> pd.DataFrame:
        """从Wind获取新闻数据

        Wind API:
            w.wset("NewsList", "wind_code=SH600000;startdate=2024-01-01;enddate=2024-06-01")
            获取新闻列表 + AI情感评分
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        try:
            from WindPy import w
            if not w.isconnected():
                w.start()

            all_news = []
            for code in codes:
                wind_code = self._to_wind_code(code)
                result = w.wset(
                    "NewsList",
                    f"wind_code={wind_code};startdate={start};enddate={end}",
                )
                if result.ErrorCode != 0:
                    logger.warning(f"Wind news fetch failed for {code}: {result.ErrorCode}")
                    continue

                df = pd.DataFrame(
                    result.Data,
                    index=result.Fields,
                ).T
                if df.empty:
                    continue
                df.columns = result.Fields
                df["code"] = code
                all_news.append(df)

            if not all_news:
                logger.warning("No news data fetched")
                return pd.DataFrame()

            df = pd.concat(all_news, ignore_index=True)
            self._save(df, "news_raw")
            logger.info(f"Fetched {len(df)} news records for {len(codes)} codes")
            return df

        except ImportError:
            logger.warning("WindPy not installed — run mock generator for offline dev")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return pd.DataFrame()

    def compute_sentiment_factor(
        self,
        codes: List[str],
        end_date: str,
        lookback: int = 20,
    ) -> pd.Series:
        """计算新闻情感因子

        因子逻辑：
        - 滚动窗口内新闻情感评分均值 → stock-level sentiment
        - 新闻数量 → attention factor
        - 情感波动 → sentiment dispersion

        Returns:
            pd.Series: index=code, value=sentiment_score [-1, 1]
        """
        df = self._load("news_raw")
        if df is None or df.empty:
            return pd.Series(dtype=float)

        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - timedelta(days=lookback)

        if "publish_time" in df.columns:
            df["publish_time"] = pd.to_datetime(df["publish_time"])
            mask = (df["publish_time"] >= start_dt) & (df["publish_time"] <= end_dt)
            df = df[mask]

        sentiment_col = self._find_sentiment_col(df)
        if sentiment_col is None:
            logger.warning("No sentiment column found in news data")
            return pd.Series(dtype=float)

        results = {}
        for code in codes:
            code_df = df[df["code"] == code]
            if code_df.empty:
                results[code] = 0.0
                continue
            avg_sentiment = code_df[sentiment_col].mean()
            results[code] = float(avg_sentiment) if not np.isnan(avg_sentiment) else 0.0

        return pd.Series(results, name="news_sentiment")

    def compute_attention_factor(
        self,
        codes: List[str],
        end_date: str,
        lookback: int = 20,
    ) -> pd.Series:
        """计算新闻关注度因子（新闻数量标准化）"""
        df = self._load("news_raw")
        if df is None or df.empty:
            return pd.Series(dtype=float)

        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - timedelta(days=lookback)

        if "publish_time" in df.columns:
            df["publish_time"] = pd.to_datetime(df["publish_time"])
            mask = (df["publish_time"] >= start_dt) & (df["publish_time"] <= end_dt)
            df = df[mask]

        counts = df.groupby("code").size()
        results = {}
        for code in codes:
            results[code] = float(counts.get(code, 0))

        series = pd.Series(results, name="news_attention")
        if series.max() > 0:
            series = series / series.max()
        return series

    def _find_sentiment_col(self, df: pd.DataFrame) -> Optional[str]:
        """自动识别情感评分列"""
        candidates = [
            "sentiment_score", "sentiment", "ai_score",
            "emotion_score", "polarity", "score",
        ]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _to_wind_code(self, code: str) -> str:
        """Qlib码转Wind码: SH600000 → 600000.SH"""
        if code.startswith("SH"):
            return code[2:] + ".SH"
        elif code.startswith("SZ"):
            return code[2:] + ".SZ"
        return code

    def _save(self, df: pd.DataFrame, name: str):
        path = self.data_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)

    def _load(self, name: str) -> Optional[pd.DataFrame]:
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)
