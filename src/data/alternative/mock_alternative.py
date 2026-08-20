"""离线另类数据生成器

为没有Wind环境的开发场景生成合成另类数据：
- 新闻舆情: 随机情感评分 [-1, 1] + 新闻数量
- 公告事件: 随机事件类型 + 正/负面标记
- 研报评级: 随机评级(买入/增持/中性/减持) + 目标价
- 宏观数据: 随机宏观指标时间序列

Usage:
    gen = MockAlternativeGenerator()
    gen.generate_all(codes=["SH600000", ...], start="2024-01-01", end="2024-06-01")
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

from .news_handler import DATA_DIR as NEWS_DIR
from .announcement_handler import DATA_DIR as ANN_DIR, EVENT_TYPES
from .research_report_handler import DATA_DIR as REPORT_DIR, RATING_MAP
from .macro_handler import DATA_DIR as MACRO_DIR, MACRO_INDICATORS

logger = logging.getLogger(__name__)

RATINGS = list(RATING_MAP.keys())


class MockAlternativeGenerator:
    """生成合成另类数据用于离线开发

    生成数据存储为Parquet格式，与Wind真实数据格式一致，
    各Handler的compute_*方法可直接消费。
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_all(
        self,
        codes: List[str],
        start: str = "2024-01-01",
        end: str = "2024-06-30",
    ) -> dict:
        """一键生成全部另类数据

        Returns:
            dict: 各数据类别的生成记录数
        """
        counts = {}
        counts["news"] = self.generate_news(codes, start, end)
        counts["announcements"] = self.generate_announcements(codes, start, end)
        counts["ratings"] = self.generate_ratings(codes)
        counts["macro"] = self.generate_macro(start, end)
        logger.info(f"Mock alternative data generated: {counts}")
        return counts

    def generate_news(self, codes: List[str], start: str, end: str) -> int:
        """生成合成新闻舆情数据"""
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        n_days = (end_dt - start_dt).days

        records = []
        for code in codes:
            n_news = self.rng.randint(5, 30)
            for _ in range(n_news):
                offset = self.rng.randint(0, max(n_days, 1))
                pub_time = start_dt + timedelta(days=offset)
                sentiment = self.rng.normal(0.1, 0.5)
                sentiment = np.clip(sentiment, -1, 1)
                records.append({
                    "code": code,
                    "publish_time": pub_time,
                    "title": f"Mock news for {code}",
                    "source": self.rng.choice(["sina", "eastmoney", "cnstock", "stcn"]),
                    "sentiment_score": round(sentiment, 4),
                    "ai_score": round(sentiment, 4),
                })

        df = pd.DataFrame(records)
        NEWS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(NEWS_DIR / "news_raw.parquet", index=False)
        return len(df)

    def generate_announcements(self, codes: List[str], start: str, end: str) -> int:
        """生成合成公告事件数据"""
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        n_days = (end_dt - start_dt).days

        records = []
        for code in codes:
            n_events = self.rng.randint(3, 15)
            for _ in range(n_events):
                offset = self.rng.randint(0, max(n_days, 1))
                event_date = start_dt + timedelta(days=offset)
                event_type = self.rng.choice(EVENT_TYPES)
                records.append({
                    "code": code,
                    "announce_date": event_date,
                    "event_type": event_type,
                    "title": f"Mock {event_type} announcement for {code}",
                })

        df = pd.DataFrame(records)
        ANN_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ANN_DIR / "announcements_raw.parquet", index=False)
        return len(df)

    def generate_ratings(self, codes: List[str]) -> int:
        """生成合成研报评级数据"""
        records = []
        for code in codes:
            rating = self.rng.choice(RATINGS)
            target_price = self.rng.uniform(10, 500)
            eps_fy1 = self.rng.uniform(0.1, 5.0)
            rev_fy1 = self.rng.uniform(1e8, 1e10)
            rating_count = self.rng.randint(1, 30)
            records.append({
                "code": code,
                "trade_date": datetime.now().strftime("%Y-%m-%d"),
                "rating_latest": rating,
                "rating_change": self.rng.choice(["up", "down", "init", ""]),
                "target_price": round(target_price, 2),
                "west_avgeps_fy1": round(eps_fy1, 4),
                "west_avgrev_fy1": round(rev_fy1, 2),
                "rating_count": rating_count,
            })

        df = pd.DataFrame(records)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(REPORT_DIR / "ratings_raw.parquet", index=False)
        return len(df)

    def generate_macro(self, start: str = "2015-01-01", end: str = None) -> int:
        """生成合成宏观经济指标数据"""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        dates = pd.date_range(start, end, freq="ME")
        n = len(dates)

        data = {}
        for name in MACRO_INDICATORS:
            base = self.rng.uniform(-5, 15)
            noise = self.rng.normal(0, 0.5, n)
            trend = np.linspace(0, self.rng.uniform(-3, 3), n)
            series = base + trend + noise
            if name in ["cpi_yoy", "ppi_yoy"]:
                series = np.clip(series, -10, 15)
            elif name in ["gdp_yoy"]:
                series = np.clip(series, -5, 12)
            elif name in ["m2_yoy", "m1_yoy"]:
                series = np.clip(series, -5, 25)
            elif name in ["shibor_3m", "lpr_1y", "treasury_10y"]:
                series = np.clip(series, 0, 10)
            elif name in ["cny_usd"]:
                series = np.clip(series, 6, 8)
            elif name in ["brent_oil"]:
                series = np.clip(series, 20, 150)
            elif name in ["gold"]:
                series = np.clip(series, 1000, 3000)
            elif name in ["copper", "rebar"]:
                series = np.clip(series, 3000, 80000)
            else:
                series = np.clip(series, 30, 60)
            data[name] = np.round(series, 4)

        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"
        MACRO_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(MACRO_DIR / "macro_raw.parquet")
        return len(df)
