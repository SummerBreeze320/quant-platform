"""公告事件数据处理器

从Wind获取公司公告/权益事件数据，包括：
1. 公告列表（类型、时间、标题）
2. 权益事件（分红、配股、减持等）

数据流：Wind WSET → Parquet存储 → 事件因子计算
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/alternative/announcements")

EVENT_TYPES = [
    " dividend", "split", "rights_issue", "buyback",
    "major_shareholder_change", "earnings_forecast", "M&A",
    "management_change", "lawsuit", "delisting_warning",
]


class AnnouncementHandler:
    """公告事件数据获取与因子计算

    Wind API:
        w.wset("EquityEvent", "wind_code=600000.SH;startdate=2024-01-01")
        获取权益事件（分红/配股/增发/回购等）

    Usage:
        handler = AnnouncementHandler()
        handler.fetch_announcements(codes=["SH600000"], start="2024-01-01")
        factor = handler.compute_event_factor(codes, end_date="2024-06-01")
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_announcements(
        self, codes: List[str], start: str, end: str = None
    ) -> pd.DataFrame:
        """从Wind获取公告/权益事件数据"""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        try:
            from WindPy import w
            if not w.isconnected():
                w.start()

            all_events = []
            for code in codes:
                wind_code = self._to_wind_code(code)
                result = w.wset(
                    "EquityEvent",
                    f"wind_code={wind_code};startDate={start};endDate={end}",
                )
                if result.ErrorCode != 0:
                    continue

                df = pd.DataFrame(result.Data, index=result.Fields).T
                if df.empty:
                    continue
                df.columns = result.Fields
                df["code"] = code
                all_events.append(df)

            if not all_events:
                return pd.DataFrame()

            df = pd.concat(all_events, ignore_index=True)
            self._save(df, "announcements_raw")
            logger.info(f"Fetched {len(df)} announcement records")
            return df

        except ImportError:
            logger.warning("WindPy not installed — run mock generator for offline dev")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Announcement fetch error: {e}")
            return pd.DataFrame()

    def compute_event_factor(
        self,
        codes: List[str],
        end_date: str,
        lookback: int = 30,
    ) -> pd.Series:
        """计算公告事件因子

        因子逻辑：
        - 滚动窗口内公告数量 → event_count（关注度代理）
        - 重大事件类型哑变量 → event_type_dummy
        - 正面/负面事件频率比 → event_sentiment_proxy
        """
        df = self._load("announcements_raw")
        if df is None or df.empty:
            return pd.Series(dtype=float)

        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - timedelta(days=lookback)

        date_col = self._find_date_col(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
            df = df[mask]

        type_col = self._find_type_col(df)

        results = {}
        for code in codes:
            code_df = df[df["code"] == code]
            n_events = len(code_df)

            if type_col and n_events > 0:
                positive_types = ["dividend", "buyback", "earnings_forecast"]
                negative_types = ["delisting_warning", "lawsuit", "major_shareholder_change"]
                types = code_df[type_col].str.lower().tolist()
                pos = sum(1 for t in types if any(p in t for p in positive_types))
                neg = sum(1 for t in types if any(n in t for n in negative_types))
                if pos + neg > 0:
                    results[code] = (pos - neg) / (pos + neg)
                else:
                    results[code] = np.log1p(n_events)
            else:
                results[code] = np.log1p(n_events)

        return pd.Series(results, name="announcement_event")

    def compute_dividend_factor(
        self,
        codes: List[str],
        end_date: str,
        lookback: int = 365,
    ) -> pd.Series:
        """计算分红事件因子（近一年是否有分红）"""
        df = self._load("announcements_raw")
        if df is None or df.empty:
            return pd.Series(dtype=float)

        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - timedelta(days=lookback)
        date_col = self._find_date_col(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]

        type_col = self._find_type_col(df)
        results = {}
        for code in codes:
            code_df = df[df["code"] == code]
            if type_col and not code_df.empty:
                has_div = code_df[type_col].str.lower().str.contains("dividend").any()
                results[code] = 1.0 if has_div else 0.0
            else:
                results[code] = 0.0
        return pd.Series(results, name="dividend_indicator")

    def _find_date_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["announce_date", "event_date", "date", "publish_date"]:
            if col in df.columns:
                return col
        return None

    def _find_type_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["event_type", "announcement_type", "type", "title"]:
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
        self.data_dir.joinpath(f"{name}.parquet").write_bytes(b"")
        df.to_parquet(self.data_dir / f"{name}.parquet", index=False)

    def _load(self, name: str) -> Optional[pd.DataFrame]:
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)
