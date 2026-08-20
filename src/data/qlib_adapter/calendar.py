"""Qlib trade calendar generator.

Generates calendars/day.txt containing all A-share trade dates.
Uses Wind wsd on SH000001 (Shanghai Composite) as the canonical
trade-day reference, with a fallback to Chinese holiday rules.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CalendarGenerator:
    """Generate Qlib-format trade calendars."""

    def __init__(self, qlib_dir: str = "data/qlib_bin"):
        self.qlib_dir = Path(qlib_dir)
        self.calendar_dir = self.qlib_dir / "calendars"
        self.calendar_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_wind(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-12-31",
    ) -> List[str]:
        """Generate trade calendar from Wind.

        Uses SH000001 (Shanghai Composite Index) close prices to
        identify actual trade days — only dates where the index
        traded are included.
        """
        from ..wind.wind_client import WindClient

        client = WindClient()
        client.connect()

        try:
            df = client.wsd(
                codes="000001.SH",
                fields="close",
                start_time=start_date,
                end_time=end_date,
                options="Fill=Previous",
            )
        finally:
            pass

        if df.empty:
            logger.warning("Wind returned empty calendar data, using fallback.")
            return self._generate_fallback(start_date, end_date)

        trade_dates = df.dropna(subset=["close"]).index
        dates_str = [d.strftime("%Y-%m-%d") for d in trade_dates]
        self._write_calendar(dates_str, "day")
        logger.info(f"Generated day calendar from Wind: {len(dates_str)} trade days")
        return dates_str

    def generate_from_date_list(
        self,
        dates: List[str],
        freq: str = "day",
    ) -> None:
        """Generate calendar from an explicit date list."""
        self._write_calendar(dates, freq)

    def _generate_fallback(self, start_date: str, end_date: str) -> List[str]:
        """Generate approximate trade calendar without Wind.

        Excludes weekends and major Chinese holidays. This is a
        rough approximation — use Wind for production accuracy.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        all_days = pd.bdate_range(start, end)

        holiday_ranges = self._get_holiday_ranges(start.year, end.year)
        trade_days = []
        for day in all_days:
            is_holiday = False
            for h_start, h_end in holiday_ranges:
                if h_start <= day <= h_end:
                    is_holiday = True
                    break
            if not is_holiday:
                trade_days.append(day.strftime("%Y-%m-%d"))

        self._write_calendar(trade_days, "day")
        logger.info(f"Generated fallback calendar: {len(trade_days)} trade days")
        return trade_days

    def _get_holiday_ranges(self, start_year: int, end_year: int) -> list:
        """Approximate Chinese public holiday date ranges.

        These are rough estimates based on typical patterns.
        For production, always use Wind-sourced calendars.
        """
        holidays = []
        for year in range(start_year, end_year + 1):
            holidays.extend([
                (pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-01-03")),    # New Year
                (pd.Timestamp(f"{year}-02-10"), pd.Timestamp(f"{year}-02-17")),   # Spring Festival (approx)
                (pd.Timestamp(f"{year}-04-04"), pd.Timestamp(f"{year}-04-06")),   # Qingming
                (pd.Timestamp(f"{year}-05-01"), pd.Timestamp(f"{year}-05-05")),   # Labour Day
                (pd.Timestamp(f"{year}-06-10"), pd.Timestamp(f"{year}-06-12")),   # Dragon Boat
                (pd.Timestamp(f"{year}-09-15"), pd.Timestamp(f"{year}-09-17")),   # Mid-Autumn
                (pd.Timestamp(f"{year}-10-01"), pd.Timestamp(f"{year}-10-07")),   # National Day
            ])
        return holidays

    def _write_calendar(self, dates: List[str], freq: str = "day") -> None:
        path = self.calendar_dir / f"{freq}.txt"
        path.write_text("\n".join(dates), encoding="utf-8")
        logger.info(f"Calendar written: {path} ({len(dates)} dates)")

    def load_calendar(self) -> List[str]:
        """Load the day calendar from file."""
        path = self.calendar_dir / "day.txt"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").strip().split("\n")
