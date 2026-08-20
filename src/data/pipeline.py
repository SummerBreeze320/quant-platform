"""Data pipeline orchestrator: Wind fetch -> clean -> Qlib bin -> calendar/instruments."""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

from .wind.wind_client import WindClient
from .wind.stock_data import StockDataFetcher
from .wind.etf_data import ETFDataFetcher
from .qlib_adapter.converter import WindToQlibConverter
from .qlib_adapter.calendar import CalendarGenerator
from .qlib_adapter.instrument import InstrumentGenerator

logger = logging.getLogger(__name__)


class DataPipeline:
    """Data pipeline: Wind -> clean -> Qlib bin -> calendar/instruments"""

    def __init__(self, qlib_dir: str = "data/qlib_bin", raw_dir: str = "data/raw"):
        self.qlib_dir = Path(qlib_dir)
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.client = WindClient()
        self.stock_fetcher = StockDataFetcher(self.client)
        self.etf_fetcher = ETFDataFetcher(self.client)
        self.converter = WindToQlibConverter(str(self.qlib_dir))
        self.features_dir = self.converter.features_dir
        self.calendar_gen = CalendarGenerator(str(self.qlib_dir))
        self.instrument_gen = InstrumentGenerator(str(self.qlib_dir))
        self._calendar_dates: Optional[list] = None

    def _connect(self):
        if not self.client.is_connected():
            self.client.connect()

    @property
    def calendar_dates(self) -> list:
        """Get calendar dates, loading from file if needed."""
        if self._calendar_dates is None:
            self._calendar_dates = self.calendar_gen.load_calendar()
        return self._calendar_dates

    def init_calendar(self, start_date: str = "2015-01-01", end_date: str = "2026-12-31"):
        logger.info(f"Initializing trade calendar ({start_date} to {end_date})...")
        self._connect()
        self._calendar_dates = self.calendar_gen.generate_from_wind(start_date, end_date)
        self.converter.load_calendar()
        logger.info(f"Trade calendar initialized: {len(self._calendar_dates)} days")

    def init_stock_data(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-08-01",
        index_code: Optional[str] = None,
        batch_size: int = 50,
    ) -> int:
        self._connect()
        if index_code:
            codes = self.stock_fetcher.fetch_index_constituents(index_code)
            tag = index_code.replace(".", "")
        else:
            codes = self.stock_fetcher.fetch_all_stocks()
            tag = "all"

        total = len(codes)
        logger.info(f"Initializing stock data: {tag}, {total} codes")
        all_instruments = []
        start_dates = []
        success_count = 0

        for i in range(0, total, batch_size):
            batch = codes[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total - 1) // batch_size + 1
            logger.info(f"Stock batch {batch_num}/{total_batches} ({len(batch)} codes)")

            data = self.stock_fetcher.fetch_daily_batch(batch, start_date, end_date)
            for code, df in data.items():
                try:
                    result = self.converter.convert_stock(
                        code, df, calendar_dates=self.calendar_dates
                    )
                    if result is not None:
                        qlib_code = WindToQlibConverter._normalize_code(code)
                        all_instruments.append(qlib_code)
                        start_idx = result[0]
                        cal = self.calendar_dates
                        if cal and start_idx < len(cal):
                            start_dates.append(cal[start_idx])
                        else:
                            start_dates.append(start_date)
                        success_count += 1
                except Exception as e:
                    logger.error(f"Convert failed {code}: {e}")
            time.sleep(1)

        if all_instruments:
            self.instrument_gen.generate(
                tag, all_instruments, start_dates=start_dates,
                end_dates=end_date,
            )
        logger.info(f"Stock data done: {success_count}/{total} converted")
        return success_count

    def init_etf_data(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-08-01",
        etf_list: Optional[list] = None,
        batch_size: int = 30,
    ) -> int:
        self._connect()
        if etf_list:
            codes = etf_list
        else:
            codes = self.etf_fetcher.fetch_all_etfs()
        total = len(codes)
        logger.info(f"Initializing ETF data: {total} ETFs")
        all_instruments = []
        start_dates = []
        success_count = 0

        for i in range(0, total, batch_size):
            batch = codes[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total - 1) // batch_size + 1
            logger.info(f"ETF batch {batch_num}/{total_batches} ({len(batch)} codes)")

            data = self.etf_fetcher.fetch_etf_daily_batch(batch, start_date, end_date)
            for code, df in data.items():
                try:
                    result = self.converter.convert_stock(
                        code, df, calendar_dates=self.calendar_dates
                    )
                    if result is not None:
                        qlib_code = WindToQlibConverter._normalize_code(code)
                        all_instruments.append(qlib_code)
                        start_idx = result[0]
                        cal = self.calendar_dates
                        if cal and start_idx < len(cal):
                            start_dates.append(cal[start_idx])
                        else:
                            start_dates.append(start_date)
                        success_count += 1
                except Exception as e:
                    logger.error(f"Convert failed {code}: {e}")
            time.sleep(1)

        if all_instruments:
            self.instrument_gen.generate(
                "etf", all_instruments, start_dates=start_dates,
                end_dates=end_date,
            )
        logger.info(f"ETF data done: {success_count}/{total} converted")
        return success_count

    def update_daily(self, trade_date: Optional[str] = None) -> dict:
        """Incremental update for a single trade date.

        Appends new data to existing bin files by re-fetching the
        full date range for each instrument and re-converting.
        For production, a more efficient incremental append should
        be implemented.
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Daily update for {trade_date}")
        self._connect()

        results = {"stocks": 0, "etfs": 0}

        stock_codes = self.stock_fetcher.fetch_all_stocks()
        stock_data = self.stock_fetcher.fetch_daily_batch(
            stock_codes, trade_date, trade_date
        )
        results["stocks"] = self.converter.convert_batch(
            stock_data, calendar_dates=self.calendar_dates
        )
        logger.info(f"Updated {results['stocks']} stocks")

        etf_codes = self.etf_fetcher.fetch_all_etfs()
        etf_data = self.etf_fetcher.fetch_etf_daily_batch(
            etf_codes, trade_date, trade_date
        )
        results["etfs"] = self.converter.convert_batch(
            etf_data, calendar_dates=self.calendar_dates
        )
        logger.info(f"Updated {results['etfs']} ETFs")

        return results

    def update_incremental(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """Efficient incremental update: only fetch new dates since last update.

        Reads existing bin files to determine last date, then fetches only
        the gap and merges with existing data before re-converting.

        Args:
            start_date: Override the auto-detected start date.
            end_date: End date (default: today).

        Returns:
            Dict with stocks/etfs counts.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        self._connect()
        self.converter.load_calendar()
        cal = self.calendar_dates

        results = {"stocks": 0, "etfs": 0, "skipped": 0}

        # Detect last date from existing bin files
        if start_date is None:
            sample_bin = self.features_dir / "SH600000" / "close.day.bin"
            if sample_bin.exists():
                import numpy as np
                arr = np.fromfile(str(sample_bin), dtype="<f4")
                n_valid = np.searchsorted(arr, -1e6, side="right")
                if n_valid > 0 and n_valid <= len(cal):
                    last_cal_date = cal[n_valid - 1]
                    start_date = last_cal_date.strftime("%Y-%m-%d")
                    logger.info(f"Incremental update: last data at {start_date}, updating to {end_date}")
                else:
                    start_date = "2015-01-01"
            else:
                start_date = "2015-01-01"

        if start_date >= end_date:
            logger.info("Data already up to date, skipping.")
            return results

        # Fetch and update stocks
        stock_codes = self.stock_fetcher.fetch_all_stocks()
        for i in range(0, len(stock_codes), 50):
            batch = stock_codes[i:i + 50]
            data = self.stock_fetcher.fetch_daily_batch(batch, start_date, end_date)
            results["stocks"] += self.converter.convert_batch(
                data, calendar_dates=cal
            )
            time.sleep(0.5)

        # Fetch and update ETFs
        etf_codes = self.etf_fetcher.fetch_all_etfs()
        for i in range(0, len(etf_codes), 30):
            batch = etf_codes[i:i + 30]
            data = self.etf_fetcher.fetch_etf_daily_batch(batch, start_date, end_date)
            results["etfs"] += self.converter.convert_batch(
                data, calendar_dates=cal
            )
            time.sleep(0.5)

        logger.info(f"Incremental update complete: {results}")
        return results

    def verify_data(self, sample_codes: Optional[list] = None) -> list:
        logger.info("Verifying data integrity...")
        if not sample_codes:
            sample_codes = ["SH600000", "SZ000001", "SH510300"]
        results = []
        for code in sample_codes:
            close_bin = self.features_dir / code / "close.day.bin"
            ok = close_bin.exists() and close_bin.stat().st_size > 8
            results.append({"code": code, "valid": ok})
        valid_count = sum(1 for r in results if r["valid"])
        logger.info(f"Verification: {valid_count}/{len(results)} valid")

        cal_path = self.qlib_dir / "calendars" / "day.txt"
        if cal_path.exists():
            lines = cal_path.read_text(encoding="utf-8").strip().split("\n")
            logger.info(f"Calendar: {len(lines)} trade days, {lines[0]} to {lines[-1]}")

        inst_dir = self.qlib_dir / "instruments"
        if inst_dir.exists():
            for f in inst_dir.glob("*.txt"):
                count = len(f.read_text(encoding="utf-8").strip().split("\n"))
                logger.info(f"Instrument {f.name}: {count} entries")
        return results

    def run_full_init(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-08-01",
        include_etf: bool = True,
        index_only: Optional[str] = None,
    ) -> dict:
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Quant Platform - Full Data Initialization")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("=" * 60)

        stats = {"calendar": 0, "stocks": 0, "etfs": 0}
        try:
            self._connect()
            self.init_calendar(start_date, end_date)
            stats["calendar"] = len(self.calendar_dates)
            stats["stocks"] = self.init_stock_data(
                start_date, end_date, index_code=index_only
            )
            if include_etf:
                stats["etfs"] = self.init_etf_data(start_date, end_date)
            self.verify_data()
            elapsed = time.time() - start_time
            logger.info(f"Full initialization complete in {elapsed:.0f}s")
            logger.info(f"Stats: {stats}")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
        finally:
            self.client.disconnect()
        return stats

    def disconnect(self):
        self.client.disconnect()
