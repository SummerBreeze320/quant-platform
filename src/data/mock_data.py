"""Mock data generator for offline development.

Generates synthetic A-share and ETF OHLCV data that follows a
realistic random-walk with volatility clustering. Converts to Qlib
bin format so the entire pipeline (factors, backtest, dashboard) can
be tested without a running Wind terminal.

Usage:
    from src.data.mock_data import MockDataGenerator
    gen = MockDataGenerator()
    gen.generate_qlib_data(n_stocks=50, n_etfs=20, start="2020-01-01", end="2024-12-31")
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from .qlib_adapter.converter import WindToQlibConverter
from .qlib_adapter.calendar import CalendarGenerator
from .qlib_adapter.instrument import InstrumentGenerator

logger = logging.getLogger(__name__)

# Representative A-share stock codes (large caps)
MOCK_STOCKS = [
    "600000.SH", "600519.SH", "600036.SH", "600276.SH", "600030.SH",
    "601318.SH", "601166.SH", "600887.SH", "600016.SH", "601288.SH",
    "000001.SZ", "000002.SZ", "000333.SZ", "000651.SZ", "000858.SZ",
    "002594.SZ", "000568.SZ", "002415.SZ", "000725.SZ", "002230.SZ",
    "300750.SZ", "300059.SZ", "300015.SZ", "300760.SZ", "300124.SZ",
]

# Representative ETF codes
MOCK_ETFS = [
    "510300.SH", "510050.SH", "510500.SH", "588000.SH", "588050.SH",
    "159915.SZ", "159949.SZ", "512100.SH", "512760.SH", "515790.SH",
    "510310.SH", "515330.SH", "512010.SH", "512200.SH", "515050.SH",
    "512880.SH", "515950.SH", "516160.SH", "513100.SH", "513050.SH",
]


class MockDataGenerator:
    """Generate synthetic market data for development and testing."""

    def __init__(self, qlib_dir: str = "data/qlib_bin"):
        self.qlib_dir = Path(qlib_dir)
        self.converter = WindToQlibConverter(str(self.qlib_dir))
        self.calendar_gen = CalendarGenerator(str(self.qlib_dir))
        self.instrument_gen = InstrumentGenerator(str(self.qlib_dir))

    def _generate_trade_calendar(
        self, start: str = "2020-01-01", end: str = "2024-12-31"
    ) -> List[str]:
        """Generate trade calendar (business days, approx holidays)."""
        dates = self.calendar_gen._generate_fallback(start, end)
        self.converter.load_calendar()
        return dates

    def _generate_ohlcv(
        self,
        n_days: int,
        init_price: float = 10.0,
        annual_vol: float = 0.25,
        annual_drift: float = 0.05,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV via geometric Brownian motion.

        Includes volatility clustering (GARCH-like) for realism.
        """
        rng = np.random.default_rng(seed)
        dt = 1 / 252
        mu = annual_drift
        sigma = annual_vol

        # Volatility clustering (GARCH-like, ensure positive)
        base_vol = sigma * np.sqrt(dt)
        vol_cluster = np.ones(n_days) * base_vol
        for i in range(1, n_days):
            vol_cluster[i] = 0.94 * vol_cluster[i-1] + 0.06 * base_vol + rng.normal(0, base_vol * 0.1)
        vol_cluster = np.maximum(vol_cluster, base_vol * 0.3)  # floor at 30% of base

        # Random walk with drift
        log_returns = rng.normal(mu * dt, vol_cluster)
        log_prices = np.log(init_price) + np.cumsum(log_returns)
        close = np.exp(log_prices)

        # Generate OHLC from close
        intraday_range = np.abs(rng.normal(0, 0.01, n_days))
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        high = np.maximum(open_, close) * (1 + intraday_range)
        low = np.minimum(open_, close) * (1 - intraday_range)
        volume = (rng.lognormal(15, 0.5, n_days) * (1 + np.abs(log_returns) * 10)).astype(float)
        amount = volume * close
        pct_chg = np.concatenate([[0], np.diff(close) / close[:-1] * 100])
        turn = np.abs(rng.normal(1, 0.3, n_days))
        vwap = (high + low + close) / 3

        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "pct_chg": pct_chg,
            "turn": turn,
            "vwap": vwap,
        })
        return df

    def _generate_fundamentals(
        self,
        n_days: int,
        close: np.ndarray,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic fundamental data.

        Creates plausible PE/PB/PS/ROE/ROA/etc values that
        are loosely correlated with price movements.
        """
        rng = np.random.default_rng(seed)
        n = n_days

        # Market cap = close * mock_shares (shares ~ 1-10亿)
        mock_shares = rng.uniform(1e8, 1e10)
        total_mv = close * mock_shares
        circ_mv = total_mv * rng.uniform(0.6, 0.95)

        # PE: log-normal around 15-30
        pe_ttm = np.exp(rng.normal(np.log(20), 0.3, n))
        pe_ttm = np.maximum(pe_ttm, 5.0)

        # PB: log-normal around 1-5
        pb_lf = np.exp(rng.normal(np.log(2), 0.3, n))
        pb_lf = np.maximum(pb_lf, 0.5)

        # PS: log-normal around 1-10
        ps_ttm = np.exp(rng.normal(np.log(3), 0.3, n))
        ps_ttm = np.maximum(ps_ttm, 0.5)

        # Dividend yield: 0-5%
        dividend_yield = np.abs(rng.normal(1.5, 0.8, n))
        dividend_yield = np.clip(dividend_yield, 0, 8)

        # ROE: 5-25%
        roe = rng.normal(12, 4, n)
        roe = np.clip(roe, -5, 35)

        # ROA: 2-15%
        roa = roe * rng.uniform(0.3, 0.6)

        # Debt ratio: 20-70%
        debt_ratio = rng.normal(45, 10, n)
        debt_ratio = np.clip(debt_ratio, 10, 85)

        # Growth rates: -20% to +50%
        revenue_yoy = rng.normal(10, 15, n)
        profit_yoy = rng.normal(8, 20, n)

        return pd.DataFrame({
            "pe_ttm": pe_ttm,
            "pb_lf": pb_lf,
            "ps_ttm": ps_ttm,
            "dividend_yield": dividend_yield,
            "total_mv": total_mv,
            "circ_mv": circ_mv,
            "roe": roe,
            "roa": roa,
            "debt_ratio": debt_ratio,
            "revenue_yoy": revenue_yoy,
            "profit_yoy": profit_yoy,
        })

    def generate_qlib_data(
        self,
        n_stocks: int = 25,
        n_etfs: int = 20,
        start: str = "2020-01-01",
        end: str = "2024-12-31",
        seed: int = 42,
    ) -> dict:
        """Generate complete Qlib dataset with mock data.

        Creates calendar, instruments, and features directories
        with synthetic OHLCV data for the specified number of
        stocks and ETFs.

        Returns:
            Stats dict with counts.
        """
        logger.info("=" * 60)
        logger.info("Generating mock Qlib data for offline development")
        logger.info(f"Stocks: {n_stocks}, ETFs: {n_etfs}, {start} to {end}")
        logger.info("=" * 60)

        # 1. Generate calendar
        cal_dates = self._generate_trade_calendar(start, end)
        n_days = len(cal_dates)
        logger.info(f"Trade calendar: {n_days} days")

        # 2. Generate stock data
        stocks = MOCK_STOCKS[:n_stocks] if n_stocks <= len(MOCK_STOCKS) else MOCK_STOCKS
        stock_instruments = []
        for i, code in enumerate(stocks):
            df = self._generate_ohlcv(
                n_days, init_price=10 + i * 2,
                annual_vol=0.20 + 0.01 * i,
                annual_drift=0.05 + 0.001 * (i % 5),
                seed=seed + i,
            )
            df.index = pd.to_datetime(cal_dates)
            # 添加基本面数据
            fund_df = self._generate_fundamentals(
                n_days, df["close"].values, seed=seed + 500 + i
            )
            fund_df.index = df.index
            df = pd.concat([df, fund_df], axis=1)
            result = self.converter.convert_stock(code, df, calendar_dates=cal_dates)
            if result:
                stock_instruments.append(WindToQlibConverter._normalize_code(code))

        self.instrument_gen.generate(
            "all", stock_instruments,
            start_dates=[cal_dates[0]] * len(stock_instruments),
            end_dates=cal_dates[-1],
        )
        logger.info(f"Generated {len(stock_instruments)} stocks")

        # 3. Generate ETF data
        etfs = MOCK_ETFS[:n_etfs] if n_etfs <= len(MOCK_ETFS) else MOCK_ETFS
        etf_instruments = []
        for i, code in enumerate(etfs):
            df = self._generate_ohlcv(
                n_days, init_price=1.0 + i * 0.1,
                annual_vol=0.15 + 0.005 * i,
                annual_drift=0.03 + 0.002 * (i % 3),
                seed=seed + 100 + i,
            )
            df.index = pd.to_datetime(cal_dates)
            # Add ETF-specific fields
            df["nav"] = df["close"] * (1 + np.random.default_rng(seed + 200 + i).normal(0, 0.001, n_days))
            df["discount"] = (df["close"] - df["nav"]) / df["nav"] * 100
            result = self.converter.convert_stock(code, df, calendar_dates=cal_dates)
            if result:
                etf_instruments.append(WindToQlibConverter._normalize_code(code))

        self.instrument_gen.generate(
            "etf", etf_instruments,
            start_dates=[cal_dates[0]] * len(etf_instruments),
            end_dates=cal_dates[-1],
        )
        logger.info(f"Generated {len(etf_instruments)} ETFs")

        # 4. Verify
        self.converter.verify_data = lambda sample_codes=None: self._verify(
            stock_instruments[:3] + etf_instruments[:3]
        )
        self._verify(stock_instruments[:3] + etf_instruments[:3])

        stats = {
            "calendar_days": n_days,
            "stocks": len(stock_instruments),
            "etfs": len(etf_instruments),
            "total_instruments": len(stock_instruments) + len(etf_instruments),
        }
        logger.info(f"Mock data generation complete: {stats}")
        return stats

    def _verify(self, codes: List[str]):
        for code in codes:
            close_bin = self.converter.features_dir / code / "close.day.bin"
            if close_bin.exists() and close_bin.stat().st_size > 8:
                logger.info(f"Verified {code}: close.day.bin OK")
            else:
                logger.warning(f"Verify failed {code}: close.day.bin missing")
