"""从 Wind 终端拉取真实行情+基本面数据，写入 Qlib bin 格式。

Usage:
    python scripts/pull_wind_data.py
    python scripts/pull_wind_data.py --start 2020-01-01 --end 2024-12-31
"""
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STOCKS = [
    "600000.SH", "600016.SH", "600030.SH", "600036.SH", "600276.SH",
    "600519.SH", "600887.SH", "601166.SH", "601288.SH", "601318.SH",
    "000001.SZ", "000002.SZ", "000333.SZ", "000568.SZ", "000651.SZ",
    "000725.SZ", "000858.SZ", "002230.SZ", "002415.SZ", "002594.SZ",
    "300015.SZ", "300059.SZ", "300124.SZ", "300750.SZ", "300760.SZ",
]

ETFS = [
    "510050.SH", "510300.SH", "510310.SH", "510500.SH", "512010.SH",
    "512100.SH", "512200.SH", "512760.SH", "512880.SH", "513050.SH",
    "513100.SH", "515050.SH", "515330.SH", "515790.SH", "515950.SH",
    "516160.SH", "588000.SH", "588050.SH",
    "159915.SZ", "159949.SZ",
]

OHLCV_FIELDS = "open,high,low,close,volume,amount,pct_chg,turn,vwap"
FUNDAMENTAL_FIELDS = "pe_ttm,pb_lf,ps_ttm,dividend_yield,total_mv,circ_mv,roe,roa,debt_ratio,revenue_yoy,profit_yoy"


def pull_stock_data(client, code, start, end):
    """拉取单只股票的 OHLCV + 基本面数据"""
    all_data = {}

    # OHLCV
    try:
        df = client.wsd(codes=code, fields=OHLCV_FIELDS, start_time=start, end_time=end, options="Fill=Previous")
        if not df.empty:
            for col in df.columns:
                all_data[col] = df[col]
    except Exception as e:
        logger.warning(f"OHLCV pull failed for {code}: {e}")

    # 基本面
    try:
        df2 = client.wsd(codes=code, fields=FUNDAMENTAL_FIELDS, start_time=start, end_time=end, options="Fill=Previous")
        if not df2.empty:
            for col in df2.columns:
                all_data[col] = df2[col]
    except Exception as e:
        logger.warning(f"Fundamental pull failed for {code}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.DataFrame(all_data)
    combined.index = pd.to_datetime(combined.index)
    return combined


def pull_etf_data(client, code, start, end):
    """拉取 ETF 数据（只有 OHLCV，无基本面）"""
    all_data = {}
    try:
        df = client.wsd(codes=code, fields=OHLCV_FIELDS, start_time=start, end_time=end, options="Fill=Previous")
        if not df.empty:
            for col in df.columns:
                all_data[col] = df[col]
    except Exception as e:
        logger.warning(f"ETF data pull failed for {code}: {e}")

    if not all_data:
        return pd.DataFrame()
    combined = pd.DataFrame(all_data)
    combined.index = pd.to_datetime(combined.index)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Pull Wind data to Qlib bin format")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--qlib-dir", default="data/qlib_bin", help="Qlib data directory")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))

    # 设置 Wind DLL 路径
    wind_base = r"D:\Wind\Wind.NET.Client\WindNET"
    for subdir in ["x64", "bin"]:
        d = os.path.join(wind_base, subdir)
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + ";" + os.environ.get("PATH", "")

    from src.data.wind.wind_client import WindClient
    from src.data.qlib_adapter.converter import WindToQlibConverter
    from src.data.qlib_adapter.calendar import CalendarGenerator
    from src.data.qlib_adapter.instrument import InstrumentGenerator

    # 连接 Wind
    client = WindClient()
    logger.info("Connecting to Wind terminal...")
    client.connect()
    if not client.is_connected():
        logger.error("Wind terminal connection failed!")
        return
    logger.info("Wind connected!")

    # 1. 生成交易日历
    logger.info(f"Generating trade calendar ({args.start} ~ {args.end})...")
    cal_gen = CalendarGenerator(args.qlib_dir)
    dates = cal_gen.generate_from_wind(start_date=args.start, end_date=args.end)
    logger.info(f"Calendar: {len(dates)} trade days ({dates[0]} ~ {dates[-1]})")

    # 2. 拉取所有股票数据
    all_codes = STOCKS + ETFS
    all_data = {}

    for i, code in enumerate(all_codes):
        logger.info(f"[{i+1}/{len(all_codes)}] Pulling {code}...")
        if code in STOCKS:
            df = pull_stock_data(client, code, args.start, args.end)
        else:
            df = pull_etf_data(client, code, args.start, args.end)

        if df.empty:
            logger.warning(f"No data for {code}, skipping.")
            continue

        all_data[code] = df
        logger.info(f"  {code}: {len(df)} rows, {list(df.columns)}")

    logger.info(f"Pulled data for {len(all_data)}/{len(all_codes)} instruments.")

    # 3. 写入 Qlib bin
    logger.info("Converting to Qlib bin format...")
    converter = WindToQlibConverter(args.qlib_dir)
    n_success = converter.convert_batch(all_data, calendar_dates=dates)
    logger.info(f"Converted {n_success}/{len(all_data)} instruments to Qlib bin.")

    # 4. 生成 instruments 文件
    logger.info("Generating instrument files...")
    inst_gen = InstrumentGenerator(args.qlib_dir)

    qlib_stocks = [WindToQlibConverter._normalize_code(c) for c in STOCKS if c in all_data]
    qlib_etfs = [WindToQlibConverter._normalize_code(c) for c in ETFS if c in all_data]
    qlib_all = qlib_stocks + qlib_etfs

    inst_gen.generate("stock", qlib_stocks, end_dates=args.end)
    inst_gen.generate("etf", qlib_etfs, end_dates=args.end)
    inst_gen.generate("all", qlib_all, end_dates=args.end)

    logger.info(f"Instruments: {len(qlib_all)} total ({len(qlib_stocks)} stocks + {len(qlib_etfs)} ETFs)")

    # 5. 断开 Wind
    client.disconnect()
    logger.info("Wind disconnected.")

    # 6. 验证
    logger.info("=== Verification ===")
    from src.core import ensure_qlib, get_ohlcv, list_instruments
    ensure_qlib(args.qlib_dir)

    instruments = list_instruments(as_list=True)
    logger.info(f"Qlib instruments: {len(instruments)}")

    if instruments:
        sample = instruments[0]
        df = get_ohlcv(sample)
        if df is not None and not df.empty:
            logger.info(f"Sample {sample}: {len(df)} days")
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  First: {df.iloc[0].to_dict()}")
            logger.info(f"  Last:  {df.iloc[-1].to_dict()}")
        else:
            logger.warning(f"No data for sample {sample}")

    logger.info("=== Done! ===")


if __name__ == "__main__":
    main()
