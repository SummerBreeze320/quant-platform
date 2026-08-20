"""Data incremental update script.

Runs after market close to update the latest trade day data.

Usage:
    python scripts/update_data.py --date 2026-08-18
    python scripts/update_data.py  # uses today's date
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.pipeline import DataPipeline
from src.utils.logger import setup_logger

logger = setup_logger("INFO")


def main():
    parser = argparse.ArgumentParser(description="Update quant data")
    parser.add_argument("--date", default=None, help="Trade date YYYY-MM-DD")
    parser.add_argument("--qlib-dir", default="data/qlib_bin")
    args = parser.parse_args()

    trade_date = args.date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Updating data for {trade_date}")

    pipeline = DataPipeline(qlib_dir=args.qlib_dir)
    results = pipeline.update_daily(trade_date=trade_date)
    pipeline.disconnect()

    logger.info(f"Update complete: {results}")


if __name__ == "__main__":
    main()
