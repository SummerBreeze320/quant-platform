"""Data initialization script.

Usage:
    # Full init from Wind (requires Wind terminal running)
    python scripts/init_data.py --market all --start 2015-01-01 --end 2026-08-01

    # Index only (e.g. CSI 300)
    python scripts/init_data.py --market stock --index 000300.SH

    # Mock data for offline development (no Wind needed)
    python scripts/init_data.py --mock --stocks 25 --etfs 20

    # Verify existing data
    python scripts/init_data.py --verify
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger("INFO")


def cmd_init_wind(args):
    """Full data initialization from Wind terminal."""
    from src.data.pipeline import DataPipeline

    pipeline = DataPipeline(qlib_dir=args.qlib_dir, raw_dir="data/raw")
    stats = pipeline.run_full_init(
        start_date=args.start,
        end_date=args.end,
        include_etf=not args.stock_only,
        index_only=args.index,
    )
    logger.info(f"Initialization stats: {stats}")
    return stats


def cmd_init_mock(args):
    """Generate mock data for offline development."""
    from src.data.mock_data import MockDataGenerator

    gen = MockDataGenerator(qlib_dir=args.qlib_dir)
    stats = gen.generate_qlib_data(
        n_stocks=args.stocks,
        n_etfs=args.etfs,
        start=args.start,
        end=args.end,
        seed=args.seed,
    )
    logger.info(f"Mock data stats: {stats}")
    return stats


def cmd_verify(args):
    """Verify existing data integrity."""
    from src.data.pipeline import DataPipeline

    pipeline = DataPipeline(qlib_dir=args.qlib_dir)
    results = pipeline.verify_data()
    valid = sum(1 for r in results if r["valid"])
    total = len(results)
    logger.info(f"Verification: {valid}/{total} instruments valid")
    return 0 if valid == total else 1


def main():
    parser = argparse.ArgumentParser(
        description="Initialize quant data from Wind or generate mock data"
    )
    parser.add_argument("--qlib-dir", default="data/qlib_bin")
    parser.add_argument("--mock", action="store_true", help="Generate mock data instead of Wind")
    parser.add_argument("--verify", action="store_true", help="Verify data integrity")
    parser.add_argument("--market", choices=["stock", "etf", "all"], default="all")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--index", default=None, help="Index code, e.g. 000300.SH")
    parser.add_argument("--stock-only", action="store_true")
    parser.add_argument("--stocks", type=int, default=25, help="Number of mock stocks")
    parser.add_argument("--etfs", type=int, default=20, help="Number of mock ETFs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mock data")
    args = parser.parse_args()

    if args.verify:
        return cmd_verify(args)
    if args.mock:
        return cmd_init_mock(args)
    return cmd_init_wind(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
