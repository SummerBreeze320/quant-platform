"""Trading Engine startup script.

Usage:
    python scripts/run_trading_engine.py --mode auto
    python scripts/run_trading_engine.py --mode manual
    python scripts/run_trading_engine.py --mode manual --date 2024-06-01

Modes:
    auto   — Start scheduler, run daily at scheduled times
    manual — Run one trading day immediately
    status — Show current engine status
"""
import argparse
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Trading Engine Launcher")
    parser.add_argument(
        "--mode",
        choices=["auto", "manual", "status"],
        default="manual",
        help="Run mode: auto=scheduler, manual=one-shot, status=status only",
    )
    parser.add_argument("--date", default=None, help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--topk", type=int, default=5, help="Number of stocks")
    parser.add_argument("--benchmark", default="SH000300", help="Benchmark code")
    parser.add_argument("--model-path", default=None, help="Model file path")
    parser.add_argument("--handler", default="alpha158", help="Data handler type")

    args = parser.parse_args()

    from src.execution.trading_engine import TradingEngine
    from src.execution.scheduler import DailyScheduler, ScheduleConfig

    if args.mode == "status":
        engine = TradingEngine(
            init_capital=args.capital,
            benchmark=args.benchmark,
            topk=args.topk,
            model_path=args.model_path,
            handler_type=args.handler,
        )
        status = engine.get_status()
        print("\n=== Trading Engine Status ===")
        for k, v in status.items():
            print(f"  {k}: {v}")
        return

    # Create and initialize engine
    engine = TradingEngine(
        init_capital=args.capital,
        benchmark=args.benchmark,
        topk=args.topk,
        model_path=args.model_path,
        handler_type=args.handler,
    )

    if args.mode == "manual":
        logger.info("=== Manual Mode: Single Trading Day ===")
        record = engine.run_daily(args.date)

        print(f"\n=== Trading Day Result ===")
        print(f"  Date: {record.trade_date}")
        print(f"  Phase: {record.phase.value}")
        print(f"  NAV: {record.nav:.2f}")
        print(f"  PnL: {record.pnl_pct:.2%}")
        print(f"  Orders: {len(record.orders)}")
        if record.error:
            print(f"  Error: {record.error}")

    elif args.mode == "auto":
        logger.info("=== Auto Mode: Starting Scheduler ===")
        scheduler = DailyScheduler(ScheduleConfig())
        scheduler.setup(engine)
        scheduler.start()

        print("\nScheduler started. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Stopping scheduler...")
            scheduler.stop()
            print("Scheduler stopped.")


if __name__ == "__main__":
    main()
