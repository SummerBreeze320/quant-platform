"""Quant Platform CLI — aligned with seven-layer architecture.

Seven layers: Data → Factor → Model → Strategy → Backtest → Execution → Dashboard

Usage:
    python -m src init-data [--start 2015-01-01] [--end 2026-08-01]
    python -m src update-data [--date 2026-08-01]
    python -m src verify-data
    python -m src list-factors
    python -m src backtest --strategy etf_rotation --code SH510300
    python -m src multi-factor-bt --combiner ic_weighted --top-n 10
    python -m src generate-signal --top-n 5
    python -m src trading-sim --initial-cash 1000000
    python -m src serve [--port 8000]
    python -m src mock-data [--stocks 25] [--etfs 20]
    python -m src mine-factors [--iterations 10]
    python -m src evolve-models [--iterations 5]
"""
import sys
import argparse
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
logger = setup_logger("INFO")

QLIB_DIR = "data/qlib_bin"


def cmd_init_data(args):
    from src.data.pipeline import DataPipeline
    pipeline = DataPipeline(qlib_dir=args.qlib_dir, raw_dir="data/raw")
    pipeline.run_full_init(
        start_date=args.start, end_date=args.end,
        include_etf=not args.stock_only, index_only=args.index,
    )


def cmd_update_data(args):
    from src.data.pipeline import DataPipeline
    pipeline = DataPipeline(qlib_dir=args.qlib_dir)
    pipeline.update_daily(trade_date=args.date)
    pipeline.disconnect()


def cmd_verify_data(args):
    from src.data.pipeline import DataPipeline
    pipeline = DataPipeline(qlib_dir=args.qlib_dir)
    results = pipeline.verify_data()
    pipeline.disconnect()
    valid = sum(1 for r in results if r["valid"])
    total = len(results)
    return 0 if valid == total else 1


def cmd_list_factors(args):
    from src.research.factor_library import list_factors, FACTOR_CATEGORIES
    factors = list_factors()
    print(f"\n{'名称':<25} {'类别':<15} {'表达式'}")
    print("-" * 80)
    for f in factors:
        name = f.get("name", "")
        cat = f.get("category", "")
        expr = f.get("expression", "")
        print(f"{name:<25} {cat:<15} {expr[:40]}")
    print(f"\n共 {len(factors)} 个因子")
    print(f"因子类别: {', '.join(FACTOR_CATEGORIES.keys())}")
    print()


def cmd_backtest(args):
    from src.backtest.backtest_engine import BacktestEngine
    from src.research.strategy_unified import create_strategy

    engine = BacktestEngine(qlib_dir=args.qlib_dir)

    params = {}
    if args.strategy == "topk_dropout":
        strategy = create_strategy("topk_dropout", topk=5, n_drop=1)
    elif args.strategy == "enhanced_indexing":
        strategy = create_strategy(
            "enhanced_indexing",
            benchmark=args.benchmark,
            tracking_error_limit=0.05,
        )
    else:
        etf_codes = [args.code] if args.code else ["SH510300"]
        strategy = create_strategy(
            "etf_rotation",
            etf_codes=etf_codes,
            lookback=20,
            top_k=1,
        )

    result = engine.run_strategy_backtest(
        strategy=strategy,
        code=args.code,
        start_date=args.start,
        end_date=args.end,
        init_cash=args.capital,
    )

    if "error" in result:
        logger.error(f"Backtest failed: {result['error']}")
        return 1

    m = result.get("metrics", {})
    print("\n" + "=" * 50)
    print(f"策略回测结果: {result.get('strategy', args.strategy)}")
    print("=" * 50)
    print(f"  标的: {result.get('code', args.code)}")
    print(f"  初始资金: ¥{args.capital:,.0f}")
    print(f"  最终价值: ¥{result.get('final_value', 0):,.0f}")
    print(f"  总收益: {m.get('total_return', 0):.4f}")
    print(f"  年化收益: {m.get('annual_return', 0):.4f}")
    print(f"  夏普比率: {m.get('sharpe_ratio', 0):.4f}")
    print(f"  最大回撤: {m.get('max_drawdown', 0):.4f}")
    print(f"  数据点: {result.get('data_points', 0)}")
    print("=" * 50)
    return 0


def cmd_multi_factor_bt(args):
    from src.execution.signal_generator import SignalGenerator
    from src.backtest.backtest_engine import BacktestEngine
    from src.backtest.risk_metrics import calc_all_risk_metrics
    from src.core import list_instruments, ensure_qlib
    import pandas as pd

    ensure_qlib(args.qlib_dir)
    all_inst = list_instruments()
    stock_codes = [c for c in all_inst if c.startswith("SH6") or c.startswith("SZ0")][:args.n_stocks]
    logger.info(f"Selected {len(stock_codes)} stocks: {stock_codes}")

    gen = SignalGenerator(
        qlib_dir=args.qlib_dir,
        combiner_method=args.combiner,
        top_n=args.top_n,
        max_position=args.max_position,
        forward_period=args.forward_period,
    )
    signal = gen.generate(codes=stock_codes, objective=args.objective)

    if "error" in signal:
        logger.error(f"Signal generation failed: {signal['error']}")
        return 1

    print("\n" + "=" * 50)
    print("多因子组合回测结果")
    print("=" * 50)
    print(f"  信号日期: {signal['signal_date']}")
    print(f"  选股数量: {signal['n_selected']}")
    print(f"  股票池: {signal['n_universe']}")
    print(f"  合成方法: {signal['combiner_method']}")

    print("\n目标持仓:")
    for stock, weight in sorted(signal["target_weights"].items(),
                                 key=lambda x: -x[1]):
        print(f"  {stock}: {weight:.2%}")

    fw = signal.get("factor_weights", {})
    if fw:
        print("\n因子权重:")
        for name, val in fw.items():
            print(f"  {name}: {val:.4f}")

    engine = BacktestEngine(qlib_dir=args.qlib_dir)
    target_weights = signal.get("target_weights", {})
    if target_weights:
        primary_code = list(target_weights.keys())[0]
        bt_result = engine.run_simple_backtest(
            signals=pd.DataFrame({primary_code: 1.0}, index=[signal["signal_date"]]),
            init_cash=args.capital,
        )
        if "error" not in bt_result:
            nav_series = bt_result.get("equity_curve", pd.Series(dtype=float))
            daily_ret = bt_result.get("daily_returns", pd.Series(dtype=float))
            rm = calc_all_risk_metrics(daily_ret) if len(daily_ret) > 0 else {}
            print("\n风险指标:")
            print(f"  总收益: {rm.get('total_return', 0):.4f}")
            print(f"  年化收益: {rm.get('annual_return', 0):.4f}")
            print(f"  夏普比率: {rm.get('sharpe_ratio', 0):.4f}")
            print(f"  最大回撤: {rm.get('max_drawdown', 0):.4f}")
            print(f"  Sortino: {rm.get('sortino_ratio', 0):.4f}")
            print(f"  Calmar: {rm.get('calmar_ratio', 0):.4f}")
    print("=" * 50)
    return 0


def cmd_generate_signal(args):
    from src.execution.signal_generator import SignalGenerator
    from src.core import list_instruments, ensure_qlib

    ensure_qlib(args.qlib_dir)
    all_inst = list_instruments()
    codes = [c for c in all_inst if c.startswith("SH6") or c.startswith("SZ0")][:args.n_stocks]

    gen = SignalGenerator(
        qlib_dir=args.qlib_dir,
        combiner_method=args.combiner,
        top_n=args.top_n,
        max_position=args.max_position,
    )
    signal = gen.generate(codes=codes, as_of_date=args.as_of_date)

    if "error" in signal:
        logger.error(f"Signal generation failed: {signal['error']}")
        return 1

    print("\n" + "=" * 50)
    print(f"交易信号 ({signal['signal_date']})")
    print("=" * 50)
    print(f"  信号生成时间: {signal['generated_at']}")
    print(f"  合成方法: {signal['combiner_method']}")
    print(f"  选股数量: {signal['n_selected']}")
    print(f"  股票池: {signal['n_universe']}")

    print("\n目标持仓:")
    for stock, weight in sorted(signal["target_weights"].items(),
                                 key=lambda x: -x[1]):
        print(f"  {stock}: {weight:.2%}")

    if signal.get("factor_weights"):
        print("\n因子权重:")
        for name, w in signal["factor_weights"].items():
            print(f"  {name}: {w:.4f}")
    print("=" * 50)
    return 0


def cmd_trading_sim(args):
    from src.execution.signal_generator import SignalGenerator
    from src.execution.trading_gateway import SimulatedBroker, OrderManager
    from src.execution.portfolio_scheduler import PortfolioScheduler
    from src.execution.trade_logger import TradeLogger
    from src.execution.monitor import TradingMonitor
    from src.core import list_instruments, get_ohlcv, ensure_qlib
    import pandas as pd

    logger.info("=== 模拟盘交易启动 ===")

    ensure_qlib(args.qlib_dir)
    all_inst = list_instruments()
    codes = [c for c in all_inst if c.startswith("SH6") or c.startswith("SZ0")][:15]

    price_data = {}
    for code in codes:
        try:
            df = get_ohlcv(code)
            if df is not None and len(df) > 20:
                price_data[code] = pd.DataFrame({
                    "open": df["open"], "high": df["high"], "low": df["low"],
                    "close": df["close"], "volume": df["volume"],
                })
        except Exception:
            pass

    if not price_data:
        logger.error("无可用数据")
        return 1

    trade_logger = TradeLogger("data/trade_logs")
    broker = SimulatedBroker(
        initial_cash=args.initial_cash,
        price_data=price_data,
        logger=trade_logger,
    )
    broker.connect()
    order_manager = OrderManager(broker, trade_logger)
    scheduler = PortfolioScheduler(broker, trade_logger=trade_logger)
    monitor = TradingMonitor(trade_logger)

    sample_df = list(price_data.values())[0]
    all_dates = sample_df.index
    all_dates = all_dates[all_dates >= pd.Timestamp(args.start)]
    all_dates = all_dates[all_dates <= pd.Timestamp(args.end)]

    date_series = pd.Series(all_dates, index=all_dates)
    rebalance_dates = date_series.groupby(date_series.index.to_period("M")).last().values

    nav_history = []
    n_rebalances = 0

    for date in all_dates:
        broker.set_current_date(date)
        is_rebalance = any(pd.Timestamp(d) == date for d in rebalance_dates)

        if is_rebalance:
            gen = SignalGenerator(
                qlib_dir=args.qlib_dir,
                combiner_method=args.combiner,
                top_n=args.top_n,
                max_position=args.max_position,
            )
            signal = gen.generate(codes=codes, as_of_date=date.strftime("%Y-%m-%d"))

            if "error" not in signal:
                target_weights = signal["target_weights"]
                account = broker.get_account()
                orders = scheduler.generate_rebalance_plan(
                    target_weights, account["total_value"]
                )
                exec_result = scheduler.execute_rebalance(orders, order_manager)
                monitor.update_state(
                    nav=account["total_value"],
                    target_weights=target_weights,
                    data_time=pd.Timestamp(date).to_pydatetime(),
                )
                n_rebalances += 1
                logger.info(f"调仓 {date.strftime('%Y-%m-%d')}: "
                             f"提交{exec_result['submitted']}笔")

        account = broker.get_account()
        nav_history.append({
            "date": date.strftime("%Y-%m-%d"),
            "nav": account["total_value"],
        })

    final = broker.get_account()
    positions = broker.get_position()
    order_summary = order_manager.get_order_summary()
    monitor_status = monitor.get_status()

    print("\n" + "=" * 50)
    print("模拟盘交易结果")
    print("=" * 50)
    print(f"  初始资金: ¥{args.initial_cash:,.0f}")
    print(f"  最终资产: ¥{final['total_value']:,.0f}")
    print(f"  总盈亏: ¥{final['total_pnl']:,.0f} ({final['total_pnl_pct']:.2%})")
    print(f"  调仓次数: {n_rebalances}")
    print(f"  提交订单: {order_summary['total_submitted']}")
    print(f"  已成交: {order_summary['filled_count']}")
    print(f"  告警数: {monitor_status['n_alerts']}")
    print(f"  当前回撤: {monitor_status['current_drawdown']:.2%}")

    if positions:
        print("\n持仓明细:")
        for stock, pos in positions.items():
            print(f"  {stock}: {pos['volume']:.0f}股, "
                  f"成本{pos['cost_price']:.2f}, "
                  f"盈亏¥{pos['pnl']:,.0f} ({pos['pnl_pct']:.2%})")
    print("=" * 50)
    return 0


def cmd_serve(args):
    import uvicorn
    uvicorn.run("src.dashboard.api.main:app", host=args.host, port=args.port, reload=args.reload)


def cmd_mock_data(args):
    from src.data.mock_data import MockDataGenerator
    gen = MockDataGenerator(qlib_dir=args.qlib_dir)
    stats = gen.generate_qlib_data(
        n_stocks=args.stocks, n_etfs=args.etfs,
        start=args.start, end=args.end, seed=args.seed,
    )
    logger.info(f"Mock data generated: {stats}")


def cmd_run_factor_mining(args):
    from src.rd_agent.factor_runner import FactorRunner
    runner = FactorRunner()
    results = runner.run_factor_mining(max_iterations=args.iterations, scenario=args.scenario)
    if results.get("success"):
        factors = runner.get_best_factors(results, min_ic=args.min_ic)
        print(f"\nBest factors ({len(factors)}):")
        for f in factors:
            print(f"  {f['name']}: IC={f['ic']:.4f}, ICIR={f['icir']:.4f}")
    else:
        logger.error(f"Factor mining failed: {results.get('error')}")
        return 1
    return 0


def cmd_evolve_models(args):
    from src.rd_agent.model_runner import ModelRunner
    runner = ModelRunner()
    results = runner.run_model_evolution(
        max_iterations=args.iterations,
        base_model=args.base_model,
    )
    if results.get("success"):
        config = runner.get_model_config(results)
        print(f"\nBest model config: {config}")
    else:
        logger.error(f"Model evolution failed: {results.get('error')}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(prog="quant-platform", description="AI Quant Platform — 七层架构")
    parser.add_argument("--qlib-dir", default=QLIB_DIR)
    subparsers = parser.add_subparsers(dest="command")

    # ─── Data Layer ───
    p = subparsers.add_parser("init-data", help="初始化数据(Wind→Qlib bin)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2026-08-01")
    p.add_argument("--index", default=None)
    p.add_argument("--stock-only", action="store_true")
    p.set_defaults(func=cmd_init_data)

    p = subparsers.add_parser("update-data", help="增量更新")
    p.add_argument("--date", default=None)
    p.set_defaults(func=cmd_update_data)

    p = subparsers.add_parser("verify-data", help="验证数据完整性")
    p.set_defaults(func=cmd_verify_data)

    p = subparsers.add_parser("mock-data", help="生成Mock数据(离线开发)")
    p.add_argument("--stocks", type=int, default=25)
    p.add_argument("--etfs", type=int, default=20)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_mock_data)

    # ─── Factor Layer ───
    p = subparsers.add_parser("list-factors", help="列出Qlib因子库")
    p.set_defaults(func=cmd_list_factors)

    # ─── Strategy + Backtest Layer ───
    p = subparsers.add_parser("backtest", help="策略回测(Qlib原生引擎)")
    p.add_argument("--strategy", default="etf_rotation",
                   choices=["etf_rotation", "topk_dropout", "enhanced_indexing"])
    p.add_argument("--code", default="SH510300")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2026-08-01")
    p.add_argument("--benchmark", default="SH000300")
    p.add_argument("--capital", type=float, default=1_000_000)
    p.set_defaults(func=cmd_backtest)

    p = subparsers.add_parser("multi-factor-bt", help="多因子组合回测")
    p.add_argument("--combiner", default="ic_weighted", choices=["ic_weighted", "equal_weight"])
    p.add_argument("--objective", default="max_sharpe", choices=["max_sharpe", "min_vol", "equal_weight"])
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--max-position", type=float, default=0.15)
    p.add_argument("--n-stocks", type=int, default=20)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--forward-period", type=int, default=5)
    p.set_defaults(func=cmd_multi_factor_bt)

    # ─── Execution Layer ───
    p = subparsers.add_parser("generate-signal", help="生成交易信号")
    p.add_argument("--combiner", default="ic_weighted", choices=["ic_weighted", "equal_weight"])
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--max-position", type=float, default=0.15)
    p.add_argument("--as-of-date", default=None)
    p.add_argument("--n-stocks", type=int, default=15)
    p.set_defaults(func=cmd_generate_signal)

    p = subparsers.add_parser("trading-sim", help="模拟盘交易")
    p.add_argument("--initial-cash", type=float, default=1_000_000)
    p.add_argument("--combiner", default="ic_weighted", choices=["ic_weighted", "equal_weight"])
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--max-position", type=float, default=0.15)
    p.add_argument("--start", default="2021-06-01")
    p.add_argument("--end", default="2024-12-31")
    p.set_defaults(func=cmd_trading_sim)

    # ─── RD-Agent ───
    p = subparsers.add_parser("mine-factors", help="RD-Agent因子挖掘")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--scenario", default="fin_factor", choices=["fin_factor", "fin_model", "fin_quant"])
    p.add_argument("--min-ic", type=float, default=0.03)
    p.set_defaults(func=cmd_run_factor_mining)

    p = subparsers.add_parser("evolve-models", help="RD-Agent模型进化")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--base-model", default="lightgbm", choices=["lightgbm", "xgboost"])
    p.set_defaults(func=cmd_evolve_models)

    # ─── Dashboard ───
    p = subparsers.add_parser("serve", help="启动Dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
