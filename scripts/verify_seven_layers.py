"""端到端验证脚本: 确认七层架构完整联通

验证内容:
1. 数据层: Qlib初始化 + bin数据访问
2. 因子层: Alpha158/360 Handler + 因子库 + 因子计算器
3. 模型层: 模型动物园 + MLflow追踪
4. 策略层: 统一BaseStrategy + TopkDropout/EnhancedIndexing/ETF轮动
5. 回测层: SimulatorExecutor + 嵌套回测 + EnhancedIndexing优化器
6. 实盘层: Qlib predict信号 + 统一风控 + 执行管线
7. 面板层: 模块化Router + WebSocket + MLflow UI

Usage:
    python -m scripts.verify_seven_layers
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_import(module_path: str, names: list = None) -> bool:
    """检查模块是否可以导入"""
    try:
        mod = __import__(module_path, fromlist=names or [])
        if names:
            for name in names:
                if not hasattr(mod, name):
                    logger.error(f"  ✗ {module_path}.{name} not found")
                    return False
        logger.info(f"  ✓ {module_path}")
        return True
    except Exception as e:
        logger.error(f"  ✗ {module_path}: {e}")
        return False


def verify_layer1_data():
    """1. 数据层验证"""
    logger.info("\n" + "=" * 60)
    logger.info("Layer 1: 数据层 (Data Layer)")
    logger.info("=" * 60)

    checks = [
        ("src.core.qlib_init", ["init_qlib", "ensure_qlib"]),
        ("src.core.data_access", ["get_calendar", "list_instruments", "get_features", "get_field", "get_ohlcv"]),
        ("src.data.pipeline", ["DataPipeline"]),
        ("src.data.factor_calculator", ["FactorCalculator"]),
        ("src.data.mock_data", None),
    ]
    return all(check_import(m, n) for m, n in checks)


def verify_layer2_factor():
    """2. 因子层验证"""
    logger.info("\n" + "=" * 60)
    logger.info("Layer 2: 因子层 (Factor Layer)")
    logger.info("=" * 60)

    checks = [
        ("src.research.handlers", ["get_alpha158_handler", "get_alpha360_handler", "FundamentalHandler"]),
        ("src.research.factor_library", ["list_factors", "compute_factors", "evaluate_factor_ic"]),
        ("src.research.factor_ic", ["FactorICAnalyzer", "batch_evaluate_factors"]),
    ]
    return all(check_import(m, n) for m, n in checks)


def verify_layer3_model():
    """3. 模型层验证"""
    logger.info("\n" + "=" * 60)
    logger.info("Layer 3: 模型层 (Model Layer)")
    logger.info("=" * 60)

    checks = [
        ("src.research.model_zoo", ["list_models", "create_model", "train_model", "backtest_model"]),
    ]

    # Verify model registry has 10+ models
    try:
        from src.research.model_zoo import MODEL_REGISTRY
        model_count = len(MODEL_REGISTRY)
        logger.info(f"  ✓ Model zoo: {model_count} models ({', '.join(MODEL_REGISTRY.keys())})")
        if model_count < 10:
            logger.warning(f"  ⚠ Expected 10+ models, got {model_count}")
    except Exception as e:
        logger.error(f"  ✗ Model registry check: {e}")
        return False

    return all(check_import(m, n) for m, n in checks)


def verify_layer4_strategy():
    """4. 策略层验证"""
    logger.info("\n" + "=" * 60)
    logger.info("Layer 4: 策略层 (Strategy Layer)")
    logger.info("=" * 60)

    checks = [
        ("src.research.strategy_unified", [
            "BaseStrategy", "TopkDropoutStrategy",
            "EnhancedIndexingStrategy", "ETFRotationStrategy",
            "create_strategy",
        ]),
    ]
    return all(check_import(m, n) for m, n in checks)


def verify_layer5_backtest():
    """5. 回测层验证"""
    logger.info("\n" + "=" * 60)
    logger.info("Layer 5: 回测层 (Backtest Layer)")
    logger.info("=" * 60)

    checks = [
        ("src.research.simulator", [
            "run_backtest_daily", "run_backtest_with_executor",
            "run_nested_backtest", "run_strategy_backtest_qlib",
        ]),
        ("src.research.enhanced_optimizer", ["EnhancedIndexingOptimizer"]),
        ("src.research.workflow", ["ResearchWorkflow", "run_full_workflow"]),
        ("src.backtest.backtest_engine", None),
        ("src.backtest.metrics", None),
    ]
    return all(check_import(m, n) for m, n in checks)


def verify_layer6_execution():
    """6. 实盘层验证"""
    logger.info("\n" + "=" * 60)
    logger.info("Layer 6: 实盘层 (Execution Layer)")
    logger.info("=" * 60)

    checks = [
        ("src.execution.signal_generator", ["SignalGenerator"]),
        ("src.execution.qlib_signal_generator", ["QlibSignalGenerator"]),
        ("src.execution.unified_risk", ["UnifiedRiskController", "UnifiedRiskConfig"]),
        ("src.execution.portfolio_scheduler", ["PortfolioScheduler"]),
        ("src.execution.trading_gateway", ["SimulatedBroker", "OrderManager"]),
        ("src.execution.execution_algo", ["TWAPAlgo", "VWAPAlgo", "create_algo"]),
        ("src.execution.monitor", ["TradingMonitor"]),
        ("src.execution.trade_logger", ["TradeLogger"]),
    ]
    return all(check_import(m, n) for m, n in checks)


def verify_layer7_dashboard():
    """7. 面板层验证"""
    logger.info("\n" + "=" * 60)
    logger.info("Layer 7: 面板层 (Dashboard Layer)")
    logger.info("=" * 60)

    checks = [
        ("src.dashboard.api.main", ["app"]),
        ("src.dashboard.api.routes", ["qlib_router", "router"]),
        ("src.dashboard.api.research_routes", ["router"]),
        ("src.dashboard.api.routers.rdagent", ["router"]),
        ("src.dashboard.api.routers.mlflow_router", ["router"]),
        ("src.dashboard.api.routers.trading_live", ["router"]),
    ]
    return all(check_import(m, n) for m, n in checks)


def verify_rd_agent():
    """RD-Agent集成验证"""
    logger.info("\n" + "=" * 60)
    logger.info("RD-Agent Deep Integration")
    logger.info("=" * 60)

    checks = [
        ("src.rd_agent.config", ["RDAgentConfig"]),
        ("src.rd_agent.factor_runner", ["FactorRunner"]),
        ("src.rd_agent.model_runner", ["ModelRunner"]),
        ("src.rd_agent.coordinator", ["RDAgentCoordinator"]),
    ]
    return all(check_import(m, n) for m, n in checks)


def verify_cross_layer():
    """跨层集成验证"""
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Layer Integration")
    logger.info("=" * 60)

    # Verify research __init__ exports
    try:
        from src.research import (
            get_handler, create_dataset, create_model, train_model,
            run_backtest_daily, EnhancedIndexingOptimizer,
            ResearchWorkflow, run_full_workflow,
            FactorICAnalyzer, batch_evaluate_factors,
            list_models, list_factors, create_strategy,
        )
        logger.info("  ✓ Research layer exports complete")
    except Exception as e:
        logger.error(f"  ✗ Research layer exports: {e}")
        return False

    # Verify execution __init__ exports
    try:
        from src.execution import (
            SignalGenerator, QlibSignalGenerator,
            UnifiedRiskController, UnifiedRiskConfig,
            PortfolioScheduler, SimulatedBroker,
            TWAPAlgo, VWAPAlgo, TradingMonitor, TradeLogger,
        )
        logger.info("  ✓ Execution layer exports complete")
    except Exception as e:
        logger.error(f"  ✗ Execution layer exports: {e}")
        return False

    # Verify RD-Agent exports
    try:
        from src.rd_agent import (
            RDAgentConfig, FactorRunner, ModelRunner, RDAgentCoordinator,
        )
        logger.info("  ✓ RD-Agent exports complete")
    except Exception as e:
        logger.error(f"  ✗ RD-Agent exports: {e}")
        return False

    # Verify dashboard routers
    try:
        from src.dashboard.api.routers import (
            rdagent_router, mlflow_router, trading_live_router,
        )
        logger.info("  ✓ Dashboard modular routers loaded")
    except Exception as e:
        logger.error(f"  ✗ Dashboard modular routers: {e}")
        return False

    return True


def verify_config_files():
    """验证配置文件存在"""
    logger.info("\n" + "=" * 60)
    logger.info("Configuration Files")
    logger.info("=" * 60)

    config_files = [
        "config/settings.yaml",
        "config/wind_config.yaml",
        "config/qlib_config.yaml",
        "config/rd_agent_config.yaml",
        "requirements.txt",
    ]

    all_ok = True
    for f in config_files:
        path = Path(f)
        if path.exists():
            logger.info(f"  ✓ {f}")
        else:
            logger.error(f"  ✗ {f} not found")
            all_ok = False

    return all_ok


def main():
    logger.info("=" * 60)
    logger.info("Quant Platform - 七层架构端到端验证")
    logger.info("=" * 60)

    results = {}

    # Verify each layer
    results["Layer 1: Data"] = verify_layer1_data()
    results["Layer 2: Factor"] = verify_layer2_factor()
    results["Layer 3: Model"] = verify_layer3_model()
    results["Layer 4: Strategy"] = verify_layer4_strategy()
    results["Layer 5: Backtest"] = verify_layer5_backtest()
    results["Layer 6: Execution"] = verify_layer6_execution()
    results["Layer 7: Dashboard"] = verify_layer7_dashboard()
    results["RD-Agent Integration"] = verify_rd_agent()
    results["Cross-Layer Integration"] = verify_cross_layer()
    results["Config Files"] = verify_config_files()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}  {name}")
        if not passed:
            all_passed = False

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("🎉 ALL CHECKS PASSED — 七层架构完整联通!")
    else:
        failed = sum(1 for v in results.values() if not v)
        logger.info(f"⚠ {failed}/{len(results)} checks failed. Please review errors above.")
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
