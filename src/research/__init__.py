"""Research layer: DataHandlers, model zoo, unified strategies, factor library, simulator.

Phase 2 of the quant-platform redesign:
- handlers.py: Alpha158/Alpha360/FundamentalHandler wrappers
- model_zoo.py: 10+ Qlib models with MLflow tracking
- strategy_unified.py: Unified BaseStrategy + TopkDropout/EnhancedIndexing/ETF rotation
- factor_library.py: SOTA factor expressions + IC evaluation
- simulator.py: SimulatorExecutor + nested backtest (daily + intraday)
- enhanced_optimizer.py: EnhancedIndexing portfolio optimizer
- workflow.py: End-to-end research workflow (handler→dataset→model→backtest→report)
- factor_ic.py: Batch factor IC analysis pipeline
"""
from .handlers import (
    get_alpha158_handler,
    get_alpha360_handler,
    FundamentalHandler,
    create_dataset,
    get_handler,
)
from .model_zoo import (
    list_models,
    create_model,
    train_model,
    backtest_model,
    create_topk_dropout_strategy,
)
from .strategy_unified import (
    BaseStrategy,
    TopkDropoutStrategy,
    EnhancedIndexingStrategy,
    ETFRotationStrategy,
    ETFSignalAdapter,
    FactorSignalAdapter,
    create_strategy,
)
from .factor_library import (
    list_factors,
    compute_factors,
    get_alpha158_field_list,
    evaluate_factor_ic,
)
from .simulator import (
    run_backtest_daily,
    run_backtest_with_executor,
    run_nested_backtest,
    run_strategy_backtest_qlib,
)
from .enhanced_optimizer import EnhancedIndexingOptimizer
from .workflow import (
    ResearchWorkflow,
    run_full_workflow,
)
from .factor_ic import (
    FactorICAnalyzer,
    batch_evaluate_factors,
)
