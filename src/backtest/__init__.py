"""Backtest layer: Qlib-native simulator, metrics, risk, attribution.

Aligned with seven-layer architecture. Old custom backtest engine,
portfolio builder, and optimizer have been removed in favor of
Qlib-native SimulatorExecutor and EnhancedIndexingOptimizer.
"""
from .backtest_engine import BacktestEngine
from .metrics import calc_all_metrics, calc_annual_return, calc_sharpe_ratio, calc_max_drawdown
from .constraints import PortfolioConstraints, default_constraints
from .cost_model import CostModel, create_default_cost_model
from .risk_metrics import calc_all_risk_metrics, calc_sortino_ratio, calc_calmar_ratio
from .attribution import (
    BrinsonAttribution,
    FactorReturnAttribution,
    IndustryAttribution,
    BarraRiskDecomposition,
    calc_factor_returns,
)
