"""Agent层 — AI驱动的量化研究智能体

8个专业化Agent组成策略研发闭环:
- ResearchAgent:   主控Agent，协调全流程
- FactorAgent:     因子空间搜索 + IC筛选
- ModelAgent:      模型空间搜索 + 超参优化
- PortfolioAgent:  组合空间搜索 + 权重优化
- RiskAgent:       市场状态识别 + 动态仓位
- BacktestAgent:   批量回测 + 综合评分
- CriticAgent:     过拟合校验 + 策略评审
- ExperimentAgent: 实验编排 + 结果比较

Usage:
    from src.agents import ResearchAgent, InvestmentObjective

    agent = ResearchAgent()
    agent.initialize()
    results = agent.execute(InvestmentObjective(goal="max_return"))
"""
from .base import BaseAgent, AgentState
from .contracts import (
    MarketState,
    RiskPreference,
    ExperimentStatus,
    InvestmentObjective,
    FactorSearchRequest,
    FactorSet,
    ModelSearchRequest,
    ModelConfig,
    PortfolioSearchRequest,
    PortfolioConfig,
    RiskAssessmentRequest,
    RiskAssessment,
    ExperimentPlan,
    BacktestRequest,
    BacktestMetrics,
    BacktestResult,
    ReviewRequest,
    ReviewResult,
    ExperimentResult,
)
from .market_state import MarketStateRecognizer
from .research_agent import ResearchAgent
from .factor_agent import FactorAgent
from .model_agent import ModelAgent
from .portfolio_agent import PortfolioAgent
from .risk_agent import RiskAgent
from .backtest_agent import BacktestAgent
from .critic_agent import CriticAgent
from .experiment_agent import ExperimentAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentState",
    # Contracts
    "MarketState",
    "RiskPreference",
    "ExperimentStatus",
    "InvestmentObjective",
    "FactorSearchRequest",
    "FactorSet",
    "ModelSearchRequest",
    "ModelConfig",
    "PortfolioSearchRequest",
    "PortfolioConfig",
    "RiskAssessmentRequest",
    "RiskAssessment",
    "ExperimentPlan",
    "BacktestRequest",
    "BacktestMetrics",
    "BacktestResult",
    "ReviewRequest",
    "ReviewResult",
    "ExperimentResult",
    # Agents
    "MarketStateRecognizer",
    "ResearchAgent",
    "FactorAgent",
    "ModelAgent",
    "PortfolioAgent",
    "RiskAgent",
    "BacktestAgent",
    "CriticAgent",
    "ExperimentAgent",
]
