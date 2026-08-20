"""Agent输入输出数据结构契约

定义所有Agent之间通信的Pydantic模型，确保类型安全和接口一致性。
每个Agent的输入输出都严格使用此处定义的数据结构。
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class MarketState(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


class RiskPreference(str, Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    APPROVED = "approved"
    REJECTED = "rejected"


class HoldingPeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class InvestmentObjective(BaseModel):
    """用户投资目标 — 整个系统的输入起点

    支持自然语言解析为结构化目标，如:
      "寻找A股稳定Alpha, 年化15%, 最大回撤<10%, 夏普>1.5"
    """
    universe: str = "A_share"
    goal: str = "stable_alpha"
    target_return: Optional[float] = None
    max_drawdown: float = 0.20
    min_sharpe: Optional[float] = None
    turnover_limit: float = 0.50
    tracking_error_limit: float = 0.05
    risk_preference: RiskPreference = RiskPreference.BALANCED
    benchmark: str = "SH000300"
    holding_period: HoldingPeriod = HoldingPeriod.DAILY
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    init_cash: float = 1000000
    # 自然语言原始输入
    raw_input: Optional[str] = None
    constraints: Dict[str, float] = Field(
        default_factory=lambda: {
            "max_drawdown": 0.20,
            "max_turnover": 0.50,
            "tracking_error_limit": 0.05,
        }
    )


class FactorSearchRequest(BaseModel):
    """ResearchAgent → FactorAgent"""
    universe: str = "A_share"
    categories: List[str] = Field(
        default_factory=lambda: ["price_volume", "momentum", "volatility", "fundamental", "alternative"]
    )
    max_factors: int = 20
    ic_threshold: float = 0.02
    icir_threshold: float = 0.3
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class FactorSet(BaseModel):
    """FactorAgent → ResearchAgent"""
    factors: List[str] = Field(default_factory=list)
    ic_scores: Dict[str, float] = Field(default_factory=dict)
    factor_weights: Dict[str, float] = Field(default_factory=dict)
    category_coverage: Dict[str, int] = Field(default_factory=dict)


class ModelSearchRequest(BaseModel):
    """ResearchAgent → ModelAgent"""
    model_types: List[str] = Field(
        default_factory=lambda: ["lightgbm", "xgboost", "transformer", "lstm"]
    )
    param_search: bool = True
    cv_folds: int = 5
    factor_set: Optional[FactorSet] = None
    universe: str = "all"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ModelConfig(BaseModel):
    """ModelAgent → ResearchAgent"""
    model_type: str = "lightgbm"
    params: Dict[str, Any] = Field(default_factory=dict)
    cv_score: float = 0.0
    ic_score: float = 0.0
    feature_importance: Dict[str, float] = Field(default_factory=dict)


class PortfolioSearchRequest(BaseModel):
    """ResearchAgent → PortfolioAgent"""
    n_holdings_range: List[int] = Field(default_factory=lambda: [5, 10, 20, 50])
    weight_schemes: List[str] = Field(
        default_factory=lambda: ["equal_weight", "score_weighted", "enhanced_indexing", "min_variance"]
    )
    rebalance_freq: List[str] = Field(default_factory=lambda: ["D", "W", "M"])
    max_position: float = 0.15
    tracking_error_limit: float = 0.05
    benchmark: str = "SH000300"
    factor_set: Optional[FactorSet] = None
    model_setting: Optional[ModelConfig] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PortfolioConfig(BaseModel):
    """PortfolioAgent → ResearchAgent"""
    n_holdings: int = 10
    weight_scheme: str = "enhanced_indexing"
    rebalance_freq: str = "W"
    max_position: float = 0.15
    tracking_error_limit: float = 0.05
    weights: Dict[str, float] = Field(default_factory=dict)


class RiskAssessmentRequest(BaseModel):
    """ResearchAgent → RiskAgent

    支持两种模式:
    1. 自动模式: 只传 benchmark + as_of_date, Agent 内部调用 MarketStateRecognizer
    2. 直接模式: 直接传入市场指标 (波动率/趋势/恐慌/成交量), 跳过 recognizer
    """
    benchmark: str = "SH000300"
    as_of_date: Optional[str] = None
    portfolio_config: Optional[PortfolioConfig] = None
    # 直接输入模式 (P5增强: 用户/AI主管可直接传入市场指标)
    volatility: Optional[float] = None
    trend_signal: Optional[float] = None
    panic_index: Optional[float] = None
    volume_ratio: Optional[float] = None


class RiskAssessment(BaseModel):
    """RiskAgent → ResearchAgent

    输出仓位决策: Bull=100%, Sideways=50%, Bear=20%
    """
    market_state: MarketState = MarketState.SIDEWAYS
    position_ratio: float = 0.5
    risk_score: float = 0.5
    volatility_20d: float = 0.0
    trend_signal: float = 0.0
    volume_ratio: float = 1.0
    panic_index: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    # 仓位决策依据
    position_reason: str = ""


class ExperimentPlan(BaseModel):
    """ResearchAgent → ExperimentAgent: 一组实验方案"""
    experiment_id: str
    factor_set: FactorSet
    model_setting: ModelConfig
    portfolio_config: PortfolioConfig
    risk_assessment: Optional[RiskAssessment] = None
    objective: Optional[InvestmentObjective] = None


class BacktestRequest(BaseModel):
    """ExperimentAgent → BacktestAgent"""
    experiment_id: str
    factor_names: List[str] = Field(default_factory=list)
    model_type: str = "lightgbm"
    model_params: Dict[str, Any] = Field(default_factory=dict)
    n_holdings: int = 10
    weight_scheme: str = "enhanced_indexing"
    rebalance_freq: str = "W"
    max_position: float = 0.15
    tracking_error_limit: float = 0.05
    start_date: str = "2015-01-01"
    end_date: str = "2026-08-19"
    init_cash: float = 1000000
    benchmark: str = "SH000300"


class BacktestMetrics(BaseModel):
    """回测指标结果 — 验证策略质量, 不只看收益"""
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    turnover: float = 0.0
    win_rate: float = 0.0
    ic_mean: float = 0.0
    icir: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    transaction_cost: float = 0.0
    net_return_after_cost: float = 0.0
    # 日内分析指标
    intraday_entry_accuracy: float = 0.0
    intraday_exit_accuracy: float = 0.0


class BacktestResult(BaseModel):
    """BacktestAgent → ExperimentAgent"""
    experiment_id: str
    metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)
    score: float = 0.0
    net_value_series: List[float] = Field(default_factory=list)
    drawdown_series: List[float] = Field(default_factory=list)
    status: str = "completed"
    error: Optional[str] = None


class ReviewRequest(BaseModel):
    """ExperimentAgent → CriticAgent"""
    experiment_id: str
    metrics: BacktestMetrics
    factor_names: List[str] = Field(default_factory=list)
    backtest_period: str = ""
    train_test_split: Optional[Dict[str, str]] = None


class ReviewResult(BaseModel):
    """CriticAgent → ExperimentAgent"""
    experiment_id: str
    approved: bool = False
    overfitting_score: float = 0.0
    generalization_score: float = 0.0
    return_randomness_score: float = 0.0
    reasons: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """完整实验结果 — 贯穿整个闭环"""
    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    factor_set: Optional[FactorSet] = None
    model_setting: Optional[ModelConfig] = None
    portfolio_config: Optional[PortfolioConfig] = None
    risk_assessment: Optional[RiskAssessment] = None
    backtest_result: Optional[BacktestResult] = None
    review_result: Optional[ReviewResult] = None
    score: float = 0.0
    created_at: Optional[str] = None
    # P5增强: 代码生成 + 实验对比
    generated_code: Optional[str] = None
    code_language: str = "python"
    experiment_description: str = ""


class ExperimentComparison(BaseModel):
    """多实验对比表 — ExperimentAgent 输出"""
    comparisons: List[Dict[str, Any]] = Field(default_factory=list)
    best_experiment_id: str = ""
    best_score: float = 0.0
    best_config: Optional[Dict[str, Any]] = None
    ranking: List[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """AI研究主管 → Agent编排: 一轮探索的计划"""
    round: int = 0
    objective: Optional[InvestmentObjective] = None
    search_direction: Dict[str, str] = Field(default_factory=dict)
    feedback_from_last_round: str = ""
    budget: Dict[str, int] = Field(
        default_factory=lambda: {
            "n_factor_candidates": 10,
            "n_models": 4,
            "n_portfolio_configs": 3,
        }
    )
    proposed_experiments: List[Dict[str, Any]] = Field(default_factory=list)


class ResearchState(BaseModel):
    """LangGraph 状态机状态 — 贯穿整个研究流程"""
    objective: Optional[InvestmentObjective] = None
    plan: Optional[ResearchPlan] = None
    factors: Optional[FactorSet] = None
    model: Optional[ModelConfig] = None
    portfolio: Optional[PortfolioConfig] = None
    backtest: Optional[BacktestResult] = None
    risk: Optional[RiskAssessment] = None
    review: Optional[ReviewResult] = None
    experiment: Optional[ExperimentResult] = None
    round: int = 0
    status: str = "idle"
    history: List[Dict[str, Any]] = Field(default_factory=list)
    best_result: Optional[ExperimentResult] = None
