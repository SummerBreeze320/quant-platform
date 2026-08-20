"""ResearchAgent — 主控Agent

接收用户投资目标，统筹调度全部研发Agent，主导整个策略探索实验。

工作流:
1. 解析投资目标 + 硬约束
2. 调度FactorAgent搜索因子空间
3. 调度ModelAgent搜索模型空间
4. 调度PortfolioAgent搜索组合空间
5. 调度RiskAgent评估市场状态
6. 组装ExperimentPlan交给ExperimentAgent
7. 收集实验结果，返回最优策略

包装现有模块:
- src.rd_agent.coordinator (RDAgentCoordinator) — 进化循环基础
"""
import logging
from typing import Optional, List
from datetime import datetime
from .base import BaseAgent
from .contracts import (
    InvestmentObjective,
    FactorSearchRequest,
    ModelSearchRequest,
    PortfolioSearchRequest,
    RiskAssessmentRequest,
    ExperimentPlan,
    ExperimentResult,
    ExperimentStatus,
)

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """主控Agent — 接收投资目标，调度全部Agent

    输入: InvestmentObjective
    输出: List[ExperimentResult] (最优策略集合)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="ResearchAgent", config=config)
        self._factor_agent = None
        self._model_agent = None
        self._portfolio_agent = None
        self._risk_agent = None
        self._experiment_agent = None

    def initialize(self):
        """初始化所有子Agent"""
        from .factor_agent import FactorAgent
        from .model_agent import ModelAgent
        from .portfolio_agent import PortfolioAgent
        from .risk_agent import RiskAgent
        from .experiment_agent import ExperimentAgent

        self._factor_agent = FactorAgent()
        self._model_agent = ModelAgent()
        self._portfolio_agent = PortfolioAgent()
        self._risk_agent = RiskAgent()
        self._experiment_agent = ExperimentAgent()

        for agent in [self._factor_agent, self._model_agent, self._portfolio_agent,
                      self._risk_agent, self._experiment_agent]:
            agent.initialize()

        self._logger.info("All sub-agents initialized")

    def run(self, input_data: InvestmentObjective) -> List[ExperimentResult]:
        """执行完整策略探索实验

        流程: 目标解析 → 因子搜索 → 模型搜索 → 组合搜索 → 风险评估 → 实验执行
        """
        if self._factor_agent is None:
            self.initialize()

        self._logger.info(f"ResearchAgent started: goal={input_data.goal} universe={input_data.universe}")

        factor_request = FactorSearchRequest(
            universe=input_data.universe,
            start_date=input_data.start_date,
            end_date=input_data.end_date,
        )
        factor_set = self._factor_agent.execute(factor_request)
        self._logger.info(f"Factor search done: {len(factor_set.factors)} factors")

        model_request = ModelSearchRequest(
            factor_set=factor_set,
            universe=input_data.universe,
            start_date=input_data.start_date,
            end_date=input_data.end_date,
        )
        model_config = self._model_agent.execute(model_request)
        self._logger.info(f"Model search done: {model_config.model_type}")

        portfolio_request = PortfolioSearchRequest(
            max_position=input_data.constraints.get("max_position", 0.15),
            tracking_error_limit=input_data.constraints.get("tracking_error_limit", 0.05),
            benchmark=input_data.benchmark,
            factor_set=factor_set,
            model_setting=model_config,
            start_date=input_data.start_date,
            end_date=input_data.end_date,
        )
        portfolio_config = self._portfolio_agent.execute(portfolio_request)
        self._logger.info(
            f"Portfolio search done: {portfolio_config.n_holdings} holdings, "
            f"{len(portfolio_config.weights)} weights"
        )

        risk_request = RiskAssessmentRequest(
            benchmark=input_data.benchmark,
            portfolio_config=portfolio_config,
        )
        risk_assessment = self._risk_agent.execute(risk_request)
        self._logger.info(
            f"Risk assessment: {risk_assessment.market_state.value} "
            f"position={risk_assessment.position_ratio:.0%}"
        )

        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        plan = ExperimentPlan(
            experiment_id=experiment_id,
            factor_set=factor_set,
            model_setting=model_config,
            portfolio_config=portfolio_config,
            risk_assessment=risk_assessment,
            objective=input_data,
        )

        result = self._experiment_agent.execute(plan)

        self._logger.info(
            f"Research complete: {result.experiment_id} "
            f"status={result.status.value} score={result.score:.4f}"
        )

        return [result]

    def run_batch(
        self, input_data: InvestmentObjective, n_experiments: int = 3
    ) -> List[ExperimentResult]:
        """批量执行多组实验(不同因子/模型变体)"""
        if self._factor_agent is None:
            self.initialize()

        results = []
        category_variants = [
            ["momentum", "volatility"],
            ["price_volume", "volume"],
            ["momentum", "fundamental", "alternative"],
        ]

        for i in range(min(n_experiments, len(category_variants))):
            self._logger.info(f"--- Experiment {i+1}/{n_experiments}: {category_variants[i]} ---")

            factor_request = FactorSearchRequest(
                universe=input_data.universe,
                categories=category_variants[i],
                start_date=input_data.start_date,
                end_date=input_data.end_date,
            )
            factor_set = self._factor_agent.execute(factor_request)

            model_request = ModelSearchRequest(
                factor_set=factor_set,
                universe=input_data.universe,
                start_date=input_data.start_date,
                end_date=input_data.end_date,
            )
            model_config = self._model_agent.execute(model_request)

            portfolio_request = PortfolioSearchRequest(
                benchmark=input_data.benchmark,
                factor_set=factor_set,
                model_setting=model_config,
                start_date=input_data.start_date,
                end_date=input_data.end_date,
            )
            portfolio_config = self._portfolio_agent.execute(portfolio_request)

            risk_request = RiskAssessmentRequest(
                benchmark=input_data.benchmark,
                portfolio_config=portfolio_config,
            )
            risk_assessment = self._risk_agent.execute(risk_request)

            plan = ExperimentPlan(
                experiment_id=f"exp_batch_{i+1}_{datetime.now().strftime('%H%M%S')}",
                factor_set=factor_set,
                model_setting=model_config,
                portfolio_config=portfolio_config,
                risk_assessment=risk_assessment,
                objective=input_data,
            )

            result = self._experiment_agent.execute(plan)
            results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        self._logger.info(
            f"Batch complete: {len(results)} experiments, "
            f"best score={results[0].score:.4f}" if results else "No results"
        )

        return results

    def run_rdagent_evolution(
        self, max_iterations: int = 3
    ) -> List[ExperimentResult]:
        """RD-Agent自动进化循环 — 因子挖掘→模型进化→策略优化

        调用RDAgentCoordinator进行自动化实验，不依赖人工因子/模型预设。
        """
        if self._experiment_agent is None:
            self.initialize()
        return self._experiment_agent.run_rdagent_evolution(max_iterations)

    def run_orchestrated(
        self,
        objective: InvestmentObjective,
        n_configs: int = 3,
    ) -> List[ExperimentResult]:
        """通过Orchestrator执行完整闭环流水线

        流程: 目标 → 搜索空间采样 → Agent链 → 评分 → 评审
              → Walk-Forward验证 → 策略注册 → 集成池构建
        """
        from .orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(n_configs=n_configs)
        orchestrator._initialize_agents()
        orchestrator._factor_agent = self._factor_agent
        orchestrator._model_agent = self._model_agent
        orchestrator._portfolio_agent = self._portfolio_agent
        orchestrator._risk_agent = self._risk_agent
        orchestrator._experiment_agent = self._experiment_agent

        results = orchestrator.run(objective)
        self._logger.info(
            f"Orchestrated run: {len(results)} configs, "
            f"{sum(1 for r in results if r.status == ExperimentStatus.APPROVED)} approved"
        )
        return results

    def run_closed_loop(
        self,
        objective: InvestmentObjective,
        n_iterations: int = 3,
    ) -> List[List[ExperimentResult]]:
        """多轮闭环迭代 — 每轮基于上轮结果优化搜索

        每轮迭代:
        1. 采样策略空间
        2. 执行Agent链(因子→模型→组合→风控→回测)
        3. Critic评审
        4. Walk-Forward验证
        5. 注册通过的策略
        6. 构建集成池
        """
        from .orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(n_configs=3)
        all_iterations = orchestrator.evolve(objective, n_iterations=n_iterations)

        self._logger.info(
            f"Closed-loop complete: {n_iterations} iterations, "
            f"total {sum(len(it) for it in all_iterations)} configs"
        )
        return all_iterations

    def get_agent_status(self) -> dict:
        """获取所有子Agent状态"""
        agents = {
            "factor": self._factor_agent,
            "model": self._model_agent,
            "portfolio": self._portfolio_agent,
            "risk": self._risk_agent,
            "experiment": self._experiment_agent,
        }
        return {
            name: agent.get_info() if agent else {"status": "not_initialized"}
            for name, agent in agents.items()
        }
