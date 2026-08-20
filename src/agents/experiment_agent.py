"""ExperimentAgent — RD-Agent自动实验 + 代码生成 + 实验对比

生成多组对照实验方案，调度RD-Agent运行，对比结果择优保留。

P5增强:
- 代码自动生成 (LLM生成因子/模型代码)
- 多实验对比表 (并排比较指标)
- 最佳实验结构化返回 (best_config)
- 实验提案模板 (动量窗口5天/10天/增加新闻因子)

包装现有模块:
- src.rd_agent.coordinator (RDAgentCoordinator)
- src.rd_agent.factor_runner (FactorRunner)
- src.rd_agent.model_runner (ModelRunner)
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from .base import BaseAgent
from .contracts import (
    ExperimentPlan,
    ExperimentResult,
    ExperimentComparison,
    BacktestRequest,
    BacktestResult,
    ReviewRequest,
    ReviewResult,
    ExperimentStatus,
    FactorSet,
    ModelConfig,
    PortfolioConfig,
)

logger = logging.getLogger(__name__)

# 实验提案模板
EXPERIMENT_TEMPLATES = {
    "momentum_5d": {
        "description": "动量窗口5天",
        "factor_changes": {"rocp_5": "Ref($close, 5) / $close - 1"},
        "model": "lightgbm",
        "params": {"num_leaves": 64, "learning_rate": 0.05},
    },
    "momentum_10d": {
        "description": "动量窗口10天",
        "factor_changes": {"rocp_10": "Ref($close, 10) / $close - 1"},
        "model": "lightgbm",
        "params": {"num_leaves": 64, "learning_rate": 0.05},
    },
    "momentum_20d": {
        "description": "动量窗口20天",
        "factor_changes": {"rocp_20": "Ref($close, 20) / $close - 1"},
        "model": "lightgbm",
        "params": {"num_leaves": 128, "learning_rate": 0.03},
    },
    "add_news_factor": {
        "description": "增加新闻情感因子",
        "factor_changes": {"news_sentiment": "External(news_sentiment_score)"},
        "model": "lightgbm",
        "params": {"num_leaves": 64, "learning_rate": 0.05},
    },
    "xgboost_model": {
        "description": "XGBoost模型替代LightGBM",
        "factor_changes": {},
        "model": "xgboost",
        "params": {"max_depth": 6, "n_estimators": 200},
    },
    "transformer_model": {
        "description": "Transformer模型捕捉时序",
        "factor_changes": {},
        "model": "transformer",
        "params": {"d_model": 64, "nhead": 4, "num_layers": 2},
    },
}


class ExperimentAgent(BaseAgent):
    """自动实验Agent — RD-Agent集成 + 代码生成

    输入: ExperimentPlan (单组实验方案) 或 List[ExperimentPlan] (多组对比)
    输出: ExperimentResult (单组) 或 ExperimentComparison (多组对比)

    工作流:
    1. 接收ExperimentPlan (或自动生成实验提案)
    2. 生成实验代码 (LLM生成因子/模型Python代码)
    3. 调度BacktestAgent逐个回测
    4. 调度CriticAgent评审
    5. 对比结果，输出最优方案 (best_config)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="ExperimentAgent", config=config)
        self._backtest_agent = None
        self._critic_agent = None

    def initialize(self):
        from .backtest_agent import BacktestAgent
        from .critic_agent import CriticAgent
        self._backtest_agent = BacktestAgent()
        self._critic_agent = CriticAgent()

    def run(self, input_data: ExperimentPlan) -> ExperimentResult:
        """执行实验"""
        if self._backtest_agent is None:
            self.initialize()

        exp_id = input_data.experiment_id
        now = datetime.now().isoformat()

        result = ExperimentResult(
            experiment_id=exp_id,
            status=ExperimentStatus.RUNNING,
            factor_set=input_data.factor_set,
            model_setting=input_data.model_setting,
            portfolio_config=input_data.portfolio_config,
            risk_assessment=input_data.risk_assessment,
            created_at=now,
        )

        bt_request = BacktestRequest(
            experiment_id=exp_id,
            factor_names=input_data.factor_set.factors if input_data.factor_set else [],
            model_type=input_data.model_setting.model_type if input_data.model_setting else "lightgbm",
            model_params=input_data.model_setting.params if input_data.model_setting else {},
            n_holdings=input_data.portfolio_config.n_holdings if input_data.portfolio_config else 10,
            weight_scheme=input_data.portfolio_config.weight_scheme if input_data.portfolio_config else "enhanced_indexing",
            rebalance_freq=input_data.portfolio_config.rebalance_freq if input_data.portfolio_config else "W",
            max_position=input_data.portfolio_config.max_position if input_data.portfolio_config else 0.15,
            start_date=input_data.objective.start_date if input_data.objective else "2020-01-01",
            end_date=input_data.objective.end_date if input_data.objective else "2024-12-31",
            init_cash=input_data.objective.init_cash if input_data.objective else 1000000,
            benchmark=input_data.objective.benchmark if input_data.objective else "SH000300",
        )

        bt_result = self._backtest_agent.execute(bt_request)
        result.backtest_result = bt_result
        result.score = bt_result.score

        if bt_result.status == "error":
            result.status = ExperimentStatus.REJECTED
            return result

        review_request = ReviewRequest(
            experiment_id=exp_id,
            metrics=bt_result.metrics,
            factor_names=bt_request.factor_names,
            backtest_period=f"{bt_request.start_date} ~ {bt_request.end_date}",
        )
        review_result = self._critic_agent.execute(review_request)
        result.review_result = review_result

        if review_result.approved:
            result.status = ExperimentStatus.APPROVED
            logger.info(f"Experiment {exp_id} APPROVED (score={bt_result.score:.4f})")
        else:
            result.status = ExperimentStatus.REJECTED
            logger.info(f"Experiment {exp_id} REJECTED (overfit={review_result.overfitting_score:.2f})")

        return result

    def run_batch(
        self,
        plans: List[ExperimentPlan],
    ) -> ExperimentComparison:
        """P5: 批量执行多组实验并对比

        执行流程:
        1. 逐个运行实验 → 获取 ExperimentResult 列表
        2. 构建对比表 (各实验指标并排)
        3. 选出最佳实验 → 结构化返回 best_config

        Returns:
            ExperimentComparison (含对比表 + best_experiment_id + best_config)
        """
        if self._backtest_agent is None:
            self.initialize()

        results: List[ExperimentResult] = []
        for plan in plans:
            logger.info(f"Running experiment: {plan.experiment_id}")
            result = self.run(plan)
            results.append(result)

        return self._build_comparison(results)

    def generate_experiments(
        self,
        base_factors: List[str],
        model_type: str = "lightgbm",
        n_variants: int = 3,
    ) -> List[ExperimentPlan]:
        """P5: 自动生成实验提案

        基于现有因子和模型，自动生成对照实验方案:
        - 不同动量窗口 (5天/10天/20天)
        - 增加另类数据因子
        - 不同模型 (LGB/XGB/Transformer)
        """
        templates = list(EXPERIMENT_TEMPLATES.keys())[:n_variants]
        plans = []

        for i, key in enumerate(templates):
            tpl = EXPERIMENT_TEMPLATES[key]
            factors = list(base_factors)
            for fname in tpl.get("factor_changes", {}):
                if fname not in factors:
                    factors.append(fname)

            plan = ExperimentPlan(
                experiment_id=f"exp_{key}_{i+1}",
                factor_set=FactorSet(factors=factors),
                model_setting=ModelConfig(
                    model_type=tpl.get("model", model_type),
                    params=tpl.get("params", {}),
                ),
                portfolio_config=PortfolioConfig(),
            )
            plans.append(plan)
            logger.info(f"Generated experiment: {plan.experiment_id} ({tpl['description']})")

        return plans

    def generate_code(self, experiment: ExperimentPlan) -> str:
        """P5: 生成实验的 Python 代码

        根据实验方案生成可执行的因子计算 + 模型训练代码。
        如果安装了 LangChain，使用 LLM 生成; 否则使用模板生成。
        """
        try:
            return self._generate_code_llm(experiment)
        except Exception:
            return self._generate_code_template(experiment)

    def _generate_code_llm(self, experiment: ExperimentPlan) -> str:
        """使用 LangChain LLM 生成代码"""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        factors = experiment.factor_set.factors if experiment.factor_set else []
        model = experiment.model_setting.model_type if experiment.model_setting else "lightgbm"
        params = experiment.model_setting.params if experiment.model_setting else {}

        prompt = ChatPromptTemplate.from_template("""
你是量化策略代码生成器。根据以下实验方案生成 Python 代码:

因子列表: {factors}
模型类型: {model}
模型参数: {params}

要求:
1. 使用 Qlib 框架
2. 包含因子计算 (Alpha158 表达式)
3. 包含模型训练和预测
4. 包含回测调用
5. 代码可直接运行

只输出 Python 代码，不要解释。
""")

        chain = prompt | llm
        response = chain.invoke({
            "factors": ", ".join(factors),
            "model": model,
            "params": str(params),
        })
        return response.content

    def _generate_code_template(self, experiment: ExperimentPlan) -> str:
        """模板生成代码 (LLM 不可用时的 fallback)"""
        factors = experiment.factor_set.factors if experiment.factor_set else []
        model = experiment.model_setting.model_type if experiment.model_setting else "lightgbm"
        params = experiment.model_setting.params if experiment.model_setting else {}

        code = f'''# Auto-generated experiment code
# Experiment: {experiment.experiment_id}
# Model: {model}
# Factors: {", ".join(factors)}

from src.core import ensure_qlib
from src.research.handlers import get_handler, create_dataset
from src.research.model_zoo import create_model, train_model, backtest_model

ensure_qlib("data/qlib_bin")

# 1. Prepare data handler
handler = get_handler(
    handler_type="alpha158",
    instruments="all",
    start_time="2020-01-01",
    end_time="2024-12-31",
)

segments = {{
    "train": ("2020-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2023-12-31"),
    "test":  ("2024-01-01", "2024-12-31"),
}}
dataset = create_dataset(handler, segments)

# 2. Train model
model = create_model("{model}", {params})
model.fit(dataset)

# 3. Predict
pred = model.predict(dataset, segment="test")

# 4. Backtest
result = backtest_model(model, dataset, topk=5, n_drop=1)
print(f"Sharpe: {{result.get('sharpe_ratio', 'N/A')}}")
'''
        return code

    def _build_comparison(self, results: List[ExperimentResult]) -> ExperimentComparison:
        """构建实验对比表"""
        comparisons = []
        valid_results = [r for r in results if r.backtest_result and r.backtest_result.metrics]

        for r in valid_results:
            m = r.backtest_result.metrics
            comparisons.append({
                "experiment_id": r.experiment_id,
                "status": r.status.value,
                "score": round(r.score, 4),
                "annual_return": round(m.annual_return, 4),
                "sharpe": round(m.sharpe_ratio, 4),
                "max_drawdown": round(m.max_drawdown, 4),
                "win_rate": round(m.win_rate, 4),
                "turnover": round(m.turnover, 4),
                "ic_mean": round(m.ic_mean, 4),
                "transaction_cost": round(m.transaction_cost, 6),
                "net_return": round(m.net_return_after_cost, 4),
                "approved": r.review_result.approved if r.review_result else False,
            })

        comparisons.sort(key=lambda x: x["score"], reverse=True)
        ranking = [c["experiment_id"] for c in comparisons]

        best = comparisons[0] if comparisons else {}
        best_config = None
        if best:
            best_result = next(r for r in results if r.experiment_id == best["experiment_id"])
            best_config = {
                "experiment_id": best["experiment_id"],
                "model": best_result.model_setting.model_type if best_result.model_setting else "lightgbm",
                "params": best_result.model_setting.params if best_result.model_setting else {},
                "factors": best_result.factor_set.factors if best_result.factor_set else [],
                "score": best["score"],
                "sharpe": best["sharpe"],
                "annual_return": best["annual_return"],
            }

        comp = ExperimentComparison(
            comparisons=comparisons,
            best_experiment_id=best.get("experiment_id", ""),
            best_score=best.get("score", 0.0),
            best_config=best_config,
            ranking=ranking,
        )

        logger.info(
            f"Experiment comparison: {len(comparisons)} experiments, "
            f"best={comp.best_experiment_id} (score={comp.best_score:.4f})"
        )
        return comp

    def run_rdagent_evolution(
        self, max_iterations: int = 3
    ) -> List[ExperimentResult]:
        """直接调用RD-Agent进化循环(高级模式)"""
        logger.info(f"Starting RD-Agent evolution: max_rounds={max_iterations}")
        try:
            from src.rd_agent.coordinator import RDAgentCoordinator

            coordinator = RDAgentCoordinator()
            full_result = coordinator.run_evolution_loop(max_rounds=max_iterations)

            factor_history = coordinator.get_factor_history() or []
            model_history = coordinator.get_model_history() or []
            best_config = coordinator.get_best_config()

            results: List[ExperimentResult] = []

            rounds_data = (
                full_result.get("rounds", []) if isinstance(full_result, dict) else []
            )
            n_rounds = len(rounds_data)

            for i in range(n_rounds):
                rd = rounds_data[i]
                factor_entry = factor_history[i] if i < len(factor_history) else {}
                model_entry = model_history[i] if i < len(model_history) else {}

                exp_id = f"rdagent_round_{i+1}"
                best_ic = rd.get("best_factor_ic", factor_entry.get("best_ic", 0))
                best_sharpe = rd.get("best_model_sharpe", model_entry.get("best_sharpe", 0))
                score = (best_ic + best_sharpe) / 2 if (best_ic or best_sharpe) else 0

                factors_found = factor_entry.get("factors_found", rd.get("factors_found", 0))
                model_name = model_entry.get("model_name", rd.get("best_model", "unknown"))

                results.append(ExperimentResult(
                    experiment_id=exp_id,
                    status=ExperimentStatus.COMPLETED,
                    score=round(score, 4),
                    created_at=datetime.now().isoformat(),
                ))

                logger.info(
                    f"  Round {i+1}: factors={factors_found} "
                    f"model={model_name} ic={best_ic:.4f} sharpe={best_sharpe:.4f}"
                )

            if best_config:
                logger.info(
                    f"RD-Agent best config: {best_config.get('model_name', 'N/A')} "
                    f"applied to qlib_config.yaml"
                )

            logger.info(f"RD-Agent evolution completed: {len(results)} rounds")
            return results

        except Exception as e:
            logger.error(f"RD-Agent evolution failed: {e}", exc_info=True)
            return []
