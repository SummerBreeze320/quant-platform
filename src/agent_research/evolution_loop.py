from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from src.agent_research.hypothesis_agent import HypothesisAgent
from src.agent_research.factor_coder_agent import FactorCoderAgent
from src.agent_research.sandbox_evaluator import SandboxEvaluator
from src.models.factor import FactorMetadata
from src.common.logger import logger

class EvolutionLoop:
    """Manages the hypothesis -> code -> test -> reflect -> persist R&D evolution cycle."""

    def __init__(
        self,
        db_session: Optional[Session] = None,
        llm_client: Optional[Any] = None,
        min_abs_ic: float = 0.035,
        min_icir: float = 0.5,
        max_correlation: float = 0.6
    ):
        self.db = db_session
        self.hypothesis_agent = HypothesisAgent(llm_client=llm_client)
        self.coder_agent = FactorCoderAgent(llm_client=llm_client)
        self.evaluator = SandboxEvaluator(
            min_abs_ic=min_abs_ic,
            min_icir=min_icir,
            max_correlation=max_correlation
        )

    def run_single_round(
        self,
        theme: str = "reversal",
        market_context: str = "A-Share CSI 300"
    ) -> Dict[str, Any]:
        """Runs a single round of automated factor R&D."""
        logger.info(f"Starting RD-Agent research round on theme '{theme}'...")

        # 1. Propose hypothesis
        hypothesis = self.hypothesis_agent.propose_hypothesis(theme, market_context)
        hyp_name = hypothesis.get("hypothesis_name", f"factor_{theme}")
        logger.info(f"Proposed hypothesis: {hyp_name} ({hypothesis.get('description')})")

        # 2. Generate code / expression
        expression = self.coder_agent.generate_code(hypothesis)
        logger.info(f"Generated Qlib expression: {expression}")

        # 3. Sandbox evaluation
        metrics = self.evaluator.evaluate_expression(expression)
        logger.info(f"Evaluation metrics: Rank IC={metrics.get('rank_ic')}, ICIR={metrics.get('icir')}")

        # 4. Check quality gate
        passed, reason = self.evaluator.check_quality_gate(metrics)

        factor_record = None
        if passed:
            logger.info(f"Factor '{hyp_name}' PASSED quality gate! Registering to database...")
            if self.db is not None:
                # Ensure unique factor name
                base_name = hyp_name
                counter = 1
                while self.db.query(FactorMetadata).filter_by(name=hyp_name).first() is not None:
                    hyp_name = f"{base_name}_v{counter}"
                    counter += 1

                factor_record = FactorMetadata(
                    name=hyp_name,
                    expression=expression,
                    category=hypothesis.get("category", theme),
                    description=hypothesis.get("description"),
                    ic_mean=metrics.get("rank_ic"),
                    icir=metrics.get("icir"),
                    t_stat=metrics.get("t_stat"),
                    extra_metrics=metrics,
                    created_by="RD-Agent",
                    is_active=True
                )
                self.db.add(factor_record)
                self.db.commit()
                logger.info(f"Successfully committed factor '{hyp_name}' (ID: {factor_record.id}).")

            return {
                "success": True,
                "factor_name": hyp_name,
                "expression": expression,
                "hypothesis": hypothesis,
                "metrics": metrics,
                "reason": reason
            }
        else:
            logger.warning(f"Factor '{hyp_name}' did not meet quality gate: {reason}")
            return {
                "success": False,
                "factor_name": hyp_name,
                "expression": expression,
                "hypothesis": hypothesis,
                "metrics": metrics,
                "reason": reason
            }

    def run_multi_rounds(self, rounds: int = 3, theme: str = "reversal") -> List[Dict[str, Any]]:
        """Runs multiple rounds and returns all results."""
        results = []
        for r in range(rounds):
            logger.info(f"--- RD-Agent Iteration {r + 1}/{rounds} ---")
            res = self.run_single_round(theme=theme)
            results.append(res)
        return results
