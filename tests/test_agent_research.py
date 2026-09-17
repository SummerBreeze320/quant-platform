import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.base import Base
from src.models.factor import FactorMetadata
from src.agent_research.prompt_templates import HYPOTHESIS_SYSTEM_PROMPT, FACTOR_CODER_PROMPT
from src.agent_research.hypothesis_agent import HypothesisAgent
from src.agent_research.factor_coder_agent import FactorCoderAgent
from src.agent_research.sandbox_evaluator import SandboxEvaluator
from src.agent_research.evolution_loop import EvolutionLoop

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()

def test_prompt_templates_exist():
    assert "quantitative" in HYPOTHESIS_SYSTEM_PROMPT.lower()
    assert "qlib" in FACTOR_CODER_PROMPT.lower()

def test_hypothesis_agent_fallback():
    agent = HypothesisAgent(llm_client=None)
    hyp = agent.propose_hypothesis(theme="reversal", market_context="A-Share CSI 300")
    assert "hypothesis_name" in hyp
    assert "description" in hyp
    assert "category" in hyp
    assert hyp["category"] == "reversal"

def test_factor_coder_agent_clean_expression():
    coder = FactorCoderAgent(llm_client=None)
    # Test cleaning markdown codeblocks
    raw_response = "```qlib\n($close - Ref($close, 5)) / Ref($close, 5)\n```"
    clean_expr = coder.clean_expression(raw_response)
    assert clean_expr == "($close - Ref($close, 5)) / Ref($close, 5)"

def test_sandbox_evaluator_quality_gate():
    evaluator = SandboxEvaluator(min_abs_ic=0.035, min_icir=0.5, max_correlation=0.6)
    
    # Passing metrics
    good_metrics = {"rank_ic": 0.045, "icir": 0.75, "t_stat": 2.8}
    passed, reason = evaluator.check_quality_gate(good_metrics, correlation_with_existing=0.3)
    assert passed is True

    # Failing metrics (low IC)
    bad_metrics = {"rank_ic": 0.015, "icir": 0.2, "t_stat": 0.8}
    passed, reason = evaluator.check_quality_gate(bad_metrics, correlation_with_existing=0.2)
    assert passed is False
    assert "IC" in reason

def test_evolution_loop_single_round(db_session):
    loop = EvolutionLoop(db_session=db_session, llm_client=None)
    
    # Mock evaluate to return high quality factor
    def mock_evaluate(expr):
        return {"rank_ic": 0.048, "normal_ic": 0.045, "icir": 0.82, "t_stat": 3.1, "positive_ic_ratio": 0.65}

    loop.evaluator.evaluate_expression = mock_evaluate
    
    result = loop.run_single_round(theme="momentum")
    assert result["success"] is True
    assert result["factor_name"] is not None
    
    # Verify factor is registered in DB
    saved = db_session.query(FactorMetadata).filter_by(name=result["factor_name"]).first()
    assert saved is not None
    assert saved.created_by == "RD-Agent"
    assert saved.ic_mean == 0.048
