from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.common.db import get_db
from src.common.logger import logger
from src.models.factor import FactorMetadata
from src.agent_research.rdagent import is_rdagent_available, run_rdagent_factor_loop
from src.service.schemas.agent_schema import AgentTaskRequest, AgentTaskResponse
from src.service.schemas.factor_schema import FactorResponse

router = APIRouter(prefix="/rd-agent", tags=["RD-Agent Research"])

@router.post("/tasks", response_model=AgentTaskResponse)
def trigger_agent_exploration(request: AgentTaskRequest, db: Session = Depends(get_db)):
    """Triggers automated factor exploration using Microsoft RD-Agent.

    Requires LLM API key to be configured in config/.env.
    """
    if not is_rdagent_available():
        raise HTTPException(
            status_code=503,
            detail="RD-Agent requires LLM API key. Configure OPENAI_API_KEY and CHAT_MODEL in config/.env.",
        )

    logger.info(f"Using Microsoft RD-Agent for factor research (rounds={request.rounds}, theme={request.theme}).")
    results = run_rdagent_factor_loop(
        rounds=request.rounds,
        theme=request.theme,
        db=db,
    )
    passed_count = sum(1 for r in results if r.get("success"))

    return {
        "status": "COMPLETED",
        "rounds": request.rounds,
        "passed_count": passed_count,
        "results": results
    }

@router.get("/factors", response_model=List[FactorResponse])
def get_agent_created_factors(db: Session = Depends(get_db)):
    """Retrieves all factors created and verified by RD-Agent."""
    return db.query(FactorMetadata).filter_by(created_by="RD-Agent").order_by(FactorMetadata.id.desc()).all()
