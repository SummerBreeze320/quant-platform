from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.common.db import get_db
from src.models.factor import FactorMetadata
from src.agent_research.evolution_loop import EvolutionLoop
from src.service.schemas.agent_schema import AgentTaskRequest, AgentTaskResponse
from src.service.schemas.factor_schema import FactorResponse

router = APIRouter(prefix="/rd-agent", tags=["RD-Agent Research"])

@router.post("/tasks", response_model=AgentTaskResponse)
def trigger_agent_exploration(request: AgentTaskRequest, db: Session = Depends(get_db)):
    """Triggers multi-round automated factor exploration with RD-Agent."""
    loop = EvolutionLoop(db_session=db)
    results = loop.run_multi_rounds(rounds=request.rounds, theme=request.theme)
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
