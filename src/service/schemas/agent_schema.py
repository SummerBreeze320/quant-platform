from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class AgentTaskRequest(BaseModel):
    theme: str = "reversal"
    rounds: int = 1
    market_context: str = "A-Share CSI 300"

class AgentTaskResponse(BaseModel):
    status: str
    rounds: int
    passed_count: int
    results: List[Dict[str, Any]]
