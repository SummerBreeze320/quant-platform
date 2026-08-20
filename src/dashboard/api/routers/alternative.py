"""Alternative data API router.

Endpoints for news sentiment, announcements, research reports,
and macro economic data — extending the Data Layer beyond Qlib bin.
"""
import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from src.data.alternative import (
    AlternativeDataPipeline,
    MockAlternativeGenerator,
    NewsHandler,
    AnnouncementHandler,
    ResearchReportHandler,
    MacroHandler,
    MACRO_INDICATORS,
)
from src.research.factor_library import ALTERNATIVE_FACTORS, list_factors

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alternative", tags=["alternative-data"])


class InitRequest(BaseModel):
    codes: List[str] = []
    start: str = "2024-01-01"
    end: Optional[str] = None
    use_mock: bool = True


class FactorComputeRequest(BaseModel):
    codes: List[str] = []
    end_date: Optional[str] = None
    lookback: int = 20


@router.get("/health")
async def alt_health():
    """Check alternative data availability."""
    pipeline = AlternativeDataPipeline()
    checks = pipeline.verify_data([])
    return {
        "status": "ok",
        "data_sources": {
            "news": checks.get("news", False),
            "announcements": checks.get("announcements", False),
            "ratings": checks.get("ratings", False),
            "macro": checks.get("macro", False),
        },
        "factor_count": len(ALTERNATIVE_FACTORS),
        "indicators": len(MACRO_INDICATORS),
    }


@router.get("/factors")
async def list_alt_factors():
    """List all alternative factors."""
    return {"factors": list_factors(category="alternative")}


@router.post("/init")
async def init_data(req: InitRequest):
    """Initialize alternative data (full fetch or mock)."""
    if not req.codes:
        from src.core import list_instruments
        ensure = __import__("src.core", fromlist=["ensure_qlib"]).ensure_qlib
        ensure()
        req.codes = list_instruments(as_list=True)

    if req.use_mock:
        gen = MockAlternativeGenerator()
        counts = gen.generate_all(req.codes, req.start, req.end or datetime.now().strftime("%Y-%m-%d"))
    else:
        pipeline = AlternativeDataPipeline()
        counts = pipeline.run_full_init(req.codes, req.start, req.end)

    return {"status": "ok", "counts": counts}


@router.post("/update")
async def update_data(req: InitRequest):
    """Incremental update alternative data."""
    pipeline = AlternativeDataPipeline(use_mock=req.use_mock)
    if not req.codes:
        from src.core import list_instruments, ensure_qlib
        ensure_qlib()
        req.codes = list_instruments(as_list=True)
    counts = pipeline.update_incremental(req.codes)
    return {"status": "ok", "counts": counts}


@router.get("/news")
async def get_news(
    code: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """Get news data with sentiment scores."""
    handler = NewsHandler()
    df = handler._load("news_raw")
    if df is None or df.empty:
        return {"data": [], "total": 0}

    if code:
        df = df[df["code"] == code]

    df = df.head(limit)
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
    return {"data": records, "total": len(records)}


@router.get("/announcements")
async def get_announcements(
    code: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """Get announcement/event data."""
    handler = AnnouncementHandler()
    df = handler._load("announcements_raw")
    if df is None or df.empty:
        return {"data": [], "total": 0}

    if code:
        df = df[df["code"] == code]

    df = df.head(limit)
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
    return {"data": records, "total": len(records)}


@router.get("/ratings")
async def get_ratings(
    code: Optional[str] = Query(None),
):
    """Get research report ratings and consensus estimates."""
    handler = ResearchReportHandler()
    df = handler._load("ratings_raw")
    if df is None or df.empty:
        return {"data": [], "total": 0}

    if code:
        df = df[df["code"] == code]

    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
    return {"data": records, "total": len(records)}


@router.get("/macro")
async def get_macro(
    indicator: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
):
    """Get macro economic indicators."""
    handler = MacroHandler()
    df = handler._load("macro_raw")
    if df is None or df.empty:
        return {"data": [], "total": 0, "indicators": list(MACRO_INDICATORS.keys())}

    if indicator and indicator in df.columns:
        df = df[[indicator]].dropna().tail(limit)
    else:
        df = df.tail(limit)

    records = []
    for idx, row in df.iterrows():
        record = {"date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx)}
        for col in df.columns:
            record[col] = float(row[col]) if not str(row[col]).startswith("Na") else None
        records.append(record)

    return {"data": records, "total": len(records), "indicators": list(df.columns)}


@router.post("/factors/compute")
async def compute_factors(req: FactorComputeRequest):
    """Compute alternative factors for given codes."""
    from src.research.factor_library import compute_alternative_factors

    if not req.codes:
        from src.core import list_instruments, ensure_qlib
        ensure_qlib()
        req.codes = list_instruments(as_list=True)

    end_date = req.end_date or datetime.now().strftime("%Y-%m-%d")

    df = compute_alternative_factors(
        codes=req.codes,
        end_date=end_date,
        lookback=req.lookback,
    )

    if df is None or df.empty:
        return {"factors": {}, "codes": req.codes}

    import numpy as np
    df = df.replace([np.inf, -np.inf], np.nan)
    factors = {}
    for code, row in df.iterrows():
        factors[code] = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}

    return {
        "factors": factors,
        "factor_names": list(df.columns),
        "codes": list(df.index),
    }
