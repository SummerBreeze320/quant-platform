"""
Adapter for Microsoft RD-Agent (rdagent package v0.8.0).

Wraps the real rdagent FactorRDLoop, syncs LLM/Qlib config from our Settings,
runs the async evolution loop, and extracts discovered factors into FactorMetadata.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.common.config import get_settings
from src.common.logger import logger
from src.models.factor import FactorMetadata


def _sync_env_to_rdagent():
    """Populate rdagent's environment variables from our Settings / .env file."""
    settings = get_settings()

    env_map = {
        "CHAT_MODEL": settings.LLM_MODEL,
        "OPENAI_API_KEY": settings.LLM_API_KEY,
        "OPENAI_API_BASE": settings.LLM_BASE_URL,
    }
    for key, value in env_map.items():
        if value and key not in os.environ:
            os.environ[key] = value

    os.environ.setdefault("QLIB_FACTOR_TRAIN_START", "2020-01-01")
    os.environ.setdefault("QLIB_FACTOR_TRAIN_END", "2023-12-31")
    os.environ.setdefault("QLIB_FACTOR_VALID_START", "2024-01-01")
    os.environ.setdefault("QLIB_FACTOR_VALID_END", "2024-12-31")
    os.environ.setdefault("QLIB_FACTOR_TEST_START", "2025-01-01")
    os.environ.setdefault("QLIB_FACTOR_TEST_END", "2025-12-31")


def is_rdagent_available() -> bool:
    """Check if rdagent is installed AND an LLM API key is configured."""
    settings = get_settings()
    if not settings.LLM_API_KEY:
        return False
    try:
        import rdagent  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_factor_from_experiment(experiment) -> Optional[Dict[str, Any]]:
    """Extract factor metadata from an rdagent Experiment object."""
    factor_info = {
        "factor_name": None,
        "expression": None,
        "description": None,
        "metrics": {},
    }

    hypothesis = getattr(experiment, "hypothesis", None)
    if hypothesis is not None:
        factor_info["description"] = getattr(hypothesis, "background", None) or str(hypothesis)
        new_hyp = getattr(hypothesis, "new_hypothesis", None)
        if new_hyp:
            factor_info["factor_name"] = new_hyp[:100] if isinstance(new_hyp, str) else str(new_hyp)[:100]

    sub_tasks = getattr(experiment, "sub_tasks", [])
    if sub_tasks:
        for task in sub_tasks:
            task_name = getattr(task, "name", None) or str(task)[:100]
            if not factor_info["factor_name"]:
                factor_info["factor_name"] = task_name

            competition = getattr(task, "competition", None)
            if competition:
                factor_info["expression"] = str(competition)

    sub_results = getattr(experiment, "sub_results", {})
    if sub_results:
        factor_info["metrics"] = {str(k): float(v) for k, v in sub_results.items()}

    running_info = getattr(experiment, "running_info", None)
    if running_info and running_info.result:
        factor_info["metrics"]["running_result"] = str(running_info.result)[:500]

    if not factor_info["factor_name"]:
        factor_info["factor_name"] = f"rdagent_factor_{id(experiment) % 10000}"

    return factor_info


def _sync_factor_to_db(factor_info: Dict[str, Any], feedback, db: Session) -> Optional[FactorMetadata]:
    """Create or update a FactorMetadata record from extracted factor info."""
    factor_name = factor_info.get("factor_name") or "unnamed_factor"
    expression = factor_info.get("expression") or ""

    existing = db.query(FactorMetadata).filter_by(factor_name=factor_name).first()
    if existing:
        existing.expression = expression
        existing.description = factor_info.get("description") or existing.description
        existing.metrics = factor_info.get("metrics", existing.metrics)
        existing.is_effective = getattr(feedback, "decision", False)
        db.commit()
        db.refresh(existing)
        logger.info(f"Updated factor '{factor_name}' (id={existing.id}).")
        return existing

    record = FactorMetadata(
        factor_name=factor_name,
        expression=expression,
        description=factor_info.get("description") or "",
        factor_type="alpha",
        created_by="RD-Agent",
        is_effective=getattr(feedback, "decision", False),
        metrics=factor_info.get("metrics", {}),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(f"Created factor '{factor_name}' (id={record.id}, effective={record.is_effective}).")
    return record


def run_rdagent_factor_loop(
    rounds: int = 1,
    theme: Optional[str] = None,
    db: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """
    Run the Microsoft RD-Agent factor research loop.

    Parameters:
        rounds: Number of evolution loops (mapped to rdagent's loop_n).
        theme: Optional research theme (unused by rdagent, logged for reference).
        db: Database session for persisting discovered factors.

    Returns:
        List of result dicts with keys: factor_name, expression, decision, observations.
    """
    _sync_env_to_rdagent()

    from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop

    if theme:
        logger.info(f"RD-Agent factor loop starting with theme='{theme}', rounds={rounds}")

    FACTOR_PROP_SETTING.evolving_n = rounds

    loop = FactorRDLoop(FACTOR_PROP_SETTING)

    try:
        asyncio.run(loop.run(loop_n=rounds))
    except SystemExit:
        logger.warning("RD-Agent loop exited early (SystemExit).")
    except KeyboardInterrupt:
        logger.warning("RD-Agent loop interrupted by user.")
    except Exception as e:
        logger.error(f"RD-Agent loop failed: {e}")
        raise

    results = []
    trace = loop.trace

    for i, (experiment, feedback) in enumerate(trace.hist):
        factor_info = _extract_factor_from_experiment(experiment)
        if not factor_info:
            continue

        result_entry = {
            "factor_name": factor_info.get("factor_name"),
            "expression": factor_info.get("expression"),
            "decision": getattr(feedback, "decision", False),
            "observations": getattr(feedback, "observations", ""),
            "hypothesis_evaluation": getattr(feedback, "hypothesis_evaluation", ""),
            "new_hypothesis": getattr(feedback, "new_hypothesis", ""),
            "metrics": factor_info.get("metrics", {}),
            "success": getattr(feedback, "decision", False),
            "round": i + 1,
        }
        results.append(result_entry)

        if db is not None:
            _sync_factor_to_db(factor_info, feedback, db)

    sota_hyp, sota_exp = trace.get_sota_hypothesis_and_experiment()
    if sota_exp and db is not None:
        sota_info = _extract_factor_from_experiment(sota_exp)
        if sota_info:
            sota_info["factor_name"] = (sota_info.get("factor_name") or "sota_factor") + "_sota"
            _sync_factor_to_db(sota_info, type("F", (), {"decision": True})(), db)

    logger.info(f"RD-Agent loop complete: {len(results)} factors discovered, "
                f"{sum(1 for r in results if r['success'])} passed quality gate.")

    return results
