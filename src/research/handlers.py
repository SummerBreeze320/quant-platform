"""Qlib DataHandler wrappers for research and backtesting.

Provides unified access to Alpha158, Alpha360, and fundamental data handlers.
All handlers wrap Qlib's DataHandler API and integrate with our core data layer.
"""
import logging
from typing import Optional, List, Dict, Any

import pandas as pd

from src.core import ensure_qlib

logger = logging.getLogger(__name__)


def get_alpha158_handler(
    start_time: str = "2010-01-01",
    end_time: str = "2024-12-31",
    fit_start_time: str = "2010-01-01",
    fit_end_time: str = "2018-12-31",
    instruments: str = "csi300",
    infer_processors: Optional[List] = None,
    learn_processors: Optional[List] = None,
    label: Optional[List] = None,
) -> Any:
    """Create Alpha158 DataHandler.

    Alpha158 provides 158 hand-crafted features including price, volume,
    technical indicators, and rolling statistics.

    Args:
        start_time/end_time: Full data range.
        fit_start_time/fit_end_time: Training period (for processor fitting).
        instruments: "csi300", "csi500", "csi800", "all" or custom list.
        infer_processors: Processors for inference data (e.g. CSZScoreNorm).
        learn_processors: Processors for training data (e.g. DropnaLabel).
        label: Label definition, default ["Ref($close, -2)/Ref($close, -1)-1"].
    """
    ensure_qlib()
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    if label is None:
        label = ["Ref($close, -2) / Ref($close, -1) - 1"]

    if infer_processors is None:
        from qlib.data.dataset.processor import CSZScoreNorm
        infer_processors = [CSZScoreNorm(fields_group="feature")]

    if learn_processors is None:
        from qlib.data.dataset.processor import DropnaLabel, CSRankNorm
        learn_processors = [DropnaLabel(), CSRankNorm(fields_group="label")]

    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        label=label,
    )
    logger.info(f"Alpha158 handler created: {instruments}, {start_time}→{end_time}")
    return handler


def get_alpha360_handler(
    start_time: str = "2010-01-01",
    end_time: str = "2024-12-31",
    fit_start_time: str = "2010-01-01",
    fit_end_time: str = "2018-12-31",
    instruments: str = "csi300",
    infer_processors: Optional[List] = None,
    learn_processors: Optional[List] = None,
    label: Optional[List] = None,
) -> Any:
    """Create Alpha360 DataHandler.

    Alpha360 provides 360 features: 6 fields × 60 days of raw price/volume data.
    Better suited for deep learning models (LSTM, Transformer, GATs).
    """
    ensure_qlib()
    from qlib.contrib.data.handler import Alpha360

    if label is None:
        label = ["Ref($close, -2) / Ref($close, -1) - 1"]

    if infer_processors is None:
        from qlib.data.dataset.processor import CSZScoreNorm
        infer_processors = [CSZScoreNorm(fields_group="feature")]

    if learn_processors is None:
        from qlib.data.dataset.processor import DropnaLabel, CSRankNorm
        learn_processors = [DropnaLabel(), CSRankNorm(fields_group="label")]

    handler = Alpha360(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        label=label,
    )
    logger.info(f"Alpha360 handler created: {instruments}, {start_time}→{end_time}")
    return handler


class FundamentalHandler:
    """Custom DataHandler for fundamental factors (PE, PB, ROE, etc.).

    Wraps Qlib's expression engine to compute fundamental features
    from bin data written by the WindToQlibConverter.

    Three tiers of fields:
    1. Raw fundamentals: PE, PB, ROE, etc. (directly from bin files)
    2. Derived fundamentals: ROE decomposition, growth rates, etc. (Qlib expressions)
    3. Quality scores: Piotroski F-score, Altman Z-score approximations
    """

    RAW_FIELDS = [
        "pe_ttm", "pb_lf", "ps_ttm", "dividend_yield",
        "total_mv", "circ_mv", "roe", "roa",
        "debt_ratio", "revenue_yoy", "profit_yoy",
    ]

    DERIVED_FIELDS = [
        # Valuation ratios
        "earnings_yield",  # 1/PE
        "book_yield",       # 1/PB
        "sales_yield",       # 1/PS
        # Growth momentum
        "revenue_accel",    # YoY revenue change acceleration
        "profit_margin",    # Net profit margin
        # Leverage & liquidity
        "equity_multiplier",  # Total assets / equity
        "current_ratio",      # Current assets / current liabilities
        # Quality
        "roe_stability",      # Std of ROE (lower = more stable)
        "earnings_quality",   # Operating CF / Net income
    ]

    FUNDAMENTAL_FIELDS = RAW_FIELDS + DERIVED_FIELDS

    # Qlib expressions for derived fields (computed from raw bin data)
    DERIVED_EXPRESSIONS = {
        "earnings_yield": "1 / $pe_ttm",
        "book_yield": "1 / $pb_lf",
        "sales_yield": "1 / $ps_ttm",
        "revenue_accel": "$revenue_yoy - Ref($revenue_yoy, 4)",
        "profit_margin": "1 / $ps_ttm",
        "equity_multiplier": "1 / (1 - $debt_ratio)",
        "current_ratio": "1 / $debt_ratio",
        "roe_stability": "Std($roe, 4)",
        "earnings_quality": "$roa",
    }

    def __init__(
        self,
        instruments: str = "all",
        start_time: str = "2010-01-01",
        end_time: str = "2024-12-31",
        fit_start_time: str = "2010-01-01",
        fit_end_time: str = "2018-12-31",
        fields: Optional[List[str]] = None,
        label: Optional[List[str]] = None,
        include_derived: bool = True,
    ):
        ensure_qlib()
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel, CSRankNorm

        if fields is None:
            fields = self.RAW_FIELDS
            if include_derived:
                fields = fields + self.DERIVED_FIELDS

        if label is None:
            label = ["Ref($close, -2) / Ref($close, -1) - 1"]

        # Build Qlib expression field list
        # For raw fields: "$pe_ttm", etc.
        # For derived fields: use expressions from DERIVED_EXPRESSIONS
        qlib_fields = []
        for f in fields:
            if f in self.DERIVED_EXPRESSIONS:
                qlib_fields.append(self.DERIVED_EXPRESSIONS[f])
            elif f.startswith("$") or "(" in f:
                qlib_fields.append(f)
            else:
                qlib_fields.append(f"${f}")

        infer_processors = [CSZScoreNorm(fields_group="feature")]
        learn_processors = [DropnaLabel(), CSRankNorm(fields_group="label")]

        self.handler = DataHandlerLP(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            fields=qlib_fields,
            label=label,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )
        self._fields = fields
        self._qlib_fields = qlib_fields
        logger.info(f"FundamentalHandler created: {len(fields)} fields ({len(self.RAW_FIELDS)} raw + {len(self.DERIVED_FIELDS)} derived), instruments={instruments}")

    def fetch(self, col_set: str = "all") -> pd.DataFrame:
        return self.handler.fetch(col_set=col_set)

    def get_groups(self) -> Dict:
        return self.handler.get_groups()

    @property
    def fields(self) -> List[str]:
        return self._fields

    @property
    def qlib_fields(self) -> List[str]:
        """Return the Qlib expression version of fields."""
        return self._qlib_fields

    def get_field_category(self, field_name: str) -> str:
        """Classify a field as 'raw', 'derived', or 'unknown'."""
        if field_name in self.RAW_FIELDS:
            return "raw"
        elif field_name in self.DERIVED_FIELDS:
            return "derived"
        return "unknown"


def create_dataset(
    handler: Any,
    segments: Optional[Dict] = None,
) -> Any:
    """Create a DatasetH from a handler with train/valid/test segments.

    Args:
        handler: A DataHandler instance (Alpha158, Alpha360, FundamentalHandler, etc.)
        segments: Dict like {"train": ("2010-01-01", "2017-12-31"), ...}
                  Default: 6/2/2 split.
    """
    ensure_qlib()
    from qlib.data.dataset import DatasetH

    if segments is None:
        segments = {
            "train": ("2010-01-01", "2017-12-31"),
            "valid": ("2018-01-01", "2019-12-31"),
            "test": ("2020-01-01", "2024-12-31"),
        }

    dataset = DatasetH(handler=handler, segments=segments)
    logger.info(f"Dataset created with segments: {list(segments.keys())}")
    return dataset


def get_handler(
    handler_type: str = "alpha158",
    **kwargs,
) -> Any:
    """Factory function to create a DataHandler by type.

    Args:
        handler_type: "alpha158", "alpha360", or "fundamental".
        **kwargs: Passed to the handler constructor.
    """
    if handler_type == "alpha158":
        return get_alpha158_handler(**kwargs)
    elif handler_type == "alpha360":
        return get_alpha360_handler(**kwargs)
    elif handler_type == "fundamental":
        return FundamentalHandler(**kwargs)
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")
