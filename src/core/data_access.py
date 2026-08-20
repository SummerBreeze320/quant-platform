"""Unified data access layer — Qlib native API wrappers.

All data reading in the platform goes through this module.
Replaces the old WindToQlibConverter.read_instrument() / list_instruments().

Key mappings:
  converter.read_instrument(code, "close")  →  get_field(code, "close")
  converter.list_instruments()               →  list_instruments()
  converter.read_instrument(code, "open"/...)→  get_ohlcv(code)

Qlib expression syntax:
  Raw fields:     "$close", "$open", "$volume", "$high", "$low"
  Expressions:    "Ref($close, 5)", "Mean($close, 20)", "$close/$open"
  Alpha158-style: pass field list to get_features()
"""
import logging
from typing import List, Optional, Union
import pandas as pd

from .qlib_init import ensure_qlib

logger = logging.getLogger(__name__)


def _ensure_field_expr(field: str) -> str:
    """Convert a bare field name to a Qlib expression.

    "close" → "$close", "$close" → "$close" (idempotent).
    Complex expressions like "Ref($close, 5)" are passed through.
    """
    if field.startswith("$") or "(" in field:
        return field
    return f"${field}"


def normalize_code(code: str) -> str:
    """Convert Wind code to Qlib code format.

    "600000.SH" → "SH600000", "000001.SZ" → "SZ000001".
    Already-normalized codes pass through unchanged.
    """
    if "." not in code:
        return code
    code_part, market = code.rsplit(".", 1)
    return f"{market}{code_part}"


def get_calendar(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    freq: str = "day",
) -> list:
    """Get trading calendar (list of pd.Timestamp).

    Args:
        start_time: Start date string (e.g. "2024-01-01"). None = from beginning.
        end_time: End date string. None = to end.
        freq: "day" or "1min".
    """
    ensure_qlib()
    from qlib.data import D

    return D.calendar(start_time=start_time, end_time=end_time, freq=freq)


def list_instruments(market: str = "all", as_list: bool = True) -> Union[list, dict]:
    """List all instruments in the given market.

    Args:
        market: "all" for all stocks, "etf" for ETFs.
        as_list: True → return list of code strings. False → return config dict.
    """
    ensure_qlib()
    from qlib.data import D

    instruments = D.instruments(market=market)
    if as_list:
        return D.list_instruments(instruments, as_list=True)
    return instruments


def get_features(
    instruments: Union[str, List[str]],
    fields: Union[str, List[str]],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    freq: str = "day",
    disk_cache: int = 0,
) -> pd.DataFrame:
    """Get feature data for multiple instruments.

    Returns a MultiIndex DataFrame (instrument, datetime).

    Args:
        instruments: Single code or list of codes (Qlib format: SH600000).
        fields: Single field or list. Bare names auto-converted to expressions.
                "close" → "$close", "Ref($close,5)" passed through.
        start_time: Start date. None = from beginning.
        end_time: End date. None = to end.
        freq: "day" or "1min".
        disk_cache: 0=skip, 1=use/generate, 2=update server cache.
    """
    ensure_qlib()
    from qlib.data import D

    if isinstance(instruments, str):
        instruments = [instruments]
    if isinstance(fields, str):
        fields = [fields]
    fields = [_ensure_field_expr(f) for f in fields]

    return D.features(
        instruments,
        fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        disk_cache=disk_cache,
    )


def get_field(
    code: str,
    field: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    freq: str = "day",
) -> Optional[pd.Series]:
    """Get a single field for a single instrument as a Series.

    Drop-in replacement for WindToQlibConverter.read_instrument().
    Returns a Series indexed by datetime, or None if no data.

    Args:
        code: Instrument code (accepts both "SH600000" and "600000.SH").
        field: Field name ("close", "open", etc.) or Qlib expression.
    """
    code = normalize_code(code)
    df = get_features([code], [field], start_time, end_time, freq=freq)
    if df is None or df.empty:
        return None

    # Extract the single field Series (drop the instrument level)
    expr = _ensure_field_expr(field)
    try:
        s = df.xs(code, level="instrument")[expr]
        s.name = field
        return s
    except KeyError:
        # Fallback: return first column
        s = df.iloc[:, 0]
        s.name = field
        return s


def get_ohlcv(
    code: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    freq: str = "day",
) -> pd.DataFrame:
    """Get OHLCV data for a single instrument as a DataFrame.

    Replacement for the common pattern:
        o = converter.read_instrument(code, "open")
        h = converter.read_instrument(code, "high")
        ...
        df = pd.DataFrame({"open": o, "high": h, ...})

    Returns a DataFrame indexed by datetime with columns: open, high, low, close, volume.
    """
    code = normalize_code(code)
    df = get_features(
        [code],
        ["open", "high", "low", "close", "volume"],
        start_time,
        end_time,
        freq=freq,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Unstack: (instrument, datetime) → datetime-indexed DataFrame
    result = df.xs(code, level="instrument")
    # Rename $close → close, etc.
    result.columns = [c.lstrip("$") for c in result.columns]
    return result
