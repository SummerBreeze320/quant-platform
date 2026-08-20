"""SOTA factor library: Qlib expression-based factors + custom factors.

Provides a catalog of factor definitions that can be computed via
Qlib's expression engine (D.features()) or custom Python calculations.
"""
import logging
from typing import Optional, List, Dict, Any, Union
import pandas as pd

from src.core import ensure_qlib, get_features, get_ohlcv

logger = logging.getLogger(__name__)

# Qlib expression-based factors (computed via D.features)
QLIB_EXPRESSION_FACTORS: Dict[str, str] = {
    # Price-volume factors
    "kmid": "($close - $open) / $open",
    "klen": "($high - $low) / $open",
    "kup": "($high - $open) / $open",
    "klow": "($open - $low) / $open",
    "ksft": "($close - $low) / $open",
    "cfh": "($high - $close) / $open",
    "cfl": "($close - $low) / $open",
    "open0": "$open / Ref($close, 1) - 1",
    "high0": "$high / Ref($close, 1) - 1",
    "low0": "$low / Ref($close, 1) - 1",

    # Momentum factors
    "rocp_5": "Ref($close, 5) / $close - 1",
    "rocp_10": "Ref($close, 10) / $close - 1",
    "rocp_20": "Ref($close, 20) / $close - 1",
    "rocp_60": "Ref($close, 60) / $close - 1",
    "ma_5": "Mean($close, 5) / $close - 1",
    "ma_20": "Mean($close, 20) / $close - 1",
    "ma_60": "Mean($close, 60) / $close - 1",
    "ema_12": "EMA($close, 12) / $close - 1",
    "ema_26": "EMA($close, 26) / $close - 1",

    # Volatility factors
    "std_5": "Std($close, 5) / $close",
    "std_20": "Std($close, 20) / $close",
    "std_60": "Std($close, 60) / $close",
    "vol_std_5": "Std($volume, 5) / Mean($volume, 5)",
    "vol_std_20": "Std($volume, 20) / Mean($volume, 20)",

    # Volume factors
    "vwap_div_close": "$vwap / $close - 1",
    "vol_ratio_5": "$volume / Mean($volume, 5)",
    "vol_ratio_20": "$volume / Mean($volume, 20)",
    "amount_ratio_5": "$amount / Mean($amount, 5)",

    # Technical indicators
    "rsi_14": "100 - 100 / (1 + Mean(Max($close - Ref($close, 1), 0), 14) / Mean(Abs($close - Ref($close, 1)), 14))",
    "bb_up_20": "Mean($close, 20) + 2 * Std($close, 20)",
    "bb_low_20": "Mean($close, 20) - 2 * Std($close, 20)",
    "bb_width": "(Mean($close, 20) + 2 * Std($close, 20)) / (Mean($close, 20) - 2 * Std($close, 20)) - 1",

    # Fundamental factors (if bin data available)
    "pe_ratio": "$pe_ttm",
    "pb_ratio": "$pb_lf",
    "ps_ratio": "$ps_ttm",
    "dividend_yield": "$dividend_yield",
    "market_cap": "$total_mv",
    "roe": "$roe",
    "roa": "$roa",
    "debt_ratio": "$debt_ratio",
    "revenue_growth": "$revenue_yoy",
    "profit_growth": "$profit_yoy",
}

# Alternative factors (computed from non-Qlib data sources, not expression-based)
ALTERNATIVE_FACTORS: Dict[str, str] = {
    "news_sentiment": "Wind AI情感评分滚动均值",
    "news_attention": "新闻数量标准化(关注度)",
    "announcement_event": "公告事件正负面频率比",
    "dividend_indicator": "近一年分红事件哑变量",
    "consensus_rating": "卖方一致评级[-2,2]",
    "target_price_upside": "目标价上行空间",
    "analyst_attention": "覆盖机构数标准化",
    "macro_cpi_mom": "CPI环比变化",
    "macro_ppi_mom": "PPI环比变化",
    "macro_pmi_trend": "PMI趋势",
    "macro_m2_trend": "M2同比趋势",
    "macro_shibor_mom": "Shibor 3M环比变化",
}

# Factor categories
FACTOR_CATEGORIES: Dict[str, List[str]] = {
    "price_volume": ["kmid", "klen", "kup", "klow", "ksft", "cfh", "cfl", "open0", "high0", "low0"],
    "momentum": ["rocp_5", "rocp_10", "rocp_20", "rocp_60", "ma_5", "ma_20", "ma_60", "ema_12", "ema_26"],
    "volatility": ["std_5", "std_20", "std_60", "vol_std_5", "vol_std_20"],
    "volume": ["vwap_div_close", "vol_ratio_5", "vol_ratio_20", "amount_ratio_5"],
    "technical": ["rsi_14", "bb_up_20", "bb_low_20", "bb_width"],
    "fundamental": ["pe_ratio", "pb_ratio", "ps_ratio", "dividend_yield", "market_cap", "roe", "roa", "debt_ratio", "revenue_growth", "profit_growth"],
    "alternative": list(ALTERNATIVE_FACTORS.keys()),
}


def list_factors(category: Optional[str] = None) -> List[Dict]:
    """List all available factors, optionally filtered by category."""
    if category:
        names = FACTOR_CATEGORIES.get(category, [])
        result = []
        for n in names:
            if category == "alternative":
                result.append({"name": n, "description": ALTERNATIVE_FACTORS[n], "category": category})
            else:
                result.append({"name": n, "expression": QLIB_EXPRESSION_FACTORS[n], "category": category})
        return result
    result = []
    for cat, names in FACTOR_CATEGORIES.items():
        for n in names:
            if cat == "alternative":
                result.append({"name": n, "description": ALTERNATIVE_FACTORS[n], "category": cat})
            else:
                result.append({"name": n, "expression": QLIB_EXPRESSION_FACTORS[n], "category": cat})
    return result


def compute_alternative_factors(
    codes: List[str],
    end_date: str,
    lookback: int = 20,
    current_prices: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute alternative data factors (news, announcements, reports, macro).

    Unlike Qlib expression factors, these are computed from Parquet-stored
    alternative data via AlternativeDataPipeline.

    Returns:
        DataFrame with index=code, columns=factor names.
    """
    from src.data.alternative import AlternativeDataPipeline

    pipeline = AlternativeDataPipeline()
    factors = pipeline.compute_all_factors(
        codes=codes,
        end_date=end_date,
        lookback=lookback,
        current_prices=current_prices,
    )

    if not factors:
        return pd.DataFrame()

    df = pd.DataFrame(factors)
    df.index.name = "code"
    return df


def compute_factors(
    instruments: Union[str, List[str]],
    factor_names: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """Compute factor values using Qlib expression engine.

    Args:
        instruments: Code or list of codes.
        factor_names: Factor names from QLIB_EXPRESSION_FACTORS. None = all.
        start_time/end_time: Date range.

    Returns:
        MultiIndex DataFrame (instrument, datetime) with factor columns.
    """
    ensure_qlib()

    if factor_names is None:
        factor_names = list(QLIB_EXPRESSION_FACTORS.keys())

    expressions = [QLIB_EXPRESSION_FACTORS[n] for n in factor_names]

    df = get_features(instruments, expressions, start_time, end_time)
    if df is not None and not df.empty:
        df.columns = factor_names
    return df


def get_alpha158_field_list() -> List[str]:
    """Get the 158 Alpha158 factor expressions.

    These match Qlib's Alpha158 handler field definitions.
    """
    fields = []
    for cat, names in FACTOR_CATEGORIES.items():
        if cat != "fundamental":
            fields.extend(names)
    return fields


# Alpha Score fusion weights (default)
ALPHA_SCORE_WEIGHTS: Dict[str, float] = {
    "price_volume": 0.35,
    "fundamental": 0.25,
    "alternative": 0.20,
    "momentum": 0.10,
    "volatility": 0.10,
}


def compute_alpha_score(
    price_volume_score: pd.Series = None,
    fundamental_score: pd.Series = None,
    alternative_score: pd.Series = None,
    momentum_score: pd.Series = None,
    volatility_score: pd.Series = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Compute multi-source Alpha Score.

    Alpha = w1*price_volume + w2*fundamental + w3*alternative
           + w4*momentum + w5*volatility

    Each input is a cross-sectional factor score (pd.Series indexed by code).
    Missing sources are excluded and weights renormalized.

    Args:
        price_volume_score: Price-volume composite factor score
        fundamental_score: Fundamental composite factor score
        alternative_score: Alternative data composite factor score
        momentum_score: Momentum composite factor score
        volatility_score: Volatility composite factor score
        weights: Custom fusion weights. Uses ALPHA_SCORE_WEIGHTS if None.

    Returns:
        pd.Series of composite alpha scores indexed by code.
    """
    w = weights or ALPHA_SCORE_WEIGHTS

    sources = {
        "price_volume": price_volume_score,
        "fundamental": fundamental_score,
        "alternative": alternative_score,
        "momentum": momentum_score,
        "volatility": volatility_score,
    }

    active = {k: v for k, v in sources.items() if v is not None and len(v) > 0}
    if not active:
        return pd.Series(dtype=float)

    total_w = sum(w.get(k, 0) for k in active)
    if total_w == 0:
        total_w = 1.0

    # Align all series to common index
    all_codes = set()
    for s in active.values():
        all_codes.update(s.index)
    common_codes = sorted(all_codes)

    alpha = pd.Series(0.0, index=common_codes, name="alpha_score")

    for cat, score in active.items():
        normalized_w = w.get(cat, 0) / total_w
        aligned = score.reindex(common_codes, fill_value=0)
        alpha += normalized_w * aligned

    return alpha


def evaluate_factor_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    periods: int = 1,
) -> Dict:
    """Evaluate factor IC, ICIR, Rank IC.

    Args:
        factor_values: Cross-sectional factor values.
        forward_returns: Forward returns for the same period.
        periods: Number of periods for IC calculation.

    Returns:
        Dict with IC, ICIR, Rank_IC, Rank_ICIR.
    """
    common = factor_values.index.intersection(forward_returns.index)
    f = factor_values.loc[common]
    r = forward_returns.loc[common]

    if len(f) < 10:
        return {"IC": 0, "ICIR": 0, "Rank_IC": 0, "Rank_ICIR": 0}

    ic = f.corr(r)
    rank_ic = f.rank().corr(r.rank())

    ic_std = f.rolling(periods).apply(lambda x: x.corr(r.loc[x.index])).std()
    icir = ic / ic_std if ic_std > 0 else 0
    rank_icir = rank_ic / ic_std if ic_std > 0 else 0

    return {
        "IC": float(ic),
        "ICIR": float(icir),
        "Rank_IC": float(rank_ic),
        "Rank_ICIR": float(rank_icir),
    }
