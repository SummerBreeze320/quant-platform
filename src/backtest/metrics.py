"""绩效指标计算"""
import numpy as np
import pandas as pd


def calc_annual_return(returns: pd.Series, periods: int = 252) -> float:
    """年化收益率"""
    cumulative = (1 + returns).prod()
    return cumulative ** (periods / len(returns)) - 1


def calc_sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    """夏普比率"""
    excess = returns - rf / periods
    return np.sqrt(periods) * excess.mean() / excess.std()


def calc_max_drawdown(returns: pd.Series) -> float:
    """最大回撤"""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def calc_win_rate(returns: pd.Series) -> float:
    """胜率"""
    return (returns > 0).mean()


def calc_turnover(weights: pd.DataFrame) -> float:
    """换手率"""
    return weights.diff().abs().sum(axis=1).mean()


def calc_all_metrics(returns: pd.Series, weights: pd.DataFrame = None) -> dict:
    """计算所有绩效指标"""
    return {
        "annual_return": round(calc_annual_return(returns), 4),
        "sharpe_ratio": round(calc_sharpe_ratio(returns), 4),
        "max_drawdown": round(calc_max_drawdown(returns), 4),
        "win_rate": round(calc_win_rate(returns), 4),
        "turnover": round(calc_turnover(weights), 4) if weights is not None else None,
        "total_days": len(returns),
        "total_return": round((1 + returns).prod() - 1, 4),
    }
