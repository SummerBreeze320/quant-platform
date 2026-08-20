"""高级风险指标

扩展基础指标，增加Sortino、Calmar、信息比率、VaR、CVaR等
"""
import numpy as np
import pandas as pd
from typing import Optional


def calc_sortino_ratio(returns: pd.Series, rf: float = 0.0,
                        periods: int = 252) -> float:
    """Sortino比率（下行风险调整）

    与Sharpe不同，仅惩罚下行波动
    """
    excess = returns - rf / periods
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = np.sqrt((downside ** 2).sum() / len(excess))
    if downside_std == 0:
        return 0.0
    return np.sqrt(periods) * excess.mean() / downside_std


def calc_calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Calmar比率 = 年化收益 / 最大回撤"""
    annual_ret = (1 + returns).prod() ** (periods / len(returns)) - 1
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    max_dd = ((cum - peak) / peak).min()
    if max_dd == 0:
        return 0.0
    return annual_ret / abs(max_dd)


def calc_information_ratio(returns: pd.Series,
                            benchmark: pd.Series,
                            periods: int = 252) -> float:
    """信息比率 = 超额收益均值 / 跟踪误差"""
    excess = returns - benchmark
    te = excess.std()
    if te == 0:
        return 0.0
    return np.sqrt(periods) * excess.mean() / te


def calc_tracking_error(returns: pd.Series,
                         benchmark: pd.Series,
                         periods: int = 252) -> float:
    """跟踪误差 = 超额收益标准差 * sqrt(periods)"""
    return (returns - benchmark).std() * np.sqrt(periods)


def calc_var(returns: pd.Series, confidence: float = 0.95,
              method: str = "historical") -> float:
    """Value at Risk

    method: "historical"(历史模拟) 或 "parametric"(参数法)
    """
    if method == "parametric":
        mean = returns.mean()
        std = returns.std()
        from scipy.stats import norm
        return mean + norm.ppf(1 - confidence) * std
    else:
        return np.percentile(returns, (1 - confidence) * 100)


def calc_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """条件VaR（Expected Shortfall）

    尾部损失超过VaR的平均值
    """
    var = calc_var(returns, confidence, "historical")
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return tail.mean()


def calc_beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Beta系数"""
    cov = returns.cov(benchmark)
    var = benchmark.var()
    if var == 0:
        return 0.0
    return cov / var


def calc_alpha(returns: pd.Series, benchmark: pd.Series,
               rf: float = 0.0, periods: int = 252) -> float:
    """Jensen's Alpha

    超额收益中无法被市场Beta解释的部分
    """
    beta = calc_beta(returns, benchmark)
    excess_ret = returns.mean() - rf / periods
    excess_bench = benchmark.mean() - rf / periods
    return periods * (excess_ret - beta * excess_bench)


def calc_max_drawdown_duration(returns: pd.Series) -> int:
    """最大回撤持续天数"""
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    in_dd = cum < peak
    max_duration = 0
    current = 0
    for v in in_dd:
        if v:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_duration


def calc_downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """下行偏差"""
    downside = returns[returns < mar]
    if len(downside) == 0:
        return 0.0
    return np.sqrt((downside ** 2).mean())


def calc_monthly_returns(returns: pd.Series) -> pd.Series:
    """月度收益率"""
    monthly = (1 + returns).cumprod().resample("M").last().pct_change().dropna()
    return monthly


def calc_all_risk_metrics(returns: pd.Series,
                           benchmark: Optional[pd.Series] = None) -> dict:
    """计算全部高级风险指标"""
    result = {
        "sortino_ratio": round(calc_sortino_ratio(returns), 4),
        "calmar_ratio": round(calc_calmar_ratio(returns), 4),
        "var_95": round(calc_var(returns, 0.95), 4),
        "cvar_95": round(calc_cvar(returns, 0.95), 4),
        "max_dd_duration": calc_max_drawdown_duration(returns),
        "downside_deviation": round(calc_downside_deviation(returns), 4),
    }
    if benchmark is not None:
        result["information_ratio"] = round(calc_information_ratio(returns, benchmark), 4)
        result["tracking_error"] = round(calc_tracking_error(returns, benchmark), 4)
        result["beta"] = round(calc_beta(returns, benchmark), 4)
        result["alpha"] = round(calc_alpha(returns, benchmark), 4)
    return result
