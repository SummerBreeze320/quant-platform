import math
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class RiskMetricsSummary(BaseModel):
    """投资组合多维风险与收益核心指标汇总"""
    total_return: float = Field(description="累计收益率")
    annualized_return: float = Field(description="年化复合收益率 (基于 252 交易日)")
    annualized_volatility: float = Field(description="年化波动率")
    max_drawdown: float = Field(description="最大回撤 (绝对值正数，如 0.15 表示 15%)")
    max_drawdown_duration: int = Field(default=0, description="最大回撤持续交易日天数")
    sharpe_ratio: float = Field(description="年化夏普比率")
    sortino_ratio: float = Field(description="年化索提诺比率 (仅考核下行波动)")
    calmar_ratio: float = Field(description="卡玛比率 (年化收益 / 最大回撤)")
    information_ratio: Optional[float] = Field(default=None, description="信息比率 (相对基准)")
    beta: Optional[float] = Field(default=None, description="相对基准的系统性风险 Beta")
    alpha: Optional[float] = Field(default=None, description="相对基准的年化超额 Alpha")
    win_rate: float = Field(description="日度交易胜率 (盈利日占比)")
    profit_loss_ratio: float = Field(description="盈亏比 (平均盈利日收益 / 平均亏损日绝对值)")
    trading_days: int = Field(description="样本交易日总数")


class RiskAnalyticsEngine:
    """
    量化投资组合多维绩效评价与风险分析计算引擎：
    纯向量化数学计算，无缝支持净值序列（NAV）或日收益率序列（Returns）。
    """

    @classmethod
    def calculate_metrics(
        cls,
        returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
        risk_free_rate: float = 0.02,
        annual_trading_days: int = 252,
    ) -> RiskMetricsSummary:
        """计算全套量化风险与收益指标"""
        n = len(returns)
        if n == 0:
            return RiskMetricsSummary(
                total_return=0.0,
                annualized_return=0.0,
                annualized_volatility=0.0,
                max_drawdown=0.0,
                max_drawdown_duration=0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                profit_loss_ratio=0.0,
                trading_days=0,
            )

        rets = np.array(returns, dtype=np.float64)
        # 1. 累计净值序列
        nav = np.cumprod(1.0 + rets)
        total_ret = float(nav[-1] - 1.0)

        # 2. 年化收益率 (CAGR)
        if nav[-1] > 0 and n > 0:
            ann_ret = float((nav[-1]) ** (annual_trading_days / n) - 1.0)
        else:
            ann_ret = float(np.mean(rets) * annual_trading_days)

        # 3. 年化波动率
        daily_vol = float(np.std(rets, ddof=1)) if n > 1 else 0.0
        ann_vol = float(daily_vol * math.sqrt(annual_trading_days))

        # 4. 最大回撤与持续期
        peak = np.maximum.accumulate(nav)
        drawdowns = (peak - nav) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # 回撤持续期 (最长连续处于峰值下方天数)
        max_dd_duration = 0
        current_dur = 0
        for dd in drawdowns:
            if dd > 1e-6:
                current_dur += 1
                if current_dur > max_dd_duration:
                    max_dd_duration = current_dur
            else:
                current_dur = 0

        # 5. 夏普比率
        sharpe = round((ann_ret - risk_free_rate) / ann_vol, 4) if ann_vol > 1e-8 else 0.0

        # 6. 索提诺比率 (仅考虑下行负偏差)
        rf_daily = risk_free_rate / annual_trading_days
        downside_diff = np.minimum(rets - rf_daily, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside_diff ** 2))) * math.sqrt(annual_trading_days)
        if downside_dev > 1e-8:
            sortino = round((ann_ret - risk_free_rate) / downside_dev, 4)
        else:
            sortino = 99.0 if ann_ret > risk_free_rate else 0.0

        # 7. 卡玛比率 (Calmar Ratio)
        calmar = round(ann_ret / max_dd, 4) if max_dd > 1e-6 else (99.0 if ann_ret > 0 else 0.0)

        # 8. 胜率与盈亏比
        pos_rets = rets[rets > 1e-6]
        neg_rets = rets[rets < -1e-6]
        win_rate = round(len(pos_rets) / n, 4) if n > 0 else 0.0
        if len(neg_rets) > 0 and len(pos_rets) > 0:
            avg_win = float(np.mean(pos_rets))
            avg_loss = float(abs(np.mean(neg_rets)))
            pnl_ratio = round(avg_win / avg_loss, 4) if avg_loss > 1e-8 else 99.0
        else:
            pnl_ratio = 99.0 if len(pos_rets) > 0 else 0.0

        # 9. 基准相对指标 (Beta, Alpha, Information Ratio)
        beta_val = None
        alpha_val = None
        info_ratio = None

        if benchmark_returns and len(benchmark_returns) == n:
            b_rets = np.array(benchmark_returns, dtype=np.float64)
            b_nav = np.cumprod(1.0 + b_rets)
            b_ann_ret = float((b_nav[-1]) ** (annual_trading_days / n) - 1.0) if b_nav[-1] > 0 else float(np.mean(b_rets) * annual_trading_days)

            cov_mat = np.cov(rets, b_rets)
            var_b = float(np.var(b_rets, ddof=1)) if n > 1 else 0.0

            if var_b > 1e-8 and cov_mat.shape == (2, 2):
                beta_val = round(float(cov_mat[0, 1] / var_b), 4)
                # Jensen's Alpha: R_p - (r_f + beta * (R_b - r_f))
                alpha_val = round(float(ann_ret - (risk_free_rate + beta_val * (b_ann_ret - risk_free_rate))), 4)

            # 信息比率: 跟踪误差 (Tracking Error)
            active_rets = rets - b_rets
            tracking_err = float(np.std(active_rets, ddof=1)) * math.sqrt(annual_trading_days) if n > 1 else 0.0
            ann_active_ret = ann_ret - b_ann_ret
            if tracking_err > 1e-8:
                info_ratio = round(float(ann_active_ret / tracking_err), 4)
            else:
                info_ratio = 0.0

        return RiskMetricsSummary(
            total_return=round(total_ret, 4),
            annualized_return=round(ann_ret, 4),
            annualized_volatility=round(ann_vol, 4),
            max_drawdown=round(max_dd, 4),
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            beta=beta_val,
            alpha=alpha_val,
            win_rate=win_rate,
            profit_loss_ratio=pnl_ratio,
            trading_days=n,
        )

    @classmethod
    def calculate_drawdown_series(cls, nav_series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """计算逐日回撤曲线"""
        if not nav_series:
            return []
        df = pd.DataFrame(nav_series)
        if "nav" not in df.columns:
            return []

        df["peak"] = df["nav"].cummax()
        df["drawdown"] = (df["peak"] - df["nav"]) / df["peak"].replace(0, 1.0)
        return df[["date", "nav", "drawdown"]].to_dict(orient="records")

    @classmethod
    def generate_monthly_matrix(cls, daily_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        根据逐日净值或收益率记录生成月度收益率热力矩阵与年度汇总：
        输入格式: [{"date": "2026-01-05", "return": 0.012}, ...]
        输出格式:
          {
            "years": [2025, 2026],
            "matrix": {
              "2026": {"01": 0.035, "02": -0.012, ..., "total": 0.124}
            }
          }
        """
        if not daily_records:
            return {"years": [], "matrix": {}}

        df = pd.DataFrame(daily_records)
        if "date" not in df.columns or "return" not in df.columns:
            return {"years": [], "matrix": {}}

        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year.astype(str)
        df["month"] = df["date"].dt.strftime("%m")

        matrix: Dict[str, Dict[str, float]] = {}
        for yr, y_df in df.groupby("year"):
            matrix[yr] = {}
            for mo, m_df in y_df.groupby("month"):
                m_ret = float(np.prod(1.0 + m_df["return"]) - 1.0)
                matrix[yr][mo] = round(m_ret, 4)

            # 年累计收益
            y_total = float(np.prod(1.0 + y_df["return"]) - 1.0)
            matrix[yr]["total"] = round(y_total, 4)

        years = sorted(list(matrix.keys()))
        return {"years": years, "matrix": matrix}
