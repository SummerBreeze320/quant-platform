from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats


class FactorEvaluationMetrics(BaseModel):
    """量化单因子多维有效性与统计评估指标"""
    ic_mean: float = Field(description="信息系数均值 (Pearson IC)")
    ic_std: float = Field(description="信息系数标准差")
    rank_ic_mean: float = Field(description="秩相关系数均值 (Spearman Rank IC)")
    rank_ic_std: float = Field(description="秩相关系数标准差")
    icir: float = Field(description="信息比率 (年化 ICIR = IC_mean / IC_std * sqrt(252))")
    rank_icir: float = Field(description="Rank ICIR (年化)")
    t_stat: float = Field(description="t 统计量检验值")
    long_short_return: float = Field(description="Top-Quintile 对比 Bottom-Quintile 多空年化超额收益")
    turnover: float = Field(description="因子自相关换手率")
    monotonicity_score: float = Field(description="五分位单调性得分 (1.0 表示完全单调递增/递减)")
    passed_gate: bool = Field(default=False, description="是否满足准入质量门禁")
    gate_violations: List[str] = Field(default_factory=list, description="未达标门禁项说明")


class FactorEvaluator:
    """
    单因子定量有效性评估引擎：
    基于时间序列截面数据，执行纯向量化 IC、Rank IC、ICIR、分层多空收益与换手衰减统计。
    """

    @classmethod
    def evaluate(
        cls,
        factor_df: pd.DataFrame,
        forward_returns_df: pd.DataFrame,
        ic_threshold: float = 0.02,
        icir_threshold: float = 0.5,
        t_stat_threshold: float = 2.0,
        quantiles: int = 5,
    ) -> FactorEvaluationMetrics:
        """
        评估因子有效性：
        factor_df: index=日期, columns=股票代码, values=因子值
        forward_returns_df: index=日期, columns=股票代码, values=未来1期前瞻收益率
        """
        common_index = factor_df.index.intersection(forward_returns_df.index)
        common_cols = factor_df.columns.intersection(forward_returns_df.columns)

        if len(common_index) < 5 or len(common_cols) < 2:
            return FactorEvaluationMetrics(
                ic_mean=0.0,
                ic_std=0.0,
                rank_ic_mean=0.0,
                rank_ic_std=0.0,
                icir=0.0,
                rank_icir=0.0,
                t_stat=0.0,
                long_short_return=0.0,
                turnover=0.0,
                monotonicity_score=0.0,
                passed_gate=False,
                gate_violations=["样本截面不足 (需至少 5 个日期与 2 只标的)"],
            )

        f_df = factor_df.loc[common_index, common_cols].astype(float)
        r_df = forward_returns_df.loc[common_index, common_cols].astype(float)

        ic_series: List[float] = []
        rank_ic_series: List[float] = []
        top_group_rets: List[float] = []
        bottom_group_rets: List[float] = []
        quantile_group_returns: List[np.ndarray] = []

        # 逐日计算截面相关系数与分组收益
        for dt in common_index:
            f_vals = f_df.loc[dt]
            r_vals = r_df.loc[dt]

            valid_mask = f_vals.notna() & r_vals.notna() & np.isfinite(f_vals) & np.isfinite(r_vals)
            f_clean = f_vals[valid_mask].values
            r_clean = r_vals[valid_mask].values

            if (
                len(f_clean) < 2
                or np.all(f_clean == f_clean[0])
                or np.all(r_clean == r_clean[0])
                or np.std(f_clean) < 1e-12
                or np.std(r_clean) < 1e-12
            ):
                continue

            # 1. Pearson IC
            p_corr = float(np.corrcoef(f_clean, r_clean)[0, 1])
            if np.isfinite(p_corr):
                ic_series.append(p_corr)

            # 2. Spearman Rank IC
            s_corr, _ = stats.spearmanr(f_clean, r_clean)
            if np.isfinite(s_corr):
                rank_ic_series.append(float(s_corr))

            # 3. 分层收益率计算
            if len(f_clean) >= quantiles:
                ranks = stats.rankdata(f_clean) / len(f_clean)
                q_step = 1.0 / quantiles
                q_rets = []
                for q in range(quantiles):
                    q_mask = (ranks > q * q_step) & (ranks <= (q + 1) * q_step)
                    if np.any(q_mask):
                        q_rets.append(float(np.mean(r_clean[q_mask])))
                    else:
                        q_rets.append(0.0)

                quantile_group_returns.append(np.array(q_rets))
                bottom_group_rets.append(q_rets[0])
                top_group_rets.append(q_rets[-1])

        # 统计汇总
        ic_arr = np.array(ic_series) if ic_series else np.array([0.0])
        rank_ic_arr = np.array(rank_ic_series) if rank_ic_series else np.array([0.0])

        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 1e-6
        rank_ic_mean = float(np.mean(rank_ic_arr))
        rank_ic_std = float(np.std(rank_ic_arr, ddof=1)) if len(rank_ic_arr) > 1 else 1e-6

        # 年化 ICIR
        icir = (ic_mean / ic_std) * np.sqrt(252.0) if ic_std > 1e-8 else 0.0
        rank_icir = (rank_ic_mean / rank_ic_std) * np.sqrt(252.0) if rank_ic_std > 1e-8 else 0.0

        # t-statistic: IC_mean / (IC_std / sqrt(N))
        t_stat = (ic_mean / (ic_std / np.sqrt(len(ic_arr)))) if (ic_std > 1e-8 and len(ic_arr) > 1) else 0.0

        # 多空年化收益率 (Top - Bottom)
        if top_group_rets and bottom_group_rets:
            spread = np.array(top_group_rets) - np.array(bottom_group_rets)
            ls_ann_ret = float(np.mean(spread) * 252.0)
        else:
            ls_ann_ret = 0.0

        # 因子换手率 (自相关系数代理)
        auto_corrs = []
        for i in range(len(common_index) - 1):
            t1 = f_df.loc[common_index[i]].values
            t2 = f_df.loc[common_index[i + 1]].values
            valid = np.isfinite(t1) & np.isfinite(t2)
            if np.sum(valid) >= 2 and np.std(t1[valid]) > 1e-12 and np.std(t2[valid]) > 1e-12:
                c = np.corrcoef(t1[valid], t2[valid])[0, 1]
                if np.isfinite(c):
                    auto_corrs.append(float(c))
        mean_autocorr = float(np.mean(auto_corrs)) if auto_corrs else 0.5
        turnover = max(0.0, min(1.0, 1.0 - mean_autocorr))

        # 单调性得分: 考核各分位平均收益是否严格单调
        if quantile_group_returns:
            avg_q_rets = np.mean(np.array(quantile_group_returns), axis=0)
            diffs = np.diff(avg_q_rets)
            pos_ratio = float(np.sum(diffs > 0) / len(diffs))
            neg_ratio = float(np.sum(diffs < 0) / len(diffs))
            monotonicity = max(pos_ratio, neg_ratio)
        else:
            monotonicity = 0.0

        # 质量门禁准入校验
        violations = []
        if abs(rank_ic_mean) < ic_threshold and abs(ic_mean) < ic_threshold:
            violations.append(f"|IC| 均值 ({abs(ic_mean):.4f}) 未达到门禁阈值 ({ic_threshold:.4f})")
        if abs(rank_icir) < icir_threshold and abs(icir) < icir_threshold:
            violations.append(f"ICIR ({abs(icir):.2f}) 未达到稳定性门禁阈值 ({icir_threshold:.2f})")
        if abs(t_stat) < t_stat_threshold:
            violations.append(f"|t-stat| ({abs(t_stat):.2f}) 统计显著性不足 ({t_stat_threshold:.2f})")

        passed_gate = (len(violations) == 0)

        return FactorEvaluationMetrics(
            ic_mean=round(ic_mean, 4),
            ic_std=round(ic_std, 4),
            rank_ic_mean=round(rank_ic_mean, 4),
            rank_ic_std=round(rank_ic_std, 4),
            icir=round(icir, 2),
            rank_icir=round(rank_icir, 2),
            t_stat=round(t_stat, 2),
            long_short_return=round(ls_ann_ret, 4),
            turnover=round(turnover, 4),
            monotonicity_score=round(monotonicity, 2),
            passed_gate=passed_gate,
            gate_violations=violations,
        )
