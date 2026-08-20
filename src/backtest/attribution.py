"""收益与风险归因

高级基金必须定期拆解收益来源：
1. Brinson归因：分解超额收益为配置效应+选股效应+交互效应
2. Barra风险分解：系统性风险(因子) vs 个股风险
3. 因子收益归因：各因子对组合收益的贡献
4. 行业归因：各行业对收益的贡献

收益 = 行业收益 + 风格收益 + 因子收益 + 选股Alpha + 随机残差
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BrinsonAttribution:
    """Brinson-Fachler收益归因

    将超额收益分解为：
    - 配置效应（Allocation）：超配/低配行业的贡献
    - 选股效应（Selection）：行业内选股能力的贡献
    - 交互效应（Interaction）：配置和选股的交叉影响
    """

    @staticmethod
    def attribute(portfolio_weights: pd.Series,
                  benchmark_weights: pd.Series,
                  portfolio_returns: pd.Series,
                  benchmark_returns: pd.Series,
                  industry_map: pd.Series) -> pd.DataFrame:
        """
        Args:
            portfolio_weights: 组合中各股权重
            benchmark_weights: 基准中各股权重
            portfolio_returns: 各股收益率
            benchmark_returns: 基准各股收益率
            industry_map: 股票 -> 行业映射
        Returns:
            DataFrame: 各行业的配置/选股/交互效应
        """
        stocks = portfolio_weights.index.intersection(benchmark_weights.index)
        industries = industry_map.loc[industry_map.index.isin(stocks)].unique()

        results = []
        for ind in industries:
            ind_stocks = industry_map[industry_map == ind].index
            ind_stocks = ind_stocks.intersection(stocks)

            wp = portfolio_weights.loc[ind_stocks].sum()
            wb = benchmark_weights.loc[ind_stocks].sum()
            rp = (portfolio_weights.loc[ind_stocks] * portfolio_returns.loc[ind_stocks]).sum() / (wp + 1e-10)
            rb = (benchmark_weights.loc[ind_stocks] * benchmark_returns.loc[ind_stocks]).sum() / (wb + 1e-10)

            # Brinson-Fachler
            allocation = (wp - wb) * rb
            selection = wb * (rp - rb)
            interaction = (wp - wb) * (rp - rb)
            total = allocation + selection + interaction

            results.append({
                "industry": ind,
                "portfolio_weight": wp,
                "benchmark_weight": wb,
                "portfolio_return": rp,
                "benchmark_return": rb,
                "allocation_effect": allocation,
                "selection_effect": selection,
                "interaction_effect": interaction,
                "total_effect": total,
            })

        df = pd.DataFrame(results)
        # 添加合计行
        totals = {
            "industry": "Total",
            "allocation_effect": df["allocation_effect"].sum(),
            "selection_effect": df["selection_effect"].sum(),
            "interaction_effect": df["interaction_effect"].sum(),
            "total_effect": df["total_effect"].sum(),
        }
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
        return df


class FactorReturnAttribution:
    """因子收益归因

    将组合收益分解为各因子的贡献：
    贡献_i = 因子暴露_i * 因子收益_i
    """

    @staticmethod
    def attribute(returns: pd.Series,
                  factor_exposures: pd.DataFrame,
                  factor_returns: pd.Series) -> pd.DataFrame:
        """
        Args:
            returns: 组合收益率
            factor_exposures: 组合在各因子上的暴露(index=因子名, column=exposure)
            factor_returns: 各因子的收益率
        Returns:
            DataFrame: 各因子的收益贡献
        """
        contributions = []
        for factor in factor_exposures.index:
            if factor in factor_returns.index:
                exposure = factor_exposures.loc[factor]
                ret = factor_returns[factor]
                contributions.append({
                    "factor": factor,
                    "exposure": exposure,
                    "factor_return": ret,
                    "contribution": exposure * ret,
                })

        df = pd.DataFrame(contributions)
        explained = df["contribution"].sum()
        residual = returns - explained
        df = pd.concat([df, pd.DataFrame([{
            "factor": "Residual",
            "exposure": 1,
            "factor_return": residual,
            "contribution": residual,
        }])], ignore_index=True)
        return df


class IndustryAttribution:
    """行业收益归因"""

    @staticmethod
    def attribute(weights: pd.DataFrame,
                  returns: pd.DataFrame,
                  industry_map: pd.Series) -> pd.DataFrame:
        """
        Args:
            weights: 组合权重历史(date x stock)
            returns: 收益率历史(date x stock)
            industry_map: 股票 -> 行业映射
        Returns:
            DataFrame: 各行业各期的收益贡献
        """
        results = []
        for date in weights.index:
            if date not in returns.index:
                continue
            w = weights.loc[date]
            r = returns.loc[date]
            stocks = w.index.intersection(r.index).intersection(industry_map.index)

            for stock in stocks:
                ind = industry_map.get(stock, "Unknown")
                contribution = w[stock] * r[stock]
                results.append({
                    "date": date,
                    "industry": ind,
                    "stock": stock,
                    "weight": w[stock],
                    "return": r[stock],
                    "contribution": contribution,
                })

        df = pd.DataFrame(results)
        if df.empty:
            return df
        # 按行业汇总
        industry_contrib = df.groupby(["date", "industry"])["contribution"].sum().unstack("industry")
        return industry_contrib


class BarraRiskDecomposition:
    """Barra风格风险分解

    将组合风险分解为系统性风险(因子风险)和个股风险
    简化版使用5个风格因子：规模、价值、动量、波动率、流动性
    """

    STYLE_FACTORS = ["size", "value", "momentum", "volatility", "liquidity"]

    @staticmethod
    def decompose(returns: pd.Series,
                  factor_exposures: pd.DataFrame,
                  factor_cov_matrix: pd.DataFrame,
                  stock_residual_var: pd.Series) -> dict:
        """
        Args:
            returns: 组合收益率(用于计算总风险)
            factor_exposures: 组合因子暴露(index=因子, column=exposure)
            factor_cov_matrix: 因子协方差矩阵
            stock_residual_var: 个股残差方差
        Returns:
            {systemic_risk, idiosyncratic_risk, factor_contribution, total_risk}
        """
        # 系统性风险 = w' * F * w (w=因子暴露, F=因子协方差)
        exposures = factor_exposures["exposure"].values if "exposure" in factor_exposures.columns else factor_exposures.values
        systemic_var = exposures @ factor_cov_matrix.values @ exposures

        # 个股风险 = 残差方差加权
        idiosyncratic_var = stock_residual_var.sum()

        # 各因子风险贡献
        factor_contrib = {}
        for i, factor in enumerate(factor_exposures.index):
            marginal = factor_cov_matrix.values[i] @ exposures
            factor_contrib[factor] = exposures[i] * marginal

        total_var = systemic_var + idiosyncratic_var
        total_risk = np.sqrt(total_var) if total_var > 0 else 0
        systemic_risk = np.sqrt(max(systemic_var, 0))
        idiosyncratic_risk = np.sqrt(max(idiosyncratic_var, 0))

        return {
            "total_risk": total_risk,
            "systemic_risk": systemic_risk,
            "idiosyncratic_risk": idiosyncratic_risk,
            "systemic_pct": systemic_var / total_var if total_var > 0 else 0,
            "idiosyncratic_pct": idiosyncratic_var / total_var if total_var > 0 else 0,
            "factor_contribution": factor_contrib,
        }


def calc_factor_returns(factor_exposures: pd.DataFrame,
                        stock_returns: pd.Series) -> pd.Series:
    """通过截面回归计算因子收益率

    对每个截面: r = X * f + epsilon
    其中r是N个股票的收益, X是因子暴露矩阵, f是因子收益

    Args:
        factor_exposures: 股票x因子的暴露矩阵
        stock_returns: 股票收益率
    Returns:
        因子收益率(pd.Series)
    """
    common = stock_returns.dropna().index.intersection(factor_exposures.index)
    if len(common) < len(factor_exposures.columns) + 1:
        return pd.Series(dtype=float)

    y = stock_returns.loc[common].values.astype(float)
    X = factor_exposures.loc[common].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])  # 截距项

    # OLS: f = (X'X)^-1 X'y
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        factor_returns = pd.Series(
            beta[1:],
            index=factor_exposures.columns,
            name="factor_return"
        )
        return factor_returns
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float)
