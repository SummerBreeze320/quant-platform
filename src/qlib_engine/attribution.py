import numpy as np
import pandas as pd
from typing import Dict, Any, List

class BrinsonAttribution:
    """
    Brinson-Fachler (BF) Performance Attribution Model.
    Decomposes portfolio excess returns into:
      - Allocation Effect (AR): Value added by overweighting/underweighting sectors relative to benchmark.
      - Selection Effect (SR): Value added by picking outperforming stocks within sectors.
      - Interaction Effect (IR): Value added from the combined timing of allocation and stock picking.
    Identity: Allocation + Selection + Interaction == Total Excess Return (R_p - R_b)
    """

    @staticmethod
    def calculate(
        holdings_df: pd.DataFrame,
        benchmark_weights: Dict[str, float],
        benchmark_returns: Dict[str, float],
        weight_col: str = "weight_p",
        ret_col: str = "ret_p",
        industry_col: str = "industry"
    ) -> Dict[str, Any]:
        """
        Calculates Brinson attribution given portfolio holdings and benchmark industry profile.
        """
        if holdings_df.empty:
            return {
                "portfolio_return": 0.0,
                "benchmark_return": 0.0,
                "total_excess_return": 0.0,
                "total_allocation": 0.0,
                "total_selection": 0.0,
                "total_interaction": 0.0,
                "industry_details": []
            }

        # Aggregate portfolio weights and weighted returns by industry
        p_records = []
        for ind, g in holdings_df.groupby(industry_col):
            w_sum = float(g[weight_col].sum())
            if w_sum > 0:
                ret_weighted = float((g[weight_col] * g[ret_col]).sum() / w_sum)
            else:
                ret_weighted = 0.0
            p_records.append({
                industry_col: ind,
                "weight_p": w_sum,
                "ret_p": ret_weighted
            })
        p_agg = pd.DataFrame(p_records)

        all_industries = sorted(list(set(benchmark_weights.keys()).union(set(p_agg[industry_col].unique()))))

        total_b_weight = sum(benchmark_weights.values()) or 1.0
        norm_b_weights = {ind: benchmark_weights.get(ind, 0.0) / total_b_weight for ind in all_industries}

        # Benchmark total return R_b
        r_b = sum(norm_b_weights[ind] * benchmark_returns.get(ind, 0.0) for ind in all_industries)

        # Portfolio total return R_p
        p_dict = dict(zip(p_agg[industry_col], zip(p_agg["weight_p"], p_agg["ret_p"])))
        r_p = sum(p_dict.get(ind, (0.0, 0.0))[0] * p_dict.get(ind, (0.0, 0.0))[1] for ind in all_industries)

        total_excess = r_p - r_b

        industry_details = []
        total_allocation = 0.0
        total_selection = 0.0
        total_interaction = 0.0

        for ind in all_industries:
            w_p, ret_p = p_dict.get(ind, (0.0, 0.0))
            w_b = norm_b_weights.get(ind, 0.0)
            ret_b = benchmark_returns.get(ind, 0.0)

            # Brinson-Fachler formulas:
            # Allocation Effect = (w_p - w_b) * (R_b,i - R_b)
            alloc = (w_p - w_b) * (ret_b - r_b)
            # Selection Effect = w_b * (R_p,i - R_b,i)
            selec = w_b * (ret_p - ret_b)
            # Interaction Effect = (w_p - w_b) * (R_p,i - R_b,i)
            inter = (w_p - w_b) * (ret_p - ret_b)

            total_allocation += alloc
            total_selection += selec
            total_interaction += inter

            industry_details.append({
                "industry": ind,
                "weight_p": round(float(w_p), 4),
                "weight_b": round(float(w_b), 4),
                "ret_p": round(float(ret_p), 4),
                "ret_b": round(float(ret_b), 4),
                "allocation": round(float(alloc), 6),
                "selection": round(float(selec), 6),
                "interaction": round(float(inter), 6),
                "total_excess": round(float(alloc + selec + inter), 6)
            })

        return {
            "portfolio_return": round(float(r_p), 6),
            "benchmark_return": round(float(r_b), 6),
            "total_excess_return": round(float(total_excess), 6),
            "total_allocation": round(float(total_allocation), 6),
            "total_selection": round(float(total_selection), 6),
            "total_interaction": round(float(total_interaction), 6),
            "industry_details": industry_details
        }
