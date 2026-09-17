from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
import cvxpy as cp
from src.common.logger import logger

class PortfolioOptimizer:
    """
    Portfolio Optimizer enforcing industry and style neutrality.
    Anchors portfolio sector exposures strictly to benchmark weights (e.g. CSI300/CSI500)
    to eliminate sector-driven style drift and capture pure selection alpha.
    """

    @staticmethod
    def allocate_industry_neutral(
        df: pd.DataFrame,
        score_col: str = "score",
        industry_col: str = "industry",
        benchmark_industry_weights: Optional[Dict[str, float]] = None,
        top_n_per_industry: int = 3,
        symbol_col: str = "symbol"
    ) -> pd.DataFrame:
        """
        Selects top-ranked stocks within each industry and assigns weights such that
        each industry's total weight strictly matches its benchmark weight.
        """
        if df.empty:
            return pd.DataFrame(columns=[symbol_col, industry_col, "weight"])

        industries = df[industry_col].unique()

        # If benchmark weights not specified, use equal industry weighting
        if not benchmark_industry_weights:
            equal_w = 1.0 / len(industries)
            benchmark_weights = {ind: equal_w for ind in industries}
        else:
            total_target = sum(benchmark_industry_weights.get(ind, 0.0) for ind in industries)
            if total_target > 0:
                benchmark_weights = {
                    ind: benchmark_industry_weights.get(ind, 0.0) / total_target
                    for ind in industries
                }
            else:
                benchmark_weights = {ind: 1.0 / len(industries) for ind in industries}

        selected_records = []
        for ind in industries:
            ind_df = df[df[industry_col] == ind].sort_values(by=score_col, ascending=False)
            top_stocks = ind_df.head(top_n_per_industry)
            if top_stocks.empty:
                continue

            ind_weight = benchmark_weights.get(ind, 0.0)
            stock_weight = ind_weight / len(top_stocks)

            for _, row in top_stocks.iterrows():
                selected_records.append({
                    symbol_col: row[symbol_col],
                    industry_col: ind,
                    "weight": float(stock_weight)
                })

        alloc_df = pd.DataFrame(selected_records)
        if not alloc_df.empty:
            # Rescale sum of weights to 1.0
            sum_w = alloc_df["weight"].sum()
            if sum_w > 0:
                alloc_df["weight"] = alloc_df["weight"] / sum_w

        return alloc_df


class ConvexOptimizer:
    """
    Institutional Convex Portfolio Optimizer using Quadratic Programming (QP).
    Formulates a mean-variance-turnover optimization problem under Barra structural risk:
        min_w  - alpha^T w + (lambda / 2) * ( (X^T h)^T F (X^T h) + sum_i delta_i h_i^2 ) + tau * ||w - w0||_1
    Subject to:
        - 1^T w == 1.0 (Full investment)
        - min_w <= w_i <= max_w (Long-only & single stock limit)
        - -tol_ind <= X_ind^T h <= tol_ind (Industry neutrality band)
        - -tol_style <= X_style^T h <= tol_style (Style neutrality band)
        - 0.5 * ||w - w0||_1 <= turnover_limit (Optional)
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_stock_weight: float = 0.05,
        min_stock_weight: float = 0.0,
        industry_tolerance: Optional[float] = 0.01,
        style_tolerance: Optional[float] = 0.10,
        turnover_penalty: float = 0.0,
        turnover_limit: Optional[float] = None,
        solver: str = "OSQP"
    ):
        self.risk_aversion = risk_aversion
        self.max_stock_weight = max_stock_weight
        self.min_stock_weight = min_stock_weight
        self.industry_tolerance = industry_tolerance
        self.style_tolerance = style_tolerance
        self.turnover_penalty = turnover_penalty
        self.turnover_limit = turnover_limit
        self.solver = solver

    def optimize(
        self,
        alpha: Union[pd.Series, np.ndarray],
        exposures: Union[pd.DataFrame, np.ndarray],
        factor_cov: Union[pd.DataFrame, np.ndarray],
        specific_var: Union[pd.Series, np.ndarray],
        benchmark_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        prev_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        industry_factors: Optional[List[str]] = None,
        style_factors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Solves the QP problem and returns optimal portfolio weights along with risk metrics.
        """
        # Determine symbol index
        symbols = None
        if isinstance(alpha, pd.Series):
            symbols = alpha.index
            alpha_arr = alpha.values
        else:
            alpha_arr = np.array(alpha).flatten()

        n_stocks = len(alpha_arr)
        if n_stocks == 0:
            raise ValueError("Stock universe is empty.")

        # Factor names
        factor_names = []
        if isinstance(exposures, pd.DataFrame):
            factor_names = list(exposures.columns)
            X_arr = exposures.values
        else:
            X_arr = np.array(exposures)
            factor_names = [f"factor_{i}" for i in range(X_arr.shape[1])]

        n_factors = X_arr.shape[1]

        # Factor covariance matrix
        if isinstance(factor_cov, pd.DataFrame):
            F_arr = factor_cov.values
        else:
            F_arr = np.array(factor_cov)

        # Ensure F is strictly positive semi-definite
        F_arr = (F_arr + F_arr.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(F_arr)
        F_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
        F_psd = (F_psd + F_psd.T) / 2.0

        # Specific risk variance
        if isinstance(specific_var, pd.Series):
            delta_arr = specific_var.values
        else:
            delta_arr = np.array(specific_var).flatten()
        delta_arr = np.maximum(delta_arr, 1e-8)

        # Benchmark weights
        if benchmark_weights is not None:
            if isinstance(benchmark_weights, pd.Series):
                w_b = benchmark_weights.reindex(symbols).fillna(0.0).values if symbols is not None else benchmark_weights.values
            else:
                w_b = np.array(benchmark_weights).flatten()
        else:
            w_b = np.zeros(n_stocks)

        # Previous weights
        if prev_weights is not None:
            if isinstance(prev_weights, pd.Series):
                w0 = prev_weights.reindex(symbols).fillna(0.0).values if symbols is not None else prev_weights.values
            else:
                w0 = np.array(prev_weights).flatten()
        else:
            w0 = np.zeros(n_stocks)

        # Auto-adjust maximum weight if mathematically impossible to sum to 1.0
        eff_max_w = max(self.max_stock_weight, 1.0 / n_stocks + 1e-4)

        # Build QP problem
        w = cp.Variable(n_stocks)
        h = w - w_b
        y = cp.Variable(n_factors)

        # Risk term: factor risk + specific risk
        factor_risk_term = cp.quad_form(y, cp.psd_wrap(F_psd))
        specific_risk_term = cp.sum(cp.multiply(delta_arr, cp.square(h)))
        portfolio_risk = (factor_risk_term + specific_risk_term)

        # Objective
        obj_expr = -alpha_arr @ w + (self.risk_aversion / 2.0) * portfolio_risk

        if self.turnover_penalty > 0 and prev_weights is not None:
            obj_expr += self.turnover_penalty * cp.norm(w - w0, 1)

        # Constraints
        constraints = [
            cp.sum(w) == 1.0,
            w >= self.min_stock_weight,
            w <= eff_max_w,
            y == X_arr.T @ h
        ]

        if self.turnover_limit is not None and prev_weights is not None:
            constraints.append(0.5 * cp.norm(w - w0, 1) <= self.turnover_limit)

        # Industry constraints
        if industry_factors and self.industry_tolerance is not None:
            ind_idx = [factor_names.index(f) for f in industry_factors if f in factor_names]
            for idx in ind_idx:
                constraints.append(y[idx] <= self.industry_tolerance)
                constraints.append(y[idx] >= -self.industry_tolerance)

        # Style constraints
        if style_factors and self.style_tolerance is not None:
            style_idx = [factor_names.index(f) for f in style_factors if f in factor_names]
            for idx in style_idx:
                constraints.append(y[idx] <= self.style_tolerance)
                constraints.append(y[idx] >= -self.style_tolerance)

        prob = cp.Problem(cp.Minimize(obj_expr), constraints)

        status = "failed"
        try:
            prob.solve(solver=getattr(cp, self.solver, cp.OSQP), verbose=False)
            status = prob.status
        except Exception as e:
            logger.warning(f"Convex optimizer primary solve failed with exception: {e}")

        # Fallback if infeasible: relax industry and style bounds
        if status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Convex optimization status '{status}' - attempting relaxed solve.")
            relaxed_constraints = [
                cp.sum(w) == 1.0,
                w >= self.min_stock_weight,
                w <= eff_max_w,
                y == X_arr.T @ h
            ]
            relaxed_prob = cp.Problem(cp.Minimize(obj_expr), relaxed_constraints)
            try:
                relaxed_prob.solve(solver=cp.OSQP, verbose=False)
                status = "relaxed_optimal" if relaxed_prob.status in ["optimal", "optimal_inaccurate"] else "failed"
            except Exception as e:
                logger.error(f"Relaxed solve also failed: {e}")
                status = "failed"

        # Final extraction or fallback
        if status in ["optimal", "optimal_inaccurate", "relaxed_optimal"] and w.value is not None:
            opt_w = np.array(w.value).flatten()
            opt_w = np.maximum(opt_w, 0.0)
            sum_opt = np.sum(opt_w)
            if sum_opt > 0:
                opt_w = opt_w / sum_opt
            else:
                opt_w = np.ones(n_stocks) / n_stocks
        else:
            logger.error("QP optimization failed, falling back to equal weighting.")
            opt_w = np.ones(n_stocks) / n_stocks
            status = "fallback_equal_weight"

        # Active risk and factor exposures
        active_w = opt_w - w_b
        active_factor_exp = X_arr.T @ active_w
        pred_act_factor_var = float(active_factor_exp.T @ F_psd @ active_factor_exp)
        pred_act_spec_var = float(np.sum((active_w ** 2) * delta_arr))
        tracking_error = float(np.sqrt(max(0.0, pred_act_factor_var + pred_act_spec_var)) * np.sqrt(252.0))

        actual_turnover = float(0.5 * np.sum(np.abs(opt_w - w0))) if prev_weights is not None else 1.0
        exp_alpha = float(np.dot(alpha_arr, opt_w))

        weights_series = pd.Series(opt_w, index=symbols) if symbols is not None else pd.Series(opt_w)
        active_exp_dict = {f_name: float(active_factor_exp[i]) for i, f_name in enumerate(factor_names)}

        return {
            "weights": weights_series,
            "status": status,
            "expected_alpha": round(exp_alpha, 6),
            "predicted_tracking_error": round(tracking_error, 4),
            "active_exposures": active_exp_dict,
            "turnover": round(actual_turnover, 4)
        }
