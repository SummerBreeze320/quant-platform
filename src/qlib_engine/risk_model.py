from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from src.common.logger import logger

class BarraRiskModel:
    """
    Barra-style Structural Multi-Factor Risk Model.
    Decomposes asset returns into systematic factor returns and idiosyncratic specific returns:
        r_t = X_t * f_t + u_t
    And covariance structure into:
        V = X * F * X^T + Delta
    Where:
        - X is the N x K factor exposure matrix
        - F is the K x K factor covariance matrix
        - Delta is the N x N diagonal specific risk matrix
    """

    def __init__(self, ridge_alpha: float = 1e-6, min_eigenvalue: float = 1e-8):
        self.ridge_alpha = ridge_alpha
        self.min_eigenvalue = min_eigenvalue

    def estimate_factor_returns(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        ret_col: str = "ret",
        weight_col: Optional[str] = "market_cap"
    ) -> Dict[str, Any]:
        """
        Estimates pure factor returns for a single cross-section using Weighted Least Squares (WLS)
        with ridge regularization to handle multicollinearity among dummy/style factors.
        """
        if df.empty or ret_col not in df.columns:
            raise ValueError("Dataframe must not be empty and must contain ret_col.")

        missing_factors = [c for c in factor_cols if c not in df.columns]
        if missing_factors:
            raise ValueError(f"Missing factor columns in dataframe: {missing_factors}")

        clean_df = df.dropna(subset=[ret_col] + factor_cols).copy()
        if clean_df.empty:
            raise ValueError("All rows contain NaN in factor or return columns.")

        X = clean_df[factor_cols].values
        y = clean_df[ret_col].values
        n_samples, n_factors = X.shape

        # Weights: square root of market cap (industry standard) or equal weights
        if weight_col and weight_col in clean_df.columns:
            weights = np.sqrt(np.maximum(clean_df[weight_col].values, 0.0))
            sum_w = np.sum(weights)
            if sum_w > 0:
                weights = weights / sum_w
            else:
                weights = np.ones(n_samples) / n_samples
        else:
            weights = np.ones(n_samples) / n_samples

        # WLS Transformation: X* = W^(1/2) X, y* = W^(1/2) y
        sqrt_w = np.sqrt(weights)[:, np.newaxis]
        X_star = X * sqrt_w
        y_star = y * sqrt_w.flatten()

        # Regularized normal equation: (X*^T X* + alpha * I) f = X*^T y*
        reg_matrix = self.ridge_alpha * np.eye(n_factors)
        A = X_star.T @ X_star + reg_matrix
        b = X_star.T @ y_star

        try:
            f = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            logger.warning("Normal equation singular, falling back to pseudoinverse.")
            f = np.linalg.pinv(A) @ b

        # Compute residuals: u = y - X f
        fitted = X @ f
        residuals = y - fitted

        # Weighted R-squared
        y_mean = np.sum(weights * y)
        total_ss = np.sum(weights * (y - y_mean) ** 2)
        residual_ss = np.sum(weights * (y - fitted) ** 2)
        r2 = 1.0 - (residual_ss / (total_ss + 1e-12))
        r2 = float(max(0.0, min(1.0, r2)))

        return {
            "factor_returns": pd.Series(f, index=factor_cols),
            "residuals": pd.Series(residuals, index=clean_df.index),
            "r2": r2
        }

    def compute_factor_covariance(
        self,
        factor_returns_df: pd.DataFrame,
        half_life: int = 63
    ) -> pd.DataFrame:
        """
        Computes the factor covariance matrix using Exponentially Weighted Moving Average (EWMA)
        with eigenvalue clipping to ensure strict positive semi-definiteness (PSD).
        """
        if factor_returns_df.empty:
            raise ValueError("factor_returns_df is empty.")

        n_obs, n_factors = factor_returns_df.shape
        decay = 0.5 ** (1.0 / max(half_life, 1))

        # Calculate exponential weights backwards from most recent
        weights = np.array([decay ** (n_obs - 1 - t) for t in range(n_obs)])
        weights = weights / np.sum(weights)

        f_values = factor_returns_df.values
        # Weighted mean
        weighted_mean = np.sum(weights[:, np.newaxis] * f_values, axis=0)
        demeaned = f_values - weighted_mean

        # Weighted covariance matrix: sum_t w_t * demeaned_t * demeaned_t^T
        cov = (demeaned.T * weights) @ demeaned

        # Guarantee symmetry
        cov = (cov + cov.T) / 2.0

        # Eigenvalue adjustment to enforce positive semi-definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        adjusted_eigenvalues = np.maximum(eigenvalues, self.min_eigenvalue)
        psd_cov = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
        psd_cov = (psd_cov + psd_cov.T) / 2.0

        return pd.DataFrame(
            psd_cov,
            index=factor_returns_df.columns,
            columns=factor_returns_df.columns
        )

    def compute_specific_risk(
        self,
        residuals_df: pd.DataFrame,
        half_life: int = 63,
        min_variance: float = 1e-6
    ) -> pd.Series:
        """
        Computes the idiosyncratic specific risk (variance) for each stock using EWMA.
        """
        if residuals_df.empty:
            raise ValueError("residuals_df is empty.")

        n_obs = len(residuals_df)
        decay = 0.5 ** (1.0 / max(half_life, 1))
        weights = np.array([decay ** (n_obs - 1 - t) for t in range(n_obs)])
        weights = weights / np.sum(weights)

        # EWMA of residual variance (residuals have 0 theoretical mean)
        res_values = residuals_df.fillna(0.0).values
        weighted_var = np.sum(weights[:, np.newaxis] * (res_values ** 2), axis=0)
        specific_var = np.maximum(weighted_var, min_variance)

        return pd.Series(specific_var, index=residuals_df.columns)

    def predict_portfolio_risk(
        self,
        weights: Union[pd.Series, np.ndarray],
        exposures: Union[pd.DataFrame, np.ndarray],
        factor_cov: Union[pd.DataFrame, np.ndarray],
        specific_var: Union[pd.Series, np.ndarray],
        benchmark_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        annualized: bool = True
    ) -> Dict[str, float]:
        """
        Calculates predicted portfolio total risk, systematic factor risk, specific risk,
        and active risk (tracking error) relative to benchmark.
        """
        w = np.array(weights).flatten()
        X = np.array(exposures)
        F = np.array(factor_cov)
        delta = np.array(specific_var).flatten()

        n_stocks = len(w)
        if benchmark_weights is not None:
            w_b = np.array(benchmark_weights).flatten()
        else:
            w_b = np.zeros(n_stocks)

        # Active weights
        h = w - w_b

        # Portfolio total risk
        w_factor_exp = X.T @ w
        total_factor_var = float(w_factor_exp.T @ F @ w_factor_exp)
        total_spec_var = float(np.sum((w ** 2) * delta))
        total_var = max(0.0, total_factor_var + total_spec_var)

        # Portfolio active risk (Tracking Error)
        h_factor_exp = X.T @ h
        act_factor_var = float(h_factor_exp.T @ F @ h_factor_exp)
        act_spec_var = float(np.sum((h ** 2) * delta))
        act_total_var = max(0.0, act_factor_var + act_spec_var)

        scale = np.sqrt(252.0) if annualized else 1.0

        return {
            "total_risk": float(np.sqrt(total_var) * scale),
            "factor_risk": float(np.sqrt(max(0.0, total_factor_var)) * scale),
            "specific_risk": float(np.sqrt(max(0.0, total_spec_var)) * scale),
            "tracking_error": float(np.sqrt(act_total_var) * scale),
            "active_factor_risk": float(np.sqrt(max(0.0, act_factor_var)) * scale),
            "active_specific_risk": float(np.sqrt(max(0.0, act_spec_var)) * scale)
        }
