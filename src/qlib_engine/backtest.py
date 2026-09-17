import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.common.logger import logger
from src.qlib_engine.neutralization import FactorNeutralizer
from src.qlib_engine.optimizer import PortfolioOptimizer, ConvexOptimizer

class BacktestEngine:
    """Portfolio simulation and performance evaluation engine."""

    def __init__(
        self,
        top_k: int = 30,
        benchmark: str = "000300.SH",
        commission_rate: float = 0.0002,
        stamp_tax_rate: float = 0.0005,
        slippage: float = 0.0005,
        neutralize: bool = False,
        industry_neutral_allocation: bool = False,
        benchmark_industry_weights: Optional[Dict[str, float]] = None,
        use_convex_optimizer: bool = False,
        risk_aversion: float = 1.0,
        max_stock_weight: float = 0.05,
        industry_tolerance: float = 0.01,
        style_tolerance: float = 0.10,
        turnover_penalty: float = 0.0
    ):
        self.top_k = top_k
        self.benchmark = benchmark
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.neutralize = neutralize
        self.industry_neutral_allocation = industry_neutral_allocation
        self.benchmark_industry_weights = benchmark_industry_weights
        self.use_convex_optimizer = use_convex_optimizer
        self.risk_aversion = risk_aversion
        self.max_stock_weight = max_stock_weight
        self.industry_tolerance = industry_tolerance
        self.style_tolerance = style_tolerance
        self.turnover_penalty = turnover_penalty

    def run_backtest(
        self,
        pred_df: pd.DataFrame,
        score_col: str = "score",
        ret_col: str = "ret",
        date_col: str = "date",
        symbol_col: str = "symbol",
        industry_col: str = "industry",
        market_cap_col: str = "market_cap"
    ) -> Dict[str, Any]:
        """
        Simulates daily rebalancing with Top-K equal weight, Industry-Neutral benchmark weighting,
        or QP Convex Optimization under Barra-style risk constraints.
        pred_df must contain [date, symbol, score, ret].
        """
        if pred_df.empty:
            raise ValueError("Prediction DataFrame is empty.")

        dates = sorted(pred_df[date_col].unique())
        prev_holdings_weights: Dict[str, float] = {}
        daily_returns = []
        daily_dates = []

        cost_per_trade = self.commission_rate + self.slippage

        for d in dates:
            day_slice = pred_df[pred_df[date_col] == d].copy()
            if day_slice.empty:
                continue

            current_score_col = score_col

            # 1. Apply Cross-Sectional Neutralization if requested
            if self.neutralize:
                has_ind = industry_col in day_slice.columns
                has_cap = market_cap_col in day_slice.columns
                day_slice = FactorNeutralizer.neutralize(
                    day_slice,
                    score_col=score_col,
                    industry_col=industry_col if has_ind else None,
                    market_cap_col=market_cap_col if has_cap else None
                )
                current_score_col = "score_neutral"

            # 2. Portfolio Construction
            if self.use_convex_optimizer:
                # Institutional QP Convex Optimization
                day_symbols = list(day_slice[symbol_col])
                stocks_df = day_slice.set_index(symbol_col)
                alpha_series = stocks_df[current_score_col]

                exposure_cols = []
                ind_factors = []
                style_factors = []

                if industry_col in stocks_df.columns:
                    ind_dummies = pd.get_dummies(stocks_df[industry_col], prefix="ind", dtype=float)
                    stocks_df = pd.concat([stocks_df, ind_dummies], axis=1)
                    ind_factors = list(ind_dummies.columns)
                    exposure_cols.extend(ind_factors)

                if market_cap_col in stocks_df.columns:
                    stocks_df["style_size"] = np.log(np.maximum(stocks_df[market_cap_col].astype(float), 1.0))
                    std = stocks_df["style_size"].std()
                    if std > 0:
                        stocks_df["style_size"] = (stocks_df["style_size"] - stocks_df["style_size"].mean()) / std
                    style_factors.append("style_size")
                    exposure_cols.append("style_size")

                if not exposure_cols:
                    stocks_df["factor_market"] = 1.0
                    exposure_cols.append("factor_market")

                X_exp = stocks_df[exposure_cols]
                k = len(exposure_cols)
                factor_cov = np.eye(k) * 0.0004
                specific_var = np.full(len(stocks_df), 0.0005)

                prev_w_series = pd.Series(prev_holdings_weights).reindex(stocks_df.index).fillna(0.0)
                b_weights = pd.Series(1.0 / len(stocks_df), index=stocks_df.index)

                opt = ConvexOptimizer(
                    risk_aversion=self.risk_aversion,
                    max_stock_weight=self.max_stock_weight,
                    industry_tolerance=self.industry_tolerance,
                    style_tolerance=self.style_tolerance,
                    turnover_penalty=self.turnover_penalty
                )

                opt_res = opt.optimize(
                    alpha=alpha_series,
                    exposures=X_exp,
                    factor_cov=factor_cov,
                    specific_var=specific_var,
                    benchmark_weights=b_weights,
                    prev_weights=prev_w_series,
                    industry_factors=ind_factors,
                    style_factors=style_factors
                )

                opt_w = opt_res["weights"]
                curr_weights = {s: float(w) for s, w in opt_w.items() if w > 1e-4}
                gross_return = float(sum(curr_weights.get(s, 0.0) * float(stocks_df.loc[s, ret_col]) for s in curr_weights))

            elif self.industry_neutral_allocation and industry_col in day_slice.columns:
                n_inds = len(day_slice[industry_col].unique())
                per_ind_k = max(1, self.top_k // max(1, n_inds))
                alloc_df = PortfolioOptimizer.allocate_industry_neutral(
                    day_slice,
                    score_col=current_score_col,
                    industry_col=industry_col,
                    benchmark_industry_weights=self.benchmark_industry_weights,
                    top_n_per_industry=per_ind_k,
                    symbol_col=symbol_col
                )
                curr_weights = dict(zip(alloc_df[symbol_col], alloc_df["weight"]))
                merged = alloc_df.merge(day_slice[[symbol_col, ret_col]], on=symbol_col, how="inner")
                gross_return = float((merged["weight"] * merged[ret_col]).sum())
            else:
                top_holdings_df = day_slice.sort_values(by=current_score_col, ascending=False).head(self.top_k)
                n_stocks = len(top_holdings_df)
                eq_weight = 1.0 / n_stocks if n_stocks > 0 else 0.0
                curr_weights = {s: eq_weight for s in top_holdings_df[symbol_col]}
                gross_return = float(top_holdings_df[ret_col].mean()) if not top_holdings_df.empty else 0.0

            # 3. Turnover and Cost Calculation
            if not prev_holdings_weights:
                turnover = 1.0
            else:
                all_symbols = set(prev_holdings_weights.keys()).union(set(curr_weights.keys()))
                turnover = float(sum(
                    abs(curr_weights.get(s, 0.0) - prev_holdings_weights.get(s, 0.0))
                    for s in all_symbols
                ) / 2.0)

            fee = turnover * (cost_per_trade + self.stamp_tax_rate / 2.0)
            net_return = gross_return - fee

            daily_returns.append(net_return)
            daily_dates.append(str(d))
            prev_holdings_weights = curr_weights

        returns_arr = np.array(daily_returns)
        cum_returns = np.cumprod(1.0 + returns_arr) - 1.0

        # Performance statistics
        n_days = len(returns_arr)
        total_return = float(cum_returns[-1]) if n_days > 0 else 0.0
        ann_factor = 252.0 / max(n_days, 1)
        annualized_return = float((1.0 + total_return) ** ann_factor - 1.0)

        std_ret = float(np.std(returns_arr))
        sharpe = float((np.mean(returns_arr) / (std_ret + 1e-8)) * np.sqrt(252)) if std_ret > 0 else 0.0

        # Maximum drawdown
        equity = 1.0 + cum_returns
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        win_rate = float(np.mean(returns_arr > 0)) if n_days > 0 else 0.0

        return {
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
            "n_trading_days": n_days,
            "cumulative_returns": {d: round(float(c), 4) for d, c in zip(daily_dates, cum_returns)},
            "daily_returns": {d: round(float(r), 4) for d, c, r in zip(daily_dates, cum_returns, daily_returns)}
        }
