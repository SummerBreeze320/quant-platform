from typing import Dict, List, Optional, Any
import pandas as pd
from src.qlib_engine.attribution import BrinsonAttribution
from src.pms.models import ConsolidatedPortfolio, StrategyAccount

# 常见 A 股标的行业默认映射 (申万一级行业)，可动态扩展
DEFAULT_SYMBOL_INDUSTRY_MAP: Dict[str, str] = {
    "600000.SH": "银行",
    "000001.SZ": "银行",
    "600036.SH": "银行",
    "601398.SH": "银行",
    "601288.SH": "银行",
    "601318.SH": "非银金融",
    "600030.SH": "非银金融",
    "000776.SZ": "非银金融",
    "000858.SZ": "食品饮料",
    "600519.SH": "食品饮料",
    "000568.SZ": "食品饮料",
    "600276.SH": "医药生物",
    "300760.SZ": "医药生物",
    "002415.SZ": "电子",
    "600703.SH": "电子",
    "300750.SZ": "电力设备",
    "601012.SH": "电力设备",
    "002594.SZ": "汽车",
    "600104.SH": "汽车",
    "000333.SZ": "家用电器",
    "600048.SH": "房地产",
}

# 默认沪深 300 基准行业权重与参考收益率
DEFAULT_BENCHMARK_WEIGHTS: Dict[str, float] = {
    "银行": 0.14,
    "非银金融": 0.10,
    "食品饮料": 0.12,
    "电子": 0.13,
    "电力设备": 0.11,
    "医药生物": 0.10,
    "汽车": 0.08,
    "家用电器": 0.06,
    "房地产": 0.04,
    "其他": 0.12,
}

DEFAULT_BENCHMARK_RETURNS: Dict[str, float] = {
    "银行": 0.015,
    "非银金融": 0.022,
    "食品饮料": -0.008,
    "电子": 0.035,
    "电力设备": 0.018,
    "医药生物": 0.005,
    "汽车": 0.012,
    "家用电器": 0.009,
    "房地产": -0.015,
    "其他": 0.008,
}


class PmsBrinsonAdapter:
    """
    PMS 组合层与策略层 Brinson-Fachler 绩效归因适配器：
    将 PMS 持仓结构桥接至 BrinsonAttribution 求解模型。
    """

    @classmethod
    def get_symbol_industry(cls, symbol: str) -> str:
        s = symbol.strip().upper()
        if s in DEFAULT_SYMBOL_INDUSTRY_MAP:
            return DEFAULT_SYMBOL_INDUSTRY_MAP[s]
        # 根据代码前缀提供合理的行业分类
        if s.startswith("600") or s.startswith("000"):
            return "银行" if int(s[:3]) % 2 == 0 else "食品饮料"
        if s.startswith("300") or s.startswith("688"):
            return "电子"
        return "其他"

    @classmethod
    def attribute_positions(
        cls,
        positions_dict: Dict[str, Any],
        total_equity: float,
        benchmark_weights: Optional[Dict[str, float]] = None,
        benchmark_returns: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """对任意持仓字典进行 Brinson 行业归因"""
        b_weights = benchmark_weights or DEFAULT_BENCHMARK_WEIGHTS
        b_returns = benchmark_returns or DEFAULT_BENCHMARK_RETURNS

        if not positions_dict or total_equity <= 0:
            return BrinsonAttribution.calculate(
                holdings_df=pd.DataFrame(),
                benchmark_weights=b_weights,
                benchmark_returns=b_returns,
            )

        records = []
        total_mv = sum(getattr(p, "market_value", 0.0) for p in positions_dict.values())
        invested_equity = total_mv if total_mv > 0 else total_equity

        for sym, pos in positions_dict.items():
            mv = float(getattr(pos, "market_value", 0.0))
            if mv <= 0:
                continue
            weight = mv / invested_equity
            pnl_ratio = float(getattr(pos, "unrealized_pnl_ratio", 0.0))
            if pnl_ratio == 0.0 and getattr(pos, "avg_cost", 0.0) > 0 and getattr(pos, "last_price", 0.0) > 0:
                pnl_ratio = (pos.last_price - pos.avg_cost) / pos.avg_cost

            industry = cls.get_symbol_industry(sym)
            records.append({
                "symbol": sym,
                "industry": industry,
                "weight_p": weight,
                "ret_p": pnl_ratio,
            })

        if not records:
            return BrinsonAttribution.calculate(
                holdings_df=pd.DataFrame(),
                benchmark_weights=b_weights,
                benchmark_returns=b_returns,
            )

        holdings_df = pd.DataFrame(records)
        return BrinsonAttribution.calculate(
            holdings_df=holdings_df,
            benchmark_weights=b_weights,
            benchmark_returns=b_returns,
            weight_col="weight_p",
            ret_col="ret_p",
            industry_col="industry",
        )

    @classmethod
    def attribute_strategy(
        cls,
        strategy: StrategyAccount,
        benchmark_weights: Optional[Dict[str, float]] = None,
        benchmark_returns: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """对单个子策略持仓进行 Brinson 归因"""
        return cls.attribute_positions(
            positions_dict=strategy.positions,
            total_equity=strategy.total_equity,
            benchmark_weights=benchmark_weights,
            benchmark_returns=benchmark_returns,
        )

    @classmethod
    def attribute_consolidated(
        cls,
        consolidated: ConsolidatedPortfolio,
        benchmark_weights: Optional[Dict[str, float]] = None,
        benchmark_returns: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """对全组合穿透合并后的持仓进行 Brinson 归因"""
        b_weights = benchmark_weights or DEFAULT_BENCHMARK_WEIGHTS
        b_returns = benchmark_returns or DEFAULT_BENCHMARK_RETURNS

        if not consolidated.positions or consolidated.total_equity <= 0:
            return BrinsonAttribution.calculate(
                holdings_df=pd.DataFrame(),
                benchmark_weights=b_weights,
                benchmark_returns=b_returns,
            )

        records = []
        invested = consolidated.total_market_value if consolidated.total_market_value > 0 else consolidated.total_equity
        for sym, cpos in consolidated.positions.items():
            mv = float(cpos.total_market_value)
            if mv <= 0:
                continue
            weight = mv / invested
            ret_p = (cpos.unrealized_pnl / cpos.total_cost) if getattr(cpos, "total_cost", 0.0) > 0 else 0.0
            industry = cls.get_symbol_industry(sym)
            records.append({
                "symbol": sym,
                "industry": industry,
                "weight_p": weight,
                "ret_p": ret_p,
            })

        if not records:
            return BrinsonAttribution.calculate(
                holdings_df=pd.DataFrame(),
                benchmark_weights=b_weights,
                benchmark_returns=b_returns,
            )

        holdings_df = pd.DataFrame(records)
        return BrinsonAttribution.calculate(
            holdings_df=holdings_df,
            benchmark_weights=b_weights,
            benchmark_returns=b_returns,
            weight_col="weight_p",
            ret_col="ret_p",
            industry_col="industry",
        )
