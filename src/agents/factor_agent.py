"""FactorAgent — 因子空间搜索

在因子空间中搜索有效因子，输出IC筛选后的因子集合。

包装现有模块:
- src.research.factor_library (QLIB_EXPRESSION_FACTORS, ALTERNATIVE_FACTORS, FACTOR_CATEGORIES)
- src.research.factor_ic (FactorICAnalyzer)
- src.rd_agent.factor_runner (FactorRunner, RD-Agent自动挖掘)
"""
import logging
from typing import Optional
from .base import BaseAgent
from .contracts import FactorSearchRequest, FactorSet

logger = logging.getLogger(__name__)


class FactorAgent(BaseAgent):
    """因子空间搜索Agent

    搜索空间:
    - price_volume: K线比率、开盘参考
    - momentum: rocp/ma/ema
    - volatility: std/vol_std
    - volume: vwap/vol_ratio
    - technical: rsi/bollinger
    - fundamental: pe/pb/roe
    - alternative: news_sentiment/announcement/consensus_rating/macro

    输入: FactorSearchRequest
    输出: FactorSet (IC筛选后的因子+权重)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="FactorAgent", config=config)

    def run(self, input_data: FactorSearchRequest) -> FactorSet:
        """执行因子搜索"""
        from src.research.factor_library import (
            FACTOR_CATEGORIES,
            QLIB_EXPRESSION_FACTORS,
            ALTERNATIVE_FACTORS,
        )
        from src.core import list_instruments, ensure_qlib

        ensure_qlib()
        categories = input_data.categories
        all_factors = []
        category_coverage = {}

        for cat in categories:
            factors = FACTOR_CATEGORIES.get(cat, [])
            all_factors.extend(factors)
            category_coverage[cat] = len(factors)

        if len(all_factors) > input_data.max_factors:
            all_factors = all_factors[: input_data.max_factors]
            logger.info(f"Truncated to {input_data.max_factors} factors")

        ic_scores = {}
        factor_weights = {}

        try:
            from src.research.factor_ic import FactorICAnalyzer

            codes = list_instruments(as_list=True)
            analyzer = FactorICAnalyzer()
            ic_df = analyzer.evaluate(
                instruments=codes,
                factor_names=[f for f in all_factors if f in QLIB_EXPRESSION_FACTORS],
                start_time=input_data.start_date,
                end_time=input_data.end_date,
            )

            if ic_df is not None and not ic_df.empty:
                for _, row in ic_df.iterrows():
                    fname = row.get("factor", "")
                    if not fname:
                        continue
                    ic = float(row.get("IC", 0) or 0)
                    icir = float(row.get("ICIR", 0) or 0)
                    if abs(ic) >= input_data.ic_threshold and abs(icir) >= input_data.icir_threshold:
                        ic_scores[fname] = round(ic, 4)
                        factor_weights[fname] = round(abs(ic) / (abs(icir) + 0.01), 4)

        except Exception as e:
            logger.warning(f"IC analysis failed, using all factors: {e}")

        if not factor_weights:
            logger.info("No factors passed IC threshold, using all with equal weights")
            for f in all_factors:
                ic_scores[f] = 0.0
                factor_weights[f] = 1.0 / max(len(all_factors), 1)

        total_w = sum(factor_weights.values())
        if total_w > 0:
            factor_weights = {k: v / total_w for k, v in factor_weights.items()}

        selected = list(factor_weights.keys())
        logger.info(f"FactorAgent selected {len(selected)} factors from {len(all_factors)}")

        return FactorSet(
            factors=selected,
            ic_scores=ic_scores,
            factor_weights=factor_weights,
            category_coverage=category_coverage,
        )
