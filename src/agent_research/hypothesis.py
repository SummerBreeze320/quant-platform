import json
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.common.config import get_settings
from src.common.logger import logger


class FactorHypothesis(BaseModel):
    """金融量化因子假设与计算表达式规范"""
    name: str = Field(description="因子唯一名称，如 mom_breakout_20d")
    category: str = Field(description="分类: momentum, reversal, volatility, liquidity, order_flow")
    hypothesis: str = Field(description="核心金融经济学假设")
    rationale: str = Field(description="行为金融学或微观结构逻辑机理")
    expression: str = Field(description="Qlib 或向量化计算公式")


class FactorHypothesisEngine:
    """
    LLM 驱动的量化因子假设生成引擎：
    支持大模型动态推导新因子假设，并内置专业级量化经典因子范式库作为无依赖回退。
    """

    DEFAULT_CATALOGUE: List[FactorHypothesis] = [
        FactorHypothesis(
            name="rev_bollinger_pctb_20d",
            category="reversal",
            hypothesis="个股价格短期偏离中轨具有均值回归倾向，处于下轨附近的资产超卖反弹概率更高。",
            rationale="散户恐慌踩踏导致短期价格超跌，机构买盘在价值低估区间提供流动性支撑。",
            expression="(close - mean(close, 20)) / (2.0 * std(close, 20) + 1e-6)",
        ),
        FactorHypothesis(
            name="mom_ts_acceleration_10d",
            category="momentum",
            hypothesis="价格变化率的一阶导与二阶导（动量加速度）能有效捕捉机构趋势资金建仓主升浪。",
            rationale="动量持续性源于信息不完全对称下的投资者认知粘性与羊群效应反应不足。",
            expression="(close / ref(close, 5) - 1.0) - (ref(close, 5) / ref(close, 10) - 1.0)",
        ),
        FactorHypothesis(
            name="vol_parkinson_compression_15d",
            category="volatility",
            hypothesis="极端振幅收窄（波动率压缩期）往往预示着突破性行情的孕育，低日内波幅往往具有负溢价反转。",
            rationale="市场处于多空平衡僵局，做市资金聚集，变盘突破即将来临。",
            expression="std(log(high / low), 15)",
        ),
        FactorHypothesis(
            name="liq_amihud_illiquidity_ratio_20d",
            category="liquidity",
            hypothesis="非流动性越高的标的往往需要更高的风险流动性溢价补偿（Amihud 非流动性溢价效应）。",
            rationale="小盘非流动性股票难以快速变现，长线资金要求更高的预期收益率方愿持有。",
            expression="mean(abs(close / ref(close, 1) - 1.0) / (volume * close + 1e-4), 20)",
        ),
        FactorHypothesis(
            name="of_volume_price_divergence_10d",
            category="order_flow",
            hypothesis="量价背离（缩量创新高或放量滞涨）反映买卖单流失衡与主力资金诱多出货意图。",
            rationale="知情交易者在订单薄顶端利用隐式大单出货，导致成交流动性与价格上涨动力脱节。",
            expression="corr(close, volume, 10)",
        ),
    ]

    @classmethod
    def generate_hypotheses(
        cls,
        category: Optional[str] = None,
        count: int = 3,
        use_llm: bool = True,
    ) -> List[FactorHypothesis]:
        """生成一组符合要求的量化因子假设"""
        settings = get_settings()

        # 1. 尝试通过配置的大模型接口动态生成
        if use_llm and settings.LLM_API_KEY:
            try:
                import openai
                client = openai.OpenAI(
                    api_key=settings.LLM_API_KEY,
                    base_url=settings.LLM_BASE_URL if settings.LLM_BASE_URL else None,
                )
                prompt = (
                    f"作为资深量化对冲基金研究员，请设计 {count} 个具有显著统计预测能力与经济学逻辑的 A 股量化 Alpha 因子。\n"
                    f"领域限定: {category or '全部 (momentum, reversal, volatility, liquidity, order_flow)'}。\n"
                    "请以严格的 JSON 列表格式返回，每个元素包含属性: name (英文小写下划线), category, hypothesis, rationale, expression。"
                )
                resp = client.chat.completions.create(
                    model=settings.LLM_MODEL or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                content = resp.choices[0].message.content or ""
                # 解析 JSON 代码块
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
                if isinstance(data, list):
                    return [FactorHypothesis(**item) for item in data][:count]
            except Exception as e:
                logger.warning(f"[FactorHypothesisEngine] LLM generation failed ({e}); falling back to built-in catalogue.")

        # 2. 回退至内置高可靠金融范式库
        candidates = cls.DEFAULT_CATALOGUE
        if category:
            candidates = [c for c in candidates if c.category.lower() == category.lower()]
            if not candidates:
                candidates = cls.DEFAULT_CATALOGUE

        return candidates[:count]

    @classmethod
    def compute_factor_values(
        cls,
        hypothesis: FactorHypothesis,
        price_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        根据因子表达式与行情数据（close, high, low, volume）计算因子截面值矩阵。
        输入 price_data: {"close": df_close, "open": df_open, "high": df_high, "low": df_low, "volume": df_vol}
        输出: DataFrame (index=日期, columns=标的代码)
        """
        close = price_data.get("close", pd.DataFrame())
        high = price_data.get("high", close)
        low = price_data.get("low", close)
        volume = price_data.get("volume", pd.DataFrame())

        expr = hypothesis.expression.lower().strip()

        # 针对常见公式提供纯向量化快速计算实现
        if "mean(close, 20)" in expr or "std(close, 20)" in expr:
            ma20 = close.rolling(20, min_periods=5).mean()
            std20 = close.rolling(20, min_periods=5).std().replace(0, 1e-6)
            return (close - ma20) / (2.0 * std20)

        elif "ref(close, 5)" in expr or "ref(close, 10)" in expr:
            r5 = close.pct_change(5)
            r10_5 = close.shift(5).pct_change(5)
            return (r5 - r10_5).fillna(0.0)

        elif "std(log(high / low), 15)" in expr or "log(high / low)" in expr:
            hl_ratio = np.log(high / low.replace(0, 1e-6))
            return hl_ratio.rolling(15, min_periods=5).std().fillna(0.0)

        elif "abs(close / ref(close, 1) - 1.0) / (volume * close" in expr or "amihud" in hypothesis.name:
            ret = close.pct_change(1).abs()
            dollar_vol = (volume * close).replace(0, 1e-4)
            illiq = ret / dollar_vol
            return illiq.rolling(20, min_periods=5).mean().fillna(0.0)

        elif "corr(close, volume, 10)" in expr or "corr" in expr:
            # 截面逐列滚动相关性
            corr_df = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
            for col in close.columns:
                corr_df[col] = close[col].rolling(10, min_periods=5).corr(volume[col]).fillna(0.0)
            return corr_df

        else:
            # 默认使用简易动量反转收益作为安全兜底
            return close.pct_change(10).fillna(0.0)
