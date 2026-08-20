"""RiskAgent — 市场状态识别 + 动态仓位 + 恐慌指标

识别市场环境，输出动态仓位比例，作为交易前置风控。

包装现有模块:
- src.agents.market_state (MarketStateRecognizer)
- src.execution.unified_risk (UnifiedRiskController)

P5增强:
- 支持直接输入模式 (波动率/趋势/恐慌/成交量直接传入)
- 新增恐慌指标 (VIX类, 基于波动率+跌幅+成交量异动合成)
- 仓位决策依据输出 (为什么是100%/50%/20%)
- 多因子加权风险评分

仓位枚举:
 Bull → 100%
 Sideways → 50%
 Bear/HighVol → 20%
"""
import logging
from typing import Optional
from .base import BaseAgent
from .contracts import RiskAssessmentRequest, RiskAssessment, MarketState

logger = logging.getLogger(__name__)

POSITION_RATIO_MAP = {
    MarketState.BULL: 1.0,
    MarketState.SIDEWAYS: 0.5,
    MarketState.LOW_VOL: 0.7,
    MarketState.BEAR: 0.2,
    MarketState.HIGH_VOL: 0.2,
}


class RiskAgent(BaseAgent):
    """风险Agent — 市场状态识别 + 动态仓位输出 + 恐慌指标

    输入: RiskAssessmentRequest (支持自动模式和直接输入模式)
    输出: RiskAssessment (市场状态 + 仓位比例 + 风险分数 + 恐慌指标 + 告警 + 决策依据)

    两种模式:
    1. 自动模式: 只传 benchmark + as_of_date, 内部调用 MarketStateRecognizer
    2. 直接模式: 直接传入 volatility/trend_signal/panic_index/volume_ratio
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="RiskAgent", config=config)
        self._recognizer = None

    def initialize(self):
        """初始化市场状态识别器"""
        try:
            from .market_state import MarketStateRecognizer
            self._recognizer = MarketStateRecognizer()
            self._logger.info("MarketStateRecognizer initialized")
        except Exception as e:
            self._logger.warning(f"MarketStateRecognizer init failed: {e}")

    def run(self, input_data: RiskAssessmentRequest) -> RiskAssessment:
        """执行风险评估

        自动判断模式:
        - 如果 request 中有 volatility/trend_signal 等直接输入 → 直接模式
        - 否则 → 自动模式 (调用 MarketStateRecognizer)
        """
        if input_data.volatility is not None or input_data.panic_index is not None:
            return self._run_direct(input_data)

        if self._recognizer is None:
            self.initialize()

        if self._recognizer is None:
            return RiskAssessment(
                market_state=MarketState.SIDEWAYS,
                position_ratio=0.5,
                risk_score=0.5,
                warnings=["Market recognizer unavailable, using default"],
                position_reason="识别器不可用, 默认震荡市50%仓位",
            )

        signals = self._recognizer.get_signals(as_of_date=input_data.as_of_date)
        market_state = MarketState(signals.get("state", "sideways"))

        position_ratio = POSITION_RATIO_MAP.get(market_state, 0.5)
        panic_index = self._calc_panic_index(signals)
        risk_score = self._calc_risk_score(signals, market_state, panic_index)
        warnings = self._generate_warnings(signals, market_state, input_data, panic_index)
        reason = self._build_position_reason(market_state, signals, panic_index)

        assessment = RiskAssessment(
            market_state=market_state,
            position_ratio=position_ratio,
            risk_score=risk_score,
            volatility_20d=signals.get("volatility", 0),
            trend_signal=signals.get("trend", 0),
            volume_ratio=signals.get("volume_ratio", 1),
            panic_index=panic_index,
            warnings=warnings,
            position_reason=reason,
        )

        logger.info(
            f"RiskAgent: state={market_state.value} position={position_ratio:.0%} "
            f"risk_score={risk_score:.2f} panic={panic_index:.2f}"
        )
        return assessment

    def _run_direct(self, req: RiskAssessmentRequest) -> RiskAssessment:
        """直接输入模式: 使用请求中传入的指标直接评估"""
        vol = req.volatility or 0.2
        trend = req.trend_signal or 0.0
        vol_ratio = req.volume_ratio or 1.0
        panic = req.panic_index if req.panic_index is not None else self._calc_panic_index({
            "volatility": vol, "trend": trend, "volume_ratio": vol_ratio,
        })

        market_state = self._classify_market(vol, trend, vol_ratio, panic)
        position_ratio = POSITION_RATIO_MAP.get(market_state, 0.5)
        risk_score = self._calc_risk_score(
            {"volatility": vol, "trend": trend, "volume_ratio": vol_ratio},
            market_state, panic,
        )

        signals = {"volatility": vol, "trend": trend, "volume_ratio": vol_ratio}
        warnings = self._generate_warnings(signals, market_state, req, panic)
        reason = self._build_position_reason(market_state, signals, panic)

        logger.info(
            f"RiskAgent(direct): state={market_state.value} position={position_ratio:.0%} "
            f"panic={panic:.2f}"
        )
        return RiskAssessment(
            market_state=market_state,
            position_ratio=position_ratio,
            risk_score=risk_score,
            volatility_20d=vol,
            trend_signal=trend,
            volume_ratio=vol_ratio,
            panic_index=panic,
            warnings=warnings,
            position_reason=reason,
        )

    def _classify_market(self, vol: float, trend: float, vol_ratio: float, panic: float) -> MarketState:
        """根据指标直接分类市场状态"""
        if panic > 0.6 or vol > 0.35:
            if trend < -0.05:
                return MarketState.BEAR
            return MarketState.HIGH_VOL
        if trend > 0.02 and vol < 0.20:
            return MarketState.BULL
        if vol < 0.12:
            return MarketState.LOW_VOL
        return MarketState.SIDEWAYS

    def _calc_panic_index(self, signals: dict) -> float:
        """计算恐慌指标 [0, 1]

        合成公式: 波动率分量 + 跌幅分量 + 成交量异动分量
        类似 VIX 的概念, 但使用 A 股可得数据合成
        """
        vol = signals.get("volatility", 0.2)
        trend = signals.get("trend", 0.0)
        vol_ratio = signals.get("volume_ratio", 1.0)

        vol_component = min(max(vol - 0.15, 0) / 0.25, 0.4)
        drop_component = min(max(-trend, 0) * 3, 0.3)
        volume_component = min(max(vol_ratio - 1.5, 0) * 0.3, 0.3)

        return min(vol_component + drop_component + volume_component, 1.0)

    def _calc_risk_score(self, signals: dict, state: MarketState, panic: float = 0.0) -> float:
        """计算综合风险分数 [0, 1], 越高越危险

        P5增强: 纳入恐慌指标, 多因子加权
        """
        vol = signals.get("volatility", 0.2)
        trend = abs(signals.get("trend", 0))
        vol_ratio = signals.get("volume_ratio", 1.0)

        score = 0.0
        score += min(vol * 2, 0.35)
        score += min(trend, 0.20)
        score += min(max(vol_ratio - 1, 0) * 0.3, 0.20)
        score += min(panic * 0.25, 0.25)

        if state in (MarketState.BEAR, MarketState.HIGH_VOL):
            score += 0.15

        return min(score, 1.0)

    def _generate_warnings(
        self, signals: dict, state: MarketState, request: RiskAssessmentRequest, panic: float = 0.0,
    ) -> list:
        """生成风控告警"""
        warnings = []
        vol = signals.get("volatility", 0)
        trend = signals.get("trend", 0)

        if vol > 0.35:
            warnings.append(f"高波动警告: 20日年化波动率 {vol:.1%}")
        if trend < -0.05:
            warnings.append(f"下行趋势警告: MA20低于MA60 {abs(trend):.1%}")
        if panic > 0.5:
            warnings.append(f"恐慌指标偏高: {panic:.2f}, 市场情绪不稳定")
        if state == MarketState.BEAR:
            warnings.append("熊市环境: 建议仓位降至20%")
        if state == MarketState.HIGH_VOL:
            warnings.append("高波动环境: 建议降低仓位并增加对冲")

        if request.portfolio_config:
            max_pos = request.portfolio_config.max_position
            if max_pos > 0.15:
                warnings.append(f"单只持仓 {max_pos:.0%} 超过15%上限建议")

        return warnings

    def _build_position_reason(self, state: MarketState, signals: dict, panic: float) -> str:
        """构建仓位决策依据"""
        vol = signals.get("volatility", 0.2)
        trend = signals.get("trend", 0.0)
        vol_ratio = signals.get("volume_ratio", 1.0)
        ratio = POSITION_RATIO_MAP.get(state, 0.5)

        reasons = []
        if state == MarketState.BULL:
            reasons.append(f"牛市环境(趋势{trend:+.2%}, 波动{vol:.1%})")
        elif state == MarketState.BEAR:
            reasons.append(f"熊市环境(趋势{trend:+.2%}, 恐慌{panic:.2f})")
        elif state == MarketState.SIDEWAYS:
            reasons.append(f"震荡市(波动{vol:.1%}, 量比{vol_ratio:.1f})")
        elif state == MarketState.HIGH_VOL:
            reasons.append(f"高波动(波动{vol:.1%}, 恐慌{panic:.2f})")
        elif state == MarketState.LOW_VOL:
            reasons.append(f"低波动(波动{vol:.1%})")

        return f"{' → '.join(reasons)} → 仓位{ratio:.0%}"

    def check_position_limit(
        self, target_weights: dict, position_ratio: float
    ) -> dict:
        """根据仓位比例裁剪目标持仓权重"""
        adjusted = {
            code: w * position_ratio for code, w in target_weights.items()
        }
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total * position_ratio for k, v in adjusted.items()}

        logger.info(
            f"Position adjusted by {position_ratio:.0%}: "
            f"{len(adjusted)} holdings, total={sum(adjusted.values()):.2%}"
        )
        return adjusted

    def live_risk_check(
        self,
        target_weights: dict,
        as_of_date: Optional[str] = None,
        direct_inputs: Optional[dict] = None,
    ) -> dict:
        """P5: 实盘风控前置检查 — TradingEngine调用

        Args:
            target_weights: 目标持仓权重 {code: weight}
            as_of_date: 检查日期
            direct_inputs: 直接输入模式 {volatility, trend_signal, panic_index, volume_ratio}

        Returns:
            {approved, position_ratio, market_state, risk_score, panic_index,
             warnings, adjusted_weights, position_reason}
        """
        from src.agents.contracts import RiskAssessmentRequest

        req_kwargs = {"benchmark": "SH000300", "as_of_date": as_of_date}
        if direct_inputs:
            req_kwargs.update({
                "volatility": direct_inputs.get("volatility"),
                "trend_signal": direct_inputs.get("trend_signal"),
                "panic_index": direct_inputs.get("panic_index"),
                "volume_ratio": direct_inputs.get("volume_ratio"),
            })

        assessment = self.execute(RiskAssessmentRequest(**req_kwargs))

        position_ratio = assessment.position_ratio
        adjusted_weights = self.check_position_limit(target_weights, position_ratio)
        approved = assessment.risk_score < 0.7 and position_ratio > 0.1

        return {
            "approved": approved,
            "position_ratio": position_ratio,
            "market_state": assessment.market_state.value,
            "risk_score": assessment.risk_score,
            "panic_index": assessment.panic_index,
            "volatility_20d": assessment.volatility_20d,
            "warnings": assessment.warnings,
            "adjusted_weights": adjusted_weights,
            "position_reason": assessment.position_reason,
        }
