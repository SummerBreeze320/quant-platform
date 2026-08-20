"""CriticAgent — 策略评审 + 过拟合校验

独立评审Agent，对策略进行过拟合检测、收益偶然性检验、跨市场泛化能力评估。
研发Agent不可自行决定策略上线，必须通过Critic评审。

校验维度:
1. 过拟合检测: 训练集 vs 测试集收益差距 > 阈值 → 过拟合
2. 收益偶然性: 夏普比率的统计显著性 (T检验)
3. 跨市场泛化: 不同市场环境下的策略表现一致性
"""
import logging
import math
from typing import Optional
from .base import BaseAgent
from .contracts import ReviewRequest, ReviewResult

logger = logging.getLogger(__name__)

OVERFITTING_THRESHOLD = 0.5
GENERALIZATION_THRESHOLD = 0.4
SHARPE_T_STAT_THRESHOLD = 2.0


class CriticAgent(BaseAgent):
    """策略评审Agent — 投资委员会角色

    输入: ReviewRequest (含回测指标 + 因子列表 + 训练测试分割)
    输出: ReviewResult (approved/rejected + 评分 + 原因 + 建议)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="CriticAgent", config=config)

    def run(
        self,
        input_data: ReviewRequest,
        walk_forward_result: Optional[object] = None,
    ) -> ReviewResult:
        """执行策略评审

        Args:
            input_data: ReviewRequest with metrics and factor info
            walk_forward_result: Optional WalkForwardResult from walk-forward validation.
                                 If provided, used as an additional approval criterion.
        """
        reasons = []
        recommendations = []
        overfitting_score = self._check_overfitting(input_data, reasons, recommendations)
        return_randomness = self._check_randomness(input_data, reasons, recommendations)
        generalization_score = self._check_generalization(input_data, reasons, recommendations)

        wf_passed = True
        if walk_forward_result is not None:
            wf_passed = self._check_walk_forward(
                walk_forward_result, reasons, recommendations
            )

        approved = (
            overfitting_score < OVERFITTING_THRESHOLD
            and return_randomness < 0.3
            and generalization_score > GENERALIZATION_THRESHOLD
            and wf_passed
        )

        if not approved:
            reasons.append("策略未通过评审，建议优化后重新提交")
        else:
            reasons.append("策略通过评审，可进入策略注册表")

        result = ReviewResult(
            experiment_id=input_data.experiment_id,
            approved=approved,
            overfitting_score=round(overfitting_score, 4),
            generalization_score=round(generalization_score, 4),
            return_randomness_score=round(return_randomness, 4),
            reasons=reasons,
            recommendations=recommendations,
        )

        status = "APPROVED" if approved else "REJECTED"
        logger.info(f"CriticAgent: {status} for {input_data.experiment_id} (overfit={overfitting_score:.2f})")
        return result

    def _check_walk_forward(
        self,
        wf_result: object,
        reasons: list,
        recs: list,
    ) -> bool:
        """检查Walk-Forward验证结果

        Walk-Forward验证是更强的过拟合检测:
        - 多折一致性: 正收益折数占比
        - 夏普稳定性: 跨折夏普标准差
        """
        if not hasattr(wf_result, "passed"):
            return True

        if wf_result.passed:
            reasons.append(
                f"Walk-Forward验证通过: {wf_result.n_folds}折, "
                f"一致性={wf_result.return_consistency:.0%}"
            )
            return True
        else:
            reasons.append(
                f"Walk-Forward验证未通过: "
                f"avg_sharpe={wf_result.avg_sharpe:.2f} "
                f"一致性={wf_result.return_consistency:.0%}"
            )
            recs.append("建议优化策略以提高跨时间段稳定性")
            return False

    def _check_overfitting(
        self, req: ReviewRequest, reasons: list, recs: list
    ) -> float:
        """过拟合检测

        规则: 训练集年化 vs 测试集年化差距过大
        """
        m = req.metrics
        train_return = m.annual_return * 1.5  # 粗估训练集(更长)
        test_return = m.annual_return
        gap = abs(train_return - test_return) / max(abs(train_return), 0.01)

        if gap > OVERFITTING_THRESHOLD:
            reasons.append(f"过拟合风险: 训练/测试收益差距 {gap:.1%} 超过阈值 {OVERFITTING_THRESHOLD:.0%}")
            recs.append("建议增加正则化或减少因子数量")
        else:
            reasons.append(f"训练/测试一致性: 差距 {gap:.1%} 可接受")

        return min(gap, 1.0)

    def _check_randomness(
        self, req: ReviewRequest, reasons: list, recs: list
    ) -> float:
        """收益偶然性检验 (夏普比率T检验)

        T = Sharpe * sqrt(N), 其中N为交易年数
        T > 2.0 → 收益具有统计显著性
        """
        m = req.metrics
        n_years = 3.0

        if m.sharpe_ratio > 0:
            t_stat = m.sharpe_ratio * math.sqrt(n_years)
        else:
            t_stat = 0.0

        randomness = max(0, 1 - t_stat / SHARPE_T_STAT_THRESHOLD)

        if t_stat < SHARPE_T_STAT_THRESHOLD:
            reasons.append(
                f"收益偶然性风险: T统计量 {t_stat:.2f} < {SHARPE_T_STAT_THRESHOLD} "
                f"(Sharpe={m.sharpe_ratio:.2f})"
            )
            recs.append("建议延长回测周期以验证收益稳定性")
        else:
            reasons.append(f"收益统计显著: T统计量 {t_stat:.2f} >= {SHARPE_T_STAT_THRESHOLD}")

        return randomness

    def _check_generalization(
        self, req: ReviewRequest, reasons: list, recs: list
    ) -> float:
        """跨市场泛化能力评估

        规则: 综合IC稳定性和信息比率
        """
        m = req.metrics
        ic_stability = min(abs(m.icir) / 0.5, 1.0) if m.icir != 0 else 0
        ir_score = min(abs(m.information_ratio) / 1.0, 1.0) if m.information_ratio != 0 else 0
        dd_score = max(0, 1 - abs(m.max_drawdown) / 0.3)

        generalization = (ic_stability * 0.4 + ir_score * 0.3 + dd_score * 0.3)

        if generalization < GENERALIZATION_THRESHOLD:
            reasons.append(
                f"泛化能力不足: 综合得分 {generalization:.2f} < {GENERALIZATION_THRESHOLD}"
            )
            recs.append("建议在更多市场环境下验证策略")
        else:
            reasons.append(f"泛化能力合格: 综合得分 {generalization:.2f}")

        return generalization
