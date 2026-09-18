from typing import Dict, List, Optional
from src.pms.models import AllocationMethod, CapitalAllocationPlan, CashTransfer

class CapitalAllocator:
    """多策略动态配资引擎"""

    def allocate_equal_weight(self, strategies: List[str]) -> Dict[str, float]:
        if not strategies:
            return {}
        weight = round(1.0 / len(strategies), 6)
        ratios = {sid: weight for sid in strategies}
        # 归一化确保合计为 1.0
        total = sum(ratios.values())
        if total > 0:
            ratios = {sid: round(w / total, 6) for sid, w in ratios.items()}
        return ratios

    def allocate_fixed_budget(
        self,
        strategies: List[str],
        base_ratios: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        if not strategies:
            return {}
        base = base_ratios or {}
        raw = {sid: max(base.get(sid, 0.0), 0.0) for sid in strategies}
        total = sum(raw.values())
        if total == 0:
            return self.allocate_equal_weight(strategies)
        return {sid: round(w / total, 6) for sid, w in raw.items()}

    def allocate_risk_parity(
        self,
        strategy_volatilities: Dict[str, float],
        min_vol: float = 0.0001
    ) -> Dict[str, float]:
        """风险平价（波动率倒数加权）: w_i = (1 / sigma_i) / sum(1 / sigma_j)"""
        if not strategy_volatilities:
            return {}
        inv_vols = {}
        for sid, vol in strategy_volatilities.items():
            safe_vol = max(vol, min_vol)
            inv_vols[sid] = 1.0 / safe_vol
        total_inv = sum(inv_vols.values())
        if total_inv <= 0:
            return self.allocate_equal_weight(list(strategy_volatilities.keys()))
        return {sid: round(val / total_inv, 6) for sid, val in inv_vols.items()}

    def allocate_sharpe_weighted(
        self,
        strategy_sharpes: Dict[str, float],
        min_floor: float = 0.01
    ) -> Dict[str, float]:
        """夏普比率动态动量加权: w_i = max(Sharpe_i, min_floor) / sum(...)"""
        if not strategy_sharpes:
            return {}
        adjusted = {sid: max(sharpe, min_floor) for sid, sharpe in strategy_sharpes.items()}
        total = sum(adjusted.values())
        if total <= 0:
            return self.allocate_equal_weight(list(strategy_sharpes.keys()))
        return {sid: round(val / total, 6) for sid, val in adjusted.items()}

    def generate_plan(
        self,
        master_equity: float,
        current_equities: Dict[str, float],
        method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
        investable_ratio: float = 0.95,
        base_ratios: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        sharpes: Optional[Dict[str, float]] = None,
    ) -> CapitalAllocationPlan:
        """根据母账户总权益与策略指标，计算目标配置与划转差额"""
        strategies = list(current_equities.keys())
        if not strategies:
            return CapitalAllocationPlan(method=method, target_ratios={})

        if method == AllocationMethod.EQUAL_WEIGHT:
            target_ratios = self.allocate_equal_weight(strategies)
        elif method == AllocationMethod.FIXED_BUDGET:
            target_ratios = self.allocate_fixed_budget(strategies, base_ratios)
        elif method == AllocationMethod.RISK_PARITY:
            target_ratios = self.allocate_risk_parity(volatilities or {})
        elif method == AllocationMethod.SHARPE_WEIGHTED:
            target_ratios = self.allocate_sharpe_weighted(sharpes or {})
        else:
            target_ratios = self.allocate_equal_weight(strategies)

        investable_fund = master_equity * investable_ratio
        transfers: List[CashTransfer] = []

        for sid in strategies:
            ratio = target_ratios.get(sid, 0.0)
            target_cap = investable_fund * ratio
            current_cap = current_equities.get(sid, 0.0)
            delta = round(target_cap - current_cap, 2)
            transfers.append(
                CashTransfer(
                    strategy_id=sid,
                    transfer_amount=delta,
                    reason=f"{method.value} 调资: 目标={target_cap:.2f}, 当前={current_cap:.2f}, 差额={delta:.2f}"
                )
            )

        return CapitalAllocationPlan(
            method=method,
            target_ratios=target_ratios,
            transfers=transfers
        )
