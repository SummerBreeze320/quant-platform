from typing import Dict, List, Optional
from src.pms.models import (
    MasterAccount,
    ConsolidatedPosition,
    ConsolidatedPortfolio,
)
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import CircuitBreakerLevel

class PortfolioAggregator:
    """跨策略全局穿透持仓聚合器与母账户宏观风控红线检测"""

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreakerManager] = None,
        max_stock_concentration: float = 0.15,
        max_leverage_ratio: float = 1.0,
    ):
        self.circuit_breaker = circuit_breaker
        self.max_stock_concentration = max_stock_concentration
        self.max_leverage_ratio = max_leverage_ratio

    def aggregate(self, master: MasterAccount) -> ConsolidatedPortfolio:
        """穿透合并所有子策略持仓并执行宏观合规红线与熔断检查"""
        consolidated_pos: Dict[str, ConsolidatedPosition] = {}

        total_sub_cash = 0.0
        for sid, strat in master.strategies.items():
            if not strat.is_active:
                continue
            total_sub_cash += strat.current_cash
            for sym, pos in strat.positions.items():
                if sym not in consolidated_pos:
                    consolidated_pos[sym] = ConsolidatedPosition(
                        symbol=sym,
                        total_volume=0,
                        total_market_value=0.0,
                        weight_in_master=0.0,
                        contributing_strategies=[],
                    )
                cpos = consolidated_pos[sym]
                cpos.total_volume += pos.total_volume
                cpos.total_market_value += pos.market_value
                cpos.total_cost += round(getattr(pos, "avg_cost", 0.0) * pos.total_volume, 2)
                cpos.unrealized_pnl += round(getattr(pos, "unrealized_pnl", 0.0), 2)
                if sid not in cpos.contributing_strategies:
                    cpos.contributing_strategies.append(sid)

        total_cash = round(master.reserve_cash + total_sub_cash, 2)
        total_mv = round(sum(p.total_market_value for p in consolidated_pos.values()), 2)
        total_equity = round(total_cash + total_mv, 2)

        # 计算各标的在母账户维度的权重
        for cpos in consolidated_pos.values():
            if total_equity > 0:
                cpos.weight_in_master = round(cpos.total_market_value / total_equity, 4)
            else:
                cpos.weight_in_master = 0.0

        # 杠杆率 = 持仓总市值 / 总资产
        leverage_ratio = round(total_mv / total_equity, 4) if total_equity > 0 else 0.0

        # 计算集中度指标
        weights = sorted([p.weight_in_master for p in consolidated_pos.values()], reverse=True)
        max_conc = weights[0] if weights else 0.0
        top5_conc = round(sum(weights[:5]), 4) if weights else 0.0

        # 宏观风控红线扫描
        macro_alerts: List[str] = []
        for sym, cpos in consolidated_pos.items():
            if cpos.weight_in_master > self.max_stock_concentration:
                macro_alerts.append(
                    f"CRITICAL: 标的 {sym} 穿透持仓占比 {cpos.weight_in_master:.2%} 超过宏观红线 {self.max_stock_concentration:.2%} (持股量: {cpos.total_volume}, 涉及策略: {cpos.contributing_strategies})"
                )

        if leverage_ratio > self.max_leverage_ratio:
            macro_alerts.append(
                f"CRITICAL: 母账户总杠杆率 {leverage_ratio:.2f} 突破风控红线 {self.max_leverage_ratio:.2f}"
            )

        # 母级日内回撤全局熔断联动
        cb_level = 0
        if self.circuit_breaker is not None:
            cb_state = self.circuit_breaker.update_equity(master.master_id, total_equity)
            cb_level = cb_state.level.value
            if cb_state.level in [CircuitBreakerLevel.ORANGE_RESTRICT_BUY, CircuitBreakerLevel.RED_HALT]:
                macro_alerts.append(
                    f"HALT: 母账户触发日内回撤全局熔断 [{cb_state.level.name}], 回撤: {cb_state.max_drawdown:.2%}"
                )

        return ConsolidatedPortfolio(
            master_id=master.master_id,
            total_equity=total_equity,
            total_cash=total_cash,
            total_market_value=total_mv,
            leverage_ratio=leverage_ratio,
            positions=consolidated_pos,
            max_stock_concentration=max_conc,
            top5_concentration=top5_conc,
            circuit_breaker_level=cb_level,
            macro_alerts=macro_alerts,
        )
