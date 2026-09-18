from datetime import datetime
from typing import Dict, List, Optional
from src.common.logger import logger
from src.pms.models import (
    MasterAccount,
    StrategyAccount,
    StrategyType,
    CashTransfer,
    CapitalAllocationPlan,
    StrategyPerformance,
)
from src.pms.allocator import CapitalAllocator
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.circuit_breaker import CircuitBreakerManager

class PortfolioManager:
    """资产组合与母子账户生命周期管理器"""

    def __init__(
        self,
        master_id: str = "master_default",
        initial_reserve: float = 10_000_000.0,
        broker: Optional[PaperBroker] = None,
        allocator: Optional[CapitalAllocator] = None,
        circuit_breaker: Optional[CircuitBreakerManager] = None,
    ):
        self.master = MasterAccount(
            master_id=master_id,
            reserve_cash=initial_reserve,
            total_equity=initial_reserve,
        )
        self.broker = broker
        self.allocator = allocator or CapitalAllocator()
        self.circuit_breaker = circuit_breaker

    def deposit_to_master(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("注资金额必须大于0")
        self.master.reserve_cash += amount
        self.refresh_total_equity()
        return self.master.reserve_cash

    def register_strategy(
        self,
        strategy_id: str,
        name: str,
        strategy_type: StrategyType = StrategyType.ALPHA,
        initial_budget: float = 0.0,
    ) -> StrategyAccount:
        """注册并激活子策略账户"""
        if strategy_id in self.master.strategies:
            raise ValueError(f"策略 ID '{strategy_id}' 已存在")
        if self.broker is not None and strategy_id in self.broker.accounts:
            raise ValueError(f"交易账户 '{strategy_id}' 已存在，不能通过注册策略覆盖其资金与持仓")
        if initial_budget > self.master.reserve_cash:
            raise ValueError(
                f"预备金不足: 母账户可用 {self.master.reserve_cash:.2f}, 所需 {initial_budget:.2f}"
            )

        self.master.reserve_cash -= initial_budget
        account = StrategyAccount(
            strategy_id=strategy_id,
            name=name,
            strategy_type=strategy_type,
            allocated_budget=initial_budget,
            current_cash=initial_budget,
            total_equity=initial_budget,
            positions={},
            is_active=True,
            updated_at=datetime.now().isoformat(),
        )
        self.master.strategies[strategy_id] = account

        if self.broker is not None:
            self.broker.create_account(account_id=strategy_id, initial_cash=initial_budget)
        if self.circuit_breaker is not None:
            self.circuit_breaker.update_equity(strategy_id, initial_budget)

        self.refresh_total_equity()
        logger.info(
            f"已注册子策略 [{strategy_id}] {name}，初始预算: {initial_budget:.2f}，母账户预备金剩余: {self.master.reserve_cash:.2f}"
        )
        return account

    def transfer_cash(self, strategy_id: str, amount: float, reason: str = "") -> CashTransfer:
        """母子账户间资金划转: amount > 0 为注资, amount < 0 为抽资"""
        if strategy_id not in self.master.strategies:
            raise ValueError(f"未找到策略 ID '{strategy_id}'")

        strat = self.master.strategies[strategy_id]
        if self.broker is not None:
            b_acc = self.broker.get_account(strategy_id)
            strat.current_cash = b_acc.available_cash
            strat.total_equity = b_acc.total_equity
            strat.positions = b_acc.positions
        equity_before = strat.total_equity

        if amount > 0:
            if self.master.reserve_cash < amount:
                raise ValueError(
                    f"母账户预备金不足: 可用 {self.master.reserve_cash:.2f}, 所需 {amount:.2f}"
                )
            self.master.reserve_cash -= amount
            strat.current_cash += amount
            strat.allocated_budget += amount
            strat.total_equity += amount
            if self.broker is not None:
                b_acc = self.broker.get_account(strategy_id)
                b_acc.available_cash += amount
                b_acc.total_equity += amount
        elif amount < 0:
            withdraw = abs(amount)
            if strat.current_cash < withdraw:
                raise ValueError(
                    f"策略现金不足以抽资: 策略可用现金 {strat.current_cash:.2f}, 请求抽取 {withdraw:.2f}"
                )
            strat.current_cash -= withdraw
            strat.allocated_budget = max(0.0, strat.allocated_budget - withdraw)
            strat.total_equity = max(0.0, strat.total_equity - withdraw)
            self.master.reserve_cash += withdraw
            if self.broker is not None:
                b_acc = self.broker.get_account(strategy_id)
                b_acc.available_cash -= withdraw
                b_acc.total_equity -= withdraw

        if self.circuit_breaker is not None and amount != 0:
            self.circuit_breaker.adjust_cash_flow(strategy_id, equity_before, strat.total_equity)
        strat.updated_at = datetime.now().isoformat()
        self.refresh_total_equity()

        transfer = CashTransfer(
            strategy_id=strategy_id,
            transfer_amount=amount,
            reason=reason or ("注资" if amount > 0 else "抽资"),
        )
        return transfer

    def apply_allocation_plan(self, plan: CapitalAllocationPlan) -> List[CashTransfer]:
        """批量应用执行资金配资划转计划"""
        executed: List[CashTransfer] = []
        # 先执行抽资以释放预备金，再执行注资
        withdrawals = [t for t in plan.transfers if t.transfer_amount < 0]
        deposits = [t for t in plan.transfers if t.transfer_amount > 0]

        for t in withdrawals:
            rec = self.transfer_cash(t.strategy_id, t.transfer_amount, t.reason)
            executed.append(rec)

        for t in deposits:
            rec = self.transfer_cash(t.strategy_id, t.transfer_amount, t.reason)
            executed.append(rec)

        return executed

    def sync_from_broker(self) -> None:
        """从仿真交易柜台同步最新资金与持仓"""
        if self.broker is None:
            return

        for sid, strat in self.master.strategies.items():
            if not strat.is_active:
                continue
            b_acc = self.broker.get_account(sid)
            strat.current_cash = b_acc.available_cash
            strat.positions = b_acc.positions
            strat.total_equity = b_acc.total_equity
            strat.updated_at = datetime.now().isoformat()

        self.refresh_total_equity()

    def refresh_total_equity(self) -> float:
        """重新汇总母账户总资产"""
        sub_equity = sum(s.total_equity for s in self.master.strategies.values() if s.is_active)
        self.master.total_equity = round(self.master.reserve_cash + sub_equity, 2)
        self.master.updated_at = datetime.now().isoformat()
        return self.master.total_equity

    def get_strategy_equities(self) -> Dict[str, float]:
        return {
            sid: s.total_equity
            for sid, s in self.master.strategies.items()
            if s.is_active
        }

    def get_performance_summary(self) -> List[StrategyPerformance]:
        """计算各子策略累计收益与母账户贡献度"""
        self.refresh_total_equity()
        master_eq = self.master.total_equity
        summaries: List[StrategyPerformance] = []

        for sid, strat in self.master.strategies.items():
            budget = strat.allocated_budget if strat.allocated_budget > 0 else 1.0
            cum_ret = (strat.total_equity - budget) / budget
            contrib = (strat.total_equity / master_eq) if master_eq > 0 else 0.0
            summaries.append(
                StrategyPerformance(
                    strategy_id=sid,
                    name=strat.name,
                    allocated_budget=round(strat.allocated_budget, 2),
                    current_equity=round(strat.total_equity, 2),
                    cumulative_return=round(cum_ret, 4),
                    contribution_to_master=round(contrib, 4),
                )
            )

        return summaries
