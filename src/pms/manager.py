from datetime import datetime
from typing import Dict, List, Optional, Any
from src.common.logger import logger
from src.pms.models import (
    MasterAccount,
    StrategyAccount,
    StrategyType,
    CashTransfer,
    CapitalAllocationPlan,
    StrategyPerformance,
    DailyNavRecord,
)
from src.pms.allocator import CapitalAllocator
from src.pms.risk_analytics import RiskAnalyticsEngine, RiskMetricsSummary
from src.pms.attribution_adapter import PmsBrinsonAdapter
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
        storage: Optional[Any] = None,
    ):
        self.master = MasterAccount(
            master_id=master_id,
            reserve_cash=initial_reserve,
            total_equity=initial_reserve,
        )
        self.broker = broker
        self.allocator = allocator or CapitalAllocator()
        self.circuit_breaker = circuit_breaker
        self.storage = storage
        self._nav_history: Dict[str, List[DailyNavRecord]] = {}


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
        if self.storage is not None:
            self.storage.save_transfer(transfer, master_id=self.master.master_id)
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

    def record_daily_nav(
        self,
        account_id: str,
        date: str,
        equity: Optional[float] = None,
        benchmark_return: float = 0.0,
    ) -> DailyNavRecord:
        """记录指定账户（策略或 master）的当日单位净值快照"""
        if account_id == "master":
            curr_eq = equity if equity is not None else self.refresh_total_equity()
        elif account_id in self.master.strategies:
            self.sync_from_broker()
            curr_eq = equity if equity is not None else self.master.strategies[account_id].total_equity
        else:
            if self.broker and account_id in self.broker.accounts:
                curr_eq = equity if equity is not None else self.broker.accounts[account_id].total_equity
            else:
                curr_eq = equity if equity is not None else 0.0

        if account_id not in self._nav_history:
            self._nav_history[account_id] = []

        history = self._nav_history[account_id]
        if not history:
            nav = 1.0
            daily_ret = 0.0
        else:
            prev = history[-1]
            daily_ret = round((curr_eq - prev.equity) / prev.equity, 6) if prev.equity > 0 else 0.0
            nav = round(prev.nav * (1.0 + daily_ret), 4)

        rec = DailyNavRecord(
            date=date,
            account_id=account_id,
            equity=round(curr_eq, 2),
            nav=nav,
            daily_return=round(daily_ret, 6),
            benchmark_return=round(benchmark_return, 6),
        )
        history.append(rec)
        return rec

    def record_all_daily_nav(self, date: str, benchmark_return: float = 0.0) -> Dict[str, DailyNavRecord]:
        """批量留存母账户及所有激活子策略的当日净值快照"""
        self.sync_from_broker()
        results: Dict[str, DailyNavRecord] = {}
        # 1. 记录 master
        results["master"] = self.record_daily_nav("master", date=date, benchmark_return=benchmark_return)
        # 2. 记录各策略
        for sid, strat in self.master.strategies.items():
            if strat.is_active:
                results[sid] = self.record_daily_nav(sid, date=date, benchmark_return=benchmark_return)
        return results

    def get_nav_history(self, account_id: str) -> List[DailyNavRecord]:
        """获取指定账户历史净值序列"""
        return self._nav_history.get(account_id, [])

    def get_risk_analytics(
        self,
        account_id: str,
        risk_free_rate: float = 0.02,
    ) -> RiskMetricsSummary:
        """获取指定策略或母账户的多维量化风险指标 (Sharpe, Sortino, MaxDD, Calmar, Alpha, Beta)"""
        history = self.get_nav_history(account_id)
        if len(history) >= 2:
            rets = [r.daily_return for r in history[1:]]
            b_rets = [r.benchmark_return for r in history[1:]]
            return RiskAnalyticsEngine.calculate_metrics(
                returns=rets,
                benchmark_returns=b_rets if any(b != 0 for b in b_rets) else None,
                risk_free_rate=risk_free_rate,
            )

        # 若历史净值点不足 2 个，利用当前累计收益与基础估值合成单点概览
        self.sync_from_broker()
        if account_id == "master":
            eq = self.master.total_equity
            budget = self.master.reserve_cash + sum(s.allocated_budget for s in self.master.strategies.values())
        elif account_id in self.master.strategies:
            strat = self.master.strategies[account_id]
            eq = strat.total_equity
            budget = strat.allocated_budget if strat.allocated_budget > 0 else eq
        else:
            eq = 0.0
            budget = 1.0

        cum_ret = (eq - budget) / budget if budget > 0 else 0.0
        return RiskMetricsSummary(
            total_return=round(cum_ret, 4),
            annualized_return=round(cum_ret, 4),
            annualized_volatility=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            win_rate=1.0 if cum_ret > 0 else 0.0,
            profit_loss_ratio=99.0 if cum_ret > 0 else 0.0,
            trading_days=len(history),
        )

    def get_monthly_returns_matrix(self, account_id: str) -> Dict[str, Any]:
        """获取指定账户的月度收益矩阵与年度汇总"""
        history = self.get_nav_history(account_id)
        records = [{"date": r.date, "return": r.daily_return} for r in history]
        return RiskAnalyticsEngine.generate_monthly_matrix(records)

    def get_brinson_attribution(
        self,
        strategy_id: Optional[str] = None,
        benchmark_weights: Optional[Dict[str, float]] = None,
        benchmark_returns: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """对子策略或全组合进行 Brinson-Fachler 行业绩效归因"""
        self.sync_from_broker()
        if strategy_id and strategy_id in self.master.strategies:
            strat = self.master.strategies[strategy_id]
            return PmsBrinsonAdapter.attribute_strategy(
                strategy=strat,
                benchmark_weights=benchmark_weights,
                benchmark_returns=benchmark_returns,
            )

        # 默认对穿透合并后的投资组合进行全量归因
        from src.pms.aggregator import PortfolioAggregator
        consolidated = PortfolioAggregator().aggregate(self.master)
        return PmsBrinsonAdapter.attribute_consolidated(
            consolidated=consolidated,
            benchmark_weights=benchmark_weights,
            benchmark_returns=benchmark_returns,
        )
