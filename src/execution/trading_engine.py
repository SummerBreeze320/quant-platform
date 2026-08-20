"""Trading Engine — 日频交易引擎 (T+1执行)

Daily production pipeline orchestrator with T+1 execution:

  Day T 09:00  Pre-market — refresh market data, compute factors
  Day T 15:00  Close — Feature → Model → Portfolio → Generate orders (pending)
  Day T+1 09:30 Open — Execute pending BUY/SELL orders at open price
  Day T+1 15:30 Post-close review — PnL, decay detection, RD-Agent trigger

The T+1 flow:
1. Orders generated at T-day close are stored as pending
2. Pending orders execute at T+1 open price
3. A-share rule: T+1 settlement (buy today, can sell tomorrow)

The engine coordinates:
  SignalGenerator → RiskAgent → PortfolioScheduler → Broker → PerformanceReview
"""
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradingPhase(str, Enum):
    IDLE = "idle"
    OPEN_EXECUTE = "open_execute"       # 09:30 执行昨日待执行订单
    PRE_MARKET = "pre_market"           # 09:00 数据更新
    DATA_READY = "data_ready"
    SIGNAL_GENERATED = "signal_generated"
    RISK_CHECKED = "risk_checked"
    ORDERS_GENERATED = "orders_generated"  # 15:00 生成订单(待执行)
    POST_CLOSE = "post_close"           # 15:30 盘后复盘
    ERROR = "error"


@dataclass
class PendingOrder:
    """待执行订单 — T日生成, T+1日开盘执行"""
    stock: str
    side: str  # "buy" / "sell"
    target_weight: float
    current_weight: float
    delta_weight: float
    generated_date: str        # 订单生成日期
    estimated_price: float     # 生成时的估算价格
    estimated_value: float     # 估算金额
    volume: float              # 目标股数
    status: str = "pending"    # pending / executed / cancelled / rejected
    executed_price: float = 0.0
    executed_value: float = 0.0
    executed_at: Optional[str] = None
    reject_reason: str = ""


@dataclass
class TradingDayRecord:
    """One day's trading record."""
    trade_date: str
    phase: TradingPhase = TradingPhase.IDLE
    signal: Optional[Dict] = None
    risk_assessment: Optional[Dict] = None
    orders: List[Dict] = field(default_factory=list)           # 今日生成的订单(待执行)
    executed_orders: List[Dict] = field(default_factory=list)  # 今日执行的订单(来自昨日)
    pending_orders: List[Dict] = field(default_factory=list)   # 今日收盘时的待执行订单
    nav: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    review: Optional[Dict] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TradingEngine:
    """Daily trading engine — T+1 production pipeline.

    Usage:
        engine = TradingEngine(init_capital=1_000_000)
        engine.run_daily()  # Run one full trading day
    """

    def __init__(
        self,
        init_capital: float = 1_000_000,
        benchmark: str = "SH000300",
        topk: int = 5,
        max_position: float = 0.15,
        model_path: Optional[str] = None,
        handler_type: str = "alpha158",
    ):
        self.init_capital = init_capital
        self.benchmark = benchmark
        self.topk = topk
        self.max_position = max_position
        self.model_path = model_path
        self.handler_type = handler_type

        self.current_nav = init_capital
        self.current_cash = init_capital
        self.current_positions: Dict[str, Dict] = {}
        self.pending_orders: List[PendingOrder] = []
        self.history: List[TradingDayRecord] = []
        self.phase = TradingPhase.IDLE

        self._signal_gen = None
        self._risk_agent = None
        self._scheduler = None
        self._review = None
        self._initialized = False

    def initialize(self):
        from src.execution.qlib_signal_generator import QlibSignalGenerator
        from src.agents.risk_agent import RiskAgent
        from src.execution.review import PerformanceReview

        self._signal_gen = QlibSignalGenerator(
            model_path=self.model_path,
            handler_type=self.handler_type,
            topk=self.topk,
            max_position=self.max_position,
        )
        self._risk_agent = RiskAgent()
        self._risk_agent.initialize()
        self._review = PerformanceReview()

        self._initialized = True
        logger.info(
            f"TradingEngine initialized: capital={self.init_capital:.0f} "
            f"topk={self.topk} benchmark={self.benchmark}"
        )

    def run_daily(self, trade_date: Optional[str] = None) -> TradingDayRecord:
        """Run one complete trading day with T+1 execution.

        Flow:
        1. Execute pending orders from previous day (09:30 open)
        2. Pre-market data refresh (09:00 — actually runs after open exec)
        3. Signal generation: Feature → Model → Portfolio (15:00)
        4. Risk assessment
        5. Generate orders (store as pending, DON'T execute)
        6. Post-close review (15:30)

        Args:
            trade_date: Date string (YYYY-MM-DD). None = today.

        Returns:
            TradingDayRecord with all phases recorded.
        """
        if not self._initialized:
            self.initialize()

        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        record = TradingDayRecord(
            trade_date=trade_date,
            started_at=datetime.now().isoformat(),
        )

        logger.info(f"=== Trading Day: {trade_date} (T+1 mode) ===")

        try:
            # Phase 1: Execute pending orders from yesterday (at open)
            record.phase = TradingPhase.OPEN_EXECUTE
            executed = self._execute_pending_orders(trade_date)
            record.executed_orders = executed

            # Phase 2: Pre-market data refresh
            record.phase = TradingPhase.PRE_MARKET
            self._pre_market(trade_date)

            # Phase 3: Signal generation (Feature → Model → Portfolio)
            record.phase = TradingPhase.DATA_READY
            signal = self._generate_signal(trade_date)
            record.signal = signal

            if "error" in signal:
                record.phase = TradingPhase.ERROR
                record.error = signal["error"]
                self.history.append(record)
                return record

            record.phase = TradingPhase.SIGNAL_GENERATED

            # Phase 4: Risk assessment
            risk = self._check_risk(trade_date)
            record.risk_assessment = risk
            record.phase = TradingPhase.RISK_CHECKED

            # Phase 5: Adjust positions based on risk
            target_weights = signal.get("target_weights", {})
            if risk and risk.get("position_ratio", 1.0) < 1.0:
                target_weights = self._risk_agent.check_position_limit(
                    target_weights, risk["position_ratio"]
                )

            # Phase 6: Generate orders (DON'T execute — T+1)
            orders = self._generate_orders(target_weights, trade_date)
            record.orders = orders
            record.phase = TradingPhase.ORDERS_GENERATED

            # Store pending orders for next day execution
            record.pending_orders = [
                {
                    "stock": o["stock"],
                    "side": o["side"],
                    "target_weight": o["target_weight"],
                    "delta_weight": o["delta_weight"],
                    "estimated_price": o["estimated_price"],
                    "estimated_value": o["estimated_value"],
                    "volume": o["volume"],
                    "generated_date": trade_date,
                    "status": "pending",
                }
                for o in orders
            ]

            # Phase 7: Post-close review
            record.phase = TradingPhase.POST_CLOSE
            review = self._post_close(trade_date, record)
            record.review = review

            # Update NAV (include position market value)
            self._update_nav()
            record.nav = self.current_nav
            record.pnl = self.current_nav - self.init_capital
            record.pnl_pct = (self.current_nav - self.init_capital) / self.init_capital
            record.completed_at = datetime.now().isoformat()

        except Exception as e:
            record.phase = TradingPhase.ERROR
            record.error = str(e)
            logger.error(f"Trading day failed: {e}", exc_info=True)

        self.history.append(record)
        self.phase = record.phase

        logger.info(
            f"Trading day {trade_date} complete: "
            f"phase={record.phase.value} nav={record.nav:.0f} "
            f"pnl={record.pnl_pct:.2%} "
            f"executed={len(record.executed_orders)} "
            f"pending={len(record.pending_orders)}"
        )
        return record

    def _execute_pending_orders(self, trade_date: str) -> List[Dict]:
        """Phase 1: Execute pending orders at open price.

        T+1 execution: orders generated yesterday are executed today at open.
        A-share rules:
        - 先卖后买 (sell first to free up cash)
        - T+1: stocks bought yesterday can be sold today
        """
        if not self.pending_orders:
            logger.info(f"[09:30] No pending orders to execute for {trade_date}")
            return []

        logger.info(f"[09:30] Executing {len(self.pending_orders)} pending orders (T+1 open)")

        # Sort: sells first (free cash), then buys
        sorted_orders = sorted(
            self.pending_orders,
            key=lambda o: 0 if o.side == "sell" else 1
        )

        executed = []
        remaining_pending = []

        for order in sorted_orders:
            # Get open price (use estimated price as fallback if no live data)
            open_price = self._get_open_price(order.stock, trade_date) or order.estimated_price

            if open_price <= 0:
                order.status = "rejected"
                order.reject_reason = "No open price available"
                remaining_pending.append(order)
                continue

            # Check if stock is tradable (not suspended, not limit up/down)
            if not self._is_tradable(order.stock, trade_date, open_price):
                order.status = "cancelled"
                order.reject_reason = "Stock not tradable (suspended or limit)"
                remaining_pending.append(order)
                continue

            # Execute order
            order.executed_price = open_price
            order.executed_value = order.estimated_value  # Approximate
            order.executed_at = datetime.now().isoformat()
            order.status = "executed"

            # Update positions and cash
            if order.side == "buy":
                self.current_cash -= order.executed_value
                self.current_positions[order.stock] = {
                    "weight": order.target_weight,
                    "price": open_price,
                    "volume": order.volume,
                    "cost_price": open_price,
                    "buy_date": trade_date,
                }
                logger.info(
                    f"  BUY {order.stock}: {order.volume:.0f} shares "
                    f"@ {open_price:.2f} = {order.executed_value:.0f}"
                )
            else:  # sell
                self.current_cash += order.executed_value
                if order.target_weight > 0:
                    if order.stock in self.current_positions:
                        self.current_positions[order.stock]["weight"] = order.target_weight
                        self.current_positions[order.stock]["price"] = open_price
                else:
                    self.current_positions.pop(order.stock, None)
                logger.info(
                    f"  SELL {order.stock}: {order.volume:.0f} shares "
                    f"@ {open_price:.2f} = {order.executed_value:.0f}"
                )

            executed.append({
                "stock": order.stock,
                "side": order.side,
                "executed_price": open_price,
                "executed_value": order.executed_value,
                "volume": order.volume,
                "generated_date": order.generated_date,
                "executed_date": trade_date,
                "status": "executed",
            })

        # Update pending orders (keep non-executed ones)
        self.pending_orders = remaining_pending

        n_buy = sum(1 for o in executed if o["side"] == "buy")
        n_sell = sum(1 for o in executed if o["side"] == "sell")
        logger.info(
            f"[09:30] Executed {len(executed)} orders "
            f"({n_buy} buy, {n_sell} sell), "
            f"{len(remaining_pending)} still pending"
        )

        return executed

    def _get_open_price(self, stock: str, trade_date: str) -> float:
        """Get opening price for a stock on trade_date."""
        try:
            from src.core import get_ohlcv
            df = get_ohlcv(stock)
            if df is not None and not df.empty:
                ts = pd.Timestamp(trade_date)
                if ts in df.index:
                    return float(df.loc[ts, "open"])
                # Fallback to closest date before
                before = df[df.index <= ts]
                if not before.empty:
                    return float(before.iloc[-1]["open"])
        except Exception as e:
            logger.debug(f"Open price lookup failed for {stock}: {e}")
        return 0.0

    def _is_tradable(self, stock: str, trade_date: str, price: float) -> bool:
        """Check if stock is tradable (not suspended, not at limit)."""
        try:
            from src.core import get_ohlcv
            df = get_ohlcv(stock)
            if df is None or df.empty:
                return True  # Allow if we can't check

            ts = pd.Timestamp(trade_date)
            if ts in df.index:
                row = df.loc[ts]
                high = float(row.get("high", 0))
                low = float(row.get("low", 0))
                # If high == low, likely suspended
                if high > 0 and low > 0 and abs(high - low) < 0.01:
                    return False
                # Check limit up/down (10% daily limit for A-share)
                prev_close = float(row.get("close", 0))
                if prev_close > 0:
                    change_pct = (price - prev_close) / prev_close
                    if abs(change_pct) >= 0.099:
                        return False
        except Exception:
            pass
        return True

    def _pre_market(self, trade_date: str):
        """Phase 2: Pre-market data refresh."""
        from src.core import ensure_qlib

        ensure_qlib()
        logger.info(f"[09:00] Pre-market: data refreshed for {trade_date}")

    def _generate_signal(self, trade_date: str) -> Dict:
        """Phase 3: Generate trading signal — Feature → Model → Portfolio."""
        logger.info(f"[15:00] Signal generation: Feature → Model → Portfolio...")

        try:
            signal = self._signal_gen.generate(
                instruments="all",
                as_of_date=trade_date,
            )
            logger.info(
                f"Signal: {signal.get('n_selected', 0)} positions, "
                f"date={signal.get('signal_date', 'N/A')}"
            )
            return signal
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {"error": str(e)}

    def _check_risk(self, trade_date: str) -> Dict:
        """Phase 4: Risk assessment."""
        from src.agents.contracts import RiskAssessmentRequest

        logger.info("[15:00] Risk assessment...")

        try:
            req = RiskAssessmentRequest(
                benchmark=self.benchmark,
                as_of_date=trade_date,
            )
            assessment = self._risk_agent.execute(req)
            result = {
                "market_state": assessment.market_state.value,
                "position_ratio": assessment.position_ratio,
                "risk_score": assessment.risk_score,
                "volatility_20d": assessment.volatility_20d,
                "panic_index": assessment.panic_index,
                "position_reason": assessment.position_reason,
                "warnings": assessment.warnings,
            }
            logger.info(
                f"Risk: state={result['market_state']} "
                f"position={result['position_ratio']:.0%} "
                f"panic={result['panic_index']:.2f}"
            )
            return result
        except Exception as e:
            logger.warning(f"Risk assessment failed: {e}")
            return {
                "market_state": "sideways",
                "position_ratio": 0.5,
                "risk_score": 0.5,
                "volatility_20d": 0,
                "panic_index": 0,
                "position_reason": "Risk assessment error, default 50%",
                "warnings": [f"Risk assessment error: {e}"],
            }

    def _generate_orders(self, target_weights: Dict[str, float], trade_date: str) -> List[Dict]:
        """Phase 6: Generate rebalance orders — T日生成, T+1日执行.

        Unlike the old _rebalance, this does NOT execute orders immediately.
        Orders are stored as pending for next-day open execution.
        """
        logger.info(f"[15:00] Generate orders: {len(target_weights)} target positions (T+1 pending)")

        orders = []
        total_value = self.current_nav

        all_stocks = set(target_weights.keys()) | set(self.current_positions.keys())

        for stock in all_stocks:
            tw = target_weights.get(stock, 0.0)
            cw = self.current_positions.get(stock, {}).get("weight", 0.0)
            delta = tw - cw

            if abs(delta) < 1e-4:
                continue

            price = self.current_positions.get(stock, {}).get("price", 0)
            if price <= 0:
                # Try to get latest price
                price = self._get_latest_price(stock) or 10.0

            delta_value = delta * total_value
            delta_volume = abs(delta_value / price) if price > 0 else 0

            # Round to lot size (100 shares for A-share)
            delta_volume = int(delta_volume / 100) * 100

            order = {
                "stock": stock,
                "side": "buy" if delta > 0 else "sell",
                "target_weight": tw,
                "current_weight": cw,
                "delta_weight": delta,
                "estimated_price": price,
                "estimated_value": abs(delta_value),
                "volume": delta_volume,
                "generated_date": trade_date,
                "status": "pending",
            }
            orders.append(order)

        # Store as pending orders for T+1 execution
        for o in orders:
            self.pending_orders.append(PendingOrder(
                stock=o["stock"],
                side=o["side"],
                target_weight=o["target_weight"],
                current_weight=o["current_weight"],
                delta_weight=o["delta_weight"],
                generated_date=trade_date,
                estimated_price=o["estimated_price"],
                estimated_value=o["estimated_value"],
                volume=o["volume"],
            ))

        n_buy = sum(1 for o in orders if o["side"] == "buy")
        n_sell = sum(1 for o in orders if o["side"] == "sell")
        logger.info(
            f"Orders generated (pending for T+1): {len(orders)} "
            f"({n_buy} buy, {n_sell} sell)"
        )
        return orders

    def _get_latest_price(self, stock: str) -> float:
        """Get latest close price for a stock."""
        try:
            from src.core import get_ohlcv
            df = get_ohlcv(stock)
            if df is not None and not df.empty:
                return float(df.iloc[-1]["close"])
        except Exception:
            pass
        return 0.0

    def _update_nav(self):
        """Update NAV to include both cash and position market value."""
        position_value = 0.0
        for stock, pos in self.current_positions.items():
            price = pos.get("price", 0)
            volume = pos.get("volume", 0)
            position_value += price * volume

        self.current_nav = self.current_cash + position_value

    def _post_close(self, trade_date: str, record: TradingDayRecord) -> Dict:
        """Phase 7: Post-close review."""
        logger.info("[15:30] Post-close review...")

        try:
            review = self._review.review_day(
                trade_date=trade_date,
                nav=self.current_nav,
                init_capital=self.init_capital,
                positions=self.current_positions,
                orders=record.executed_orders,  # Review executed orders
                signal=record.signal,
                risk=record.risk_assessment,
            )
            return review
        except Exception as e:
            logger.warning(f"Post-close review failed: {e}")
            return {"error": str(e)}

    def get_pending_orders(self) -> List[Dict]:
        """Get current pending orders (generated but not yet executed)."""
        return [
            {
                "stock": o.stock,
                "side": o.side,
                "target_weight": o.target_weight,
                "delta_weight": o.delta_weight,
                "estimated_price": o.estimated_price,
                "estimated_value": o.estimated_value,
                "volume": o.volume,
                "generated_date": o.generated_date,
                "status": o.status,
            }
            for o in self.pending_orders
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            "phase": self.phase.value,
            "nav": self.current_nav,
            "cash": self.current_cash,
            "init_capital": self.init_capital,
            "pnl_pct": (self.current_nav - self.init_capital) / self.init_capital,
            "n_positions": len(self.current_positions),
            "n_pending_orders": len(self.pending_orders),
            "n_trading_days": len(self.history),
            "last_trade_date": self.history[-1].trade_date if self.history else None,
            "last_phase": self.history[-1].phase.value if self.history else None,
        }

    def get_positions(self) -> List[Dict]:
        """Get current positions detail."""
        return [
            {
                "stock": stock,
                "weight": pos.get("weight", 0),
                "price": pos.get("price", 0),
                "volume": pos.get("volume", 0),
                "market_value": pos.get("price", 0) * pos.get("volume", 0),
                "cost_price": pos.get("cost_price", 0),
                "buy_date": pos.get("buy_date", ""),
            }
            for stock, pos in self.current_positions.items()
        ]

    def get_history(self, n_days: int = 30) -> List[Dict]:
        """Get recent trading history."""
        recent = self.history[-n_days:]
        return [
            {
                "trade_date": r.trade_date,
                "phase": r.phase.value,
                "nav": r.nav,
                "pnl_pct": r.pnl_pct,
                "n_orders_generated": len(r.orders),
                "n_orders_executed": len(r.executed_orders),
                "n_pending": len(r.pending_orders),
                "error": r.error,
            }
            for r in recent
        ]
