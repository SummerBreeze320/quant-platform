import uuid
import time
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from src.common.logger import logger
from src.execution_engine.models import (
    Order, Trade, AlgoType, OrderDirection, OrderStatus
)
from src.risk_engine.models import OrderRiskRequest, CircuitBreakerLevel


class ExecutionTaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class SlicedExecutionTask(BaseModel):
    task_id: str
    account_id: str
    algo_type: AlgoType = AlgoType.TWAP
    symbol: str
    direction: OrderDirection
    total_volume: int
    filled_volume: int = 0
    total_slices: int
    completed_slices: int = 0
    interval_seconds: float = 1.0
    status: ExecutionTaskStatus = ExecutionTaskStatus.PENDING
    trades: List[Trade] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class TimeSlicedScheduler:
    """时间切片算法执行调度器。
    
    负责按预设时间步长（Interval）异步分发 TWAP / VWAP 子切片订单，
    在切片间隔期间支持动态风控监测、熔断拦截以及中途主动撤单。
    """

    def __init__(self):
        self.tasks: Dict[str, SlicedExecutionTask] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def get_task(self, task_id: str) -> Optional[SlicedExecutionTask]:
        with self._lock:
            task = self.tasks.get(task_id)
            return task.model_copy(deep=True) if task else None

    def list_tasks(self, account_id: Optional[str] = None) -> List[SlicedExecutionTask]:
        with self._lock:
            tasks = list(self.tasks.values())
            if account_id:
                tasks = [t for t in tasks if t.account_id == account_id]
            return [t.model_copy(deep=True) for t in tasks]

    def cancel_task(self, task_id: str) -> bool:
        """撤销正在进行的切片执行任务，中断后续子切片报单"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            if task.status in [ExecutionTaskStatus.COMPLETED, ExecutionTaskStatus.CANCELLED, ExecutionTaskStatus.FAILED]:
                return False

            event = self._cancel_events.get(task_id)
            if event:
                event.set()
            task.status = ExecutionTaskStatus.CANCELLED
            task.updated_at = datetime.now().isoformat()
            logger.info(f"[Scheduler] Task {task_id} manually cancelled ({task.completed_slices}/{task.total_slices} slices completed).")
            return True

    def schedule_execution(
        self,
        parent_order: Order,
        slices: List[Order],
        interval_seconds: float,
        coordinator,
        run_async: bool = True,
    ) -> SlicedExecutionTask:
        """注册并启动时间切片执行任务"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = SlicedExecutionTask(
            task_id=task_id,
            account_id=parent_order.account_id,
            algo_type=parent_order.algo_type,
            symbol=parent_order.symbol,
            direction=parent_order.direction,
            total_volume=parent_order.volume,
            total_slices=len(slices),
            interval_seconds=max(0.0, interval_seconds),
            status=ExecutionTaskStatus.PENDING,
        )

        cancel_event = threading.Event()

        with self._lock:
            self.tasks[task_id] = task
            self._cancel_events[task_id] = cancel_event

        if run_async:
            thread = threading.Thread(
                target=self._run_slices_worker,
                args=(task_id, slices, interval_seconds, coordinator, cancel_event),
                daemon=True,
                name=f"twap_worker_{task_id}",
            )
            thread.start()
        else:
            self._run_slices_worker(task_id, slices, interval_seconds, coordinator, cancel_event)

        return self.get_task(task_id)

    def _run_slices_worker(
        self,
        task_id: str,
        slices: List[Order],
        interval_seconds: float,
        coordinator,
        cancel_event: threading.Event,
    ):
        with self._lock:
            task = self.tasks[task_id]
            task.status = ExecutionTaskStatus.RUNNING
            task.updated_at = datetime.now().isoformat()

        logger.info(f"[Scheduler] Started task {task_id}: {len(slices)} slices, interval={interval_seconds}s.")

        for idx, sub_order in enumerate(slices):
            if cancel_event.is_set():
                logger.info(f"[Scheduler] Task {task_id} halted due to cancellation signal.")
                break

            # 1. 事前熔断检查 (Circuit Breaker check)
            cb_state = coordinator.circuit_breaker.get_state(task.account_id)
            if cb_state.level == CircuitBreakerLevel.RED_HALT:
                with self._lock:
                    task.status = ExecutionTaskStatus.CANCELLED
                    task.errors.append("RED_HALT 熔断已激活，终止切片挂单")
                    task.updated_at = datetime.now().isoformat()
                logger.warning(f"[Scheduler] Task {task_id} aborted by RED_HALT circuit breaker.")
                break
            elif (
                cb_state.level == CircuitBreakerLevel.ORANGE_RESTRICT_BUY
                and sub_order.direction == OrderDirection.BUY
            ):
                with self._lock:
                    task.status = ExecutionTaskStatus.CANCELLED
                    task.errors.append("ORANGE 熔断已激活，限制买入建仓")
                    task.updated_at = datetime.now().isoformat()
                logger.warning(f"[Scheduler] Task {task_id} aborted by ORANGE circuit breaker.")
                break

            # 2. 事前风控检查 (Pre-trade risk check)
            acc = coordinator.gateway.get_account(task.account_id)
            curr_pos_vol = acc.positions.get(sub_order.symbol).total_volume if acc.positions.get(sub_order.symbol) else 0

            risk_req = OrderRiskRequest(
                account_id=task.account_id,
                order_id=sub_order.order_id,
                symbol=sub_order.symbol,
                direction=sub_order.direction.value,
                price=sub_order.price,
                volume=sub_order.volume,
                current_position=curr_pos_vol,
                total_equity=acc.total_equity,
                available_cash=acc.available_cash,
            )
            risk_res = coordinator.risk_checker.check_order(risk_req)
            if not risk_res.passed:
                with self._lock:
                    task.errors.append(f"切片 {idx+1} 风控拒单: {'; '.join(risk_res.violations)}")
                    task.completed_slices += 1
                    task.updated_at = datetime.now().isoformat()
                continue

            # 3. 提交切片至柜台
            trade = coordinator.gateway.submit_order(sub_order)
            with self._lock:
                if trade:
                    task.trades.append(trade)
                    task.filled_volume += trade.volume
                    # 4. 更新熔断器水位
                    updated_acc = coordinator.gateway.get_account(task.account_id)
                    coordinator.circuit_breaker.update_equity(task.account_id, updated_acc.total_equity)
                else:
                    task.errors.append(f"切片 {idx+1} 柜台未能成交")

                task.completed_slices += 1
                task.updated_at = datetime.now().isoformat()

            # 5. 等待切片时间间隔
            if idx < len(slices) - 1 and interval_seconds > 0:
                is_cancelled = cancel_event.wait(timeout=interval_seconds)
                if is_cancelled:
                    break

        with self._lock:
            if task.status == ExecutionTaskStatus.RUNNING:
                task.status = ExecutionTaskStatus.COMPLETED
            task.updated_at = datetime.now().isoformat()
            logger.info(f"[Scheduler] Task {task_id} finished: status={task.status.value}, filled={task.filled_volume}/{task.total_volume}")
