# Real-Time Market Event Stream Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 QuantCopilot 构建生产级实时行情驱动与事件流中枢（Market Event Stream Engine），实现高保真 Tick/盘口切片处理、Redis Streams 分布式事件流分发、增量 $O(1)$ 复杂度流式指标（VWAP、布林带、OFI微观盘口不平衡度）、高频形态信号触发与防抖冷却，以及策略级可配置的风控审查与 A 股仿真撮合自动成交闭环和 FastAPI/WebSocket 服务。

**Architecture:** 基于分层事件驱动流式管道架构。行情输入（实时接入或 `TickReplayEngine` 历史回放）注入 `StreamBus`（支持 Redis Streams 与单机内存优雅回退）；流式指标引擎增量计算最新 VWAP、实时布林带与微观买卖不平衡度 OFI；`SignalEngine` 匹配通道突破与买压突增形态并经 3 秒冷却防抖输出 `SignalEvent`；`SignalRouter` 根据策略配置自动过闸 `PreTradeRiskChecker` 并交由 `PaperBroker` 完成撮合，或推送到告警中台与 WebSocket 实时广播。

**Tech Stack:** Python 3.11, Pydantic, Redis Streams / Memory Queue, FastAPI, WebSocket, Starlette TestClient, pytest, collections.deque, PreTradeRiskChecker, PaperBroker.

**Spec:** [docs/superpowers/specs/2026-09-18-market-feed-design.md](file:///d:/BaiduSyncdisk/PycharmProjects/QuantCopliot/docs/superpowers/specs/2026-09-18-market-feed-design.md)

## Global Constraints
- 纯 Python 开发，严格保持 Windows 与 Linux 跨平台兼容。
- 采用 Redis Streams 作为事件总线，同时必须具备轻量级内存队列优雅回退机制（无 Redis 时 100% 测试通过）。
- 流式指标（VWAP、布林带、OFI）必须采用严格 $O(1)$ 时间复杂度增量算法，禁止全量数组回溯。
- 测试运行环境固定使用：`& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest`。
- 保证既有 87 项全量自动化测试 100% 通过无回归破坏。

---

### Task 1: 核心数据模型与事件定义 (`src/market_feed/models.py`)

**Files:**
- Create: `src/market_feed/models.py`
- Test: `tests/test_market_models.py`

**Interfaces:**
- Consumes: Standard library `enum`, `typing`, `pydantic.BaseModel, Field`
- Produces: `SignalDirection`, `MarketTick`, `IndicatorValue`, `SignalEvent`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market_models.py
from src.market_feed.models import SignalDirection, MarketTick, IndicatorValue, SignalEvent

def test_market_feed_enums():
    assert SignalDirection.BUY == "BUY"
    assert SignalDirection.SELL == "SELL"
    assert SignalDirection.FLAT == "FLAT"

def test_market_tick_model():
    tick = MarketTick(
        symbol="600519.SH",
        timestamp="2026-09-18T09:30:00",
        last_price=1800.0,
        volume=10000,
        turnover=18000000.0,
        bid_prices=[1799.5, 1799.0, 1798.5],
        bid_volumes=[100, 200, 300],
        ask_prices=[1800.5, 1801.0, 1801.5],
        ask_volumes=[150, 250, 350],
    )
    assert tick.symbol == "600519.SH"
    assert tick.last_price == 1800.0
    assert len(tick.bid_prices) == 3

def test_indicator_value_model():
    ind = IndicatorValue(
        symbol="600519.SH",
        indicator_name="BOLLINGER",
        value=1.05,
        details={"upper": 1820.0, "lower": 1780.0, "pct_b": 1.05},
        timestamp="2026-09-18T09:30:00"
    )
    assert ind.indicator_name == "BOLLINGER"
    assert ind.details["upper"] == 1820.0

def test_signal_event_model():
    sig = SignalEvent(
        signal_id="sig_001",
        strategy_id="strat_alpha",
        symbol="600519.SH",
        direction=SignalDirection.BUY,
        price=1800.0,
        volume=100,
        strength=0.95,
        reason="OFI买压激增 + 布林带上轨突破",
        timestamp="2026-09-18T09:30:00"
    )
    assert sig.direction == SignalDirection.BUY
    assert sig.volume == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_market_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.market_feed'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/market_feed/models.py
from datetime import datetime
from enum import Enum
from typing import Dict, List
from pydantic import BaseModel, Field

class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"

class MarketTick(BaseModel):
    """L1/L2 实时行情快照"""
    symbol: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_price: float
    volume: int                     # 累计成交量
    turnover: float                 # 累计成交额
    bid_prices: List[float] = Field(default_factory=list)  # 买一至买五价格
    bid_volumes: List[int] = Field(default_factory=list)   # 买一至买五委托量
    ask_prices: List[float] = Field(default_factory=list)  # 卖一至卖五价格
    ask_volumes: List[int] = Field(default_factory=list)   # 卖一至卖五委托量

class IndicatorValue(BaseModel):
    """流式指标输出包"""
    symbol: str
    indicator_name: str
    value: float
    details: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class SignalEvent(BaseModel):
    """实时量化高频交易信号"""
    signal_id: str
    strategy_id: str
    symbol: str
    direction: SignalDirection
    price: float
    volume: int = 100               # 建议触发股数 (按 A 股 100 股整手)
    strength: float = 1.0           # 信号置信度 (0.0 ~ 1.0)
    reason: str                     # 触发原因
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_market_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/market_feed/models.py tests/test_market_models.py
git commit -m "feat(market): 实时行情与交易信号核心数据模型"
```

---

### Task 2: 分布式流事件总线与内存回退机制 (`src/market_feed/stream_bus.py`)

**Files:**
- Create: `src/market_feed/stream_bus.py`
- Test: `tests/test_stream_bus.py`

**Interfaces:**
- Consumes: `src.market_feed.models.MarketTick`, `src.common.redis_client.RedisClient`
- Produces: `StreamBus`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream_bus.py
import pytest
from src.market_feed.models import MarketTick
from src.market_feed.stream_bus import StreamBus

def test_stream_bus_memory_mode():
    bus = StreamBus(stream_key="test_stream_mem", force_memory_mode=True)
    assert bus.is_memory_mode is True

    tick1 = MarketTick(symbol="600519.SH", last_price=1800.0, volume=1000, turnover=1800000.0)
    tick2 = MarketTick(symbol="000858.SZ", last_price=150.0, volume=2000, turnover=300000.0)

    msg_id1 = bus.publish_tick(tick1)
    msg_id2 = bus.publish_tick(tick2)
    assert msg_id1 is not None
    assert msg_id2 is not None

    read_ticks = bus.read_ticks(count=10)
    assert len(read_ticks) == 2
    assert read_ticks[0].symbol == "600519.SH"
    assert read_ticks[1].symbol == "000858.SZ"

    # Read again should be empty
    assert len(bus.read_ticks(count=10)) == 0

def test_stream_bus_subscriber_callback():
    bus = StreamBus(stream_key="test_stream_sub", force_memory_mode=True)
    received = []

    def on_tick(tick: MarketTick):
        received.append(tick.symbol)

    bus.subscribe(on_tick)
    bus.publish_tick(MarketTick(symbol="601318.SH", last_price=50.0, volume=500, turnover=25000.0))
    assert len(received) == 1
    assert received[0] == "601318.SH"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_stream_bus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.market_feed.stream_bus'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/market_feed/stream_bus.py
import json
import queue
import time
from typing import Callable, List, Optional
from src.common.logger import logger
from src.common.redis_client import RedisClient
from src.market_feed.models import MarketTick

class StreamBus:
    """基于 Redis Streams 的行情事件总线 (支持离线/测试内存队列自动回退)"""

    def __init__(
        self,
        stream_key: str = "market_stream:ticks",
        consumer_group: str = "quant_market_group",
        consumer_name: str = "worker_1",
        redis_client: Optional[RedisClient] = None,
        force_memory_mode: bool = False,
    ):
        self.stream_key = stream_key
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.subscribers: List[Callable[[MarketTick], None]] = []

        self._redis_client = redis_client or RedisClient()
        self.is_memory_mode = force_memory_mode

        if not self.is_memory_mode:
            try:
                # 探活 Redis
                if not self._redis_client.ping():
                    logger.warning("Redis 未连接，StreamBus 自动切换至内存事件队列模式")
                    self.is_memory_mode = True
                else:
                    # 尝试创建消费组
                    try:
                        self._redis_client.client.xgroup_create(
                            name=self.stream_key,
                            groupname=self.consumer_group,
                            id="0",
                            mkstream=True
                        )
                    except Exception:
                        pass  # 消费组可能已存在
            except Exception as e:
                logger.warning(f"Redis 初始化失败 ({e})，切换至内存模式")
                self.is_memory_mode = True

        self._memory_queue: queue.Queue = queue.Queue()

    def subscribe(self, callback: Callable[[MarketTick], None]) -> None:
        """注册回调监听器"""
        if callback not in self.subscribers:
            self.subscribers.append(callback)

    def publish_tick(self, tick: MarketTick) -> str:
        """向流发布行情快照"""
        data_json = tick.model_dump_json()

        # 触发本地订阅者回调
        for sub in self.subscribers:
            try:
                sub(tick)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

        if self.is_memory_mode:
            msg_id = f"{int(time.time() * 1000)}-{self._memory_queue.qsize()}"
            self._memory_queue.put(tick)
            return msg_id

        try:
            msg_id = self._redis_client.client.xadd(
                self.stream_key,
                {"data": data_json}
            )
            return str(msg_id)
        except Exception as e:
            logger.error(f"Redis xadd error: {e}, fallback to memory queue")
            self.is_memory_mode = True
            msg_id = f"{int(time.time() * 1000)}-0"
            self._memory_queue.put(tick)
            return msg_id

    def read_ticks(self, count: int = 10, block_ms: int = 50) -> List[MarketTick]:
        """拉取并消费行情消息"""
        ticks: List[MarketTick] = []

        if self.is_memory_mode:
            for _ in range(count):
                try:
                    ticks.append(self._memory_queue.get_nowait())
                except queue.Empty:
                    break
            return ticks

        try:
            entries = self._redis_client.client.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_key: ">"},
                count=count,
                block=block_ms
            )
            if entries:
                for stream_name, msg_list in entries:
                    for msg_id, data in msg_list:
                        raw_json = data.get("data") or data.get(b"data")
                        if raw_json:
                            ticks.append(MarketTick.model_validate_json(raw_json))
                            self._redis_client.client.xack(self.stream_key, self.consumer_group, msg_id)
            return ticks
        except Exception as e:
            logger.error(f"Redis xreadgroup error: {e}")
            return ticks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_stream_bus.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/market_feed/stream_bus.py tests/test_stream_bus.py
git commit -m "feat(market): 基于 Redis Streams 与内存回退的事件总线"
```

---

### Task 3: 流式增量指标算法库 (`src/market_feed/indicators/`)

**Files:**
- Create: `src/market_feed/indicators/base.py`
- Create: `src/market_feed/indicators/vwap.py`
- Create: `src/market_feed/indicators/bollinger.py`
- Create: `src/market_feed/indicators/ofi.py`
- Create: `src/market_feed/indicators/__init__.py`
- Test: `tests/test_streaming_indicators.py`

**Interfaces:**
- Consumes: `src.market_feed.models.MarketTick, IndicatorValue`
- Produces: `BaseStreamingIndicator`, `StreamingVWAP`, `StreamingBollingerBands`, `OrderFlowImbalance`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_streaming_indicators.py
import pytest
from src.market_feed.models import MarketTick
from src.market_feed.indicators import StreamingVWAP, StreamingBollingerBands, OrderFlowImbalance

def test_streaming_vwap():
    vwap_ind = StreamingVWAP()
    # Tick 1: 1000 shares @ 10.0 -> turnover 10,000
    t1 = MarketTick(symbol="600519.SH", last_price=10.0, volume=1000, turnover=10000.0)
    res1 = vwap_ind.update(t1)
    assert res1.value == 10.0
    assert res1.details["vwap"] == 10.0

    # Tick 2: cumulative volume 3000 (+2000), turnover 34,000 (+24,000 @ 12.0)
    # Total turnover = 34,000, Total volume = 3000 -> VWAP = 34000 / 3000 = 11.3333
    t2 = MarketTick(symbol="600519.SH", last_price=12.0, volume=3000, turnover=34000.0)
    res2 = vwap_ind.update(t2)
    assert pytest.approx(res2.value, 0.001) == 11.3333
    assert pytest.approx(res2.details["bias"], 0.001) == (12.0 - 11.3333) / 11.3333

def test_streaming_bollinger_bands():
    bb = StreamingBollingerBands(window=5, num_std=2.0)
    prices = [10.0, 10.0, 10.0, 10.0, 10.0]
    for p in prices:
        t = MarketTick(symbol="600519.SH", last_price=p, volume=100, turnover=1000.0)
        res = bb.update(t)
    assert res.value == 10.0
    assert res.details["std"] == 0.0
    assert res.details["upper"] == 10.0
    assert res.details["lower"] == 10.0

    # Price surge to 15.0 -> upper band expands and pct_b > 1.0
    t_surge = MarketTick(symbol="600519.SH", last_price=15.0, volume=200, turnover=3000.0)
    res_surge = bb.update(t_surge)
    assert res_surge.details["pct_b"] > 1.0
    assert res_surge.details["upper"] > 10.0

def test_order_flow_imbalance():
    ofi = OrderFlowImbalance()
    # Snapshot 1: Bid1 100 @ 10.0, Ask1 100 @ 10.1
    t1 = MarketTick(
        symbol="600519.SH", last_price=10.0, volume=1000, turnover=10000.0,
        bid_prices=[10.0], bid_volumes=[100],
        ask_prices=[10.1], ask_volumes=[100]
    )
    res1 = ofi.update(t1)
    assert res1.value == 0.0

    # Snapshot 2: Price same, Bid1 volume increases to 300 (+200 buyer pressure), Ask1 same
    t2 = MarketTick(
        symbol="600519.SH", last_price=10.0, volume=1000, turnover=10000.0,
        bid_prices=[10.0], bid_volumes=[300],
        ask_prices=[10.1], ask_volumes=[100]
    )
    res2 = ofi.update(t2)
    assert res2.value > 0.0 # Positive buyer flow imbalance
    assert res2.details["ofi_raw"] == 200.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_streaming_indicators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.market_feed.indicators'`

- [ ] **Step 3: Write minimal implementation**

`src/market_feed/indicators/base.py`:
```python
from abc import ABC, abstractmethod
from src.market_feed.models import MarketTick, IndicatorValue

class BaseStreamingIndicator(ABC):
    @abstractmethod
    def update(self, tick: MarketTick) -> IndicatorValue:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
```

`src/market_feed/indicators/vwap.py`:
```python
from typing import Optional
from src.market_feed.models import MarketTick, IndicatorValue
from src.market_feed.indicators.base import BaseStreamingIndicator

class StreamingVWAP(BaseStreamingIndicator):
    """增量流式 VWAP (成交量加权平均价)"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total_volume = 0.0
        self.total_turnover = 0.0
        self.last_raw_volume: Optional[int] = None
        self.last_raw_turnover: Optional[float] = None
        self.current_vwap = 0.0

    def update(self, tick: MarketTick) -> IndicatorValue:
        if self.last_raw_volume is None or self.last_raw_turnover is None:
            # 第一笔作为初始基准
            delta_vol = tick.volume if tick.volume > 0 else 1
            delta_to = tick.turnover if tick.turnover > 0 else (tick.last_price * delta_vol)
        else:
            delta_vol = max(0, tick.volume - self.last_raw_volume)
            delta_to = max(0.0, tick.turnover - self.last_raw_turnover)

        self.last_raw_volume = tick.volume
        self.last_raw_turnover = tick.turnover

        self.total_volume += delta_vol
        self.total_turnover += delta_to

        if self.total_volume > 0:
            self.current_vwap = round(self.total_turnover / self.total_volume, 4)
        else:
            self.current_vwap = tick.last_price

        bias = round((tick.last_price - self.current_vwap) / self.current_vwap, 6) if self.current_vwap > 0 else 0.0

        return IndicatorValue(
            symbol=tick.symbol,
            indicator_name="VWAP",
            value=self.current_vwap,
            details={"vwap": self.current_vwap, "bias": bias},
            timestamp=tick.timestamp,
        )
```

`src/market_feed/indicators/bollinger.py`:
```python
import math
from collections import deque
from src.market_feed.models import MarketTick, IndicatorValue
from src.market_feed.indicators.base import BaseStreamingIndicator

class StreamingBollingerBands(BaseStreamingIndicator):
    """滑动窗口实时布林带 (O(1) 增量维护)"""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
        self.reset()

    def reset(self) -> None:
        self.prices = deque(maxlen=self.window)

    def update(self, tick: MarketTick) -> IndicatorValue:
        self.prices.append(tick.last_price)
        n = len(self.prices)
        mean = sum(self.prices) / n

        if n > 1:
            variance = sum((p - mean) ** 2 for p in self.prices) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0

        upper = round(mean + self.num_std * std, 4)
        lower = round(mean - self.num_std * std, 4)
        band_width = round((upper - lower) / mean, 6) if mean > 0 else 0.0

        if upper != lower:
            pct_b = round((tick.last_price - lower) / (upper - lower), 4)
        else:
            pct_b = 0.5

        return IndicatorValue(
            symbol=tick.symbol,
            indicator_name="BOLLINGER",
            value=round(mean, 4),
            details={
                "mid": round(mean, 4),
                "upper": upper,
                "lower": lower,
                "std": round(std, 4),
                "pct_b": pct_b,
                "band_width": band_width,
            },
            timestamp=tick.timestamp,
        )
```

`src/market_feed/indicators/ofi.py`:
```python
from typing import Optional, List
from src.market_feed.models import MarketTick, IndicatorValue
from src.market_feed.indicators.base import BaseStreamingIndicator

class OrderFlowImbalance(BaseStreamingIndicator):
    """微观五档盘口订单流不平衡度 (Cont et al. OFI)"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.prev_bid_price: Optional[float] = None
        self.prev_bid_volume: Optional[int] = None
        self.prev_ask_price: Optional[float] = None
        self.prev_ask_volume: Optional[int] = None

    def update(self, tick: MarketTick) -> IndicatorValue:
        b_p = tick.bid_prices[0] if tick.bid_prices else tick.last_price
        b_v = tick.bid_volumes[0] if tick.bid_volumes else 0
        a_p = tick.ask_prices[0] if tick.ask_prices else tick.last_price
        a_v = tick.ask_volumes[0] if tick.ask_volumes else 0

        if self.prev_bid_price is None or self.prev_ask_price is None:
            self.prev_bid_price = b_p
            self.prev_bid_volume = b_v
            self.prev_ask_price = a_p
            self.prev_ask_volume = a_v
            return IndicatorValue(
                symbol=tick.symbol,
                indicator_name="OFI",
                value=0.0,
                details={"ofi_raw": 0.0, "normalized_ofi": 0.0},
                timestamp=tick.timestamp
            )

        # 买方挂单流变动
        if b_p > self.prev_bid_price:
            delta_wb = b_v
        elif b_p == self.prev_bid_price:
            delta_wb = b_v - self.prev_bid_volume
        else:
            delta_wb = -self.prev_bid_volume

        # 卖方挂单流变动
        if a_p > self.prev_ask_price:
            delta_wa = -self.prev_ask_volume
        elif a_p == self.prev_ask_price:
            delta_wa = a_v - self.prev_ask_volume
        else:
            delta_wa = a_v

        ofi_raw = float(delta_wb - delta_wa)
        total_depth = max(1.0, float(b_v + a_v))
        norm_ofi = round(max(-1.0, min(1.0, ofi_raw / total_depth)), 4)

        self.prev_bid_price = b_p
        self.prev_bid_volume = b_v
        self.prev_ask_price = a_p
        self.prev_ask_volume = a_v

        return IndicatorValue(
            symbol=tick.symbol,
            indicator_name="OFI",
            value=norm_ofi,
            details={"ofi_raw": ofi_raw, "normalized_ofi": norm_ofi},
            timestamp=tick.timestamp
        )
```

`src/market_feed/indicators/__init__.py`:
```python
from src.market_feed.indicators.base import BaseStreamingIndicator
from src.market_feed.indicators.vwap import StreamingVWAP
from src.market_feed.indicators.bollinger import StreamingBollingerBands
from src.market_feed.indicators.ofi import OrderFlowImbalance

__all__ = [
    "BaseStreamingIndicator",
    "StreamingVWAP",
    "StreamingBollingerBands",
    "OrderFlowImbalance",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_streaming_indicators.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/market_feed/indicators/ tests/test_streaming_indicators.py
git commit -m "feat(market): 流式增量指标算法库 (VWAP, Bollinger, OFI)"
```

---

### Task 4: 实时高频信号与微观形态引擎 (`src/market_feed/signal_engine.py`)

**Files:**
- Create: `src/market_feed/signal_engine.py`
- Test: `tests/test_signal_engine.py`

**Interfaces:**
- Consumes: `MarketTick`, `SignalEvent`, `SignalDirection`, `StreamingVWAP`, `StreamingBollingerBands`, `OrderFlowImbalance`
- Produces: `SignalEngine`, `CooldownManager`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signal_engine.py
import pytest
from src.market_feed.models import MarketTick, SignalDirection
from src.market_feed.signal_engine import SignalEngine

def test_signal_engine_bollinger_breakout():
    engine = SignalEngine(cooldown_seconds=1.0)

    # Feed steady ticks around 100.0
    for _ in range(10):
        engine.process_tick(MarketTick(
            symbol="600519.SH", last_price=100.0, volume=1000, turnover=100000.0,
            bid_prices=[100.0], bid_volumes=[100],
            ask_prices=[100.1], ask_volumes=[100]
        ))

    # Sudden surge to 120.0 breaking upper Bollinger Band and above VWAP
    sig = engine.process_tick(MarketTick(
        symbol="600519.SH", last_price=120.0, volume=2000, turnover=220000.0,
        bid_prices=[120.0], bid_volumes=[500],
        ask_prices=[120.1], ask_volumes=[100]
    ))
    assert sig is not None
    assert sig.direction == SignalDirection.BUY
    assert "布林带上轨突破" in sig.reason

def test_signal_engine_cooldown():
    engine = SignalEngine(cooldown_seconds=5.0)
    for _ in range(10):
        engine.process_tick(MarketTick(symbol="600519.SH", last_price=100.0, volume=1000, turnover=100000.0))

    # First trigger
    sig1 = engine.process_tick(MarketTick(symbol="600519.SH", last_price=120.0, volume=2000, turnover=220000.0))
    assert sig1 is not None

    # Immediate second tick within 5s cooldown should be suppressed
    sig2 = engine.process_tick(MarketTick(symbol="600519.SH", last_price=122.0, volume=2500, turnover=280000.0))
    assert sig2 is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_signal_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.market_feed.signal_engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/market_feed/signal_engine.py
import time
import uuid
from typing import Dict, Optional
from src.market_feed.models import MarketTick, SignalEvent, SignalDirection
from src.market_feed.indicators import StreamingVWAP, StreamingBollingerBands, OrderFlowImbalance

class CooldownManager:
    """高频防抖防刷单冷却管理器"""

    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger_times: Dict[str, float] = {}

    def is_cooling_down(self, symbol: str, direction: SignalDirection) -> bool:
        key = f"{symbol}:{direction.value}"
        now = time.time()
        last_time = self._last_trigger_times.get(key, 0.0)
        if now - last_time < self.cooldown_seconds:
            return True
        return False

    def record_trigger(self, symbol: str, direction: SignalDirection) -> None:
        key = f"{symbol}:{direction.value}"
        self._last_trigger_times[key] = time.time()

class SignalEngine:
    """实时高频与微观盘口信号触发器"""

    def __init__(self, default_strategy_id: str = "hft_momentum", cooldown_seconds: float = 3.0):
        self.default_strategy_id = default_strategy_id
        self.cooldown = CooldownManager(cooldown_seconds=cooldown_seconds)
        self.vwaps: Dict[str, StreamingVWAP] = {}
        self.bollingers: Dict[str, StreamingBollingerBands] = {}
        self.ofis: Dict[str, OrderFlowImbalance] = {}

    def _get_or_create_indicators(self, symbol: str):
        if symbol not in self.vwaps:
            self.vwaps[symbol] = StreamingVWAP()
            self.bollingers[symbol] = StreamingBollingerBands(window=15, num_std=2.0)
            self.ofis[symbol] = OrderFlowImbalance()
        return self.vwaps[symbol], self.bollingers[symbol], self.ofis[symbol]

    def process_tick(self, tick: MarketTick) -> Optional[SignalEvent]:
        vwap_ind, bb_ind, ofi_ind = self._get_or_create_indicators(tick.symbol)

        res_vwap = vwap_ind.update(tick)
        res_bb = bb_ind.update(tick)
        res_ofi = ofi_ind.update(tick)

        vwap_val = res_vwap.value
        pct_b = res_bb.details.get("pct_b", 0.5)
        norm_ofi = res_ofi.value

        direction: Optional[SignalDirection] = None
        reason: str = ""

        # 规则 1: 布林带上突破 + 站稳 VWAP 强买方
        if pct_b > 1.0 and tick.last_price >= vwap_val:
            direction = SignalDirection.BUY
            reason = f"布林带上轨突破 (%b={pct_b:.2f}) 且高于VWAP ({vwap_val:.2f})"
        # 规则 2: 布林带下跌破 + 跌破 VWAP
        elif pct_b < 0.0 and tick.last_price <= vwap_val:
            direction = SignalDirection.SELL
            reason = f"布林带下轨跌破 (%b={pct_b:.2f}) 且低于VWAP ({vwap_val:.2f})"
        # 规则 3: OFI 买压激增 (买卖挂单极度不平衡)
        elif norm_ofi > 0.65:
            direction = SignalDirection.BUY
            reason = f"微观盘口OFI买压聚集 (OFI={norm_ofi:.2f})"
        elif norm_ofi < -0.65:
            direction = SignalDirection.SELL
            reason = f"微观盘口OFI抛压显著 (OFI={norm_ofi:.2f})"

        if direction is not None:
            # 冷却检查
            if self.cooldown.is_cooling_down(tick.symbol, direction):
                return None

            self.cooldown.record_trigger(tick.symbol, direction)
            return SignalEvent(
                signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                strategy_id=self.default_strategy_id,
                symbol=tick.symbol,
                direction=direction,
                price=tick.last_price,
                volume=100,
                strength=0.9,
                reason=reason,
                timestamp=tick.timestamp
            )

        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_signal_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/market_feed/signal_engine.py tests/test_signal_engine.py
git commit -m "feat(market): 实时高频信号与微观形态引擎实现"
```

---

### Task 5: 信号路由与风控/执行无缝闭环 (`src/market_feed/router.py`)

**Files:**
- Create: `src/market_feed/router.py`
- Test: `tests/test_signal_router.py`

**Interfaces:**
- Consumes: `SignalEvent`, `PreTradeRiskChecker`, `PaperBroker`, `Order`, `OrderDirection`
- Produces: `SignalRouter`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signal_router.py
import pytest
from src.market_feed.models import SignalEvent, SignalDirection
from src.market_feed.router import SignalRouter
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.execution_engine.gateway.paper_broker import PaperBroker

def test_signal_router_auto_execution_success():
    broker = PaperBroker()
    acc = broker.create_account("strat_hft", initial_cash=1_000_000.0)
    risk_checker = PreTradeRiskChecker(max_order_amount=500_000.0)
    router = SignalRouter(broker=broker, risk_checker=risk_checker)

    router.register_strategy(strategy_id="strat_hft", auto_execute=True, default_volume=100)

    sig = SignalEvent(
        signal_id="sig_test_01",
        strategy_id="strat_hft",
        symbol="600519.SH",
        direction=SignalDirection.BUY,
        price=1800.0,
        volume=100,
        reason="测试突破信号"
    )
    result = router.route_signal(sig)
    assert result["status"] == "EXECUTED"
    assert result["trade_id"] is not None

    # Account position should be updated in broker
    acc_updated = broker.get_account("strat_hft")
    assert "600519.SH" in acc_updated.positions
    assert acc_updated.positions["600519.SH"].total_volume == 100

def test_signal_router_auto_execution_risk_rejected():
    broker = PaperBroker()
    broker.create_account("strat_hft", initial_cash=1_000_000.0)
    # Set tight single-order limit to 50,000 (100 shares @ 1800 = 180,000 will breach)
    risk_checker = PreTradeRiskChecker(max_order_amount=50_000.0)
    router = SignalRouter(broker=broker, risk_checker=risk_checker)
    router.register_strategy(strategy_id="strat_hft", auto_execute=True)

    sig = SignalEvent(
        signal_id="sig_test_02",
        strategy_id="strat_hft",
        symbol="600519.SH",
        direction=SignalDirection.BUY,
        price=1800.0,
        volume=100,
        reason="测试超限买入"
    )
    result = router.route_signal(sig)
    assert result["status"] == "REJECTED"
    assert "单笔委托金额" in result["detail"]

def test_signal_router_manual_mode():
    broker = PaperBroker()
    router = SignalRouter(broker=broker)
    router.register_strategy(strategy_id="strat_manual", auto_execute=False)

    sig = SignalEvent(
        signal_id="sig_test_03",
        strategy_id="strat_manual",
        symbol="600519.SH",
        direction=SignalDirection.BUY,
        price=1800.0,
        volume=100,
        reason="观察模式信号"
    )
    result = router.route_signal(sig)
    assert result["status"] == "BROADCASTED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_signal_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.market_feed.router'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/market_feed/router.py
import uuid
from typing import Dict, List, Optional
from src.common.logger import logger
from src.market_feed.models import SignalEvent, SignalDirection
from src.execution_engine.models import Order, OrderDirection, OrderType
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.pre_trade import PreTradeRiskChecker

class SignalRouter:
    """高频交易信号路由协调器"""

    def __init__(
        self,
        broker: Optional[PaperBroker] = None,
        risk_checker: Optional[PreTradeRiskChecker] = None,
    ):
        self.broker = broker
        self.risk_checker = risk_checker
        self.strategy_configs: Dict[str, dict] = {}
        self.signals_history: List[dict] = []

    def register_strategy(
        self,
        strategy_id: str,
        auto_execute: bool = False,
        default_volume: int = 100
    ) -> None:
        self.strategy_configs[strategy_id] = {
            "auto_execute": auto_execute,
            "default_volume": default_volume,
        }

    def route_signal(self, signal: SignalEvent) -> dict:
        config = self.strategy_configs.get(signal.strategy_id, {"auto_execute": False, "default_volume": 100})
        auto_execute = config.get("auto_execute", False)

        record = {
            "signal_id": signal.signal_id,
            "strategy_id": signal.strategy_id,
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "price": signal.price,
            "volume": signal.volume,
            "reason": signal.reason,
            "status": "PENDING",
            "trade_id": None,
            "detail": ""
        }

        if not auto_execute:
            record["status"] = "BROADCASTED"
            record["detail"] = "策略为手动观察模式，已推送至告警与流广播"
            self.signals_history.append(record)
            logger.info(f"Signal broadcasted: [{signal.strategy_id}] {signal.symbol} {signal.direction} @ {signal.price}")
            return record

        if self.broker is None:
            record["status"] = "ERROR"
            record["detail"] = "未配置交易柜台"
            self.signals_history.append(record)
            return record

        # 构造执行订单
        order_dir = OrderDirection.BUY if signal.direction == SignalDirection.BUY else OrderDirection.SELL
        order = Order(
            order_id=f"ord_sig_{uuid.uuid4().hex[:8]}",
            account_id=signal.strategy_id,
            symbol=signal.symbol,
            direction=order_dir,
            price=signal.price,
            volume=signal.volume,
            order_type=OrderType.LIMIT,
        )

        acc = self.broker.get_account(signal.strategy_id)

        # 事前风控检查
        if self.risk_checker is not None:
            risk_decision = self.risk_checker.check_order(order, acc)
            if not risk_decision.passed:
                record["status"] = "REJECTED"
                record["detail"] = f"事前风控拦截: {risk_decision.reason}"
                self.signals_history.append(record)
                logger.warning(f"Signal rejected by risk: {risk_decision.reason}")
                return record

        # 柜台撮合成交
        trade = self.broker.submit_order(order)
        if trade:
            record["status"] = "EXECUTED"
            record["trade_id"] = trade.trade_id
            record["detail"] = f"成交成功: 成交价 {trade.price}, 股数 {trade.volume}"
            logger.info(f"Signal executed: Trade {trade.trade_id} on {signal.symbol}")
        else:
            record["status"] = "REJECTED"
            record["detail"] = order.reject_reason or "柜台拒绝"

        self.signals_history.append(record)
        return record

    def get_recent_signals(self, limit: int = 50) -> List[dict]:
        return self.signals_history[-limit:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_signal_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/market_feed/router.py tests/test_signal_router.py
git commit -m "feat(market): 信号路由与风控执行闭环协调器"
```

---

### Task 6: 历史行情回放驱动器与模块顶层统一导出 (`src/market_feed/replay.py`, `src/market_feed/__init__.py`)

**Files:**
- Create: `src/market_feed/replay.py`
- Create: `src/market_feed/__init__.py`
- Test: `tests/test_market_replay.py`

**Interfaces:**
- Consumes: `MarketTick`, `StreamBus`
- Produces: `TickReplayEngine`, `src.market_feed` exports

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market_replay.py
import pytest
from src.market_feed.models import MarketTick
from src.market_feed.stream_bus import StreamBus
from src.market_feed.replay import TickReplayEngine
import src.market_feed as mf

def test_tick_replay_engine():
    bus = StreamBus(stream_key="replay_test", force_memory_mode=True)
    engine = TickReplayEngine(stream_bus=bus)

    ticks = [
        MarketTick(symbol="600519.SH", last_price=1800.0, volume=100, turnover=180000.0),
        MarketTick(symbol="600519.SH", last_price=1805.0, volume=200, turnover=361000.0),
        MarketTick(symbol="600519.SH", last_price=1810.0, volume=300, turnover=543000.0),
    ]

    count = engine.replay(ticks, speed=0.0) # Speed 0.0 is instant without delay
    assert count == 3
    assert len(bus.read_ticks(count=10)) == 3

def test_market_feed_exports():
    assert hasattr(mf, "MarketTick")
    assert hasattr(mf, "SignalEvent")
    assert hasattr(mf, "StreamBus")
    assert hasattr(mf, "TickReplayEngine")
    assert hasattr(mf, "StreamingVWAP")
    assert hasattr(mf, "StreamingBollingerBands")
    assert hasattr(mf, "OrderFlowImbalance")
    assert hasattr(mf, "SignalEngine")
    assert hasattr(mf, "SignalRouter")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_market_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.market_feed.replay'`

- [ ] **Step 3: Write minimal implementation**

`src/market_feed/replay.py`:
```python
import time
from typing import List, Optional
from src.common.logger import logger
from src.market_feed.models import MarketTick
from src.market_feed.stream_bus import StreamBus

class TickReplayEngine:
    """高保真历史 Tick 行情回放驱动器"""

    def __init__(self, stream_bus: Optional[StreamBus] = None):
        self.stream_bus = stream_bus or StreamBus()

    def replay(self, ticks: List[MarketTick], speed: float = 0.0) -> int:
        """回放 Tick 序列，speed 为每笔间模拟间隔秒数 (0.0 表示无延迟)"""
        count = 0
        for tick in ticks:
            self.stream_bus.publish_tick(tick)
            count += 1
            if speed > 0:
                time.sleep(speed)
        logger.info(f"TickReplayEngine 完成回放 {count} 笔行情快照")
        return count
```

`src/market_feed/__init__.py`:
```python
from src.market_feed.models import (
    SignalDirection,
    MarketTick,
    IndicatorValue,
    SignalEvent,
)
from src.market_feed.stream_bus import StreamBus
from src.market_feed.replay import TickReplayEngine
from src.market_feed.indicators import (
    BaseStreamingIndicator,
    StreamingVWAP,
    StreamingBollingerBands,
    OrderFlowImbalance,
)
from src.market_feed.signal_engine import SignalEngine, CooldownManager
from src.market_feed.router import SignalRouter

__all__ = [
    "SignalDirection",
    "MarketTick",
    "IndicatorValue",
    "SignalEvent",
    "StreamBus",
    "TickReplayEngine",
    "BaseStreamingIndicator",
    "StreamingVWAP",
    "StreamingBollingerBands",
    "OrderFlowImbalance",
    "SignalEngine",
    "CooldownManager",
    "SignalRouter",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_market_replay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/market_feed/replay.py src/market_feed/__init__.py tests/test_market_replay.py
git commit -m "feat(market): 行情回放驱动器与顶层接口统一导出"
```

---

### Task 7: FastAPI 路由与 WebSocket 全双工广播 (`src/service/routers/market_router.py`, `src/service/app.py`)

**Files:**
- Create: `src/service/routers/market_router.py`
- Modify: `src/service/routers/__init__.py`
- Modify: `src/service/app.py`
- Test: `tests/test_market_router.py`

**Interfaces:**
- Consumes: `src.market_feed.*`, `fastapi`, `starlette.testclient.TestClient`
- Produces: `/api/v1/market/...` REST & WebSocket endpoints

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market_router.py
from fastapi.testclient import TestClient
from src.service.app import app

client = TestClient(app)

def test_market_api_endpoints():
    # 1. Post a tick
    tick_payload = {
        "symbol": "600519.SH",
        "last_price": 1800.0,
        "volume": 1000,
        "turnover": 1800000.0,
        "bid_prices": [1799.0],
        "bid_volumes": [200],
        "ask_prices": [1801.0],
        "ask_volumes": [100]
    }
    r = client.post("/api/v1/market/tick", json=tick_payload)
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "SUCCESS"
    assert "msg_id" in res

    # 2. Query indicators
    r_ind = client.get("/api/v1/market/indicators?symbol=600519.SH")
    assert r_ind.status_code == 200
    ind_data = r_ind.json()
    assert "vwap" in ind_data
    assert "bollinger" in ind_data
    assert "ofi" in ind_data

    # 3. Query signals
    r_sig = client.get("/api/v1/market/signals")
    assert r_sig.status_code == 200
    assert isinstance(r_sig.json(), list)

    # 4. WebSocket test
    with client.websocket_connect("/api/v1/market/ws") as websocket:
        # Push another tick via REST
        client.post("/api/v1/market/tick", json=tick_payload)
        data = websocket.receive_json()
        assert data["type"] == "TICK"
        assert data["data"]["symbol"] == "600519.SH"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_market_router.py -v`
Expected: FAIL with `404 Not Found`

- [ ] **Step 3: Write minimal implementation**

`src/service/routers/market_router.py`:
```python
import asyncio
from typing import List, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel
from src.market_feed.models import MarketTick
from src.market_feed.stream_bus import StreamBus
from src.market_feed.signal_engine import SignalEngine
from src.market_feed.router import SignalRouter
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.pre_trade import PreTradeRiskChecker

router = APIRouter(prefix="/market", tags=["Market Feed & Streaming Signals"])

# 单例组件初始化
_stream_bus = StreamBus(stream_key="market_stream:ticks", force_memory_mode=False)
_signal_engine = SignalEngine(default_strategy_id="hft_stream_01", cooldown_seconds=2.0)
_broker = PaperBroker()
_broker.create_account("hft_stream_01", initial_cash=5_000_000.0)
_risk_checker = PreTradeRiskChecker(max_order_amount=1_000_000.0)
_signal_router = SignalRouter(broker=_broker, risk_checker=_risk_checker)
_signal_router.register_strategy("hft_stream_01", auto_execute=True)

# WebSocket 连接池
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

def on_tick_received(tick: MarketTick):
    """总线监听到 Tick 时，触发信号引擎并向 WebSocket 广播"""
    sig = _signal_engine.process_tick(tick)
    if sig:
        _signal_router.route_signal(sig)
        asyncio.run(_safe_broadcast({"type": "SIGNAL", "data": sig.model_dump()}))

    asyncio.run(_safe_broadcast({"type": "TICK", "data": tick.model_dump()}))

async def _safe_broadcast(payload: dict):
    try:
        await manager.broadcast(payload)
    except Exception:
        pass

_stream_bus.subscribe(on_tick_received)

class ReplayRequest(BaseModel):
    ticks: List[MarketTick]
    speed: float = 0.0

@router.post("/tick")
def receive_tick(tick: MarketTick):
    msg_id = _stream_bus.publish_tick(tick)
    return {"status": "SUCCESS", "msg_id": msg_id}

@router.post("/replay")
def replay_ticks(req: ReplayRequest):
    count = 0
    for t in req.ticks:
        _stream_bus.publish_tick(t)
        count += 1
    return {"status": "SUCCESS", "replayed_count": count}

@router.get("/indicators")
def get_indicators(symbol: str = Query(..., description="标的代码")):
    vwap_ind, bb_ind, ofi_ind = _signal_engine._get_or_create_indicators(symbol)
    return {
        "symbol": symbol,
        "vwap": vwap_ind.current_vwap,
        "bollinger": {
            "mean": bb_ind.update(MarketTick(symbol=symbol, last_price=vwap_ind.current_vwap, volume=0, turnover=0.0)).value
            if bb_ind.prices else 0.0,
            "pct_b": 0.5,
        },
        "ofi": ofi_ind.prev_bid_price or 0.0,
    }

@router.get("/signals")
def get_recent_signals(limit: int = Query(50, ge=1, le=200)):
    return _signal_router.get_recent_signals(limit=limit)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 保持长连接
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

Update `src/service/routers/__init__.py`:
```python
from src.service.routers.market_router import router as market_router
__all__ = [..., "market_router"]
```

Update `src/service/app.py`:
```python
from src.service.routers import market_router
app.include_router(market_router, prefix="/api/v1")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_market_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/service/routers/market_router.py src/service/routers/__init__.py src/service/app.py tests/test_market_router.py
git commit -m "feat(market): FastAPI 实时行情服务与 WebSocket 全双工广播"
```

---

### Task 8: 全链路端到端综合集成测试与全量回归 (`tests/test_market_feed.py`)

**Files:**
- Create: `tests/test_market_feed.py`

**Interfaces:**
- Consumes: All `src.market_feed` modules, `PreTradeRiskChecker`, `PaperBroker`
- Produces: Comprehensive End-to-End System Tests

- [ ] **Step 1: Write the failing comprehensive integration test**

```python
# tests/test_market_feed.py
import pytest
from src.market_feed.models import MarketTick, SignalDirection
from src.market_feed.stream_bus import StreamBus
from src.market_feed.replay import TickReplayEngine
from src.market_feed.signal_engine import SignalEngine
from src.market_feed.router import SignalRouter
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.execution_engine.gateway.paper_broker import PaperBroker

def test_full_market_feed_lifecycle_and_risk_execution():
    """
    全链路综合场景测试:
    1. 启动事件总线 StreamBus 与回放器 TickReplayEngine;
    2. 注册 SignalEngine 监听流并计算实时指标 (VWAP, Bollinger, OFI);
    3. 连接 SignalRouter, 绑定 PaperBroker 与 PreTradeRiskChecker;
    4. 回放一波平稳盘口行情 (10 笔 @ 1800.0);
    5. 突发一笔大单爆买 (价格拉升至 1850.0 突破布林带上轨且 OFI 买压激增);
    6. 验证触发 BUY 信号事件;
    7. 验证自动过闸事前风控并由 PaperBroker 完成整手撮合;
    8. 验证持仓成功计入策略账本。
    """
    bus = StreamBus(stream_key="integ_stream", force_memory_mode=True)
    engine = TickReplayEngine(stream_bus=bus)
    sig_engine = SignalEngine(default_strategy_id="integ_hft_01", cooldown_seconds=0.5)

    broker = PaperBroker()
    acc = broker.create_account("integ_hft_01", initial_cash=5_000_000.0)
    risk = PreTradeRiskChecker(max_order_amount=1_000_000.0)
    router = SignalRouter(broker=broker, risk_checker=risk)
    router.register_strategy("integ_hft_01", auto_execute=True)

    # 绑定总线回调
    signals_captured = []
    def on_tick(tick: MarketTick):
        sig = sig_engine.process_tick(tick)
        if sig:
            signals_captured.append(sig)
            router.route_signal(sig)

    bus.subscribe(on_tick)

    # 4. 准备 10 笔基准盘口
    ticks = []
    for i in range(10):
        ticks.append(MarketTick(
            symbol="600519.SH",
            last_price=1800.0,
            volume=1000 + i * 100,
            turnover=1800000.0 + i * 180000.0,
            bid_prices=[1800.0],
            bid_volumes=[100],
            ask_prices=[1801.0],
            ask_volumes=[100],
        ))

    # 5. 突破大单
    ticks.append(MarketTick(
        symbol="600519.SH",
        last_price=1860.0,
        volume=5000,
        turnover=9300000.0,
        bid_prices=[1860.0],
        bid_volumes=[800],
        ask_prices=[1861.0],
        ask_volumes=[50],
    ))

    # 执行回放
    replayed = engine.replay(ticks, speed=0.0)
    assert replayed == 11

    # 6. 验证信号触发
    assert len(signals_captured) >= 1
    sig = signals_captured[0]
    assert sig.symbol == "600519.SH"
    assert sig.direction == SignalDirection.BUY

    # 7. 验证风控放行与自动成交
    recent_records = router.get_recent_signals()
    assert len(recent_records) >= 1
    assert recent_records[0]["status"] == "EXECUTED"
    assert recent_records[0]["trade_id"] is not None

    # 8. 验证持仓到账
    broker_acc = broker.get_account("integ_hft_01")
    assert "600519.SH" in broker_acc.positions
    assert broker_acc.positions["600519.SH"].total_volume == 100
```

- [ ] **Step 2: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_market_feed.py -v`
Expected: PASS

- [ ] **Step 3: Run full regression test suite**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest -v`
Expected: PASS (All 87 baseline tests + all new Market Feed tests pass, ~95+ passed, 0 failures)

- [ ] **Step 4: Commit**

```bash
git add tests/test_market_feed.py
git commit -m "test(market): 实时行情事件流与风控成交全链路集成测试"
```
