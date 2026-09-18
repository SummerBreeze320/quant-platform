# 实时行情驱动与事件流中枢（Market Event Stream Engine）设计规范

## 1. 概述与业务定位

### 1.1 业务背景
在现代量化投资与高频交易体系中，策略驱动机制从传统的**日频批处理（Batch Processing）**逐渐演进为**微秒/毫秒级事件驱动（Event-Driven Architecture）**：
- **实时 Tick 与盘口切片（L1/L2 Market Feed）**：以极高频率产生逐笔与快照数据，包含最新成交价、成交量、五档买卖委托挂单量与价格档位；
- **流式增量特征工程（Streaming Feature Engineering）**：在高频场景下，不能使用全量数组进行昂贵的历史回溯重算，必须采用严格的 $O(1)$ 复杂度在线递推算法，实时更新日内 VWAP、布林带动态通道以及盘口微观不平衡度（Order Flow Imbalance: OFI）；
- **微观信号触发与风控/执行自动化闭环**：当高频盘口或技术面突破信号触发时，支持策略级灵活配置——既可在监控大屏上实时推送预警，亦可直接无缝穿透风控中台（`PreTradeRiskChecker`）并由仿真交易柜台（`PaperBroker`）毫秒级自动撮合成交。

---

## 2. 系统拓扑与模块组织

### 2.1 目录结构
```text
src/market_feed/
├── __init__.py                # 模块统一对外导出声明
├── models.py                  # 核心数据模型 (MarketTick, OrderBook, SignalEvent, IndicatorValue)
├── stream_bus.py              # 基于 Redis Streams 的分布式事件分发总线 (含离线内存回退模式)
├── replay.py                  # 高保真历史 Tick/快照行情回放驱动器 (TickReplayEngine)
├── indicators/                # 流式增量指标算法库
│   ├── base.py                # BaseStreamingIndicator 抽象插件基类
│   ├── vwap.py                # 实时日内/滑动成交量加权均价 (StreamingVWAP)
│   ├── bollinger.py           # 实时布林带指标 (StreamingBollingerBands)
│   └── ofi.py                 # 买卖盘口微观不平衡度 (OrderFlowImbalance)
├── signal_engine.py           # 实时高频信号规则与微观形态触发器 (SignalEngine)
└── router.py                  # 信号分发与风控/执行协调器 (SignalRouter, 支持 auto_execute 开关)

src/service/routers/
└── market_router.py           # RESTful 状态/回放接口与 WebSocket 实时行情广播管道

tests/
└── test_market_feed.py        # 独立全量自动化集成测试
```

### 2.2 核心业务流程
```text
┌───────────────────────────────────────────────────────────┐
│        实时行情源推送 / 历史行情回放器 (TickReplayEngine)       │
└─────────────────────────────┬─────────────────────────────┘
                              │ MarketTick (L1/L2 快照)
                              ▼
┌───────────────────────────────────────────────────────────┐
│           Redis Streams 事件分发总线 (StreamBus)            │
│  - Stream 键名: "market_stream:ticks"                     │
│  - 消费者组: "quant_market_workers"                       │
│  - 广播至: WebSocket 管道 (前端/量化终端实时透视)              │
└──────────────┬─────────────────────────────┬──────────────┘
               │ 消费 Tick                   │ 消费 Tick
               ▼                             ▼
┌─────────────────────────────┐   ┌─────────────────────────┐
│   StreamingVWAP / Bollinger │   │   OrderFlowImbalance    │
│  (实时均线/布林带滑动计算)    │   │  (微观盘口买卖失衡指标)   │
└──────────────┬──────────────┘   └──────────┬──────────────┘
               │ 指标输出                    │ 指标输出
               └──────────────┬──────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│           实时高频信号触发器 (SignalEngine)                  │
│  - 布林带通道突破 (BollingerBreakout: %b > 1.0 且 > VWAP)   │
│  - 盘口买压突增 (OFISurge: OFI > 0.65 且放量)              │
│  - 高频防抖冷却机制 (CooldownManager: 3秒单标的冷却)        │
│  ──► 输出: SignalEvent (BUY/SELL, 建议股数, 置信度, 触发原因)│
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│              信号路由协调器 (SignalRouter)                  │
│                                                           │
│  [策略 auto_execute == True?]                             │
│     ├── YES:                                              │
│     │   1. 转换为整手 Order                                │
│     │   2. 调用 PreTradeRiskChecker 事前风控合规审查        │
│     │   3. 送入 PaperBroker / ExecutionCoordinator 撮合成交 │
│     │                                                     │
│     └── NO:                                               │
│         1. 记录信号流水                                    │
│         2. 调用 AlertManager 触发告警通知并在 WebSocket 广播 │
└───────────────────────────────────────────────────────────┘
```

---

## 3. 核心数据模型规范 (`models.py`)

### 3.1 枚举定义
```python
from enum import Enum

class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"
```

### 3.2 实体模型定义
1. **`MarketTick`（逐笔/快照行情切片）**：
   - `symbol: str`（标的代码，如 `"600519.SH"`）
   - `timestamp: str`（ISO-8601 时间戳）
   - `last_price: float`（最新成交价）
   - `volume: int`（当日累计成交股数）
   - `turnover: float`（当日累计成交金额）
   - `bid_prices: List[float]`（买一至买五价格列表）
   - `bid_volumes: List[int]`（买一至买五委托量列表）
   - `ask_prices: List[float]`（卖一至卖五价格列表）
   - `ask_volumes: List[int]`（卖一至卖五委托量列表）

2. **`IndicatorValue`（流式指标计算结果）**：
   - `symbol: str`
   - `indicator_name: str`（如 `"VWAP"`, `"BOLLINGER"`, `"OFI"`）
   - `value: float`（主指标值）
   - `details: Dict[str, float]`（衍生辅助值，如 `{"upper": 1820.0, "lower": 1780.0, "pct_b": 1.05}`）
   - `timestamp: str`

3. **`SignalEvent`（交易信号事件）**：
   - `signal_id: str`（唯一标识，UUID）
   - `strategy_id: str`（关联目标策略，如 `"csi300_alpha"`）
   - `symbol: str`
   - `direction: SignalDirection`
   - `price: float`（触发时行情最新价）
   - `volume: int = 100`（推荐执行数量，A股100股整手）
   - `strength: float = 1.0`（置信度 0.0 ~ 1.0）
   - `reason: str`（触发逻辑与描述）
   - `timestamp: str`

---

## 4. 分布式事件总线规范 (`stream_bus.py`)

1. **Redis Streams 驱动机制**：
   - 发布方法 `publish_tick(tick: MarketTick)`：调用 `client.xadd(stream_key, {"data": tick.model_dump_json()})`；
   - 消费组管理：自动创建 Consumer Group `quant_market_workers`；
   - 批量消费 `read_ticks(count: int = 10, block_ms: int = 100) -> List[MarketTick]`。
2. **纯内存测试/离线回退 (Memory Fallback)**：
   - 当本地 Redis 无法连接或不可用时（如单机离线单元测试），自动切换为进程内线程安全的 `queue.Queue` 或模拟环形缓冲区，对外暴露统一的 `publish` 与 `read` 接口，确保无外部依赖时测试 100% 通过。

---

## 5. 流式增量指标算法模型 (`indicators/`)

### 5.1 基础抽象插件 (`base.py`)
```python
from abc import ABC, abstractmethod
from src.market_feed.models import MarketTick, IndicatorValue

class BaseStreamingIndicator(ABC):
    @abstractmethod
    def update(self, tick: MarketTick) -> IndicatorValue:
        """接收单笔 Tick，以 O(1) 增量时间复杂度计算并输出最新指标值"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """日初或换日重置内部累计状态"""
        pass
```

### 5.2 实时 VWAP 指标 (`vwap.py`)
- **算法设计**：
  - 维护累计成交额增量 $S_{turnover}$ 与累计成交量增量 $S_{volume}$；
  - 每当收到第 $t$ 笔快照：
    $$\Delta Volume_t = \max(0, Volume_t - Volume_{t-1})$$
    $$\Delta Turnover_t = \max(0, Turnover_t - Turnover_{t-1})$$
    $$VWAP_t = \frac{\sum \Delta Turnover}{\sum \Delta Volume}$$
  - 计算偏离度 `bias = (last_price - vwap) / vwap`。

### 5.3 实时布林带指标 (`bollinger.py`)
- **算法设计**：
  - 滑动窗口长度 $N$（默认 20 个 Tick），标准差倍数 $k = 2.0$；
  - 采用定长双端队列 `collections.deque(maxlen=N)` 维护最新价格序列；
  - 增量更新均值 $\mu$ 与样本方差 $\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{N - 1}}$；
  - 输出中轨 $Mid = \mu$，上轨 $Upper = \mu + k\sigma$，下轨 $Lower = \mu - k\sigma$；
  - 输出相对百分比位置指标 `%b = \frac{Price - Lower}{Upper - Lower}`。

### 5.4 盘口订单流不平衡度 (`ofi.py`)
- **算法设计**：
  - 记录上一笔快照的买一价量 $(P_B^{t-1}, V_B^{t-1})$ 与卖一价量 $(P_A^{t-1}, V_A^{t-1})$；
  - 计算买方微观净增挂单量：
    $$\Delta W_B = \begin{cases} V_B^t, & P_B^t > P_B^{t-1} \\ V_B^t - V_B^{t-1}, & P_B^t = P_B^{t-1} \\ -V_B^{t-1}, & P_B^t < P_B^{t-1} \end{cases}$$
  - 计算卖方微观净增挂单量：
    $$\Delta W_A = \begin{cases} -V_A^{t-1}, & P_A^t > P_A^{t-1} \\ V_A^t - V_A^{t-1}, & P_A^t = P_A^{t-1} \\ V_A^t, & P_A^t < P_A^{t-1} \end{cases}$$
  - 盘口不平衡量 $OFI_t = \Delta W_B - \Delta W_A$；
  - 归一化至 $[-1.0, 1.0]$，正值代表微观买方动能强劲，负值代表卖方抛压显著。

---

## 6. 实时高频信号引擎规范 (`signal_engine.py`)

1. **信号触发规则**：
   - **布林带通道突破 (`BOLLINGER_BREAKOUT`)**：
     - 买入信号：`%b > 1.0`（突破上轨）且 `last_price > vwap`（处于日内 VWAP 之上强势区）；
     - 卖出信号：`%b < 0.0`（跌破下轨）且 `last_price < vwap`；
   - **盘口不平衡度脉冲 (`OFI_SURGE`)**：
     - 买入信号：标准化 $OFI > 0.65$ 且成交量较上一周期放大 $1.5$ 倍；
     - 卖出信号：标准化 $OFI < -0.65$ 且成交量放大；
2. **防抖与冷却管理 (`CooldownManager`)**：
   - 记录各标的上次触发时间 `Dict[str, float]`；
   - 默认冷却时间 3 秒，若冷却周期内收到同方向信号则自动平滑抑制，防止微观抖动引发高频刷单。

---

## 7. 信号路由与风控/仿真执行闭环 (`router.py`)

`SignalRouter` 协调信号走向：
1. **策略注册**：`register_strategy(strategy_id, auto_execute=False, default_volume=100)`；
2. **信号处理 `route_signal(signal: SignalEvent)`**：
   - 若策略配置 `auto_execute == True`：
     - 生成整手买卖订单 `Order(account_id=strategy_id, symbol=signal.symbol, direction=..., price=signal.price, volume=signal.volume)`；
     - 调用 `PreTradeRiskChecker.check_order` 执行事前风控校验（资金余额、涨跌停、限额）；
     - 若合规通过：调用 `PaperBroker.submit_order` 完成高保真撮合，计提税费并更新持仓；
     - 若合规被拒：记录拒绝原因；
   - 若策略配置 `auto_execute == False`：
     - 调用告警中台 `AlertManager.send_alert` 广播一般信号通知，留存流水供交易员核对；
   - 留存最新 500 笔信号历史供审计。

---

## 8. 历史行情回放驱动器 (`replay.py`)

`TickReplayEngine` 支持历史复盘与模拟推演：
- 接收 `List[MarketTick]` 历史切片序列；
- 支持回放倍速调节 `speed: float = 1.0`（1.0 实时、10.0 十倍速、0.0 极速离线）；
- 逐笔推入 `StreamBus` 驱动指标与信号引擎全链路运转。

---

## 9. FastAPI 服务路由与 WebSocket 广播 (`src/service/routers/market_router.py`)

| 路径 | 协议 | 描述 | 功能 |
| :--- | :--- | :--- | :--- |
| `/api/v1/market/tick` | POST | 接收单笔行情快照 | `MarketTick` 输入并推入总线 |
| `/api/v1/market/replay` | POST | 启动历史数据模拟回放 | 异步回放历史 Tick 序列 |
| `/api/v1/market/indicators` | GET | 查询指定标的最新流式指标 | 获取 VWAP, Bollinger, OFI 最新值 |
| `/api/v1/market/signals` | GET | 查询实时产生的交易信号记录 | 分页/列表审计 |
| `/api/v1/market/ws` | WebSocket | 全双工实时数据推送管道 | 前端实时订阅 Tick 与 Signal 事件 |

---

## 10. 验证与测试规范 (`tests/test_market_feed.py`)

构建自动化测试集，确保以下场景 100% 验证通过：
1. **`test_models_and_enums`**：验证数据模型约束、字段序列化与枚举；
2. **`test_stream_bus_pubsub`**：验证 Redis/内存回退事件总线的发布与批量拉取；
3. **`test_streaming_vwap_accuracy`**：验证增量 VWAP 与全量计算结果误差在 $0.001$ 以内；
4. **`test_streaming_bollinger_bands`**：验证滑动窗口均值、方差与 `%b` 相对轨线指标精度；
5. **`test_order_flow_imbalance`**：验证微观盘口挂单变动时 OFI 的正负方向与买卖压识别；
6. **`test_signal_engine_and_cooldown`**：验证布林带突破信号触发与 3 秒冷却抑制；
7. **`test_signal_router_auto_execution_and_risk_closed_loop`**：
   - 验证 `auto_execute=True` 时信号自动转换为订单、经由 `PreTradeRiskChecker` 过闸并由 `PaperBroker` 成功记账；
   - 验证 `auto_execute=False` 时仅产生告警流水；
8. **`test_market_api_and_websocket`**：验证 RESTful 接口与 WebSocket 实时广播；
9. **全量回归验证**：确保既有 87 项测试全部通过，系统全量测试总量达到约 95+ 项且 100% 通过。
