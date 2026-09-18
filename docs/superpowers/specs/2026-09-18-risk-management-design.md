# 量化风控中台与动态熔断系统（Risk Engine & Circuit Breaker）设计规范

## 1. 概述与设计目标

### 1.1 背景与定位
在生产级量化投研与实盘交易体系中，阿尔法策略与因子挖掘负责“进攻”，而风控中台（Risk Engine）负责“防守与合规底线”。任何策略模型的信号输出在进入订单执行系统（OMS/EMS）前，必须经过严格的**事前合规与资金拦截校验（Pre-Trade Risk Checks）**；在持仓与实盘交易过程中，必须有**事中/事后组合风险监控（Post-Trade Risk Monitoring）**以及**日内动态阶梯熔断（Circuit Breaker）**，以防止算法异常、模型过拟合、极端市场黑天鹅或乌龙指等造成的不可控净值崩塌。

本项目旨在为 QuantCopilot 打造一个高性能、纯 Python、生产级量化风控中台系统：
1. **高性能纯内存检查**：微秒级/毫秒级低延迟裁决，满足高频回测与实时委托拦截需求。
2. **可选分布式同步**：熔断状态与账户高水位支持可选同步至 Redis 缓存（遵循平台原有的优雅降级规范，无 Redis 时自动回退纯内存，绝不产生硬依赖）。
3. **插件化规则管道**：基于开闭原则（OCP）与单一职责（SRP），每条风控规则解耦封装，支持动态启闭、阈值调参与快速失败（Fail-Fast）模式。
4. **三级动态梯级熔断机**：根据日内动态回撤自动跃迁（预警 -> 禁开新仓 -> 全局终止），支持手动复位与跨日重置。
5. **多通道告警与 RESTful 服务**：提供环形内存历史队列、结构化日志、飞书/钉钉 Webhook 格式适配器，以及标准化 FastAPI 接口。

---

## 2. 系统架构与模块组织

### 2.1 模块目录结构
```text
src/risk_engine/
├── __init__.py                # 导出核心实体与管理类
├── models.py                  # Pydantic 校验模型与枚举定义
├── rules.py                   # BaseRiskRule 抽象基类与 6 大预置规则
├── pre_trade.py               # 事前风控协调器 PreTradeRiskChecker
├── post_trade.py              # 事后组合监控器 PostTradeRiskMonitor
├── circuit_breaker.py         # 三级熔断有限状态机 CircuitBreakerManager
└── alert.py                   # 告警事件分发与环形队列 RiskAlertManager

src/service/routers/
└── risk_router.py             # FastAPI RESTful 路由 (/api/v1/risk/...)

tests/
└── test_risk_management.py   # 独立风控自动化单元测试集
```

### 2.2 核心数据流图
```text
┌────────────────────────────────────────────────────────────────────────┐
│                        交易客户端 / 回测引擎 / API                      │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
        (1) 单笔/批量委托订单                         │ (2) 实时组合持仓快照
                    ▼                                ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────┐
│       PreTradeRiskChecker            │ │      PostTradeRiskMonitor     │
│   (事前风控规则管道: 串行/Fail-Fast)   │ │  (日内回撤/集中度/Barra风格漂移)│
├──────────────────────────────────────┤ └───────────────┬───────────────┘
│  - BlacklistRule (黑名单禁投)         │                 │
│  - MaxOrderAmountRule (单笔限额)      │                 │ 动态回撤率更新
│  - MaxStockWeightRule (持仓集中度)    │                 ▼
│  - AdvLiquidityRule (流动性/ADV占比) │ ┌───────────────────────────────┐
│  - CashSufficiencyRule (可用资金充足)│ │    CircuitBreakerManager      │
│  - CircuitBreakerRule (熔断状态拦截) │ │  (Normal/Yellow/Orange/Red)   │
└───────────────────┬──────────────────┘ └───────────────┬───────────────┘
                    │                                    │
           违规/预警/熔断事件                             │ 触发熔断升级
                    ▼                                    │
┌────────────────────────────────────────────────────────▼───────────────┐
│                         RiskAlertManager                               │
│  - 内存环形队列 (deque 200条)                                            │
│  - 结构化日志 (src.common.logger)                                       │
│  - Webhook 格式适配器 (通用JSON / 飞书富文本 / 钉钉Markdown)              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模型与枚举规范 (`models.py`)

### 3.1 枚举定义
```python
from enum import Enum, IntEnum

class RiskAction(str, Enum):
    PASS = "PASS"        # 检查通过，无合规隐患
    WARN = "WARN"        # 触发软性预警，允许执行但记录告警
    REJECT = "REJECT"    # 触发硬性拦截，禁止执行

class CircuitBreakerLevel(IntEnum):
    NORMAL = 0               # 正常交易
    YELLOW_WARN = 1          # 一级预警 (日内回撤 >= 1.5%)
    ORANGE_RESTRICT_BUY = 2  # 二级限制 (日内回撤 >= 2.5%，禁开新仓，仅允许卖出)
    RED_HALT = 3             # 三级熔断 (日内回撤 >= 3.5%，全局冻结买卖)

class AlertLevel(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
```

### 3.2 契约数据模型
1. **`OrderRiskRequest`**：
   - `account_id: str = "default"`
   - `order_id: Optional[str] = None`
   - `symbol: str`（标的代码，如 "600519.SH"）
   - `direction: str`（"BUY" 或 "SELL"）
   - `price: float`（委托单价，必须大于 0）
   - `volume: int`（委托股数，必须大于 0）
   - `current_position: int = 0`（该标的当前持仓股数）
   - `total_equity: float`（账户当前总资产净值）
   - `available_cash: float`（账户当前可用现金）
   - `adv_20: Optional[float] = None`（过去 20 日日均成交量，单位：股）
   - `timestamp: Optional[str] = None`

2. **`OrderRiskResult`**：
   - `passed: bool`（若动作为 PASS 或 WARN 则为 True，REJECT 则为 False）
   - `action: RiskAction`
   - `violations: List[str]`（硬性违规原因列表）
   - `warnings: List[str]`（软性预警描述列表）
   - `metrics: Dict[str, Any]`（检查计算过程中的各项派生指标，如 order_amount, post_weight, adv_ratio 等）

3. **`PortfolioRiskRequest`**：
   - `account_id: str = "default"`
   - `total_equity: float`
   - `positions: Dict[str, float]`（各标的市值字典，如 `{"600519.SH": 200000.0, ...}`）
   - `high_watermark: Optional[float] = None`（历史/日内最高净值，若不传则由系统内部维护状态决定）
   - `benchmark_weights: Optional[Dict[str, float]] = None`（基准指数成分权重，用于计算跟踪误差）
   - `factor_exposures: Optional[Dict[str, Dict[str, float]]] = None`（标的因子暴露矩阵）

4. **`PortfolioRiskResult`**：
   - `account_id: str`
   - `current_equity: float`
   - `high_watermark: float`
   - `drawdown: float`（当前相对最高水位的回撤比例，如 0.021 表示 2.1%）
   - `circuit_breaker_level: CircuitBreakerLevel`
   - `max_stock_weight: float`（最大单股集中度）
   - `top5_weight: float`（前 5 大重仓股合计权重）
   - `leverage_ratio: float`（持仓市值与总权益比率）
   - `active_risk: Optional[Dict[str, float]] = None`（由 Barra 模型计算的跟踪误差与特质风险）
   - `alerts: List[str]`

5. **`CircuitBreakerState`**：
   - `account_id: str`
   - `level: CircuitBreakerLevel`
   - `high_watermark: float`
   - `current_equity: float`
   - `max_drawdown: float`
   - `last_triggered_at: Optional[str]`
   - `trigger_reason: Optional[str]`

6. **`RiskAlertEvent`**：
   - `event_id: str`
   - `timestamp: str`
   - `level: AlertLevel`
   - `account_id: str`
   - `rule_name: str`
   - `message: str`
   - `metadata: Dict[str, Any]`

---

## 4. 详细模块设计与实现规范

### 4.1 规则抽象与开箱即用规则集 (`rules.py`)

#### 4.1.1 规则基类 `BaseRiskRule`
```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

class BaseRiskRule(ABC):
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def check(self, req: OrderRiskRequest, ctx: Dict[str, Any]) -> Tuple[RiskAction, Optional[str]]:
        pass
```

#### 4.1.2 预置规则实现规范
1. **`BlacklistRule`**：
   - **配置**：`blacklist: Set[str]`
   - **逻辑**：若 `req.symbol` 存在于黑名单中，返回 `(RiskAction.REJECT, f"标的 {req.symbol} 命中合规禁投黑名单")`。
2. **`MaxOrderAmountRule`**：
   - **配置**：`max_order_amount: float = 1_000_000.0`
   - **逻辑**：计算 `order_amount = req.price * req.volume`。若 `order_amount > max_order_amount`，返回 `(RiskAction.REJECT, f"委托金额 {order_amount:.2f} 超过单笔限额 {max_order_amount:.2f}")`。
3. **`MaxStockWeightRule`**：
   - **配置**：`max_stock_weight: float = 0.10`（单股占比默认不超 10%）
   - **逻辑**：仅针对买入订单。预估下单后市值 `post_mv = (req.current_position + req.volume) * req.price`。若 `post_mv / req.total_equity > max_stock_weight`，返回 `(RiskAction.REJECT, f"买入后持仓权重 {post_weight:.2%} 超过个股集中度限制 {max_stock_weight:.2%}")`。
4. **`AdvLiquidityRule`**：
   - **配置**：`max_adv_ratio: float = 0.10`（委托量不超 ADV20 的 10%）
   - **逻辑**：若提供了 `req.adv_20` 且 `req.adv_20 > 0`：计算 `adv_ratio = req.volume / req.adv_20`。若 `adv_ratio > max_adv_ratio`，返回 `(RiskAction.REJECT, f"委托数量占过去20日日均成交量比例 {adv_ratio:.2%} 超过冲击成本上限 {max_adv_ratio:.2%}")`。
5. **`CashSufficiencyRule`**：
   - **配置**：`fee_buffer_rate: float = 0.002`（预留千分之二交易佣金与杂费缓冲）
   - **逻辑**：针对买入订单。若 `req.price * req.volume * (1 + fee_buffer_rate) > req.available_cash`，返回 `(RiskAction.REJECT, "可用现金不足以支付委托金额及预估交易费用")`。
6. **`CircuitBreakerRule`**：
   - **关联**：引用 `CircuitBreakerManager`；
   - **逻辑**：查询 `req.account_id` 当前熔断级别：
     - 若处于 `RED_HALT`：无论买卖一律拦截，返回 `(RiskAction.REJECT, f"账户处于 RED_HALT 紧急熔断状态，禁止一切交易")`；
     - 若处于 `ORANGE_RESTRICT_BUY` 且 `req.direction == "BUY"`：返回 `(RiskAction.REJECT, f"账户处于 ORANGE 限制买入熔断状态，禁止新开仓/加仓")`；
     - 处于 `ORANGE` 且 `req.direction == "SELL"` 时放行；
     - 其余正常放行。

---

### 4.2 事前风控检查器 (`pre_trade.py`)

`PreTradeRiskChecker` 是整个事前拦截管道的总入口：
1. 维护规则注册表 `self.rules: List[BaseRiskRule]`；
2. 包含黑名单快捷管理方法 `add_blacklist(symbol)`、`remove_blacklist(symbol)`、`get_blacklist()`；
3. 执行方法 `check_order(request: OrderRiskRequest, fail_fast: bool = False) -> OrderRiskResult`：
   - 顺序遍历启用的规则；
   - 遇到 `RiskAction.REJECT`：
     - 添加原因至 `violations`；
     - 产生 `AlertLevel.CRITICAL` 告警事件投递至 `RiskAlertManager`；
     - 若 `fail_fast=True`，立即终止并返回结果；
   - 遇到 `RiskAction.WARN`：
     - 添加原因至 `warnings`；
     - 产生 `AlertLevel.WARN` 告警事件投递至 `RiskAlertManager`；
   - 最终判定：若 `violations` 为空且 `warnings` 为空，动作为 `PASS`，`passed=True`；若仅有 `warnings`，动作为 `WARN`，`passed=True`；若存在 `violations`，动作为 `REJECT`，`passed=False`。

---

### 4.3 动态阶梯熔断管理器 (`circuit_breaker.py`)

#### 4.3.1 状态转移与阶梯设定
默认阶梯阈值：
- `warn_drawdown = 0.015` (1.5%) -> `YELLOW_WARN`
- `restrict_buy_drawdown = 0.025` (2.5%) -> `ORANGE_RESTRICT_BUY`
- `halt_drawdown = 0.035` (3.5%) -> `RED_HALT`

#### 4.3.2 动态更新逻辑
- `update_equity(account_id: str, equity: float, high_watermark: Optional[float] = None) -> CircuitBreakerState`
  - 若 `high_watermark` 未传，则比较 `equity` 与历史记录的最大值更新 `high_watermark`；
  - 计算回撤 $dd = \frac{HW - equity}{HW}$；
  - 依照阶梯阈值计算应达到的最高熔断等级；
  - **单向熔断保护机制**：日内净值反弹时，未经风控管理员手动复位（Manual Reset），熔断等级**不可自动降级**（即触发 RED_HALT 后，即便尾盘反弹回撤缩小至 2%，仍保持 RED_HALT，必须由人工审核介入）。

#### 4.3.3 复位机制
- `reset(account_id: str, reset_watermark: bool = True) -> CircuitBreakerState`：将状态置为 `NORMAL`，若 `reset_watermark=True` 则将最高水位重置为当前最新净值。
- `daily_reset()`：所有账户状态重置为 `NORMAL`，迎接下一交易日。

#### 4.3.4 存储与降级
- 内部采用线程安全字典 `Dict[str, CircuitBreakerState]`；
- 若检测到全局 `get_redis_client()` 可用，在每次状态发生升级/复位时同步写入 `quant:risk:cb:{account_id}`，实现跨 Worker 进程共享；若 Redis 连接不可用或异常，透明回退纯内存，无日志刷屏与异常抛出。

---

### 4.4 事后组合监控器 (`post_trade.py`)

`PostTradeRiskMonitor` 负责组合全局健康度计算与 Barra 风格漂移监控：
1. 计算基本穿透指标：
   - 最大持仓个股代码及权重；
   - 前 5 大持仓累计权重合计（Top-5 Concentration）；
   - 组合总杠杆率（总持仓市值 / 账户总权益）；
2. 联动更新熔断机：
   - 将净值更新输入 `CircuitBreakerManager`，获取最新熔断等级；
3. 联动 Barra 风险模型 (`src/qlib_engine/risk_model.py`)：
   - 当请求中包含标的列表、基准权重以及外部预估的因子暴露时，调用 `BarraRiskModel.predict_portfolio_risk()` 计算预期跟踪误差（Tracking Error）与特质风险；
   - 若跟踪误差超过阈值（如年化 15%），产生预警事件。

---

### 4.5 告警管理器与分发器 (`alert.py`)

1. **`RiskAlertManager`**：
   - 内部使用 `collections.deque(maxlen=200)` 维护固定大小的环形历史队列；
   - `record_alert(event: RiskAlertEvent)`：
     - 追加至队列；
     - 记录至标准日志：`logger.warning` 或 `logger.error`；
     - 若配置了 `webhook_url`，调用适配器推送通知。
2. **多平台 Webhook 适配器**：
   - **`GenericWebhookAdapter`**：发送标准 JSON payload；
   - **`FeishuWebhookAdapter`**：转换为飞书机器人富文本交互卡片消息（包含风险等级红/橙/黄色块、违规规则、发生时间与具体信息）；
   - **`DingTalkWebhookAdapter`**：转换为钉钉 markdown 消息；
   - **容错保证**：HTTP 请求采用短超时（如 2 秒），任何网络或服务不可达异常被静默捕获并记 debug 日志，确保永远不会逆向阻塞主交易业务。

---

## 5. FastAPI 服务路由与接口规范 (`src/service/routers/risk_router.py`)

在 `src/service/app.py` 中挂载：`app.include_router(risk_router, prefix="/api/v1/risk", tags=["Risk Management"])`。

### 5.1 接口列表定义
| 路径 | 方法 | 描述 | 请求体/参数 |
| :--- | :--- | :--- | :--- |
| `/api/v1/risk/check_order` | POST | 事前订单合规与风险校验 | `OrderRiskRequest` (JSON) |
| `/api/v1/risk/check_portfolio` | POST | 组合持仓多维风控与回撤扫描 | `PortfolioRiskRequest` (JSON) |
| `/api/v1/risk/circuit_breaker/status` | GET | 查询账户熔断机当前状态 | Query param: `account_id` (可选) |
| `/api/v1/risk/circuit_breaker/reset` | POST | 人工复位指定账户熔断状态 | `{"account_id": "default", "reset_watermark": true}` |
| `/api/v1/risk/alerts` | GET | 查询内存环形历史告警记录 | Query params: `limit: int = 50`, `level: Optional[str]` |
| `/api/v1/risk/blacklist` | POST | 管理禁投股票黑名单 | `{"action": "add"|"remove"|"list", "symbols": ["600519.SH"]}` |
| `/api/v1/risk/rules/config` | POST | 动态调整风控规则开关与阈值 | `{"rule_name": "MaxOrderAmountRule", "max_order_amount": 2000000.0, "enabled": true}` |

---

## 6. 验证与测试规范 (`tests/test_risk_management.py`)

使用 pytest 编写完备的自动化测试集，确保以下关键测试用例 100% 覆盖并通过：
1. **`test_blacklist_rule`**：验证标的命中与未命中黑名单的动作（REJECT vs PASS）。
2. **`test_order_amount_rule`**：验证委托金额在界限上下（如 99.9万 vs 100.1万）的拦截行为。
3. **`test_max_stock_weight_rule`**：验证买入加仓导致仓位突破 10% 时的精准拦截。
4. **`test_adv_liquidity_rule`**：验证委托量超过 ADV20 10% 时的拦截，以及缺失 ADV 时的容错。
5. **`test_cash_sufficiency_rule`**：验证可用资金不足与充足场景的对比判断。
6. **`test_circuit_breaker_transitions`**：
   - 验证随着回撤从 1.0% -> 1.8% (YELLOW) -> 2.8% (ORANGE) -> 3.8% (RED) 的阶梯状态跃迁；
   - 验证净值回弹时的不可自动降级特性；
   - 验证调用 `reset()` 后的成功复位与最高水位同步校准。
7. **`test_circuit_breaker_order_interception`**：
   - 验证 RED 状态下买单与卖单均被拒绝；
   - 验证 ORANGE 状态下买单被拒绝、卖单允许通过；
   - 验证 NORMAL 状态下正常执行。
8. **`test_post_trade_monitor`**：验证组合集中度、Top5 汇总计算以及与 `BarraRiskModel` 的无缝协同。
9. **`test_alert_manager_ring_buffer`**：验证环形队列的最大容量限制（超出 200 自动挤出旧记录）与等级过滤。
10. **`test_fastapi_risk_endpoints`**：使用 `starlette.testclient.TestClient` 验证 `/api/v1/risk/*` 全部 7 个接口的 HTTP 交互与状态码。
11. **全量回归验证**：确保原有的 53 项自动化测试完整通过，无回归破坏。

---

## 7. 规范自审清单 (Spec Self-Review)
- [x] **无占位符**：全文无 TBD、TODO 或未定义字段。
- [x] **内部一致性**：模型字段、规则判定逻辑与 API 接口完全对应一致。
- [x] **架构聚焦与可行性**：专注纯 Python、高性能、易测且低外部依赖的风控核心体系。
- [x] **确定性与无歧义**：明确了单向熔断不可自动降级、Fail-Fast 行为、资金缓冲率、黑名单及各级阶梯阈值。
