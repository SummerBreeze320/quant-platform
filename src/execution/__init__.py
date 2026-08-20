"""实盘执行模块

七层交易系统架构：
1. 信号生成层 (signal_generator / qlib_signal_generator): 模型预测 → 目标持仓
2. 组合调度层 (portfolio_scheduler): 对比持仓，生成调仓指令
3. 交易网关层 (trading_gateway): 对接券商API，报单撤单
4. 执行算法层 (execution_algo): VWAP/TWAP拆单算法
5. 监控告警层 (monitor): 持仓偏离、风控、行情中断告警
6. 日志持久层 (trade_logger): 全链路日志记录
7. 统一风控层 (unified_risk): 贯穿全链路的风控检查
"""
from .risk_control import RiskConfig, RiskController
from .risk_manager import (
    RiskLimits,
    PreTradeRiskChecker,
    IntraTradeRiskMonitor,
    PostTradeRiskAnalyzer,
)
from .unified_risk import UnifiedRiskConfig, UnifiedRiskController
from .trade_logger import (
    TradeLogger,
    SignalRecord,
    OrderRecord,
    TradeRecord,
    AlertRecord,
)
from .trading_gateway import (
    BrokerBase,
    SimulatedBroker,
    OrderManager,
    Order,
    OrderStatus,
)
from .execution_algo import (
    ExecutionAlgo,
    TWAPAlgo,
    VWAPAlgo,
    IcebergAlgo,
    ImplementationShortfallAlgo,
    AlgoConfig,
    create_algo,
)
from .signal_generator import SignalGenerator
from .qlib_signal_generator import QlibSignalGenerator
from .portfolio_scheduler import PortfolioScheduler, RebalanceOrder
from .monitor import TradingMonitor, AlertRule
