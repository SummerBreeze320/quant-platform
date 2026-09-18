import uuid
from typing import Optional, Dict, Any, List
from src.execution_engine.models import (
    Order, Trade, Position, AccountState, OrderDirection, OrderType, OrderStatus
)
from src.execution_engine.gateway.base import BaseBrokerGateway
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.common.logger import logger


class QmtBrokerGateway(BaseBrokerGateway):
    """迅投 QMT / XtQuant 券商柜台适配器。
    
    支持与实际 MiniQMT 进程通信；在未安装 xtquant 或未配置客户端路径的环境下，
    自动平滑切换至高保真模拟柜台模式（Mock/Simulation Mode），保证系统全天候可用性。
    """

    def __init__(
        self,
        mini_qmt_path: Optional[str] = None,
        account_id: str = "default_stock",
        session_id: int = 123456,
        mock_mode: bool = False,
    ):
        self.mini_qmt_path = mini_qmt_path
        self.account_id = account_id
        self.session_id = session_id
        self.mock_mode = mock_mode
        self.is_connected = False

        self._trader = None
        self._stock_account = None
        self._mock_broker: Optional[PaperBroker] = None

        # 始终初始化一个内部 fallback 模拟柜台
        self._mock_broker = PaperBroker()

        # 尝试自适应连接
        self.connect()

    def connect(self) -> bool:
        """连接 MiniQMT 客户端或切入模拟模式"""
        if self.mock_mode:
            self.is_connected = True
            logger.info(f"[QmtGateway] Running in explicit mock simulation mode (account={self.account_id}).")
            return True

        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount

            if not self.mini_qmt_path:
                logger.info("[QmtGateway] mini_qmt_path not specified; activating high-fidelity mock broker.")
                self.mock_mode = True
                self.is_connected = True
                return True

            self._trader = XtQuantTrader(self.mini_qmt_path, self.session_id)
            self._trader.start()
            res = self._trader.connect()
            if res == 0:
                self._stock_account = StockAccount(self.account_id)
                self._trader.subscribe(self._stock_account)
                self.is_connected = True
                logger.info(f"[QmtGateway] Successfully connected to live MiniQMT at {self.mini_qmt_path}.")
                return True
            else:
                logger.warning(f"[QmtGateway] MiniQMT connect returned {res}; falling back to simulation mode.")
                self.mock_mode = True
                self.is_connected = True
                return True
        except ImportError:
            logger.info("[QmtGateway] xtquant package not installed; falling back to simulation mode.")
            self.mock_mode = True
            self.is_connected = True
            return True
        except Exception as e:
            logger.warning(f"[QmtGateway] Failed to connect to MiniQMT: {e}; falling back to simulation mode.")
            self.mock_mode = True
            self.is_connected = True
            return True

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """标准化 A 股证券代码格式，如 '600000.SH', '000001.SZ'"""
        sym = symbol.strip().upper()
        if sym.startswith(("SH", "SZ")):
            market = sym[:2]
            code = sym[2:]
            return f"{code}.{market}"
        return sym

    def submit_order(self, order: Order) -> Optional[Trade]:
        """向柜台提交委托订单"""
        if not self.is_connected:
            self.connect()

        norm_sym = self.normalize_symbol(order.symbol)
        order.symbol = norm_sym

        if self.mock_mode or self._trader is None:
            # 仿真模式下委托由底层 PaperBroker 立即执行撮合
            return self._mock_broker.submit_order(order)

        # 真实 QMT 柜台报单
        try:
            import xtquant.xtconstant as xtconstant

            direction = (
                xtconstant.STOCK_BUY
                if order.direction == OrderDirection.BUY
                else xtconstant.STOCK_SELL
            )
            price_type = (
                xtconstant.FIX_PRICE
                if order.order_type == OrderType.LIMIT
                else xtconstant.LATEST_PRICE
            )

            qmt_order_id = self._trader.order_stock(
                account=self._stock_account,
                stock_code=norm_sym,
                order_type=direction,
                order_volume=order.volume,
                price_type=price_type,
                price=order.price,
                strategy_name="QuantCopliot",
                order_remark=order.order_id,
            )

            if qmt_order_id < 0:
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"QMT 柜台拒绝报单 (error_code={qmt_order_id})"
                return None

            order.status = OrderStatus.PENDING
            logger.info(f"[QmtGateway] Submitted order {order.order_id} to QMT (qmt_order_id={qmt_order_id})")
            return None
        except Exception as e:
            logger.error(f"[QmtGateway] Order submission error: {e}")
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """撤销未成交委托"""
        if self.mock_mode or self._trader is None:
            return self._mock_broker.cancel_order(order_id)
        try:
            # 在实盘下调用 cancel_order_stock
            res = self._trader.cancel_order_stock_sysid(self._stock_account, order_id)
            return res == 0
        except Exception as e:
            logger.warning(f"[QmtGateway] Cancel order error: {e}")
            return False

    def get_account(self, account_id: str) -> AccountState:
        """获取账户资金与持仓快照"""
        if self.mock_mode or self._trader is None:
            return self._mock_broker.get_account(account_id)

        try:
            asset = self._trader.query_stock_asset(self._stock_account)
            positions = self._trader.query_stock_positions(self._stock_account)

            pos_dict: Dict[str, Position] = {}
            if positions:
                for p in positions:
                    sym = p.stock_code
                    pos_dict[sym] = Position(
                        symbol=sym,
                        total_volume=p.volume,
                        available_volume=p.can_use_volume,
                        frozen_volume=p.volume - p.can_use_volume,
                        avg_cost=float(p.open_price),
                        market_value=float(p.market_value),
                    )

            total_equity = float(asset.total_asset) if asset else 0.0
            cash = float(asset.cash) if asset else 0.0

            return AccountState(
                account_id=account_id,
                total_equity=total_equity,
                available_cash=cash,
                positions=pos_dict,
            )
        except Exception as e:
            logger.warning(f"[QmtGateway] query account error: {e}; falling back to mock state.")
            return self._mock_broker.get_account(account_id)

    def update_market_price(self, symbol: str, price: float) -> List[AccountState]:
        """更新标的最新行情价格并重新估值持仓市值与浮动盈亏"""
        # 统一由内部 broker 或 mock_broker 计算
        return self._mock_broker.update_market_price(symbol, price)

    def get_all_accounts(self) -> List[AccountState]:
        """获取所有账户状态"""
        if self.mock_mode or self._trader is None:
            return self._mock_broker.get_all_accounts()
        return [self.get_account(self.account_id)]
