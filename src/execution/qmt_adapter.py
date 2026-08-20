"""QMT（迅投）交易接口适配器

注意: 需要安装QMT客户端并开通交易权限。
QMT Python SDK通过 xtquant 接口下单。
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from xtquant import xttrader, xtdata
    QMT_AVAILABLE = True
except ImportError:
    QMT_AVAILABLE = False
    logger.info("xtquant not installed. QMT execution disabled.")


class QMTAdapter:
    """QMT交易接口"""

    def __init__(self, path: str = "", session_id: int = 1):
        if not QMT_AVAILABLE:
            raise RuntimeError("xtquant not available. Install QMT Python SDK.")

        self.path = path
        self.session_id = session_id
        self.trader = None

    def connect(self) -> bool:
        """连接QMT"""
        if not QMT_AVAILABLE:
            return False

        self.trader = xttrader.XtQuantTrader(self.path, self.session_id)
        self.trader.start()

        account = xttrader.StockAccount("YOUR_ACCOUNT_ID")
        connected = self.trader.connect()

        if connected:
            self.trader.subscribe(account)
            logger.info("QMT connected successfully.")
            return True
        else:
            logger.error("QMT connection failed.")
            return False

    def buy(self, code: str, price: float, volume: int) -> str:
        """买入"""
        if not self.trader:
            raise RuntimeError("QMT not connected.")

        order_id = self.trader.order_stock(
            account=xttrader.StockAccount("YOUR_ACCOUNT_ID"),
            order_type=xttrader.STOCK_BUY,
            stock_code=code,
            price=price,
            volume=volume,
            strategy_name="quant_platform",
        )
        logger.info(f"Buy order: {code} @ {price} x {volume}, order_id={order_id}")
        return str(order_id)

    def sell(self, code: str, price: float, volume: int) -> str:
        """卖出"""
        if not self.trader:
            raise RuntimeError("QMT not connected.")

        order_id = self.trader.order_stock(
            account=xttrader.StockAccount("YOUR_ACCOUNT_ID"),
            order_type=xttrader.STOCK_SELL,
            stock_code=code,
            price=price,
            volume=volume,
            strategy_name="quant_platform",
        )
        logger.info(f"Sell order: {code} @ {price} x {volume}, order_id={order_id}")
        return str(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if not self.trader:
            return False
        return self.trader.cancel_order_stock(
            xttrader.StockAccount("YOUR_ACCOUNT_ID"),
            int(order_id),
        )

    def get_position(self) -> list:
        """查询持仓"""
        if not self.trader:
            return []
        return self.trader.query_stock_positions(
            xttrader.StockAccount("YOUR_ACCOUNT_ID")
        )
