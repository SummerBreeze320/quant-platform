"""风控检查"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """风控配置"""
    max_position_pct: float = 0.10    # 单票最大仓位10%
    max_sector_pct: float = 0.30     # 单行业最大仓位30%
    max_total_position: float = 0.95  # 最大总仓位95%
    daily_stop_loss: float = -0.03    # 日止损-3%
    max_turnover: float = 0.50       # 单日换手率上限50%


class RiskController:
    """风控检查器"""

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    def check_order(
        self,
        code: str,
        side: str,
        price: float,
        volume: int,
        current_positions: dict,
        total_capital: float,
    ) -> tuple[bool, str]:
        """检查订单是否通过风控

        Returns:
            (passed, reason)
        """
        order_value = price * volume

        # 单票仓位检查
        current_value = current_positions.get(code, {}).get("value", 0)
        new_pct = (current_value + order_value * (1 if side == "buy" else -1)) / total_capital
        if side == "buy" and new_pct > self.config.max_position_pct:
            return False, f"单票仓位超限: {new_pct:.1%} > {self.config.max_position_pct:.1%}"

        # 总仓位检查
        total_position_value = sum(p.get("value", 0) for p in current_positions.values())
        new_total = total_position_value + order_value * (1 if side == "buy" else -1)
        if new_total / total_capital > self.config.max_total_position:
            return False, f"总仓位超限"

        return True, "OK"

    def check_daily_loss(self, daily_pnl: float, total_capital: float) -> tuple[bool, str]:
        """日止损检查"""
        daily_return = daily_pnl / total_capital
        if daily_return < self.config.daily_stop_loss:
            return False, f"触发日止损: {daily_return:.2%} < {self.config.daily_stop_loss:.2%}"
        return True, "OK"
