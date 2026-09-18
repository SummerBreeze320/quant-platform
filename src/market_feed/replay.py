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
