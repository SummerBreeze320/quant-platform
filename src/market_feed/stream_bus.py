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
