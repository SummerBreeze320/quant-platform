"""执行算法层

拆单算法，降低大单交易的冲击成本：

1. TWAP (Time-Weighted Average Price):
   将大单均匀拆分到指定时间区间内执行
   适用于流动性较好、日内波动均匀的标的

2. VWAP (Volume-Weighted Average Price):
   根据历史成交量分布，在交易量大的时段下更多单
   适用于有规律成交量模式的标的

3. Iceberg (冰山订单):
   每次只暴露小部分订单量，成交后再提交下一笔
   适用于避免暴露真实交易意图的大单

4. Implementation Shortfall (执行 shortfall):
   平衡冲击成本和时间风险，动态调整拆单速度
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlgoConfig:
    """算法配置"""
    total_volume: float          # 总订单量
    stock: str                   # 标的
    side: str                    # buy/sell
    duration_minutes: int = 240  # 执行时长(分钟)
    slices: int = 10             # 拆单次数
    max_part_of_volume: float = 0.05  # 单笔占ADV最大比例
    price_limit_pct: float = 0.02     # 价格偏离限制


class ExecutionAlgo:
    """执行算法基类"""

    name: str = "base"

    def __init__(self, config: AlgoConfig):
        self.config = config
        self._executed_volume = 0.0
        self._slice_index = 0
        self._slices: List[dict] = []

    def generate_slices(self, volume_profile: Optional[pd.Series] = None) -> List[dict]:
        """生成拆单计划"""
        raise NotImplementedError

    def get_next_slice(self) -> Optional[dict]:
        """获取下一笔拆单"""
        if self._slice_index >= len(self._slices):
            return None
        sl = self._slices[self._slice_index]
        self._slice_index += 1
        return sl

    @property
    def progress(self) -> float:
        """执行进度"""
        if self.config.total_volume == 0:
            return 0
        return self._executed_volume / self.config.total_volume

    @property
    def remaining_volume(self) -> float:
        return self.config.total_volume - self._executed_volume


class TWAPAlgo(ExecutionAlgo):
    """TWAP拆单算法

    将大单均匀拆分到时间区间内，每个时间间隔执行相同数量。
    """

    name = "TWAP"

    def generate_slices(self, volume_profile=None):
        n = self.config.slices
        vol_per_slice = self.config.total_volume / n
        interval = self.config.duration_minutes / n

        self._slices = []
        for i in range(n):
            self._slices.append({
                "slice_id": i,
                "time_offset": i * interval,
                "volume": vol_per_slice,
                "algorithm": "TWAP",
            })

        logger.info(f"TWAP拆单: {self.config.total_volume}股 → "
                     f"{n}笔 × {vol_per_slice:.0f}股, "
                     f"间隔{interval:.0f}分钟")
        return self._slices


class VWAPAlgo(ExecutionAlgo):
    """VWAP拆单算法

    根据历史成交量分布（U型曲线特征），在交易量大的时段下更多单。
    需要提供历史日内成交量分布数据。
    如果没有历史数据，使用默认的U型分布。
    """

    name = "VWAP"

    def generate_slices(self, volume_profile: Optional[pd.Series] = None):
        n = self.config.slices

        if volume_profile is not None and len(volume_profile) > 0:
            # 使用历史成交量分布
            weights = volume_profile.values.astype(float)
            # 按时间段聚合到n个切片
            chunk_size = len(weights) // n
            if chunk_size > 0:
                aggregated = np.array([
                    weights[i*chunk_size:(i+1)*chunk_size].sum()
                    for i in range(n)
                ])
            else:
                aggregated = np.array(weights[:n])
        else:
            # 默认U型分布：开盘和收盘成交量大，午盘小
            t = np.linspace(0, 1, n)
            # U型: w = a * (t - 0.5)^2 + b
            a, b = 3.0, 0.5
            weights = a * (t - 0.5) ** 2 + b
            aggregated = weights

        # 归一化
        total_weight = aggregated.sum()
        if total_weight > 0:
            volumes = (aggregated / total_weight) * self.config.total_volume
        else:
            volumes = np.ones(n) * self.config.total_volume / n

        interval = self.config.duration_minutes / n
        self._slices = []
        for i in range(n):
            self._slices.append({
                "slice_id": i,
                "time_offset": i * interval,
                "volume": float(volumes[i]),
                "algorithm": "VWAP",
            })

        logger.info(f"VWAP拆单: {self.config.total_volume}股 → "
                     f"{n}笔, 最大单笔{max(volumes):.0f}股, "
                     f"最小单笔{min(volumes):.0f}股")
        return self._slices


class IcebergAlgo(ExecutionAlgo):
    """冰山订单算法

    每次只暴露一小部分订单，成交后再提交下一笔。
    用于隐藏真实交易意图。
    """

    name = "Iceberg"

    def __init__(self, config: AlgoConfig, 
                 display_ratio: float = 0.1,
                 min_display: float = 100):
        super().__init__(config)
        self.display_ratio = display_ratio
        self.min_display = min_display

    def generate_slices(self, volume_profile=None):
        remaining = self.config.total_volume
        self._slices = []
        i = 0

        while remaining > 0:
            display_vol = max(
                remaining * self.display_ratio,
                self.min_display
            )
            display_vol = min(display_vol, remaining)

            self._slices.append({
                "slice_id": i,
                "time_offset": i * 5,  # 每笔间隔5分钟
                "volume": display_vol,
                "algorithm": "Iceberg",
                "display_volume": display_vol,
                "hidden_volume": remaining - display_vol,
            })
            remaining -= display_vol
            i += 1

        logger.info(f"冰山拆单: {self.config.total_volume}股 → "
                     f"{len(self._slices)}笔, "
                     f"每笔暴露{self.display_ratio:.0%}")
        return self._slices


class ImplementationShortfallAlgo(ExecutionAlgo):
    """执行 shortfall 算法

    平衡冲击成本和时间风险，动态调整拆单速度。
    前期快速执行（降低时间风险），后期慢速执行（降低冲击成本）。
    """

    name = "Implementation_Shortfall"

    def __init__(self, config: AlgoConfig,
                 risk_aversion: float = 1.0):
        super().__init__(config)
        self.risk_aversion = risk_aversion

    def generate_slices(self, volume_profile=None):
        n = self.config.slices
        # 前重后轻的分配：使用指数衰减
        t = np.arange(n, 0, -1)  # n, n-1, ..., 1
        weights = np.exp(-self.risk_aversion * t / n)
        weights = weights / weights.sum()
        volumes = weights * self.config.total_volume

        interval = self.config.duration_minutes / n
        self._slices = []
        for i in range(n):
            self._slices.append({
                "slice_id": i,
                "time_offset": i * interval,
                "volume": float(volumes[i]),
                "algorithm": "IS",
            })

        logger.info(f"IS拆单: {self.config.total_volume}股 → "
                     f"{n}笔, 前重后轻(λ={self.risk_aversion})")
        return self._slices


def create_algo(algo_name: str, config: AlgoConfig, **kwargs) -> ExecutionAlgo:
    """工厂函数：创建执行算法"""
    algos = {
        "TWAP": TWAPAlgo,
        "VWAP": VWAPAlgo,
        "Iceberg": IcebergAlgo,
        "IS": ImplementationShortfallAlgo,
    }
    cls = algos.get(algo_name, TWAPAlgo)
    return cls(config, **kwargs)
