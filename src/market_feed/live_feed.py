import json
import math
import random
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Set, Any

from src.common.logger import logger
from src.market_feed.models import MarketTick
from src.market_feed.stream_bus import StreamBus


def normalize_symbol_to_standard(symbol: str) -> str:
    """标准化代码格式为 600000.SH / 000001.SZ"""
    s = symbol.strip().upper()
    if s.endswith(".SH") or s.endswith(".SZ"):
        return s
    if s.startswith("SH") and len(s) == 8:
        return f"{s[2:]}.SH"
    if s.startswith("SZ") and len(s) == 8:
        return f"{s[2:]}.SZ"
    if s.startswith("6") or s.startswith("9") or s.startswith("5"):
        return f"{s}.SH"
    return f"{s}.SZ"


def standard_to_sina_code(symbol: str) -> str:
    """将标准代码转换为新浪接口格式 sh600000 / sz000001"""
    std = normalize_symbol_to_standard(symbol)
    code, mkt = std.split(".")
    return f"{mkt.lower()}{code}"


class BaseFeedAdapter(ABC):
    """行情源适配器抽象基类"""

    def __init__(self, bus: StreamBus):
        self.bus = bus
        self.subscribed_symbols: Set[str] = set()
        self._is_running = False
        self._total_ticks = 0
        self._lock = threading.Lock()

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    def subscribe(self, symbols: List[str]) -> None:
        with self._lock:
            for s in symbols:
                self.subscribed_symbols.add(normalize_symbol_to_standard(s))

    def unsubscribe(self, symbols: List[str]) -> None:
        with self._lock:
            for s in symbols:
                self.subscribed_symbols.discard(normalize_symbol_to_standard(s))

    def get_subscribed_symbols(self) -> List[str]:
        with self._lock:
            return sorted(list(self.subscribed_symbols))

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def total_ticks(self) -> int:
        return self._total_ticks

    def emit_tick(self, tick: MarketTick) -> None:
        self.bus.publish_tick(tick)
        with self._lock:
            self._total_ticks += 1


class SimulationLiveFeedAdapter(BaseFeedAdapter):
    """
    高保真波动行情仿真适配器：
    利用几何布朗运动 (GBM) 和微观离散订单簿生成高频 Tick 数据流。
    适合非交易时段、系统开发、压力测试与持续集成。
    """

    def __init__(
        self,
        bus: StreamBus,
        interval_seconds: float = 0.5,
        initial_prices: Optional[Dict[str, float]] = None,
    ):
        super().__init__(bus)
        self.interval_seconds = max(interval_seconds, 0.01)
        self._prices: Dict[str, float] = initial_prices or {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="SimulationFeedWorker", daemon=True)
        self._thread.start()
        logger.info("[SimulationLiveFeed] 仿真行情驱动器已启动")

    def stop(self) -> None:
        if not self._is_running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._is_running = False
        logger.info("[SimulationLiveFeed] 仿真行情驱动器已停止")

    def _run_loop(self) -> None:
        cumulative_volume: Dict[str, int] = {}
        cumulative_turnover: Dict[str, float] = {}

        while not self._stop_event.is_set():
            symbols = self.get_subscribed_symbols()
            now_iso = datetime.now().isoformat()

            for sym in symbols:
                if self._stop_event.is_set():
                    break

                curr_p = self._prices.get(sym, 20.0)
                # 随机游走: 微幅变动 -0.5% ~ +0.5%
                pct_change = random.gauss(0.0001, 0.002)
                new_p = round(max(curr_p * (1.0 + pct_change), 0.01), 2)
                self._prices[sym] = new_p

                # 随机生成成交量与成交额
                vol_step = random.randint(100, 2000)
                turnover_step = round(vol_step * new_p, 2)
                cumulative_volume[sym] = cumulative_volume.get(sym, 0) + vol_step
                cumulative_turnover[sym] = cumulative_turnover.get(sym, 0.0) + turnover_step

                # 构造 5 档盘口
                tick_spread = 0.01
                bid_prices = [round(new_p - (i + 1) * tick_spread, 2) for i in range(5)]
                ask_prices = [round(new_p + (i + 1) * tick_spread, 2) for i in range(5)]
                bid_volumes = [random.randint(50, 500) * 100 for _ in range(5)]
                ask_volumes = [random.randint(50, 500) * 100 for _ in range(5)]

                tick = MarketTick(
                    symbol=sym,
                    timestamp=now_iso,
                    last_price=new_p,
                    volume=cumulative_volume[sym],
                    turnover=cumulative_turnover[sym],
                    bid_prices=bid_prices,
                    bid_volumes=bid_volumes,
                    ask_prices=ask_prices,
                    ask_volumes=ask_volumes,
                )
                self.emit_tick(tick)

            # 等待步长
            self._stop_event.wait(self.interval_seconds)


class PublicWebLiveFeedAdapter(BaseFeedAdapter):
    """
    公网实时行情源降级适配器：
    利用新浪财经/腾讯开放接口拉取实时 A 股盘口快照，解析最新价与五档委托量价。
    无需任何客户端依赖，开箱即用。
    """

    def __init__(
        self,
        bus: StreamBus,
        interval_seconds: float = 1.0,
        request_timeout: float = 2.0,
    ):
        super().__init__(bus)
        self.interval_seconds = max(interval_seconds, 0.2)
        self.request_timeout = request_timeout
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="PublicWebFeedWorker", daemon=True)
        self._thread.start()
        logger.info("[PublicWebLiveFeed] 公网实时行情拉取器已启动")

    def stop(self) -> None:
        if not self._is_running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._is_running = False
        logger.info("[PublicWebLiveFeed] 公网实时行情拉取器已停止")

    def _fetch_snapshot(self, symbols: List[str]) -> List[MarketTick]:
        if not symbols:
            return []

        sina_symbols = [standard_to_sina_code(s) for s in symbols]
        sym_map = {standard_to_sina_code(s): normalize_symbol_to_standard(s) for s in symbols}
        query_str = ",".join(sina_symbols)
        url = f"http://hq.sinajs.cn/list={query_str}"

        req = urllib.request.Request(
            url,
            headers={
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            },
        )

        ticks: List[MarketTick] = []
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                content = response.read().decode("gbk", errors="ignore")

            for line in content.strip().split("\n"):
                if not line.startswith("var hq_str_"):
                    continue
                parts = line.split("=")
                if len(parts) < 2:
                    continue
                code_key = parts[0].replace("var hq_str_", "").strip()
                data_str = parts[1].strip('";')
                if not data_str:
                    continue
                fields = data_str.split(",")
                if len(fields) < 32:
                    continue

                std_sym = sym_map.get(code_key, normalize_symbol_to_standard(code_key))
                last_price = float(fields[3]) if float(fields[3]) > 0 else float(fields[2])
                volume = int(float(fields[8]))
                turnover = float(fields[9])
                date_str = fields[30]
                time_str = fields[31]
                ts = f"{date_str}T{time_str}" if date_str and time_str else datetime.now().isoformat()

                # 买一到买五
                bid_vols = [int(float(fields[10])), int(float(fields[12])), int(float(fields[14])), int(float(fields[16])), int(float(fields[18]))]
                bid_prices = [float(fields[11]), float(fields[13]), float(fields[15]), float(fields[17]), float(fields[19])]
                # 卖一到卖五
                ask_vols = [int(float(fields[20])), int(float(fields[22])), int(float(fields[24])), int(float(fields[26])), int(float(fields[28]))]
                ask_prices = [float(fields[21]), float(fields[23]), float(fields[25]), float(fields[27]), float(fields[29])]

                ticks.append(
                    MarketTick(
                        symbol=std_sym,
                        timestamp=ts,
                        last_price=last_price,
                        volume=volume,
                        turnover=turnover,
                        bid_prices=bid_prices,
                        bid_volumes=bid_vols,
                        ask_prices=ask_prices,
                        ask_volumes=ask_vols,
                    )
                )
        except Exception as e:
            logger.warning(f"[PublicWebLiveFeed] HTTP 快照拉取异常: {e}")

        return ticks

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            symbols = self.get_subscribed_symbols()
            if symbols:
                ticks = self._fetch_snapshot(symbols)
                for tick in ticks:
                    if self._stop_event.is_set():
                        break
                    self.emit_tick(tick)
            self._stop_event.wait(self.interval_seconds)


class QmtLiveFeedAdapter(BaseFeedAdapter):
    """
    迅投 QMT (XtQuant) 实盘行情订阅适配器：
    利用 xtquant.xtdata.subscribe_quote 接入本地 MiniQMT 高性能共享内存行情报文。
    若检测到环境未安装 xtquant，自动给出自适应模式提示。
    """

    def __init__(self, bus: StreamBus):
        super().__init__(bus)
        self._xtdata = None
        self._sub_seqs: Dict[str, int] = {}
        self._has_xtquant = False
        try:
            from xtquant import xtdata
            self._xtdata = xtdata
            self._has_xtquant = True
        except ImportError:
            logger.info("[QmtLiveFeed] 未检测到 xtquant 库，QmtLiveFeed 处于待命/模拟模式")

    @property
    def has_xtquant(self) -> bool:
        return self._has_xtquant

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        logger.info(f"[QmtLiveFeed] QMT 行情适配器已启动 (xtquant可用={self._has_xtquant})")
        if self._has_xtquant and self._xtdata:
            symbols = self.get_subscribed_symbols()
            for s in symbols:
                self._subscribe_xtdata(s)

    def stop(self) -> None:
        if not self._is_running:
            return
        if self._has_xtquant and self._xtdata:
            for s, seq in list(self._sub_seqs.items()):
                try:
                    self._xtdata.unsubscribe_quote(seq)
                except Exception as e:
                    logger.warning(f"[QmtLiveFeed] 退订失败 {s}: {e}")
            self._sub_seqs.clear()
        self._is_running = False
        logger.info("[QmtLiveFeed] QMT 行情适配器已停止")

    def _subscribe_xtdata(self, symbol: str) -> None:
        if not self._has_xtquant or not self._xtdata:
            return
        # xtdata 格式为 600000.SH
        try:
            seq = self._xtdata.subscribe_quote(
                stock_code=symbol,
                period="tick",
                count=0,
                callback=self._on_xtdata_callback,
            )
            self._sub_seqs[symbol] = seq
        except Exception as e:
            logger.error(f"[QmtLiveFeed] subscribe_quote {symbol} error: {e}")

    def _on_xtdata_callback(self, datas: Any) -> None:
        """处理 xtdata 推送行情回调"""
        if not isinstance(datas, dict):
            return
        for sym, tick_info in datas.items():
            if not isinstance(tick_info, dict):
                continue
            std_sym = normalize_symbol_to_standard(sym)
            last_p = float(tick_info.get("lastPrice", 0.0))
            vol = int(tick_info.get("volume", 0))
            amt = float(tick_info.get("amount", 0.0))
            ask_prices = [float(p) for p in tick_info.get("askPrice", [])[:5]]
            ask_vols = [int(v) for v in tick_info.get("askVol", [])[:5]]
            bid_prices = [float(p) for p in tick_info.get("bidPrice", [])[:5]]
            bid_vols = [int(v) for v in tick_info.get("bidVol", [])[:5]]

            ts_ms = tick_info.get("time", int(time.time() * 1000))
            dt_str = datetime.fromtimestamp(ts_ms / 1000.0).isoformat()

            tick = MarketTick(
                symbol=std_sym,
                timestamp=dt_str,
                last_price=last_p,
                volume=vol,
                turnover=amt,
                bid_prices=bid_prices,
                bid_volumes=bid_vols,
                ask_prices=ask_prices,
                ask_volumes=ask_vols,
            )
            self.emit_tick(tick)

    def subscribe(self, symbols: List[str]) -> None:
        super().subscribe(symbols)
        if self._is_running and self._has_xtquant:
            for s in symbols:
                std_s = normalize_symbol_to_standard(s)
                if std_s not in self._sub_seqs:
                    self._subscribe_xtdata(std_s)

    def unsubscribe(self, symbols: List[str]) -> None:
        super().unsubscribe(symbols)
        if self._has_xtquant and self._xtdata:
            for s in symbols:
                std_s = normalize_symbol_to_standard(s)
                seq = self._sub_seqs.pop(std_s, None)
                if seq is not None:
                    try:
                        self._xtdata.unsubscribe_quote(seq)
                    except Exception:
                        pass


class LiveFeedManager:
    """
    统一实时行情驱动协调管理器：
    负责调度 QMT、公网与仿真等不同行情源适配器，提供动态切换、订阅管理与实时吞吐监控。
    """

    def __init__(self, bus: StreamBus):
        self.bus = bus
        self._adapters: Dict[str, BaseFeedAdapter] = {
            "SIMULATION": SimulationLiveFeedAdapter(bus=bus),
            "PUBLIC": PublicWebLiveFeedAdapter(bus=bus),
            "QMT": QmtLiveFeedAdapter(bus=bus),
        }
        self._current_source = "SIMULATION"
        self._start_time: Optional[float] = None
        self._last_tick_count = 0
        self._last_tps_check = time.time()
        self._current_tps = 0.0

    @property
    def current_adapter(self) -> BaseFeedAdapter:
        return self._adapters[self._current_source]

    @property
    def current_source(self) -> str:
        return self._current_source

    def start(
        self,
        source: str = "SIMULATION",
        symbols: Optional[List[str]] = None,
        interval_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """启动行情接收服务"""
        norm_source = source.strip().upper()
        if norm_source not in self._adapters:
            raise ValueError(f"Unknown feed source '{source}'. Available: {list(self._adapters.keys())}")

        # 若当前已有运行的 adapter 且需要切换源，先停止旧源
        if self.current_adapter.is_running and norm_source != self._current_source:
            self.current_adapter.stop()

        self._current_source = norm_source
        adapter = self.current_adapter

        if interval_seconds is not None:
            if hasattr(adapter, "interval_seconds"):
                adapter.interval_seconds = max(interval_seconds, 0.01)

        if symbols:
            adapter.subscribe(symbols)

        if not adapter.is_running:
            adapter.start()
            self._start_time = time.time()
            self._last_tick_count = adapter.total_ticks
            self._last_tps_check = time.time()

        return self.get_status()

    def stop(self) -> Dict[str, Any]:
        """停止当前行情源"""
        adapter = self.current_adapter
        adapter.stop()
        self._start_time = None
        self._current_tps = 0.0
        return self.get_status()

    def subscribe(self, symbols: List[str]) -> Dict[str, Any]:
        """为当前生效适配器追加订阅标的"""
        self.current_adapter.subscribe(symbols)
        return self.get_status()

    def unsubscribe(self, symbols: List[str]) -> Dict[str, Any]:
        """从当前生效适配器退订标的"""
        self.current_adapter.unsubscribe(symbols)
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """获取当前行情驱动器的运行监控状态"""
        adapter = self.current_adapter
        now = time.time()
        time_diff = now - self._last_tps_check

        if time_diff >= 1.0:
            tick_diff = adapter.total_ticks - self._last_tick_count
            self._current_tps = round(tick_diff / time_diff, 2)
            self._last_tick_count = adapter.total_ticks
            self._last_tps_check = now

        uptime = round(now - self._start_time, 2) if self._start_time and adapter.is_running else 0.0

        return {
            "source": self._current_source,
            "is_running": adapter.is_running,
            "uptime_seconds": uptime,
            "total_ticks": adapter.total_ticks,
            "current_tps": self._current_tps,
            "subscribed_symbols": adapter.get_subscribed_symbols(),
            "available_sources": list(self._adapters.keys()),
        }
