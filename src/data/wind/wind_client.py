"""Wind Py client wrapper: connection, wsd, wset, wsq."""
import logging
import os
import time
from typing import Optional, Union, List
import pandas as pd

logger = logging.getLogger(__name__)

# Wind DLL 路径（必须在 import WindPy 之前设置）
_WIND_BASE = r"D:\Wind\Wind.NET.Client\WindNET"
_WIND_DIRS = [
    os.path.join(_WIND_BASE, "x64"),
    os.path.join(_WIND_BASE, "bin"),
]
for _d in _WIND_DIRS:
    if os.path.isdir(_d):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + ";" + os.environ.get("PATH", "")

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    logger.warning("WindPy not installed. Install: pip install WindPy")
except FileNotFoundError as e:
    WIND_AVAILABLE = False
    logger.warning(f"WindPy DLL load failed: {e}")


class WindClient:
    """WindPy client singleton"""

    _instance: Optional["WindClient"] = None
    _connected: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def connect(self, max_retries: int = 3, retry_interval: int = 5) -> bool:
        if not WIND_AVAILABLE:
            raise RuntimeError("WindPy not available. Ensure Wind terminal is running.")
        if self._connected:
            return True
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to Wind terminal (attempt {attempt + 1}/{max_retries})...")
                w.start()
                if w.isconnected():
                    self._connected = True
                    logger.info("Wind terminal connected successfully.")
                    return True
                else:
                    logger.warning(f"Wind connection attempt {attempt + 1} failed.")
                    time.sleep(retry_interval)
            except Exception as e:
                logger.error(f"Wind connection error (attempt {attempt + 1}): {e}")
                time.sleep(retry_interval)
        raise RuntimeError(f"Failed to connect Wind terminal after {max_retries} attempts.")

    def disconnect(self):
        if self._connected and WIND_AVAILABLE:
            w.stop()
            self._connected = False
            logger.info("Wind terminal disconnected.")

    def is_connected(self) -> bool:
        return self._connected and WIND_AVAILABLE and w.isconnected()

    def ensure_connection(self):
        if not self.is_connected():
            self.connect()

    def wsd(self, codes: Union[str, List[str]], fields: str,
            start_time: str, end_time: str, options: str = "") -> pd.DataFrame:
        self.ensure_connection()
        if isinstance(codes, list):
            codes = ",".join(codes)
        result = w.wsd(codes, fields, start_time, end_time, options)
        if result.ErrorCode != 0:
            raise RuntimeError(f"Wind wsd error: code={result.ErrorCode}, codes={codes}, fields={fields}")

        code_list = codes.split(",")
        is_single = len(code_list) == 1
        if is_single:
            data = {}
            for i, field in enumerate(result.Fields):
                data[field.lower()] = result.Data[i]
            df = pd.DataFrame(data, index=pd.to_datetime(result.Times))
        else:
            records = []
            for i, field in enumerate(result.Fields):
                for j, code in enumerate(code_list):
                    values = result.Data[i]
                    if isinstance(values[0], list):
                        vals = values[j]
                    else:
                        vals = values
                    for k, date in enumerate(result.Times):
                        records.append({
                            "date": date, "code": code,
                            "field": field.lower(),
                            "value": vals[k] if k < len(vals) else None,
                        })
            df = pd.DataFrame(records).pivot_table(
                index="date", columns=["code", "field"], values="value"
            )
            df.index = pd.to_datetime(df.index)
        return df

    def wset(self, report_name: str, options: str) -> pd.DataFrame:
        self.ensure_connection()
        result = w.wset(report_name, options)
        if result.ErrorCode != 0:
            raise RuntimeError(f"Wind wset error: code={result.ErrorCode}")
        if not result.Codes or not result.Data:
            logger.warning(f"Wind wset returned empty: {report_name}")
            return pd.DataFrame()
        data = {}
        for i, field in enumerate(result.Fields):
            data[field] = result.Data[i]
        return pd.DataFrame(data, index=result.Codes)

    def wsq(self, codes: str, fields: str, on_data=None):
        self.ensure_connection()
        return w.wsq(codes, fields, on_data)

    @staticmethod
    def to_wind_code(code: str) -> str:
        code = code.strip()
        if "." in code:
            return code
        if code.startswith("6"):
            return f"{code}.SH"
        elif code.startswith(("0", "3")):
            return f"{code}.SZ"
        elif code.startswith(("5", "11", "12", "13", "15", "16", "18")):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"

    @staticmethod
    def to_qlib_code(code: str) -> str:
        if "." not in code:
            return code
        code_part, market = code.rsplit(".", 1)
        return f"{market}{code_part}"
