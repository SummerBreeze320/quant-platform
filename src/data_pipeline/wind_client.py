import time
from typing import Optional, Any
from src.common.logger import logger

class WindClient:
    """Manages the connection lifecycle to the WindPy API."""

    def __init__(self, w_instance: Optional[Any] = None):
        self._connected = False
        if w_instance is not None:
            self._w = w_instance
        else:
            try:
                from WindPy import w
                self._w = w
            except ImportError:
                logger.warning("WindPy module not installed or cannot be imported. Running in offline/mock mode.")
                self._w = None

    @property
    def w(self) -> Any:
        return self._w

    def is_connected(self) -> bool:
        if self._w is None:
            return False
        try:
            return bool(self._w.isconnected())
        except Exception:
            return False

    def connect(self, timeout: int = 15) -> bool:
        """Initialize connection to Wind Terminal."""
        if self._w is None:
            logger.error("Cannot connect: WindPy instance is unavailable.")
            return False
        
        if self.is_connected():
            return True

        logger.info("Initializing WindPy connection...")
        try:
            res = self._w.start(waitTime=timeout)
            if hasattr(res, "ErrorCode") and res.ErrorCode == 0:
                self._connected = True
                logger.info("WindPy connected successfully.")
                return True
            else:
                err_code = getattr(res, "ErrorCode", "Unknown")
                logger.error(f"WindPy start failed with ErrorCode: {err_code}")
                return False
        except Exception as e:
            logger.error(f"Exception during WindPy start: {e}")
            return False

    def disconnect(self):
        """Disconnect from WindPy."""
        if self._w is not None and self.is_connected():
            try:
                self._w.stop()
                logger.info("WindPy disconnected.")
            except Exception as e:
                logger.warning(f"Error stopping WindPy: {e}")
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
