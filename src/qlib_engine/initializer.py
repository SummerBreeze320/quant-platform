from pathlib import Path
from typing import Optional
import qlib
from qlib.config import REG_CN, C
from src.common.config import get_settings
from src.common.logger import logger

_is_initialized = False

def is_qlib_initialized() -> bool:
    global _is_initialized
    return _is_initialized or getattr(C, "registered", False)

def init_qlib(provider_uri: Optional[str] = None, region: str = REG_CN, reload: bool = False):
    """Initializes Microsoft Qlib with specified data directory."""
    global _is_initialized
    
    settings = get_settings()
    data_dir = Path(provider_uri or settings.QLIB_DATA_DIR)
    
    # Ensure calendars and instruments directories exist so Qlib doesn't panic
    (data_dir / "calendars").mkdir(parents=True, exist_ok=True)
    (data_dir / "instruments").mkdir(parents=True, exist_ok=True)
    (data_dir / "features").mkdir(parents=True, exist_ok=True)

    if is_qlib_initialized() and not reload:
        # If already initialized with the same provider_uri, skip re-init
        current_uri = getattr(C, "provider_uri", None)
        if current_uri == str(data_dir):
            return

    logger.info(f"Initializing Qlib with provider_uri: {data_dir} (region: {region})")
    
    qlib.init(
        provider_uri=str(data_dir),
        region=region,
        expression_cache=None,  # Disable disk cache during testing / dynamic runs
        dataset_cache=None
    )
    _is_initialized = True
    logger.info("Qlib initialized successfully.")
