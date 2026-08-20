"""Qlib initialization wrapper.

Ensures qlib.init() is called with the correct provider_uri.
All data access in the platform goes through Qlib's D (DataProvider) singleton.

Supports re-initialization when provider_uri changes (important for tests
and switching data sources).
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_qlib_initialized = False
_current_provider_uri: Optional[str] = None
_default_provider_uri = "data/qlib_bin"


def init_qlib(
    provider_uri: str = _default_provider_uri,
    region: str = "cn",
    expression_cache: Optional[str] = None,
) -> bool:
    """Initialize Qlib with the given provider_uri.

    Re-initializes automatically when provider_uri changes.

    Args:
        provider_uri: Path to Qlib bin data directory.
        region: "cn" for China market, "us" for US market.
        expression_cache: Optional path for expression cache directory.
    """
    global _qlib_initialized, _current_provider_uri

    resolved_uri = str(Path(provider_uri).resolve())

    if _qlib_initialized and _current_provider_uri == resolved_uri:
        logger.debug("Qlib already initialized with same provider_uri, skipping.")
        return True

    import qlib
    from qlib.config import REG_CN, REG_US

    region_map = {"cn": REG_CN, "us": REG_US}
    qlib_region = region_map.get(region, REG_CN)

    init_kwargs = {
        "provider_uri": resolved_uri,
        "region": qlib_region,
    }
    if expression_cache:
        init_kwargs["expression_cache"] = expression_cache

    logger.info(f"Initializing Qlib: provider_uri={resolved_uri}, region={region}")
    qlib.init(**init_kwargs)
    _qlib_initialized = True
    _current_provider_uri = resolved_uri
    logger.info("Qlib initialized successfully.")
    return True


def ensure_qlib(provider_uri: str = _default_provider_uri) -> None:
    """Ensure Qlib is initialized with the given provider_uri.

    Re-initializes if provider_uri differs from the current one.
    """
    global _qlib_initialized
    resolved_uri = str(Path(provider_uri).resolve())
    if not _qlib_initialized or _current_provider_uri != resolved_uri:
        init_qlib(provider_uri=provider_uri)


def is_initialized() -> bool:
    """Check if Qlib has been initialized."""
    return _qlib_initialized


def reset() -> None:
    """Reset initialization state (for testing)."""
    global _qlib_initialized, _current_provider_uri
    _qlib_initialized = False
    _current_provider_uri = None
