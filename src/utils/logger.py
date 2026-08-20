"""日志配置"""
import logging
from loguru import logger
import sys


def setup_logger(level: str = "INFO"):
    """配置loguru日志"""
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")
    logger.add("logs/quant_platform.log", level=level, rotation="10 MB", retention="30 days")
    return logger
