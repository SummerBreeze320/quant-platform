"""Wind客户端测试"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch


def test_to_wind_code():
    """测试代码格式转换"""
    from src.data.wind.wind_client import WindClient

    assert WindClient.to_wind_code("600000") == "600000.SH"
    assert WindClient.to_wind_code("000001") == "000001.SZ"
    assert WindClient.to_wind_code("510300") == "510300.SH"
    assert WindClient.to_wind_code("600000.SH") == "600000.SH"


def test_wind_client_singleton():
    """测试单例模式"""
    from src.data.wind.wind_client import WindClient

    c1 = WindClient()
    c2 = WindClient()
    assert c1 is c2
