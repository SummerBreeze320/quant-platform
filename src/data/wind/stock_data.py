"""A股数据采集器

从Wind获取A股历史行情、财务数据、指数成分股。
"""
import logging
import pandas as pd
from typing import List, Optional
from .wind_client import WindClient

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """A股数据采集"""

    DAILY_FIELDS = (
        "open,high,low,close,volume,amount,pct_chg,turn,free_turn,vwap"
    )

    FUNDAMENTAL_FIELDS = (
        "pe_ttm,pb_lf,ps_ttm,dividend_yield,total_mv,circ_mv,"
        "roe,roa,debt_ratio,revenue_yoy,profit_yoy"
    )

    def __init__(self, client: Optional[WindClient] = None):
        self.client = client or WindClient()

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str = "F",
    ) -> pd.DataFrame:
        """获取日K线数据

        Args:
            code: 股票代码，如 "600000.SH"
            start_date: "2015-01-01"
            end_date: "2026-08-01"
            adjust: 复权方式 F=前复权 B=后复权 N=不复权

        Returns:
            DataFrame，索引为日期，列为OHLCV等
        """
        wind_code = WindClient.to_wind_code(code)
        options = f"PriceAdj={adjust}"

        df = self.client.wsd(
            codes=wind_code,
            fields=self.DAILY_FIELDS,
            start_time=start_date,
            end_time=end_date,
            options=options,
        )
        logger.info(f"Fetched {len(df)} rows for {wind_code} ({start_date} to {end_date})")
        return df

    def fetch_fundamental(
        self,
        code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取财务数据（日频快照）"""
        wind_code = WindClient.to_wind_code(code)
        df = self.client.wsd(
            codes=wind_code,
            fields=self.FUNDAMENTAL_FIELDS,
            start_time=start_date,
            end_time=end_date,
            options="",
        )
        logger.info(f"Fetched fundamental data for {wind_code}: {len(df)} rows")
        return df

    def fetch_index_constituents(self, index_code: str) -> List[str]:
        """获取指数成分股列表

        Args:
            index_code: 如 "000300.SH"（沪深300）

        Returns:
            成分股代码列表
        """
        df = self.client.wset(
            report_name="indexconstituent",
            options=f"index={index_code};field=wind_code",
        )

        codes = df["wind_code"].tolist() if "wind_code" in df.columns else df.iloc[:, 0].tolist()
        logger.info(f"Index {index_code} has {len(codes)} constituents")
        return codes

    def fetch_all_stocks(self) -> List[str]:
        """获取全A股代码列表"""
        df = self.client.wset(
            report_name="sectorconstituent",
            options="sectorId=a00101010000;field=wind_code",
        )
        codes = df["wind_code"].tolist() if "wind_code" in df.columns else df.iloc[:, 0].tolist()
        logger.info(f"Total A-share stocks: {len(codes)}")
        return codes

    def fetch_suspension(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取停牌信息"""
        wind_code = WindClient.to_wind_code(code)
        df = self.client.wsd(
            codes=wind_code,
            fields="trade_status",
            start_time=start_date,
            end_time=end_date,
            options="",
        )
        return df

    def fetch_daily_batch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "F",
    ) -> dict[str, pd.DataFrame]:
        """批量获取多只股票日K线数据

        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权方式

        Returns:
            {code: DataFrame} 字典
        """
        result = {}
        for code in codes:
            try:
                df = self.fetch_daily(code, start_date, end_date, adjust)
                if not df.empty:
                    result[code] = df
            except Exception as e:
                logger.error(f"Failed to fetch {code}: {e}")
        logger.info(f"Batch fetched {len(result)}/{len(codes)} stocks")
        return result
