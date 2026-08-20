"""ETF数据采集器

从Wind获取ETF行情、申赎清单、ETF成分股。
"""
import logging
import pandas as pd
from typing import List, Optional
from .wind_client import WindClient

logger = logging.getLogger(__name__)


class ETFDataFetcher:
    """ETF数据采集"""

    ETF_DAILY_FIELDS = (
        "open,high,low,close,volume,amount,pct_chg,turn,vwap"
    )

    ETF_EXTRA_FIELDS = (
        "nav,discount,nav_accumulated"
    )

    def __init__(self, client: Optional[WindClient] = None):
        self.client = client or WindClient()

    def fetch_etf_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str = "F",
    ) -> pd.DataFrame:
        """获取ETF日K线数据"""
        wind_code = WindClient.to_wind_code(code)
        fields = self.ETF_DAILY_FIELDS + "," + self.ETF_EXTRA_FIELDS
        options = f"PriceAdj={adjust}"

        df = self.client.wsd(
            codes=wind_code,
            fields=fields,
            start_time=start_date,
            end_time=end_date,
            options=options,
        )
        logger.info(f"Fetched ETF {wind_code}: {len(df)} rows")
        return df

    def fetch_all_etfs(self) -> List[str]:
        """获取全市场ETF代码列表"""
        df = self.client.wset(
            report_name="sectorconstituent",
            options="sectorId=a00100300100;field=wind_code",
        )
        codes = df["wind_code"].tolist() if "wind_code" in df.columns else df.iloc[:, 0].tolist()
        logger.info(f"Total ETFs: {len(codes)}")
        return codes

    def fetch_etf_constituents(self, etf_code: str) -> pd.DataFrame:
        """获取ETF成分股/持仓明细"""
        wind_code = WindClient.to_wind_code(etf_code)
        df = self.client.wset(
            report_name="etfconstituent",
            options=f"windcode={wind_code};field=wind_code,name,weight",
        )
        logger.info(f"ETF {wind_code} has {len(df)} constituents")
        return df

    def fetch_etf_pcflist(self, etf_code: str, date: str) -> pd.DataFrame:
        """获取ETF申赎清单（PCF）

        Args:
            etf_code: ETF代码
            date: 日期 "YYYY-MM-DD"

        Returns:
            申赎清单DataFrame
        """
        wind_code = WindClient.to_wind_code(etf_code)
        df = self.client.wset(
            report_name="etfpcflist",
            options=f"windcode={wind_code};date={date};field=wind_code,name,volume,market,cash_substitute,cash_substitute_flag",
        )
        logger.info(f"ETF {wind_code} PCF on {date}: {len(df)} items")
        return df

    def fetch_etf_daily_batch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "F",
    ) -> dict[str, pd.DataFrame]:
        """批量获取多只ETF日K线"""
        result = {}
        for code in codes:
            try:
                df = self.fetch_etf_daily(code, start_date, end_date, adjust)
                if not df.empty:
                    result[code] = df
            except Exception as e:
                logger.error(f"Failed to fetch ETF {code}: {e}")
        logger.info(f"Batch fetched {len(result)}/{len(codes)} ETFs")
        return result
