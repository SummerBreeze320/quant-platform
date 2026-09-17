import pandas as pd
from typing import List, Optional
from src.data_pipeline.wind_client import WindClient
from src.data_pipeline.transformer import DataTransformer
from src.common.logger import logger

class WindDataCollector:
    """Extracts market quotes, trading calendars and index constituents using WindPy."""

    def __init__(self, client: Optional[WindClient] = None):
        self.client = client or WindClient()

    def get_trade_days(self, start_date: str, end_date: str) -> List[str]:
        """Fetches trading days from Wind."""
        if not self.client.is_connected():
            if not self.client.connect():
                raise ConnectionError("Cannot fetch trade days: WindPy not connected.")

        w = self.client.w
        data = w.tdays(start_date, end_date, "")
        if hasattr(data, "ErrorCode") and data.ErrorCode != 0:
            raise RuntimeError(f"Wind tdays failed with ErrorCode: {data.ErrorCode}")

        if hasattr(data, "Data") and len(data.Data) > 0:
            # Format dates to YYYY-MM-DD
            dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in data.Data[0]]
            return dates
        return []

    def get_sector_constituents(self, sector_id: str = "a001010100000000", date: Optional[str] = None) -> List[str]:
        """
        Fetches constituent stock symbols for an index or sector.
        sector_id:
            - '000300.SH' (CSI 300)
            - '000905.SH' (CSI 500)
            - '000852.SH' (CSI 1000)
            - 'a001010100000000' (All A-Shares)
        """
        if not self.client.is_connected():
            if not self.client.connect():
                raise ConnectionError("WindPy not connected.")

        w = self.client.w
        options = f"date={date};sectorid={sector_id}" if date else f"sectorid={sector_id}"
        data = w.wset("sectorconstituent", options)
        
        if hasattr(data, "ErrorCode") and data.ErrorCode != 0:
            logger.warning(f"wset sectorconstituent failed for sector {sector_id}: {data.ErrorCode}")
            return []

        if hasattr(data, "Data") and len(data.Data) > 1:
            # Usually index 1 holds wind_code
            symbols = [str(s).strip() for s in data.Data[1]]
            return symbols
        return []

    def get_daily_quotes(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetches daily quotes for multiple symbols across a date range."""
        if not symbols:
            return pd.DataFrame()

        if not self.client.is_connected():
            if not self.client.connect():
                raise ConnectionError("WindPy not connected.")

        req_fields = fields or ["open", "high", "low", "close", "volume", "amt", "adjfactor", "vwap"]
        field_str = ",".join(req_fields)
        symbol_str = ",".join(symbols)

        w = self.client.w
        logger.info(f"Fetching daily quotes for {len(symbols)} symbols from {start_date} to {end_date}...")
        
        data = w.wsd(symbol_str, field_str, start_date, end_date, "adj=None;Period=D")
        
        if hasattr(data, "ErrorCode") and data.ErrorCode != 0:
            raise RuntimeError(f"Wind wsd failed with ErrorCode: {data.ErrorCode}")

        if not hasattr(data, "Data") or not data.Data:
            return pd.DataFrame()

        # Reshape Wind wsd output into DataFrame
        records = []
        codes = data.Codes if hasattr(data, "Codes") else symbols
        times = data.Times if hasattr(data, "Times") else []
        field_names = [f.upper() for f in (data.Fields if hasattr(data, "Fields") else req_fields)]

        # Note: If single symbol vs multiple symbols, Wind returns varying shapes
        if len(codes) == 1:
            code = codes[0]
            for t_idx, t in enumerate(times):
                row = {"SEC_CODE": code, "DATETIME": t}
                for f_idx, f in enumerate(field_names):
                    row[f] = data.Data[f_idx][t_idx]
                records.append(row)
        else:
            # data.Data layout: [field_0_all_dates_code_0...code_N, ...] or 2D matrix
            # Using standard pandas reconstruction
            for c_idx, code in enumerate(codes):
                for t_idx, t in enumerate(times):
                    row = {"SEC_CODE": code, "DATETIME": t}
                    for f_idx, f in enumerate(field_names):
                        # Wind wsd with multiple codes returns Data as [field_idx][time_idx * num_codes + code_idx]
                        idx = t_idx * len(codes) + c_idx
                        if idx < len(data.Data[f_idx]):
                            row[f] = data.Data[f_idx][idx]
                    records.append(row)

        raw_df = pd.DataFrame(records)
        return DataTransformer.standardize_quotes(raw_df)
