"""Test qlib.init() reading existing bin data."""
import qlib
from qlib.config import REG_CN

qlib.init(provider_uri="data/qlib_bin", region=REG_CN)

from qlib.data import D

# 1. Calendar
cal = D.calendar(start_time="2024-01-01", end_time="2024-12-31")
print(f"Calendar: {len(cal)} days, first={cal[0]}, last={cal[-1]}")

# 2. Instruments
inst = D.instruments(market="all")
inst_list = D.list_instruments(inst, as_list=True)
print(f"Instruments: {len(inst_list)} total, first 5: {inst_list[:5]}")

# 3. Features
df = D.features(["SH600000"], ["$close", "$volume", "$open"], start_time="2024-12-01", end_time="2024-12-31")
print(f"Features shape: {df.shape}")
print(df.tail(3))

# 4. Expression (Alpha158 style)
df2 = D.features(["SH600000"], ["Ref($close, 5) / $close"], start_time="2024-12-01", end_time="2024-12-31")
print(f"Expression feature shape: {df2.shape}")
print(df2.tail(3))
