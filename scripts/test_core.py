"""Test the new core data_access module."""
import sys
sys.path.insert(0, ".")

from src.core import (
    init_qlib,
    get_calendar,
    list_instruments,
    get_features,
    get_field,
    get_ohlcv,
    normalize_code,
)

# 1. Init
init_qlib(provider_uri="data/qlib_bin", region="cn")

# 2. Calendar
cal = get_calendar("2024-12-01", "2024-12-31")
print(f"Calendar: {len(cal)} days")

# 3. Instruments
inst = list_instruments()
print(f"Instruments: {len(inst)} total, first 3: {inst[:3]}")

# 4. get_field (drop-in replacement for converter.read_instrument)
s = get_field("SH600000", "close", "2024-12-01", "2024-12-31")
print(f"get_field close: {len(s)} points, last={s.iloc[-1]:.4f}")

# 5. get_ohlcv
df = get_ohlcv("SH600000", "2024-12-01", "2024-12-31")
print(f"get_ohlcv: {df.shape}, columns={list(df.columns)}")
print(df.tail(2))

# 6. normalize_code
print(f"normalize_code('600000.SH') = {normalize_code('600000.SH')}")
print(f"normalize_code('SH600000') = {normalize_code('SH600000')}")

# 7. Qlib expression
df2 = get_features(["SH600000"], ["Ref($close, 5) / $close"], "2024-12-01", "2024-12-31")
print(f"Expression: {df2.shape}")
print(df2.tail(2))

print("\nAll core tests passed!")
