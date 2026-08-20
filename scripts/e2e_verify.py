"""End-to-end verification: mock-data → qlib.init → D.features() full chain."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    print("=" * 60)
    print("Phase 1 E2E Verification: mock-data → qlib.init → D.features()")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Generate mock data
        print("\n[1/5] Generating mock data...")
        from src.data.mock_data import MockDataGenerator
        gen = MockDataGenerator(qlib_dir=tmpdir)
        stats = gen.generate_qlib_data(n_stocks=10, n_etfs=10, seed=42)
        print(f"  Stocks: {stats['stocks']}, ETFs: {stats['etfs']}")
        print(f"  Calendar days: {stats['calendar_days']}")

        # Verify files exist
        features_dir = Path(tmpdir) / "features"
        inst_dirs = sorted([d.name for d in features_dir.iterdir() if d.is_dir()])
        print(f"  Instrument dirs: {len(inst_dirs)}")
        assert len(inst_dirs) == 20, f"Expected 20, got {len(inst_dirs)}"

        cal_path = Path(tmpdir) / "calendars" / "day.txt"
        print(f"  Calendar file exists: {cal_path.exists()}")
        assert cal_path.exists()

        # Step 2: Initialize Qlib
        print("\n[2/5] Initializing Qlib...")
        from src.core import init_qlib, reset
        reset()  # Ensure clean state
        init_qlib(provider_uri=tmpdir, region="cn")
        print("  Qlib initialized successfully")

        # Step 3: Read OHLCV via core module
        print("\n[3/5] Reading OHLCV via get_ohlcv()...")
        from src.core import get_ohlcv
        df = get_ohlcv("SH600000")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.index[0]} → {df.index[-1]}")
        print(f"  Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        assert not df.empty, "OHLCV data is empty"
        assert "close" in df.columns

        # Step 4: Read via D.features() directly (Qlib native API)
        print("\n[4/5] Reading via D.features() (Qlib native)...")
        from qlib.data import D
        df2 = D.features(
            ["SH600000", "SH510300"],
            ["$close", "$volume", "Mean($close, 5)", "Ref($close, 1)"],
            freq="day",
        )
        print(f"  Shape: {df2.shape}")
        print(f"  Columns: {list(df2.columns)}")
        print(f"  Instruments: {df2.index.get_level_values('instrument').unique().tolist()}")
        assert not df2.empty
        assert "Mean($close, 5)" in df2.columns

        # Step 5: List instruments
        print("\n[5/5] Listing instruments via core module...")
        from src.core import list_instruments
        instruments = list_instruments()
        print(f"  Total instruments: {len(instruments)}")
        print(f"  Sample: {instruments[:5]}")
        assert len(instruments) > 0

    print("\n" + "=" * 60)
    print("✓ Phase 1 E2E Verification PASSED")
    print("  - Mock data generation: OK")
    print("  - Qlib initialization: OK")
    print("  - get_ohlcv() via core: OK")
    print("  - D.features() native: OK")
    print("  - list_instruments(): OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
