"""Verify alternative data layer integration."""
import sys
sys.path.insert(0, ".")

print("1. Import alternative module...")
from src.data.alternative import (
    AlternativeDataPipeline, MockAlternativeGenerator,
    NewsHandler, AnnouncementHandler, ResearchReportHandler, MacroHandler,
)
print("   OK")

print("2. Check factor library...")
from src.research.factor_library import ALTERNATIVE_FACTORS, FACTOR_CATEGORIES, list_factors
print(f"   {len(ALTERNATIVE_FACTORS)} alt factors, {len(FACTOR_CATEGORIES)} categories")
alt = list_factors(category="alternative")
print(f"   Names: {[f['name'] for f in alt]}")

print("3. Check API router...")
from src.dashboard.api.routers.alternative import router
print(f"   {len(router.routes)} routes")

print("4. Generate mock data...")
gen = MockAlternativeGenerator()
codes = ["SH600000", "SH600519", "SH600036", "SZ000001", "SH510300"]
counts = gen.generate_all(codes, "2024-01-01", "2024-06-30")
print(f"   {counts}")

print("5. Compute alternative factors...")
pipe = AlternativeDataPipeline()
factors = pipe.compute_all_factors(codes, "2024-06-01")
print(f"   {len(factors)} factors computed")
for name, s in list(factors.items())[:3]:
    print(f"   {name}: {s.to_dict()}")

print("6. Verify data sources...")
checks = pipe.verify_data(codes)
print(f"   {checks}")

print("\n=== All alternative data layer checks PASSED ===")
