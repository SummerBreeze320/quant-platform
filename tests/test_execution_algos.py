from src.execution_engine.models import Order, OrderDirection, AlgoType
from src.execution_engine.algos import DirectAlgo, TwapAlgo, VwapAlgo

def test_algos_slicing():
    parent = Order(
        order_id="p_01",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        price=100.0,
        volume=1000
    )

    # 1. DirectAlgo
    direct = DirectAlgo()
    slices_direct = direct.slice_order(parent)
    assert len(slices_direct) == 1
    assert slices_direct[0].volume == 1000

    # 2. TwapAlgo (5 slices of 200)
    twap = TwapAlgo(num_slices=5)
    slices_twap = twap.slice_order(parent)
    assert len(slices_twap) == 5
    assert sum(s.volume for s in slices_twap) == 1000
    assert all(s.volume % 100 == 0 for s in slices_twap)
    assert all(s.algo_type == AlgoType.TWAP for s in slices_twap)

    # 3. VwapAlgo (U-shape profile)
    vwap = VwapAlgo(volume_profile=[0.30, 0.15, 0.10, 0.15, 0.30])
    slices_vwap = vwap.slice_order(parent)
    assert len(slices_vwap) == 5
    assert sum(s.volume for s in slices_vwap) == 1000
    assert all(s.volume % 100 == 0 for s in slices_vwap)
    assert slices_vwap[0].volume == 300
