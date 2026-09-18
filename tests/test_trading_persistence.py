import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from src.models.base import Base
from src.models.trading import (
    TradingAccountModel,
    TradingPositionModel,
    OrderRecordModel,
    TradeRecordModel,
    PMSTransferRecordModel,
)
from src.execution_engine.models import (
    Order, Trade, Position, AccountState, OrderDirection, OrderType, OrderStatus
)
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.execution_engine.persistence import TradingStorage
from src.pms.manager import PortfolioManager
from src.service.app import create_app


@pytest.fixture
def db_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return TestingSession


def test_trading_storage_orm_crud(db_session_factory):
    storage = TradingStorage(db_session_factory)

    acc = AccountState(
        account_id="acc_test",
        available_cash=500_000.0,
        total_equity=500_000.0,
        positions={
            "600519.SH": Position(
                symbol="600519.SH", total_volume=200, available_volume=200, avg_cost=1700.0, market_value=340_000.0
            )
        },
    )
    assert storage.save_account(acc)

    # Verify query
    with db_session_factory() as session:
        db_acc = session.get(TradingAccountModel, "acc_test")
        assert db_acc is not None
        assert db_acc.available_cash == 500_000.0

        db_pos = session.execute(
            select(TradingPositionModel).where(
                TradingPositionModel.account_id == "acc_test",
                TradingPositionModel.symbol == "600519.SH",
            )
        ).scalar_one_or_none()
        assert db_pos is not None
        assert db_pos.total_volume == 200

    # Delete position
    assert storage.delete_position("acc_test", "600519.SH")
    with db_session_factory() as session:
        db_pos = session.execute(
            select(TradingPositionModel).where(
                TradingPositionModel.account_id == "acc_test",
                TradingPositionModel.symbol == "600519.SH",
            )
        ).scalar_one_or_none()
        assert db_pos is None


def test_paper_broker_persistence_and_restart_recovery(db_session_factory):
    storage = TradingStorage(db_session_factory)
    broker1 = PaperBroker(storage=storage)

    # 1. Create account and execute Buy
    broker1.create_account("strat_alpha", initial_cash=1_000_000.0)
    buy_order = Order(
        order_id="ord_buy_1",
        account_id="strat_alpha",
        symbol="000001.SZ",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=10.0,
        volume=1000,
    )
    trade1 = broker1.submit_order(buy_order)
    assert trade1 is not None
    assert trade1.volume == 1000

    # Settle overnight so shares become available for selling
    broker1.settle_overnight("strat_alpha")

    # 2. Execute partial Sell
    sell_order = Order(
        order_id="ord_sell_1",
        account_id="strat_alpha",
        symbol="000001.SZ",
        direction=OrderDirection.SELL,
        order_type=OrderType.LIMIT,
        price=11.0,
        volume=400,
    )
    trade2 = broker1.submit_order(sell_order)
    assert trade2 is not None

    acc1 = broker1.get_account("strat_alpha")
    remaining_vol = acc1.positions["000001.SZ"].total_volume
    assert remaining_vol == 600
    saved_cash = acc1.available_cash

    # 3. Simulate process crash & restart: new PaperBroker instance reading from storage
    broker2 = PaperBroker(storage=storage)
    assert "strat_alpha" not in broker2.accounts

    restored_count = broker2.restore_from_storage()
    assert restored_count == 1
    assert "strat_alpha" in broker2.accounts

    acc2 = broker2.get_account("strat_alpha")
    assert acc2.account_id == "strat_alpha"
    assert pytest.approx(acc2.available_cash, 0.01) == saved_cash
    assert "000001.SZ" in acc2.positions
    assert acc2.positions["000001.SZ"].total_volume == 600

    # Verify trades and orders can be retrieved from storage
    historical_trades = storage.get_historical_trades("strat_alpha")
    assert len(historical_trades) == 2

    historical_orders = storage.get_historical_orders("strat_alpha")
    assert len(historical_orders) == 2


def test_pms_cash_transfer_persistence(db_session_factory):
    storage = TradingStorage(db_session_factory)
    broker = PaperBroker(storage=storage)
    pms = PortfolioManager(
        master_id="master_test",
        initial_reserve=5_000_000.0,
        broker=broker,
        storage=storage,
    )

    pms.register_strategy("strat_trend", "Trend Following", initial_budget=1_000_000.0)

    # Perform injection and withdrawal
    t1 = pms.transfer_cash("strat_trend", 500_000.0, reason="追加投资")
    assert t1.transfer_amount == 500_000.0

    t2 = pms.transfer_cash("strat_trend", -200_000.0, reason="利润计提")
    assert t2.transfer_amount == -200_000.0

    transfers = storage.get_historical_transfers("strat_trend")
    assert len(transfers) == 2
    reasons = [t["reason"] for t in transfers]
    assert "追加投资" in reasons
    assert "利润计提" in reasons


def test_api_trades_and_transfers_endpoints(db_session_factory):
    app = create_app(session_factory=db_session_factory)
    client = TestClient(app)

    # 1. Register strategy via PMS
    reg_resp = client.post("/api/v1/pms/strategies", json={
        "strategy_id": "persisted_strat",
        "name": "Persisted Strategy",
        "initial_budget": 2_000_000.0
    })
    assert reg_resp.status_code == 200

    # 2. Rebalance (execute trade)
    rebalance_resp = client.post("/api/v1/execution/rebalance", json={
        "account_id": "persisted_strat",
        "target_weights": {"600000.SH": 0.10},
        "current_prices": {"600000.SH": 10.0}
    })
    assert rebalance_resp.status_code == 200
    assert len(rebalance_resp.json()["executed_trades"]) == 1

    # 3. Query historical trades from API
    trades_resp = client.get("/api/v1/execution/trades", params={"account_id": "persisted_strat"})
    assert trades_resp.status_code == 200
    trades = trades_resp.json()
    assert len(trades) >= 1
    assert trades[0]["symbol"] == "600000.SH"

    # 4. Query historical orders from API
    orders_resp = client.get("/api/v1/execution/orders", params={"account_id": "persisted_strat"})
    assert orders_resp.status_code == 200
    orders = orders_resp.json()
    assert len(orders) >= 1

    # 5. Allocate / transfer cash and query transfers
    alloc_resp = client.post("/api/v1/pms/allocate", json={
        "method": "FIXED_BUDGET",
        "base_ratios": {"persisted_strat": 1.0},
        "apply": True
    })
    assert alloc_resp.status_code == 200

    transfers_resp = client.get("/api/v1/pms/transfers", params={"strategy_id": "persisted_strat"})
    assert transfers_resp.status_code == 200
    transfers = transfers_resp.json()
    assert len(transfers) >= 1
