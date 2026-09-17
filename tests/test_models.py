import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.base import Base
from src.models.factor import FactorMetadata
from src.models.sync_log import SyncLog
from src.models.model_registry import ModelRegistry
from src.models.backtest_record import BacktestRecord

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()

def test_factor_metadata_orm(session):
    factor = FactorMetadata(
        name="rev_5d",
        expression="Ref($close, -5) / $close - 1",
        category="reversal",
        description="5-day reversal factor",
        ic_mean=0.042,
        icir=0.65,
        created_by="RD-Agent",
        is_active=True
    )
    session.add(factor)
    session.commit()
    
    saved = session.query(FactorMetadata).filter_by(name="rev_5d").first()
    assert saved is not None
    assert saved.id is not None
    assert saved.ic_mean == 0.042
    assert saved.created_at is not None

def test_sync_log_orm(session):
    log = SyncLog(
        sync_date="2026-09-14",
        sync_type="daily",
        status="SUCCESS",
        symbols_count=5000,
        message="Synced successfully"
    )
    session.add(log)
    session.commit()
    
    saved = session.query(SyncLog).filter_by(sync_date="2026-09-14").first()
    assert saved is not None
    assert saved.status == "SUCCESS"

def test_model_registry_and_backtest_orm(session):
    model = ModelRegistry(
        model_name="lgb_alpha158_v1",
        model_type="LightGBM",
        feature_set=["alpha158", "rev_5d"],
        train_start="2020-01-01",
        train_end="2025-12-31",
        model_path="data/models/lgb_v1.pkl",
        metrics={"test_ic": 0.051, "test_icir": 0.72}
    )
    session.add(model)
    session.commit()
    
    bt = BacktestRecord(
        model_id=model.id,
        strategy_name="Top30_EqualWeight",
        start_date="2026-01-01",
        end_date="2026-09-14",
        benchmark="000300.SH",
        annualized_return=0.185,
        sharpe_ratio=1.62,
        max_drawdown=-0.082,
        win_rate=0.58
    )
    session.add(bt)
    session.commit()
    
    saved_bt = session.query(BacktestRecord).filter_by(model_id=model.id).first()
    assert saved_bt is not None
    assert saved_bt.sharpe_ratio == 1.62
