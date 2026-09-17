import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.base import Base
from src.models.sync_log import SyncLog
from src.tasks.jobs import job_daily_data_sync, job_daily_model_predict, job_weekend_factor_mining
from src.tasks.scheduler import QuantScheduler

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()

def test_job_daily_data_sync_mock(db_session):
    mock_collector = MagicMock()
    mock_collector.get_daily_quotes.return_value = MagicMock(empty=True)
    mock_dumper = MagicMock()
    mock_redis = MagicMock()
    
    # Test sync job with mocks
    with patch("src.tasks.jobs.WindDataCollector", return_value=mock_collector), \
         patch("src.tasks.jobs.QlibDumper", return_value=mock_dumper), \
         patch("src.tasks.jobs.get_redis", return_value=mock_redis):
        
        # Lock context manager mock
        mock_lock = MagicMock()
        mock_redis.distributed_lock.return_value.__enter__.return_value = True
        
        result = job_daily_data_sync(trade_date="2026-09-14", db=db_session)
        assert result["status"] == "SUCCESS"
        
        # Verify sync log persisted
        log = db_session.query(SyncLog).filter_by(sync_date="2026-09-14").first()
        assert log is not None
        assert log.status == "SUCCESS"

def test_scheduler_lifecycle():
    scheduler = QuantScheduler()
    scheduler.configure_jobs()
    
    jobs = scheduler.get_jobs()
    job_ids = [j.id for j in jobs]
    assert "daily_data_sync" in job_ids
    assert "daily_model_predict" in job_ids
    assert "weekend_factor_mining" in job_ids
