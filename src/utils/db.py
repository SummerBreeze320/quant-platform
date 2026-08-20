"""Database connection utilities using SQLAlchemy."""
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """MySQL database manager"""

    def __init__(self, host: str = "127.0.0.1", port: int = 3306,
                 database: str = "quant_platform", user: str = "root",
                 password: str = ""):
        self.url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        self.engine = None
        self.Session = None

    def connect(self):
        self.engine = create_engine(self.url, pool_size=5, pool_recycle=3600)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Database connected: {self.engine.url}")

    def init_schema(self, sql_file: str = "scripts/init_db.sql"):
        sql_path = Path(sql_file)
        if not sql_path.exists():
            logger.error(f"SQL file not found: {sql_path}")
            return
        sql = sql_path.read_text(encoding="utf-8")
        with self.engine.connect() as conn:
            for statement in sql.split(";"):
                stmt = statement.strip()
                if stmt and not stmt.startswith("--"):
                    conn.execute(text(stmt))
            conn.commit()
        logger.info("Database schema initialized.")

    def get_session(self) -> Session:
        if not self.Session:
            self.connect()
        return self.Session()

    def execute(self, sql: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    def query_df(self, sql: str, params: dict = None):
        import pandas as pd
        return pd.read_sql(text(sql), self.engine, params=params)
