"""
PostgreSQL client for fundamental data, earnings, macro indicators,
and system metadata (model versions, alerts, etc.).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config.settings import settings

Base = declarative_base()


class PostgresClient:
    """
    Wraps SQLAlchemy engine for PostgreSQL.
    Silently degrades if the database is unreachable.
    """

    def __init__(self) -> None:
        self._engine = None
        self._SessionLocal = None
        self._available: Optional[bool] = None

    def _get_engine(self):
        if self._engine is None:
            try:
                self._engine = create_engine(
                    settings.postgres_dsn,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                )
                self._SessionLocal = sessionmaker(bind=self._engine)
            except Exception as exc:
                logger.error(f"PostgreSQL engine creation failed: {exc}")
        return self._engine

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        engine = self._get_engine()
        if engine is None:
            self._available = False
            return False
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._available = True
        except OperationalError:
            self._available = False
            logger.warning("PostgreSQL not reachable — fundamental data unavailable")
        return self._available

    def init_schema(self) -> None:
        """Create tables if they don't exist."""
        if not self.is_available():
            return
        ddl = """
        CREATE TABLE IF NOT EXISTS earnings (
            id          SERIAL PRIMARY KEY,
            symbol      VARCHAR(20)  NOT NULL,
            report_date DATE         NOT NULL,
            eps_actual  FLOAT,
            eps_est     FLOAT,
            revenue_actual FLOAT,
            revenue_est    FLOAT,
            surprise_pct   FLOAT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS macro_indicators (
            id          SERIAL PRIMARY KEY,
            indicator   VARCHAR(50)  NOT NULL,
            release_date DATE        NOT NULL,
            actual      FLOAT,
            expected    FLOAT,
            surprise    FLOAT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS model_registry (
            id          SERIAL PRIMARY KEY,
            model_name  VARCHAR(100) NOT NULL,
            version     VARCHAR(50)  NOT NULL,
            train_start DATE,
            train_end   DATE,
            val_sharpe  FLOAT,
            val_accuracy FLOAT,
            artifact_path TEXT,
            is_active   BOOLEAN DEFAULT FALSE,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS performance_log (
            id          SERIAL PRIMARY KEY,
            log_date    DATE         NOT NULL,
            symbol      VARCHAR(20),
            pnl         FLOAT,
            win_rate    FLOAT,
            sharpe      FLOAT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS ix_earnings_symbol ON earnings(symbol);
        CREATE INDEX IF NOT EXISTS ix_model_active ON model_registry(is_active);
        """
        with self._get_engine().connect() as conn:
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()
        logger.info("PostgreSQL schema initialised")

    # ── Generic query helpers ─────────────────────────────────────────────

    def read_sql(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        if not self.is_available():
            return pd.DataFrame()
        try:
            return pd.read_sql(query, self._get_engine(), params=params)
        except Exception as exc:
            logger.error(f"PostgreSQL read failed: {exc}")
            return pd.DataFrame()

    def execute(self, sql: str, params: Optional[Dict] = None) -> None:
        if not self.is_available():
            return
        try:
            with self._get_engine().connect() as conn:
                conn.execute(text(sql), params or {})
                conn.commit()
        except Exception as exc:
            logger.error(f"PostgreSQL execute failed: {exc}")

    # ── Domain helpers ────────────────────────────────────────────────────

    def upsert_earnings(self, symbol: str, records: List[Dict]) -> None:
        for rec in records:
            self.execute(
                """
                INSERT INTO earnings (symbol, report_date, eps_actual, eps_est,
                    revenue_actual, revenue_est, surprise_pct)
                VALUES (:symbol, :report_date, :eps_actual, :eps_est,
                    :revenue_actual, :revenue_est, :surprise_pct)
                ON CONFLICT DO NOTHING
                """,
                {"symbol": symbol, **rec},
            )

    def get_latest_earnings(self, symbol: str, n: int = 8) -> pd.DataFrame:
        return self.read_sql(
            "SELECT * FROM earnings WHERE symbol = %(symbol)s "
            "ORDER BY report_date DESC LIMIT %(n)s",
            {"symbol": symbol, "n": n},
        )

    def register_model(
        self,
        model_name: str,
        version: str,
        train_start,
        train_end,
        val_sharpe: float,
        val_accuracy: float,
        artifact_path: str,
    ) -> None:
        # Deactivate previous versions
        self.execute(
            "UPDATE model_registry SET is_active=FALSE WHERE model_name=:name",
            {"name": model_name},
        )
        self.execute(
            """
            INSERT INTO model_registry (model_name, version, train_start, train_end,
                val_sharpe, val_accuracy, artifact_path, is_active)
            VALUES (:name, :version, :ts, :te, :sharpe, :acc, :path, TRUE)
            """,
            {
                "name": model_name,
                "version": version,
                "ts": train_start,
                "te": train_end,
                "sharpe": val_sharpe,
                "acc": val_accuracy,
                "path": artifact_path,
            },
        )
