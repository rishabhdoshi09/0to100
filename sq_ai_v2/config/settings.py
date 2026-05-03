"""
Central settings — single source of truth for every config value.
All modules import `settings` from here; nothing reads os.environ directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Zerodha / Kite Connect ────────────────────────────────────────────
    kite_api_key: str = Field(default="")
    kite_api_secret: str = Field(default="")
    kite_access_token: str = Field(default="")

    # ── Optional data-source keys ─────────────────────────────────────────
    alpha_vantage_key: str = Field(default="")
    newsapi_key: str = Field(default="")
    finnhub_key: str = Field(default="")

    # ── QuestDB ───────────────────────────────────────────────────────────
    questdb_host: str = Field(default="localhost")
    questdb_port: int = Field(default=9009)      # ILP
    questdb_http_port: int = Field(default=9000)  # REST

    # ── PostgreSQL ────────────────────────────────────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="sqai")
    postgres_user: str = Field(default="sqai")
    postgres_password: str = Field(default="sqai_secret")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str = Field(default="")

    # ── Trading Universe ──────────────────────────────────────────────────
    universe: str = Field(
        default="RELIANCE,INFY,TCS,HDFCBANK,ICICIBANK,WIPRO,AXISBANK,KOTAKBANK,LT,BAJFINANCE"
    )

    @property
    def symbol_list(self) -> List[str]:
        return [s.strip() for s in self.universe.split(",") if s.strip()]

    # ── Risk ──────────────────────────────────────────────────────────────
    max_capital_exposure: float = Field(default=0.80)
    max_position_size_pct: float = Field(default=0.10)
    max_open_positions: int = Field(default=8)
    max_daily_loss_pct: float = Field(default=0.02)
    min_signal_confidence: float = Field(default=0.55)
    kelly_fraction: float = Field(default=0.25)   # fractional Kelly multiplier
    vol_target: float = Field(default=0.15)        # annualised volatility target

    # ── Backtest ──────────────────────────────────────────────────────────
    backtest_initial_capital: float = Field(default=1_000_000.0)
    backtest_slippage: float = Field(default=0.0005)
    backtest_transaction_cost: float = Field(default=0.001)
    backtest_start_date: str = Field(default="2020-01-01")
    backtest_end_date: str = Field(default="2023-12-31")

    # ── Walk-forward ──────────────────────────────────────────────────────
    walk_forward_train_years: int = Field(default=3)
    walk_forward_test_months: int = Field(default=3)

    # ── Models ────────────────────────────────────────────────────────────
    model_dir: Path = Field(default=Path("models/saved"))

    @field_validator("model_dir", mode="before")
    @classmethod
    def _make_model_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── HuggingFace / FinBERT ─────────────────────────────────────────────
    hf_model_name: str = Field(default="ProsusAI/finbert")
    hf_cache_dir: str = Field(default=".hf_cache")

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"))

    @field_validator("log_dir", mode="before")
    @classmethod
    def _make_log_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
