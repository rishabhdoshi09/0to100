"""
Central configuration — loaded once at startup from environment / .env file.
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

    # ── Zerodha ───────────────────────────────────────────────────────────────
    kite_api_key: str = Field(default="")
    kite_api_secret: str = Field(default="")
    kite_access_token: str = Field(default="")

    # ── DeepSeek ──────────────────────────────────────────────────────────────
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    deepseek_model: str = Field(default="deepseek-chat")

    # ── News ──────────────────────────────────────────────────────────────────
    news_rss_feeds: str = Field(
        default="https://feeds.feedburner.com/ndtvprofit-latest"
    )

    @property
    def rss_feed_list(self) -> List[str]:
        return [u.strip() for u in self.news_rss_feeds.split(",") if u.strip()]

    # ── Universe ──────────────────────────────────────────────────────────────
    universe: str = Field(default="RELIANCE,INFY,TCS,HDFCBANK,ICICIBANK")

    @property
    def symbol_list(self) -> List[str]:
        return [s.strip() for s in self.universe.split(",") if s.strip()]

    # ── Risk ──────────────────────────────────────────────────────────────────
    max_capital_exposure: float = Field(default=0.20)
    max_position_size_pct: float = Field(default=0.10)
    max_open_positions: int = Field(default=5)
    max_daily_loss_pct: float = Field(default=0.02)
    min_signal_confidence: float = Field(default=0.60)

    # ── Engine ────────────────────────────────────────────────────────────────
    cycle_interval_seconds: int = Field(default=300)
    exchange: str = Field(default="NSE")
    product_type: str = Field(default="MIS")

    # ── Backtest ──────────────────────────────────────────────────────────────
    backtest_slippage: float = Field(default=0.0005)
    backtest_transaction_cost: float = Field(default=0.001)
    backtest_initial_capital: float = Field(default=1_000_000.0)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"))

    @field_validator("log_dir", mode="before")
    @classmethod
    def _make_log_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
