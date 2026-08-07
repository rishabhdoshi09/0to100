"""Canonical observation contracts for QuantTerm company and market data."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class QualityStatus(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    ERROR = "ERROR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DataCapability(str, Enum):
    SECURITY_MASTER = "SECURITY_MASTER"
    DAILY_PRICES = "DAILY_PRICES"
    LIVE_QUOTES = "LIVE_QUOTES"
    FUNDAMENTALS = "FUNDAMENTALS"
    OWNERSHIP = "OWNERSHIP"
    CORPORATE_ACTIONS = "CORPORATE_ACTIONS"
    CORPORATE_EVENTS = "CORPORATE_EVENTS"
    NEWS = "NEWS"


@dataclass(frozen=True)
class ObservationMeta:
    symbol: str
    source: str
    source_date: str = ""
    retrieved_at: str = ""
    frequency: str = ""
    scope: str = ""
    currency: str = "INR"
    unit: str = ""
    period_start: str = ""
    period_end: str = ""
    quality_status: QualityStatus = QualityStatus.MISSING
    missing_reason: str = ""


@dataclass
class CompanyProfile:
    symbol: str
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    exchange: str = "NSE"
    series: str = "EQ"
    isin: str = ""
    face_value: float | None = None
    listing_date: str = ""
    fno_eligible: bool = False
    index_membership: list[str] = field(default_factory=list)
    active: bool = True
    meta: ObservationMeta | None = None


@dataclass
class MarketSnapshot:
    symbol: str
    last_price: float | None = None
    previous_close: float | None = None
    change_pct: float | None = None
    volume: float | None = None
    turnover: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    market_cap: float | None = None
    as_of: str = ""
    meta: ObservationMeta | None = None


@dataclass
class FinancialRatioSnapshot:
    symbol: str
    key: str
    label: str
    value: float | None
    formula: str = ""
    numerator: str = ""
    denominator: str = ""
    period: str = ""
    scope: str = ""
    missing_reason: str = ""
    meta: ObservationMeta | None = None


@dataclass
class DataCoverage:
    symbol: str
    identity: QualityStatus = QualityStatus.MISSING
    price_history: QualityStatus = QualityStatus.MISSING
    latest_market: QualityStatus = QualityStatus.MISSING
    fundamentals: QualityStatus = QualityStatus.MISSING
    ratios: QualityStatus = QualityStatus.MISSING
    ownership: QualityStatus = QualityStatus.MISSING
    events: QualityStatus = QualityStatus.MISSING
    corporate_actions: QualityStatus = QualityStatus.MISSING
    scan_eligible: QualityStatus = QualityStatus.MISSING
    long_term_eligible: QualityStatus = QualityStatus.MISSING
    reasons: dict[str, str] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat()
