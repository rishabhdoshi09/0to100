"""
Shared data structures for the Quant Red Flag Analyst module.

Every metric carries its own explanation (what / why / good / implication)
so the report layer never shows an unexplained number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


SEVERITY_ORDER = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Verdict(str, Enum):
    STRONG_CANDIDATE = "Strong Candidate"
    WATCHLIST = "Watchlist"
    NEUTRAL = "Neutral"
    CAUTION = "Caution"
    HIGH_RISK = "High Risk"
    AVOID = "Avoid"


@dataclass
class Metric:
    """A single analyzed metric with full explanation."""
    name: str
    value: Optional[float]
    display: str                      # human-readable value, e.g. "1.42x"
    what: str                         # what is it?
    why: str                          # why does it matter?
    good: str                         # what is considered good?
    implication: str                  # what does THIS result imply?
    score: Optional[float] = None     # 0–100 contribution, None = not scored


@dataclass
class RedFlag:
    """A detected red flag with evidence and context."""
    title: str
    severity: Severity
    evidence: str                     # human-readable summary
    why_it_matters: str
    precedent: str                    # historical precedent
    period: str = ""                  # fiscal period
    evidence_rows: list = field(default_factory=list)  # [(label, value), ...]
    sources: list = field(default_factory=list)        # statement sources used


@dataclass
class LayerResult:
    """Output of one analysis layer."""
    layer: str
    score: Optional[float]            # 0–100, None if insufficient data
    metrics: List[Metric] = field(default_factory=list)
    flags: List[RedFlag] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FundamentalData:
    """Raw inputs fetched once and shared across all layers."""
    symbol: str
    ticker: str                       # resolved data-source ticker (e.g. RELIANCE.NS)
    info: Dict[str, Any]
    income: pd.DataFrame              # annual income statement (cols = fiscal years, newest first)
    balance: pd.DataFrame
    cashflow: pd.DataFrame
    quarterly_income: pd.DataFrame
    prices: pd.DataFrame              # daily OHLCV, ascending index
    benchmark: pd.DataFrame           # index OHLCV for beta
    insider_transactions: Optional[pd.DataFrame] = None
    institutional_holders: Optional[pd.DataFrame] = None
    major_holders: Optional[pd.DataFrame] = None

    def years(self) -> List[Any]:
        """Fiscal year columns, newest first."""
        return list(self.income.columns) if not self.income.empty else []


def stmt_row(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    """Fetch a statement line item trying several label variants."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def val(series: Optional[pd.Series], col_idx: int = 0) -> Optional[float]:
    """Value at fiscal-year offset (0 = latest). None if missing/NaN."""
    if series is None or len(series) <= col_idx:
        return None
    v = series.iloc[col_idx]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return float(v)


def pct_change(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev is None or prev == 0:
        return None
    return (curr - prev) / abs(prev)
