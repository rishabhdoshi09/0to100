"""
Data Acquisition Layer (DAL) — data as a product.

Every external artifact the platform consumes flows through here and is
stored as a JSON envelope carrying full provenance:

    {
      "symbol":      "RELIANCE",
      "category":    "shareholding",
      "source":      "NSE Shareholding Pattern",
      "reliability": 95.0,
      "fetched_at":  "2026-06-11T10:42:00",
      "period":      "2026-03-31",          # the period the data describes
      "payload":     { ... }
    }

Directory layout (under sq_ai/datastore/ — `data/` is the Kite client package):
    datastore/
    ├── filings/
    ├── shareholding/
    ├── auditors/
    ├── bhavcopy/
    ├── announcements/
    ├── concalls/
    └── annual_reports/

Because every record carries `period` and `fetched_at`, the DAL natively
supports point-in-time queries (`get_as_of`) — the foundation of Research
Replay: an analysis dated 2023-06-01 only sees records knowable then.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from forensics.models import SOURCE_RELIABILITY

DATA_ROOT = Path(__file__).parent.parent / "datastore"

CATEGORIES = [
    "filings", "shareholding", "auditors", "bhavcopy",
    "announcements", "concalls", "annual_reports",
]


@dataclass
class DalRecord:
    symbol: str
    category: str
    source: str
    reliability: float
    fetched_at: str
    period: Optional[str]
    payload: Any

    @classmethod
    def from_dict(cls, d: Dict) -> "DalRecord":
        return cls(
            symbol=d.get("symbol", ""),
            category=d.get("category", ""),
            source=d.get("source", "Alternative / Web Data"),
            reliability=float(d.get("reliability", 50.0)),
            fetched_at=d.get("fetched_at", ""),
            period=d.get("period"),
            payload=d.get("payload"),
        )


def _dir(category: str) -> Path:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown DAL category '{category}'. "
                         f"Valid: {', '.join(CATEGORIES)}")
    p = DATA_ROOT / category
    p.mkdir(parents=True, exist_ok=True)
    return p


def put(category: str, symbol: str, payload: Any, source: str,
        period: Optional[str] = None) -> Path:
    """Store an artifact with full provenance. Returns the file path.

    Records are keyed by (symbol, period); re-putting the same period
    overwrites — the newest fetch of a period is the canonical one.
    """
    symbol = symbol.upper()
    rec = {
        "symbol": symbol,
        "category": category,
        "source": source,
        "reliability": SOURCE_RELIABILITY.get(source, 50.0),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "period": period,
        "payload": payload,
    }
    key = period or datetime.now().strftime("%Y-%m-%d")
    path = _dir(category) / f"{symbol}_{key}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    return path


def _load_all(category: str, symbol: str) -> List[DalRecord]:
    symbol = symbol.upper()
    recs: List[DalRecord] = []
    for path in sorted(_dir(category).glob(f"{symbol}_*.json")):
        try:
            with open(path) as f:
                recs.append(DalRecord.from_dict(json.load(f)))
        except Exception:
            continue
    return recs


def get_latest(category: str, symbol: str) -> Optional[DalRecord]:
    """Most recent record by period (falling back to fetched_at)."""
    recs = _load_all(category, symbol)
    if not recs:
        return None
    return max(recs, key=lambda r: (r.period or "", r.fetched_at))


def get_as_of(category: str, symbol: str,
              as_of: date) -> Optional[DalRecord]:
    """Latest record whose period is on or before `as_of`.

    This is the point-in-time view used by Research Replay: only data
    that described a period knowable at `as_of` is visible.
    """
    cutoff = as_of.isoformat()
    eligible = [r for r in _load_all(category, symbol)
                if (r.period or r.fetched_at[:10]) <= cutoff]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r.period or "", r.fetched_at))


def history(category: str, symbol: str) -> List[DalRecord]:
    """All records for a symbol in a category, oldest first."""
    return sorted(_load_all(category, symbol),
                  key=lambda r: (r.period or "", r.fetched_at))


def coverage_summary(symbol: str) -> Dict[str, int]:
    """How many records exist per category — the data-moat dashboard."""
    return {cat: len(_load_all(cat, symbol)) for cat in CATEGORIES}
