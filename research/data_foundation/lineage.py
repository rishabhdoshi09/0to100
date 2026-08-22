"""Lightweight provenance for research features."""
from __future__ import annotations

from typing import Any


def fundamentals_feature(
    name: str,
    *,
    current: dict | None,
    prior: dict | None,
    available_at,
    calc_version: str,
    value,
) -> dict[str, Any]:
    return {
        "feature": name,
        "value": value,
        "calc_version": calc_version,
        "source_filing": {
            "row_id": (current or {}).get("row_id"),
            "source": (current or {}).get("source"),
            "source_hash": (current or {}).get("source_hash"),
            "xbrl_url": (current or {}).get("xbrl_url"),
            "period_end": (current or {}).get("period_end"),
        },
        "prior_period": {
            "row_id": (prior or {}).get("row_id"),
            "period_end": (prior or {}).get("period_end"),
            "available_at": (prior or {}).get("available_at"),
        },
        "availability_timestamp": available_at,
    }


def price_feature(name: str, *, symbol: str, as_of: str, bar_date: str | None, source: str) -> dict[str, Any]:
    return {
        "feature": name,
        "symbol": symbol,
        "as_of": as_of,
        "bar_date": bar_date,
        "source": source,
    }
