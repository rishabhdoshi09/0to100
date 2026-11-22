"""Input provenance of feature002 shadow rows (Cycle 7).

A row frozen before ``f002-input.v1`` carries no record of which history its
features were computed from (the active Kite snapshot, the official bhavcopy
store, or the yfinance cache: scan.bulk_fetcher.get_cached). Its provenance is
UNKNOWN and stays unknown: it is never inferred from which source happens to be
authoritative today. Analyses separate rows by ``row_input_source``.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

KNOWN = ("kite_snapshot", "bhavcopy_store", "yfinance_cache")


def row_input(row: Mapping[str, Any]) -> dict:
    snap = row.get("feature_snapshot")
    if isinstance(snap, str):
        try:
            snap = json.loads(snap)
        except Exception:
            snap = {}
    inp = (snap or {}).get("input") if isinstance(snap, Mapping) else None
    return dict(inp) if isinstance(inp, Mapping) else {}


def row_input_source(row: Mapping[str, Any]) -> str:
    """The recorded source, or 'unknown' (never inferred)."""
    src = str(row_input(row).get("source") or "")
    return src if src in KNOWN else "unknown"
