"""
NSE universe loader — provides a comprehensive list of NSE-listed equity symbols
with company names. Falls back to the bundled CSV if live data is unavailable.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

_CSV_PATH = Path(__file__).parent / "nse_symbols_fallback.csv"


def _load_csv() -> Dict[str, str]:
    """Load symbol → company name mapping from the bundled CSV."""
    out: Dict[str, str] = {}
    try:
        with open(_CSV_PATH, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sym = row.get("SYMBOL", "").strip().upper()
                name = row.get("NAME", "").strip()
                if sym:
                    out[sym] = name or sym
    except Exception:
        pass
    return out


def get_nse_universe_with_names() -> Dict[str, str]:
    """
    Return a dict of {SYMBOL: "Company Name"} for NSE-listed equities.
    Uses the bundled CSV as the primary source so it works without a live
    Kite connection.
    """
    return _load_csv()


def get_nse_universe() -> List[str]:
    """Return a sorted list of NSE equity symbols."""
    return sorted(get_nse_universe_with_names().keys())
