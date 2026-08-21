"""Research scanner = production UnifiedScanner._analyze.

Canonical A–G evaluation is daily. This module never substitutes 5-day
sampling. Speed-ups must keep the same function and prove equivalence on a
frozen sample.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd


def make_production_scanner():
    from scan.unified_scanner import UnifiedScanner
    return UnifiedScanner()


def research_scanner_analyze(scanner, symbol: str, hist: pd.DataFrame | None):
    """The research-equivalent scanner **is** UnifiedScanner._analyze."""
    if scanner is None:
        return None
    return scanner._analyze(symbol, hist)


def scanner_signal_ok(hit) -> bool:
    """SEPA-001/001R/R2 baseline A: any non-empty signal list with a plan."""
    return bool(hit is not None and getattr(hit, "signals", None))


def equivalence_pairs(
    scanner,
    sample: list[tuple[str, pd.DataFrame]],
    *,
    alt: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Compare research wrapper vs UnifiedScanner._analyze on a frozen sample.

    ``alt`` defaults to the same function (identity). A faster equivalent
    must pass this before replacing the wrapper in the canonical runner.
    """
    fn = alt or research_scanner_analyze
    rows = []
    n_match = 0
    for symbol, hist in sample:
        gold = scanner._analyze(symbol, hist)
        got = fn(scanner, symbol, hist)
        g_sig = list(getattr(gold, "signals", None) or []) if gold is not None else []
        o_sig = list(getattr(got, "signals", None) or []) if got is not None else []
        g_entry = getattr(gold, "entry", None) if gold is not None else None
        o_entry = getattr(got, "entry", None) if got is not None else None
        g_stop = getattr(gold, "stop", None) if gold is not None else None
        o_stop = getattr(got, "stop", None) if got is not None else None
        match = g_sig == o_sig and g_entry == o_entry and g_stop == o_stop
        n_match += int(match)
        rows.append({
            "symbol": symbol,
            "match": match,
            "gold_signals": g_sig,
            "got_signals": o_sig,
        })
    return {
        "n": len(sample),
        "n_match": n_match,
        "equivalent": n_match == len(sample) and len(sample) > 0,
        "function": "UnifiedScanner._analyze",
        "rows": rows,
    }
