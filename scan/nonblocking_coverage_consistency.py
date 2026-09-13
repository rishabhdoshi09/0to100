"""Keep whole-market coverage observation off the synchronous repair path.

The canonical whole-market scan already requires current official NSE history before
it starts. Coverage accounting must therefore observe that authoritative cached
store immediately; it must not synchronously walk the broker/public repair ladder
before the first stock progress event.

Missing symbols remain explicit ``NO_OHLCV`` rows in the coverage ledger. Callers
that intentionally want targeted repair can still opt in with
``repair_missing=True``.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

_INSTALLED = False


@contextmanager
def _observe_nonblocking(scanner: Any, symbols: Iterable[str], *, repair_missing: bool = False):
    from scan import scan_coverage as coverage

    requested = [coverage._symbol(s) for s in symbols if coverage._symbol(s)]

    if repair_missing:
        original_context = getattr(_observe_nonblocking, "__qt_original_observe__", None)
        if original_context is None:
            raise RuntimeError("original coverage observer is unavailable")
        with original_context(scanner, requested) as probe:
            yield probe
        return

    history_repair = {
        "skipped": True,
        "reason": "whole_market_scan_nonblocking",
        "requested": len(requested),
        "attempted": 0,
        "loaded": 0,
    }

    original_analyze = getattr(scanner, "_analyze", None)
    instrumented = callable(original_analyze)
    probe = coverage.ScanCoverageProbe(
        requested,
        instrumented=instrumented,
        history_repair=history_repair,
        universe_provenance=coverage._universe_provenance(),
    )

    if not instrumented:
        yield probe
        return

    def wrapped(symbol, df):
        return probe.run_analyze(original_analyze, symbol, df)

    scanner._analyze = wrapped
    try:
        yield probe
    finally:
        scanner._analyze = original_analyze


def install() -> None:
    """Install the non-blocking coverage observer exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return

    from scan import scan_coverage as coverage

    current = coverage.observe_scanner
    if getattr(current, "__qt_nonblocking_coverage__", False):
        _INSTALLED = True
        return

    setattr(_observe_nonblocking, "__qt_original_observe__", current)
    setattr(_observe_nonblocking, "__qt_nonblocking_coverage__", True)
    coverage.observe_scanner = _observe_nonblocking
    _INSTALLED = True
