"""Cycle 12: the walk-forward measures stocks only, like the scan it certifies.

Research on an isolated copy of the production store (26 Nov): the walk-forward sampled
store_symbols() directly, and 137 of the store's 3,471 symbols are fund units (liquid-rate
funds, ETFs, BEES); 25 sat in the 800-symbol production sample. A liquid fund's near-flat NAV
gives the scanner a stop 0.01% below entry, so an ordinary move is a -282R "loss": every
trade with a stop under 1% in the test sample was LIQUID1, LIQUID or LTGILTBEES (mean -18.9R),
and they dominated the tight-stop patterns' means (FLAT_BASE, ASC_TRIANGLE). The live scan
already drops these (build_scan_payload -> _is_non_stock_fund); the evidence that certifies its
BUYs must measure the same universe.
"""
from __future__ import annotations

from scan import signal_backtest as SB


def test_the_walk_forward_samples_stocks_only(monkeypatch):
    seen = {}
    monkeypatch.setattr("data.bhavcopy_store.store_symbols",
                        lambda: ["TCS", "LIQUID", "LIQUID1", "NIFTYBEES", "GOLDETF", "INFY", "RELIANCE"])

    def inner(sc, stats, symbols, *a, **k):
        seen["symbols"] = list(symbols)
        return {"signals": {}}

    monkeypatch.setattr(SB, "_run_backtest_inner", inner)
    SB.run_backtest(max_symbols=800)
    assert sorted(seen["symbols"]) == ["INFY", "RELIANCE", "TCS"]


def test_the_report_records_its_stock_universe(monkeypatch):
    monkeypatch.setattr("data.bhavcopy_store.store_symbols", lambda: ["TCS", "LIQUID", "INFY"])
    assert SB.universe_note(["TCS", "LIQUID", "INFY"]) == {"filter": SB.UNIVERSE_FILTER, "excluded_funds": 1,
                                                           "stocks": 2}


def test_contaminated_evidence_cannot_certify():
    from scan import unified_scanner as US

    assert SB.REPORT_SCHEMA >= 7 and US.EVIDENCE_MIN_SCHEMA >= 7
