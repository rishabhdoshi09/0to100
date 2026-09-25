from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from scan.market_scan_service import _default_universe, run_whole_market_scan
from scan.scan_coverage import (
    ANALYSIS_ERROR,
    LIQUIDITY_FILTER,
    NO_OHLCV,
    NO_SETUP,
    PRICE_FILTER,
    QUALIFIED,
)


def _frame(price: float = 100.0, volume: float = 500_000.0, days: int = 90) -> pd.DataFrame:
    idx = pd.date_range("2026-04-01", periods=days, freq="B")
    close = np.linspace(price * 0.9, price, days)
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(days, volume),
        },
        index=idx,
    )


def test_default_universe_uses_canonical_membership_and_kite_only_for_names(monkeypatch):
    canonical = [f"EQ{i:03d}" for i in range(205)] + ["VERYLONGEQ01", "NONAMEEQ"]
    names = {f"EQ{i:03d}": f"Official Company {i}" for i in range(205)}
    meta = {
        "VERYLONGEQ01": {
            "exchange": "NSE",
            "instrument_type": "EQ",
            "name": "Long Symbol Ltd",
        },
        "NONAMEEQ": {
            "exchange": "NSE",
            "instrument_type": "EQ",
            "name": "",
        },
        # A broker instrument outside the canonical approved universe must not
        # silently expand the scan denominator.
        "EXTRADEBT": {
            "exchange": "NSE",
            "instrument_type": "EQ",
            "name": "Extra Debt-like Instrument",
        },
    }

    class Manager:
        _meta_map = meta

    monkeypatch.setattr("data.nse_universe.get_nse_universe", lambda: canonical)
    monkeypatch.setattr("data.nse_universe.get_nse_universe_with_names", lambda: names)
    monkeypatch.setattr("data.instruments.InstrumentManager", Manager)

    universe = _default_universe()

    assert len(universe) == len(canonical)
    assert universe["EQ001"] == "Official Company 1"
    assert universe["VERYLONGEQ01"] == "Long Symbol Ltd"
    assert universe["NONAMEEQ"] == "NONAMEEQ"
    assert "EXTRADEBT" not in universe


def test_scan_audit_accounts_for_qualified_no_setup_policy_missing_and_error(monkeypatch):
    frames = {
        "QUAL": _frame(),
        "NONE": _frame(),
        "LOWPX": _frame(price=10.0),
        "LOWLIQ": _frame(price=100.0, volume=10_000.0),
        "ERR": _frame(),
    }

    class Scanner:
        def _analyze(self, symbol, df):
            price = float(df["close"].iloc[-1])
            if price < 20:
                return None
            if float(df["volume"].tail(20).mean()) * price < 1e7:
                return None
            if symbol == "ERR":
                raise RuntimeError("boom")
            signals = ["MOMENTUM"] if symbol == "QUAL" else []
            return SimpleNamespace(
                symbol=symbol,
                signals=signals,
                reasons=["ok"] if signals else [],
                score=80 if signals else 0,
                verdict="BUY" if signals else "WATCH",
                chase_risk=False,
                price=price,
                momentum_5d=2,
                rsi=55,
                volume_ratio=1.5,
                entry=101,
                stop=95,
                target=120,
            )

        def scan(self, symbols, progress=None, prefetch=False):
            results = []
            available = [s for s in symbols if s in frames]
            if progress:
                progress(0, len(available))
            for i, symbol in enumerate(available, start=1):
                try:
                    row = self._analyze(symbol, frames[symbol])
                    if row is not None and row.signals:
                        results.append(row)
                except Exception:
                    pass
                if progress:
                    progress(i, len(available))
            return results

    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", lambda: list(frames))

    names = {symbol: symbol for symbol in ["QUAL", "NONE", "LOWPX", "LOWLIQ", "ERR", "MISS"]}
    report = run_whole_market_scan(
        universe_provider=lambda: names,
        prefetch_fn=lambda *a, **k: len(frames),
        scanner=Scanner(),
        fno_provider=lambda: set(),
        save=False,
    )

    assert report.ok
    coverage = report.payload["coverage"]
    counts = coverage["reason_counts"]
    assert counts[QUALIFIED] == 1
    assert counts[NO_SETUP] == 1
    assert counts[PRICE_FILTER] == 1
    assert counts[LIQUIDITY_FILTER] == 1
    assert counts[ANALYSIS_ERROR] == 1
    assert counts[NO_OHLCV] == 1
    assert coverage["requested"] == 6
    assert coverage["checked"] == 4  # two technically evaluated + two explicit policy checks
    assert coverage["state"] == "DEGRADED"
    # Existing universe_size/scanned contract remains "actually checked"; the
    # requested universe is separately visible instead of being conflated.
    assert report.requested_universe == 6
    assert report.universe_size == 4
    assert report.scanned == 4
    assert report.payload["requested_universe"] == 6
    assert report.payload["universe_size"] == 4


def test_scan_priority_changes_order_not_membership():
    from scan.market_scan_service import priority_ordered_symbols

    symbols = ["AAA", "BBB", "CCC", "DDD"]
    ordered = priority_ordered_symbols(
        symbols,
        scan_payload={"records": [{"symbol": "CCC", "signals": ["MOMENTUM"]}]},
        watchlist=["DDD"],
    )
    assert ordered[:2] == ["CCC", "DDD"]
    assert set(ordered) == set(symbols)
    assert len(ordered) == len(symbols)


def test_coverage_distinguishes_missing_ohlcv_from_partial_history():
    from scan.scan_coverage import ScanCoverageProbe

    short = _frame(days=20)
    probe = ScanCoverageProbe(["NEWIPO", "MISS"], instrumented=True)
    probe.run_analyze(lambda _symbol, _df: None, "NEWIPO", short)
    audit = probe.finalize([], cached=["NEWIPO"], walked_total=1)

    summary = audit["summary"]
    assert summary["data_unavailable"] == 2
    assert summary["hard_data_unavailable"] == 1
    assert summary["partial_history"] == 1
    assert summary["state"] == "DEGRADED"

    rows = {row["symbol"]: row for row in audit["ledger"]}
    assert rows["NEWIPO"]["status"] == "INSUFFICIENT_HISTORY"
    assert rows["MISS"]["status"] == "NO_OHLCV"
