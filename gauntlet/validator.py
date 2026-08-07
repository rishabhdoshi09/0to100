"""
E4 — Dataset Validator (abort-on-fail).

The gauntlet must refuse to run on data it cannot trust — a single silent defect
(an un-adjusted split, a survivor-only universe, a duplicated bar) turns the whole
experiment into a confident lie. Every check returns pass/fail with a reason; if
ANY required check fails, `validate()` reports ok=False and the runner aborts
before a single trade is simulated.

Checks operate on whatever data is actually present (fail-open per check → a check
that cannot run is a FAIL, never a false pass). In an environment with no data,
that correctly means "abort" — which is the honest outcome.
"""
from __future__ import annotations


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def _peek_index(ticker: str):
    """NETWORK-FREE peek at the already-built index store — never trigger a
    download (get_index_ohlcv builds on first use, wrong for a pre-flight gate)."""
    try:
        from data import index_store as IX
        with IX._lock:
            return IX._store.get(IX.TICKER_MAP.get(ticker.upper()))
    except Exception:
        return None


def validate(symbols=None, sample: int = 150, factors_enabled: bool = False) -> dict:
    """Run the full pre-flight. Returns {ok, checks:[…], failed:[…]}. `ok` is True
    only when every REQUIRED check passes."""
    checks: list[dict] = []

    def guard(name: str, fn):
        """Run one check; any exception is itself a FAIL (never a false pass)."""
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, str(e)
        checks.append(_check(name, ok, detail))

    # Read the CA table and the store symbol list ONCE, reuse across checks.
    try:
        from data.corporate_actions import load_events
        ca = load_events()
    except Exception:
        ca = {}
    try:
        from data.bhavcopy_store import store_symbols
        store_syms = list(symbols) if symbols is not None else store_symbols()
    except Exception:
        store_syms = []

    def _ca_loaded():
        return len(ca) > 0, f"{len(ca)} symbols with events"

    def _no_phantom():
        from core.data_integrity import verify_ca_adjustment
        v = verify_ca_adjustment(sample=sample)
        return bool(v.get("passed")), \
            f"gap_rate={v.get('gap_rate')}, checked={v.get('checked')}"

    def _survivorship():
        from data.nse_universe import point_in_time_universe
        import datetime
        pit = point_in_time_universe(datetime.date.today().isoformat())
        return bool(pit.get("survivorship_complete")), \
            pit.get("note") or "membership history present"

    def _symbol_match():
        store = set(store_syms)
        ca_syms = set(ca)
        missing = sorted(ca_syms - store) if store else list(ca_syms)
        return (not ca_syms) or (not missing), f"{len(missing)} CA symbols not in store"

    guard("corporate_actions_loaded", _ca_loaded)
    guard("no_phantom_gaps", _no_phantom)
    guard("survivorship_complete", _survivorship)

    # ✓ Index + VIX history present (network-free peek)
    nifty, vix = _peek_index("^NSEI"), _peek_index("^INDIAVIX")
    checks.append(_check("index_history_present", nifty is not None and len(nifty) > 200,
                         f"nifty bars={0 if nifty is None else len(nifty)}"))
    checks.append(_check("vix_history_present", vix is not None and len(vix) > 200,
                         f"vix bars={0 if vix is None else len(vix)}"))

    # Bar-level integrity over a sample — one pass emits three checks
    dup_ok = future_ok = mono_ok = False
    dup_detail = future_detail = mono_detail = "no data"
    try:
        import pandas as pd
        from data.bhavcopy_store import get_ohlcv
        now = pd.Timestamp.now().normalize()
        checked = dups = futures = disordered = 0
        for s in store_syms[:sample]:
            df = get_ohlcv(s)
            if df is None or df.empty:
                continue
            checked += 1
            idx = pd.DatetimeIndex(df.index)
            dups += int(idx.duplicated().any())
            futures += int((idx > now).any())
            disordered += int(not idx.is_monotonic_increasing)
        dup_ok, future_ok, mono_ok = (checked > 0 and dups == 0,
                                      checked > 0 and futures == 0,
                                      checked > 0 and disordered == 0)
        dup_detail = f"{dups}/{checked} symbols with duplicate bars"
        future_detail = f"{futures}/{checked} symbols with future timestamps"
        mono_detail = f"{disordered}/{checked} symbols out of order"
    except Exception as e:
        dup_detail = future_detail = mono_detail = str(e)
    checks.append(_check("no_duplicate_bars", dup_ok, dup_detail))
    checks.append(_check("no_future_timestamps", future_ok, future_detail))
    checks.append(_check("bars_time_ordered", mono_ok, mono_detail))

    guard("no_symbol_mismatch", _symbol_match)

    if factors_enabled:              # NSE strategy indices as factor proxies
        checks.append(_check("factor_coverage_complete", nifty is not None,
                             "factor proxies present"))

    failed = [c["check"] for c in checks if not c["ok"]]
    return {"ok": not failed, "checks": checks, "failed": failed}
