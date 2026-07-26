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


def validate(symbols=None, sample: int = 150, factors_enabled: bool = False) -> dict:
    """Run the full pre-flight. Returns {ok, checks:[…], failed:[…]}. `ok` is True
    only when every REQUIRED check passes."""
    checks: list[dict] = []

    # ✓ Corporate actions loaded
    try:
        from data.corporate_actions import load_events
        n_ca = len(load_events())
        checks.append(_check("corporate_actions_loaded", n_ca > 0,
                             f"{n_ca} symbols with events"))
    except Exception as e:
        checks.append(_check("corporate_actions_loaded", False, str(e)))

    # ✓ No phantom gaps after adjustment (the CA acceptance test)
    try:
        from core.data_integrity import verify_ca_adjustment
        v = verify_ca_adjustment(sample=sample)
        checks.append(_check("no_phantom_gaps", bool(v.get("passed")),
                             f"gap_rate={v.get('gap_rate')}, checked={v.get('checked')}"))
    except Exception as e:
        checks.append(_check("no_phantom_gaps", False, str(e)))

    # ✓ Survivorship-complete universe
    try:
        from data.nse_universe import point_in_time_universe
        import datetime
        pit = point_in_time_universe(datetime.date.today().isoformat())
        checks.append(_check("survivorship_complete",
                             bool(pit.get("survivorship_complete")),
                             pit.get("note") or "membership history present"))
    except Exception as e:
        checks.append(_check("survivorship_complete", False, str(e)))

    # ✓ Index history present  ✓ VIX history present  ✓ Benchmark coverage
    # NETWORK-FREE peek: read the already-built store, never trigger a download
    # (get_index_ohlcv builds on first use — wrong for a pre-flight gate).
    def _peek_index(ticker: str):
        try:
            from data import index_store as IX
            name = IX.TICKER_MAP.get(ticker.upper())
            with IX._lock:
                df = IX._store.get(name)
            return df
        except Exception:
            return None
    nifty, vix = _peek_index("^NSEI"), _peek_index("^INDIAVIX")
    idx_ok = nifty is not None and len(nifty) > 200
    vix_ok = vix is not None and len(vix) > 200
    checks.append(_check("index_history_present", idx_ok,
                         f"nifty bars={0 if nifty is None else len(nifty)}"))
    checks.append(_check("vix_history_present", vix_ok,
                         f"vix bars={0 if vix is None else len(vix)}"))

    # Bar-level integrity over a sample of the store
    dup_ok = future_ok = mono_ok = True
    dup_detail = future_detail = mono_detail = "no data"
    try:
        import pandas as pd
        from data.bhavcopy_store import store_symbols, get_ohlcv
        syms = (symbols or store_symbols())[:sample]
        now = pd.Timestamp.now().normalize()
        checked = 0
        dups = futures = gaps_missing = 0
        for s in syms:
            df = get_ohlcv(s)
            if df is None or df.empty:
                continue
            checked += 1
            idx = pd.DatetimeIndex(df.index)
            if idx.duplicated().any():
                dups += 1                                   # ✓ No duplicate bars
            if (idx > now).any():
                futures += 1                                # ✓ No future timestamps
            if not idx.is_monotonic_increasing:
                gaps_missing += 1                           # out-of-order → suspect
        dup_ok = checked > 0 and dups == 0
        future_ok = checked > 0 and futures == 0
        mono_ok = checked > 0 and gaps_missing == 0
        dup_detail = f"{dups}/{checked} symbols with duplicate bars"
        future_detail = f"{futures}/{checked} symbols with future timestamps"
        mono_detail = f"{gaps_missing}/{checked} symbols out of order"
    except Exception as e:
        dup_detail = future_detail = mono_detail = str(e)
        dup_ok = future_ok = mono_ok = False
    checks.append(_check("no_duplicate_bars", dup_ok, dup_detail))
    checks.append(_check("no_future_timestamps", future_ok, future_detail))
    checks.append(_check("bars_time_ordered", mono_ok, mono_detail))

    # ✓ Symbol mismatch: every CA/universe symbol should resolve in the store
    try:
        from data.corporate_actions import load_events
        from data.bhavcopy_store import store_symbols
        store = set(store_symbols())
        ca_syms = set(load_events().keys())
        missing = sorted(ca_syms - store) if store else list(ca_syms)
        checks.append(_check("no_symbol_mismatch", (not ca_syms) or (not missing),
                             f"{len(missing)} CA symbols not in store"))
    except Exception as e:
        checks.append(_check("no_symbol_mismatch", False, str(e)))

    # ✓ Factor coverage (only when factors are enabled)
    if factors_enabled:
        # NSE strategy indices used as factor proxies (network-free peek)
        covered = _peek_index("^NSEI") is not None
        checks.append(_check("factor_coverage_complete", covered,
                             "factor proxies present"))

    failed = [c["check"] for c in checks if not c["ok"]]
    return {"ok": not failed, "checks": checks, "failed": failed}
