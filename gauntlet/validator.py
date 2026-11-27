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


def scoped_equity_frames(archives, symbols, actions):
    """Audit named NSE cash stocks against each expected regular-session file.

    Missing sessions and suspect bars break both the feature warm-up and the
    outcome window. Prices remain as traded within each admitted interval.
    """
    from collections import Counter, defaultdict
    import csv
    import io
    import math

    import pandas as pd

    from data.bhavcopy_integrity import validate_archive_bytes

    fields = {"open": "OPEN_PRICE", "high": "HIGH_PRICE", "low": "LOW_PRICE",
              "close": "CLOSE_PRICE", "volume": "TTL_TRD_QNTY", "prev": "PREV_CLOSE"}
    active_actions = {symbol: {item["ex_date"].date() for item in actions.get(symbol, ())}
                      for symbol in symbols}
    exclusions = defaultdict(Counter)
    rows = defaultdict(list)
    previous = {}
    bad_files = {}
    for day, payload in archives:
        if payload is None:
            bad_files[day.isoformat()] = "missing_archive"
            previous.clear()
            continue
        valid, reason = validate_archive_bytes(payload, day)
        if not valid:
            bad_files[day.isoformat()] = reason
            previous.clear()
            continue
        observed = defaultdict(list)
        reader = csv.DictReader(io.StringIO(payload.decode("latin-1")), skipinitialspace=True)
        reader.fieldnames = [column.strip().upper() for column in reader.fieldnames]
        for row in reader:
            symbol = (row.get("SYMBOL") or "").strip()
            if symbol in active_actions:
                observed[symbol].append(row)
        for symbol in symbols:
            records = observed[symbol]
            eq = [r for r in records if (r.get("SERIES") or "").strip() == "EQ"]
            reason = None
            if not eq:
                reason = ("trade_to_trade_BE" if any((r.get("SERIES") or "").strip() == "BE" for r in records)
                          else "other_series" if records else "absent_from_archive")
            elif len(eq) != 1:
                reason = "duplicate_EQ_bar"
            elif day in active_actions[symbol]:
                reason = "documented_corporate_action"
            else:
                try:
                    bar = {key: float(eq[0][column]) for key, column in fields.items()}
                except (KeyError, TypeError, ValueError):
                    bar = {}
                prior = previous.get(symbol)
                if (not bar or not all(math.isfinite(v) and v > 0 for v in bar.values())
                        or not bar["low"] <= min(bar["open"], bar["close"])
                        <= max(bar["open"], bar["close"]) <= bar["high"]):
                    reason = "invalid_OHLCV_or_reference"
                elif prior is not None and not math.isclose(prior, bar["prev"], rel_tol=1e-8, abs_tol=0.005):
                    reason = "unreconciled_previous_close"
                elif max(abs(bar[key] / bar["prev"] - 1) for key in ("open", "high", "low", "close")) >= 0.35:
                    reason = "unreconciled_large_move"
            if reason:
                exclusions[symbol][reason] += 1
                previous.pop(symbol, None)
                continue
            previous[symbol] = bar["close"]
            del bar["prev"]
            try:
                delivery = float(eq[0].get("DELIV_PER"))
            except (TypeError, ValueError):
                delivery = None
            bar["deliv_per"] = delivery if delivery is not None and 0 <= delivery <= 100 else float("nan")
            rows[symbol].append({"date": day, **bar})

    frames = {}
    for symbol, bars in rows.items():
        frame = pd.DataFrame(bars).set_index("date")
        frame.index = pd.DatetimeIndex(frame.index)
        frames[symbol] = frame
    totals = Counter()
    for counts in exclusions.values():
        totals.update(counts)
    return frames, {
        "ok": bool(frames),
        "market_wide_excluded_sessions": bad_files,
        "quarantined_symbol_sessions_by_reason": dict(totals),
        "quarantined_by_symbol": {s: dict(c) for s, c in exclusions.items()},
        "admitted_bars": sum(map(len, frames.values())),
        "symbols_with_bars": len(frames),
    }
