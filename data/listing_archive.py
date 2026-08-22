"""Listing / delisting honesty layer + generic investability freshness.

Official EQUITY_L + delisted.csv improve *current* listing dates.
They do not complete historical survivorship while undated delists remain
omitted. v2 exists as a limited official overlay — not a manufactured
complete archive.
"""
from __future__ import annotations

from typing import Any

from data.universe_freshness import HARD_STALE_SESSIONS, investability
from data.universe_history import history_path, ledger_status


def universe_pit_class(path=None) -> dict[str, Any]:
    st = ledger_status(path)
    src = str(st.get("source") or "")
    completeness = st.get("completeness") or {}
    omitted = int(completeness.get("delisted_omitted_no_listed_date") or 0)
    official_partial = src.startswith("nse_equity_l")
    complete = bool(st.get("research_grade")) and omitted == 0 and bool(
        completeness.get("survivorship_complete")
    )
    if official_partial and not complete:
        pit_class = "PIT_DEGRADED"
        status = "RESEARCH_READY_WITH_LIMITATIONS"
        v2 = True
        note = (
            "Official EQUITY_L listing dates for current EQ plus official "
            "delist dates where a listing date is known. "
            f"{omitted} official delists omitted (no listing date). "
            "Not a complete 2019–2024 dead-name archive. "
            "Bhav-inferred sidecar remains the historical membership fallback."
        )
    elif complete:
        pit_class = "PIT_STRONG"
        status = "RESEARCH_READY"
        v2 = True
        note = "Official listing/delist archive is complete for membership."
    else:
        pit_class = "PIT_DEGRADED"
        status = "DESCRIPTIVE_ONLY"
        v2 = False
        note = (
            "Inferred first/last bhav session is not an official "
            "listing/delisting/suspension/symbol-change archive."
        )
    return {
        "path": str(history_path(path)),
        "rows": st.get("rows"),
        "source": src,
        "research_grade": bool(st.get("research_grade")),
        "pit_class": pit_class,
        "point_in_time_universe_v2": v2,
        "status": status,
        "omitted_undated_delists": omitted,
        "note": note,
        **{k: st.get(k) for k in ("date_range", "n_delisted", "generated_at") if k in st},
    }


def _last_bar_for(symbol: str, as_of: str, frame=None) -> str | None:
    if frame is not None:
        from data.universe_freshness import last_bar_date
        last = last_bar_date(frame)
        if last and last <= as_of:
            return last
        return last
    try:
        from data import bhavcopy_store as BS
        spans = BS.symbol_date_spans() or {}
        span = spans.get(str(symbol).upper()) or {}
        last = str(span.get("last") or "")[:10] or None
        if last and last > as_of:
            return None
        return last
    except Exception:
        return None


def is_investable(
    symbol: str,
    as_of,
    *,
    path=None,
    max_stale_sessions: int = HARD_STALE_SESSIONS,
    frame=None,
    calendar: list[str] | None = None,
    last_bar: str | None = None,
    suspended: bool = False,
) -> dict[str, Any]:
    """Membership + freshness. A last print is not a living listing."""
    import pandas as pd
    from data.nse_universe import point_in_time_universe

    asof = str(pd.Timestamp(as_of).date())
    pit = point_in_time_universe(as_of, path=path)
    sym = str(symbol).upper()
    members = set(pit.get("symbols") or [])
    listed = sym in members
    bar = last_bar if last_bar is not None else _last_bar_for(sym, asof, frame)
    fresh = investability(
        symbol=sym,
        as_of=asof,
        listed=listed,
        delisted=not listed and _was_delisted(sym, asof, path),
        last_bar=bar,
        calendar=calendar,
        max_stale_sessions=max_stale_sessions,
        suspended=suspended,
    )
    return {
        "symbol": sym,
        "as_of": asof,
        "in_universe": listed,
        "tradable": bool(fresh.get("tradable")),
        "freshness_reason": fresh.get("reason"),
        "stale_sessions": fresh.get("stale_sessions"),
        "last_bar": bar,
        "survivorship_complete": bool(pit.get("survivorship_complete")),
        "research_grade": bool(pit.get("research_grade")),
        "pit_class": "PIT_DEGRADED" if not pit.get("research_grade") else "PIT_STRONG",
        "max_stale_sessions_policy": max_stale_sessions,
        "note": pit.get("note"),
    }


def _was_delisted(symbol: str, as_of: str, path=None) -> bool:
    """True only when a membership row exists and its delist date has passed."""
    try:
        import json
        from data.universe_history import _coerce_payload, history_path
        p = history_path(path)
        if not p.exists():
            return False
        rows, _ = _coerce_payload(json.loads(p.read_text(encoding="utf-8")))
        for r in rows:
            if str(r.get("symbol") or "").upper() != symbol:
                continue
            d = r.get("delisted")
            return bool(d) and str(d)[:10] <= as_of
    except Exception:
        return False
    return False


def apply_freshness(
    symbols: list[str],
    as_of,
    *,
    path=None,
    frames: dict | None = None,
    calendar: list[str] | None = None,
    max_stale_sessions: int = HARD_STALE_SESSIONS,
) -> dict[str, Any]:
    """Filter a membership list with the generic stale-bar rule."""
    tradable = []
    dropped = []
    for sym in symbols:
        frame = (frames or {}).get(str(sym).upper())
        info = is_investable(
            sym, as_of, path=path, frame=frame, calendar=calendar,
            max_stale_sessions=max_stale_sessions,
        )
        if info.get("tradable"):
            tradable.append(str(sym).upper())
        else:
            dropped.append({
                "symbol": str(sym).upper(),
                "reason": info.get("freshness_reason"),
                "last_bar": info.get("last_bar"),
            })
    return {
        "as_of": str(as_of)[:10],
        "members": [str(s).upper() for s in symbols],
        "tradable": sorted(set(tradable)),
        "dropped": dropped,
        "fresh_bar_requirement": max_stale_sessions,
    }
