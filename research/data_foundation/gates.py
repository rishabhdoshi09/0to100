"""Reusable fail-closed data-quality gates for future experiments."""
from __future__ import annotations

from typing import Any

PRICE_OK = "PRICE_OK"
CA_OK = "CA_OK"
UNIVERSE_OK = "UNIVERSE_OK"
FUNDAMENTALS_OK = "FUNDAMENTALS_OK"
EARNINGS_EVENT_OK = "EARNINGS_EVENT_OK"
SECTOR_OK = "SECTOR_OK"
BENCHMARK_OK = "BENCHMARK_OK"

FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
PASS = "PASS"


def _fail(gate: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"gate": gate, "status": FAIL, "reason": reason, **extra}


def _pass(gate: str, **extra: Any) -> dict[str, Any]:
    return {"gate": gate, "status": PASS, **extra}


def _unknown(gate: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"gate": gate, "status": UNKNOWN, "reason": reason, **extra}


def price_ok(frame, *, min_bars: int = 60) -> dict[str, Any]:
    if frame is None:
        return _fail(PRICE_OK, "no_frame")
    try:
        n = len(frame)
    except Exception:
        return _fail(PRICE_OK, "unreadable_frame")
    if n < min_bars:
        return _fail(PRICE_OK, "insufficient_bars", n=n, min_bars=min_bars)
    cols = {str(c).lower() for c in getattr(frame, "columns", [])}
    if not {"open", "high", "low", "close"}.issubset(cols) and not {
        "Open", "High", "Low", "Close"
    }.issubset(set(getattr(frame, "columns", []))):
        if "close" not in cols and "Close" not in set(getattr(frame, "columns", [])):
            return _fail(PRICE_OK, "missing_ohlc")
    return _pass(PRICE_OK, n=n)


def ca_ok(symbol: str, as_of, *, require_complete: bool = False) -> dict[str, Any]:
    from data.ca_research import events_as_of, research_status
    st = research_status()
    if require_complete and not st.get("ca_complete"):
        return _fail(CA_OK, "ca_not_complete", label=st.get("label"))
    if not st.get("ca_research_acceptable") and require_complete:
        return _fail(CA_OK, "ca_not_research_acceptable")
    # Segment is OK when we only need to know crossing events are listed or absent.
    ev = events_as_of(symbol, as_of)
    return _pass(CA_OK, n_events=len(ev), label=st.get("label"),
                 complete=False, research_acceptable=st.get("ca_research_acceptable"))


def universe_ok(symbol: str, as_of, *, require_research_grade: bool = False) -> dict[str, Any]:
    from data.listing_archive import is_investable
    info = is_investable(symbol, as_of)
    if require_research_grade and not info.get("research_grade"):
        return _fail(UNIVERSE_OK, "membership_not_research_grade", **info)
    if not info.get("in_universe"):
        return _fail(UNIVERSE_OK, "not_in_universe", **info)
    if not info.get("research_grade"):
        return _unknown(UNIVERSE_OK, "pit_degraded_membership", **info)
    return _pass(UNIVERSE_OK, **info)


def fundamentals_ok(symbol: str, as_of, *, fields: tuple[str, ...] = ("basic_eps",)) -> dict[str, Any]:
    from data.pit_fundamentals import get_fundamentals
    row = get_fundamentals(symbol, as_of)
    if not row:
        return _fail(FUNDAMENTALS_OK, "no_row_known_by_as_of")
    missing = [f for f in fields if row.get(f) in (None, "")]
    if missing:
        return _fail(FUNDAMENTALS_OK, "required_fields_unknown", missing=missing,
                     available_at=row.get("available_at"))
    return _pass(FUNDAMENTALS_OK, available_at=row.get("available_at"), row_id=row.get("row_id"))


def earnings_event_ok(symbol: str, as_of) -> dict[str, Any]:
    from data.earnings_events import timeline
    evs = timeline(symbol, as_of)
    dated = [e for e in evs if e.get("announced_date")]
    if not dated:
        return _fail(EARNINGS_EVENT_OK, "no_announcement_timestamp")
    return _pass(EARNINGS_EVENT_OK, n=len(dated), last=dated[-1]["announced_date"])


def sector_ok(symbol: str, *, allow_static: bool = True) -> dict[str, Any]:
    from data.sector_map import STATIC_BACKFILL, UNKNOWN, sector_of
    info = sector_of(symbol)
    if info.get("sector") == UNKNOWN or info.get("pit_status") == UNKNOWN:
        return _fail(SECTOR_OK, "unmapped")
    if info.get("pit_status") == STATIC_BACKFILL and not allow_static:
        return _fail(SECTOR_OK, "static_backfill_not_allowed", **info)
    if info.get("pit_status") == STATIC_BACKFILL:
        return _unknown(SECTOR_OK, "static_backfill", **info)
    return _pass(SECTOR_OK, **info)


def benchmark_ok(name: str, as_of: str) -> dict[str, Any]:
    from data.benchmarks import load_index
    series = load_index(name, as_of=as_of)
    if not series.get("available"):
        return _fail(BENCHMARK_OK, "benchmark_unavailable", name=name)
    return _pass(
        BENCHMARK_OK,
        name=series["name"],
        return_kind=series["return_kind"],
        first=series["first"],
        last=series["last"],
        n=series["n"],
    )


def evaluate(
    symbol: str,
    as_of,
    *,
    required: tuple[str, ...] = (PRICE_OK, UNIVERSE_OK),
    frame=None,
) -> dict[str, Any]:
    """Fail closed: any required FAIL rejects the observation."""
    results = {}
    if PRICE_OK in required:
        results[PRICE_OK] = price_ok(frame)
    if CA_OK in required:
        results[CA_OK] = ca_ok(symbol, as_of)
    if UNIVERSE_OK in required:
        results[UNIVERSE_OK] = universe_ok(symbol, as_of)
    if FUNDAMENTALS_OK in required:
        results[FUNDAMENTALS_OK] = fundamentals_ok(symbol, as_of)
    if EARNINGS_EVENT_OK in required:
        results[EARNINGS_EVENT_OK] = earnings_event_ok(symbol, as_of)
    if SECTOR_OK in required:
        results[SECTOR_OK] = sector_ok(symbol)
    if BENCHMARK_OK in required:
        results[BENCHMARK_OK] = benchmark_ok("Nifty 500", str(as_of)[:10])
    failed = [g for g, r in results.items() if r["status"] == FAIL]
    return {
        "accepted": not failed,
        "failed": failed,
        "gates": results,
        "symbol": symbol,
        "as_of": str(as_of)[:10],
    }
