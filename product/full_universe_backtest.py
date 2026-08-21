"""Full-universe signal backtest — every bhav EQ symbol, paper research only.

Never places LIVE or paper orders. Measures scanner signal accuracy across the
whole official history store so strategy checks are not limited to a hand-picked
watchlist.
"""
from __future__ import annotations

import time
from typing import Any, Callable


def resolve_backtest_universe(scope: str = "full") -> dict[str, Any]:
    """Return the symbol list for a full / scoped signal backtest.

    Scopes:
      • ``full`` / ``bhav`` — every symbol in the local bhavcopy store
      • ``nse`` — live NSE EQ universe ∩ bhav (research-grade overlap)
      • ``nifty500`` — hardcoded NIFTY500 ∩ bhav
    """
    from data.bhavcopy_store import store_symbols

    scope_key = str(scope or "full").strip().lower()
    available = [str(s).strip().upper() for s in (store_symbols() or []) if str(s).strip()]
    if scope_key in {"full", "bhav", "all", ""}:
        symbols = available
        source = "bhavcopy_store"
    elif scope_key in {"nse", "universe"}:
        from data.nse_universe import get_nse_universe

        live = {str(s).strip().upper() for s in (get_nse_universe() or [])}
        symbols = [s for s in available if s in live]
        source = "nse_universe∩bhav"
    elif scope_key in {"nifty500", "n500"}:
        from data.nse_universe import get_nifty500_universe

        idx = {str(s).strip().upper() for s in (get_nifty500_universe() or [])}
        symbols = [s for s in available if s in idx]
        source = "nifty500∩bhav"
    else:
        raise ValueError(f"Unknown backtest scope: {scope}")

    return {
        "scope": scope_key or "full",
        "source": source,
        "available_in_store": len(available),
        "symbols": symbols,
        "count": len(symbols),
    }


def run_full_universe_backtest(
    *,
    scope: str = "full",
    sample_step: int = 5,
    lookback_sessions: int = 250,
    horizon: int = 10,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run the signal backtest on 100% of the resolved universe (default: all bhav)."""
    from scan.signal_backtest import run_backtest

    uni = resolve_backtest_universe(scope)
    symbols = list(uni["symbols"])
    if not symbols:
        return {
            "ok": False,
            "error_code": "EMPTY_UNIVERSE",
            "message": "No symbols available for full-universe backtest — prepare bhav history first.",
            "universe": uni,
            "places_orders": False,
            "live_locked": True,
        }

    if progress:
        progress(0, len(symbols), f"Starting full-universe backtest · {len(symbols)} symbols")

    t0 = time.time()
    report = run_backtest(
        sample_step=sample_step,
        lookback_sessions=lookback_sessions,
        horizon=horizon,
        max_symbols=None,
        symbols=symbols,
    )

    universe_meta = dict(report.get("universe") or {})
    universe_meta.update({
        "scope": uni["scope"],
        "source": uni["source"],
        "available_in_store": uni["available_in_store"],
        "requested": uni["count"],
        "run": int(report.get("symbols") or universe_meta.get("run") or len(symbols)),
        "truncated": bool(universe_meta.get("truncated")),
    })
    report["universe"] = universe_meta
    report["ok"] = True
    report["places_orders"] = False
    report["live_locked"] = True
    report["elapsed_s"] = round(float(report.get("elapsed_s") or (time.time() - t0)), 1)
    if progress:
        progress(
            universe_meta["run"],
            uni["count"],
            f"Done · {universe_meta['run']} symbols · {report.get('elapsed_s')}s",
        )
    return report


def backtest_status() -> dict[str, Any]:
    from scan.signal_backtest import get_state, load_report

    report = load_report() or {}
    state = get_state()
    return {
        "running": bool(state.get("running")),
        "progress": state.get("progress"),
        "total": state.get("total"),
        "current": state.get("current") or "",
        "has_report": bool(report),
        "generated_at": report.get("generated_at"),
        "symbols_run": report.get("symbols"),
        "universe": report.get("universe") or {},
        "places_orders": False,
        "live_locked": True,
    }
