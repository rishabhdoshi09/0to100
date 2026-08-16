"""Market institutional flows and options chain API for the React terminal."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query


def institutional_workspace(days: int = Query(30, ge=5, le=365)) -> dict[str, Any]:
    from data.fii_dii_store import workspace_payload

    return workspace_payload(days=max(5, min(int(days), 365)))


def fii_dii_backfill_status_workspace() -> dict[str, Any]:
    from data.fii_dii_store import backfill_status

    return backfill_status()


def fii_dii_backfill_run(days: int = Query(90, ge=30, le=365)) -> dict[str, Any]:
    from data.fii_dii_store import refresh_if_needed

    refresh = refresh_if_needed(force=True)
    return {"forced": True, "days": days, "refresh": refresh}


def options_workspace(
    symbol: str,
    spot: float | None = Query(None, description="Optional spot for ATM IV"),
    force: bool = Query(False, description="Bypass TTL cache and failure backoff"),
) -> dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    if not sym or len(sym) > 32:
        raise HTTPException(status_code=400, detail="invalid symbol")
    resolved_spot = spot
    if resolved_spot is None and sym not in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
        try:
            from data.bhavcopy_runtime import get_ohlcv

            frame = get_ohlcv(sym)
            if frame is not None and len(frame):
                resolved_spot = float(frame["close"].iloc[-1])
        except Exception:
            resolved_spot = None
    from options.chain_fetch import chain_workspace_cached

    return chain_workspace_cached(sym, spot=resolved_spot, force=force)


def options_history_workspace(
    symbol: str,
    days: int = Query(14, ge=3, le=90),
) -> dict[str, Any]:
    """Persisted EOD PCR/IV/OI snapshots for multi-day context (read-only)."""
    sym = str(symbol or "").strip().upper()
    if not sym or len(sym) > 32:
        raise HTTPException(status_code=400, detail="invalid symbol")
    try:
        from options.eod_store import history, store_status

        rows = history(sym, days=int(days))
        status = store_status()
        return {
            "available": bool(rows),
            "symbol": sym,
            "days": int(days),
            "rows": rows,
            "store": status,
            "message": (
                ""
                if rows
                else "No EOD options history yet — run python main.py options-eod or wait for the autonomy job."
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "symbol": sym,
            "days": int(days),
            "rows": [],
            "store": {},
            "message": f"Options EOD history unavailable ({exc})",
        }


def nifty_options_workspace() -> dict[str, Any]:
    return options_workspace("NIFTY")


def options_watch_eod(symbol: str) -> dict[str, Any]:
    """Enqueue a user-opened underlying for the next options EOD job."""
    from options.eod_watch import add_watch

    return add_watch(symbol)


def install_market_routes(app) -> None:
    app.add_api_route(
        "/api/market/institutional",
        institutional_workspace,
        methods=["GET"],
        name="market_institutional",
    )
    app.add_api_route(
        "/api/market/fii-dii/backfill",
        fii_dii_backfill_status_workspace,
        methods=["GET"],
        name="market_fii_dii_backfill_status",
    )
    app.add_api_route(
        "/api/market/fii-dii/backfill",
        fii_dii_backfill_run,
        methods=["POST"],
        name="market_fii_dii_backfill_run",
    )
    app.add_api_route(
        "/api/market/options/nifty",
        nifty_options_workspace,
        methods=["GET"],
        name="market_options_nifty",
    )
    app.add_api_route(
        "/api/market/options/{symbol}/history",
        options_history_workspace,
        methods=["GET"],
        name="market_options_history",
    )
    app.add_api_route(
        "/api/market/options/{symbol}/watch-eod",
        options_watch_eod,
        methods=["POST"],
        name="market_options_watch_eod",
    )
    app.add_api_route(
        "/api/market/options/{symbol}",
        options_workspace,
        methods=["GET"],
        name="market_options_symbol",
    )
