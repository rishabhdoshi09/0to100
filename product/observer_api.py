"""Lifecycle and read-only API projections installed on the terminal app."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi import Body, HTTPException, Query

import terminal_api as core
from product.data_api import install_data_routes
from product.workspace import SCANNER_MODES, build_command_center_state, scanner_rows

RUNTIME_PATH = core.ROOT / "logs" / "reconciliation" / "observer_runtime.json"
SNAPSHOT_DB = core.ROOT / "logs" / "reconciliation" / "broker_snapshots.db"

_observer_process: subprocess.Popen | None = None
_installed = False


def _json_file(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _fresh(value: Any, max_age_seconds: float = 30.0) -> bool:
    try:
        age = time.time() - float(value)
        return 0 <= age <= max_age_seconds
    except Exception:
        return False


def observer_enabled() -> bool:
    value = os.getenv("QT_ENABLE_ZERODHA_OBSERVER", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def observer_payload() -> dict[str, Any]:
    runtime = dict(_json_file(RUNTIME_PATH, {}) or {})
    running = bool(runtime.get("process_running")) and _fresh(runtime.get("heartbeat_epoch"))
    snapshots: dict[str, Any] = {
        "available": False,
        "summary": {
            "snapshots": 0,
            "account_complete_snapshots": 0,
            "protection_complete_snapshots": 0,
            "complete_snapshots": 0,
            "latest_snapshot_id": "",
            "latest_complete_snapshot_id": "",
        },
        "latest": {},
    }
    try:
        if SNAPSHOT_DB.exists():
            from execution.reconciliation.snapshot_store import BrokerSnapshotStore

            store = BrokerSnapshotStore(SNAPSHOT_DB)
            latest = store.latest()
            snapshots = {
                "available": latest is not None,
                "summary": store.summary(),
                "latest": latest or {},
            }
    except Exception as exc:
        snapshots["error"] = str(exc)

    return {
        "enabled": observer_enabled(),
        "running": running,
        "process_running": bool(runtime.get("process_running")),
        "heartbeat": runtime.get("heartbeat", ""),
        "phase": runtime.get("phase", "OFFLINE"),
        "broker_mutations_enabled": False,
        "last_result": dict(runtime.get("last_result", {}) or {}),
        "last_error": str(runtime.get("last_error", "") or ""),
        "snapshots": snapshots,
        "message": (
            "Scheduled Zerodha observation is read-only and cannot place, modify or cancel "
            "orders or GTTs."
        ),
    }


def command_center_workspace() -> dict[str, Any]:
    """Project authoritative persisted state into one coherent command surface."""
    market = core._market_payload()
    scan = core._scan_payload()
    long_term = core._long_term_payload()
    state = build_command_center_state(
        scan_payload=scan,
        long_term_payload=long_term,
        paper=core._paper_payload(),
        autonomy=core._autonomy_payload(),
        market=market,
    )
    return {"generated_at": datetime.now(timezone.utc).isoformat(), **state}


def scanner_workspace(mode: str) -> dict[str, Any]:
    """Return one server-ranked scanner mode without duplicating scan calculations."""
    requested = mode.strip().replace("_", "-")
    canonical = next(
        (item for item in SCANNER_MODES if item.lower() == requested.lower()),
        None,
    )
    if canonical is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scanner mode. Choose one of: {', '.join(SCANNER_MODES)}",
        )
    market = core._market_payload()
    scan = core._scan_payload()
    long_term = core._long_term_payload()
    rows = scanner_rows(
        canonical,
        scan_payload=scan,
        long_term_payload=long_term,
        conviction_rows=core._conviction(scan, market),
    )
    from product.radar_workspace import enrich_long_term_row, enrich_scanner_rows

    scanned_at = (
        long_term.get("scanned_at", "")
        if canonical == "Long-Term"
        else scan.get("scanned_at", "")
    )
    if canonical == "Long-Term":
        enriched = [enrich_long_term_row(dict(row), scanned_at=scanned_at) for row in rows]
    else:
        enriched = enrich_scanner_rows(rows, scanned_at=scanned_at)
    source = (
        "long_term"
        if canonical == "Long-Term"
        else "conviction"
        if canonical == "Conviction"
        else "market_scan"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": canonical,
        "source": source,
        "scanned_at": scanned_at,
        "universe_size": int(scan.get("universe_size", 0) or 0),
        "rows": enriched,
    }


def radar_home_workspace() -> dict[str, Any]:
    """Today/Setups bootstrap. Empty cards until the last scan JSON is readable."""
    scan: dict[str, Any] = {}
    sepa_cards: list[dict[str, Any]] = []
    sepa_note = "Last scan is not readable yet."
    try:
        scan = core._scan_payload()
    except Exception:
        scan = {}
    try:
        from product.sepa_setup import public_best_setups

        sepa_cards, sepa_note = public_best_setups(scan, limit=8, score_cap=24, max_seconds=2.0)
    except Exception:
        sepa_cards = []
        sepa_note = "SEPA ranking is temporarily unavailable."
    try:
        market = core._market_payload()
        if not scan:
            scan = core._scan_payload()
        long_term = core._long_term_payload()
        from product.radar_workspace import build_radar_home
        home = build_radar_home(
            scan_payload=scan,
            long_term_payload=long_term,
            market=market,
            sepa_cards=sepa_cards,
        )
    except Exception as exc:
        home = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "market_session": "",
            "market_health": "Unavailable",
            "breadth": "—",
            "nifty_change_1d": None,
            "vix": None,
            "leaders": [],
            "laggards": [],
            "scan_scanned_at": str((scan or {}).get("scanned_at", "") or ""),
            "long_term_scanned_at": "",
            "universe_size": int((scan or {}).get("universe_size", 0) or 0),
            "lanes": {"breakouts": [], "momentum": [], "long_term_picks": []},
            "counts": {"breakouts": 0, "momentum": 0, "long_term_picks": 0, "sniper_breakouts": 0},
            "best_breakout": None,
            "best_among_fundamentals": None,
            "best_of_best": [],
            "best_among_note": "Radar ranking is temporarily unavailable.",
            "sniper_candidates": [],
            "ranking_legend": {},
            "scan_shared_note": "",
            "sepa_rank_used": False,
            "error": str(exc),
        }
    home["best_setups"] = list(sepa_cards)
    home["best_setups_note"] = str(sepa_note or "")
    try:
        from product.scan_progress import read_progress
        home["scan_progress"] = read_progress()
    except Exception:
        home["scan_progress"] = {"active": False}
    try:
        from product.telegram_delivery import delivery_status
        home["telegram"] = delivery_status()
    except Exception as exc:
        home["telegram"] = {
            "configured": False,
            "state": "unavailable",
            "headline": "Telegram status unavailable",
            "detail": str(exc),
        }
    try:
        from operations.store import OperationStore
        from product.desk_pipeline import describe_desk_pipeline

        home["desk_pipeline"] = describe_desk_pipeline(OperationStore(core.OPS_DB))
    except Exception:
        home["desk_pipeline"] = None
    return home


def recommendations_workspace() -> dict[str, Any]:
    """Reco-style research categories + Active/Closed lifecycle (evidence only)."""
    from product.recommendations_workspace import build_recommendations_workspace
    return build_recommendations_workspace(
        scan_payload=core._scan_payload(),
        long_term_payload=core._long_term_payload(),
        refresh_technicals=True,
    )


def market_reports_workspace() -> dict[str, Any]:
    """Chronological Market Pulse desk from street_pulse + saved day files."""
    from product.recommendations_workspace import build_market_reports_workspace
    news: dict[str, Any] = {}
    try:
        news = core._news_payload()
    except Exception:
        news = {}
    return build_market_reports_workspace(
        persist_today=True,
        news_payload=news,
        scan_payload=core._scan_payload(),
    )


def compare_workspace(symbols: str = Query("", description="Comma-separated NSE symbols")) -> dict[str, Any]:
    from product.compare_workspace import build_compare_workspace
    parts = [item.strip() for item in str(symbols or "").split(",") if item.strip()]
    if not parts:
        raise HTTPException(status_code=400, detail="Provide symbols as comma-separated list")
    if len(parts) > 5:
        raise HTTPException(status_code=400, detail="Compare up to 5 symbols at once")
    return build_compare_workspace(parts)


def watchlist_workspace() -> dict[str, Any]:
    from product.watchlist_store import list_items
    from product.radar_workspace import enrich_scan_row

    scan = core._scan_payload()
    scan_at = str(scan.get("scanned_at", "") or "")
    scan_map = {str(r.get("symbol", "")).upper(): r for r in scan.get("records", []) or []}
    items = []
    for row in list_items():
        sym = str(row.get("symbol", "")).upper()
        scan_row = scan_map.get(sym)
        enriched = enrich_scan_row(scan_row or {"symbol": sym}, scanned_at=scan_at) if scan_row else {"symbol": sym}
        items.append({**row, "snapshot": enriched})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        "count": len(items),
    }


def watchlist_add(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from product.watchlist_store import add_item
    symbol = str(payload.get("symbol", "") or "")
    try:
        item = add_item(
            symbol,
            buy_low=payload.get("buy_zone_low"),
            buy_high=payload.get("buy_zone_high"),
            target=payload.get("target_price"),
            stop=payload.get("stop_price"),
            notes=str(payload.get("notes", "") or ""),
            added_price=payload.get("added_price"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"accepted": True, "item": item}


def watchlist_remove(row_id: int) -> dict[str, Any]:
    from product.watchlist_store import remove_item
    removed = remove_item(row_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    return {"accepted": True, "removed_id": row_id}


def drain_telegram_on_startup() -> None:
    """Replay last-scan Telegram alerts once the desk API is up.

    Autonomy also drains on start/tick. This catches the case where Telegram
    was connected after the scan and the supervisor is not running.
    """
    def _run() -> None:
        time.sleep(2.0)
        try:
            from product.telegram_delivery import drain_scan_alerts
            sent = drain_scan_alerts(min_interval_s=0.0) or {}
            reason = str(sent.get("reason") or "")
            if reason and reason not in {"already_sent", "no_candidates", "no_scan", "in_progress"}:
                print(
                    f"[TELEGRAM] desk drain · setups={int(sent.get('setup') or 0)} · "
                    f"near-breakout={int(sent.get('prebreakout') or 0)} · {reason}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[TELEGRAM] desk drain failed: {type(exc).__name__}: {exc}", flush=True)

    threading.Thread(target=_run, name="telegram-scan-drain", daemon=True).start()


def ensure_observer_worker() -> dict[str, Any]:
    global _observer_process
    payload = observer_payload()
    if not observer_enabled() or payload.get("running"):
        return payload
    if _observer_process is not None and _observer_process.poll() is None:
        return payload
    _observer_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "operations.zerodha_observer"],
        cwd=str(core.ROOT),
        env=os.environ.copy(),
    )
    deadline = time.time() + 2.5
    while time.time() < deadline:
        time.sleep(0.1)
        payload = observer_payload()
        if payload.get("running"):
            break
        if _observer_process.poll() is not None:
            break
    return payload


def stop_observer_worker() -> None:
    global _observer_process
    if _observer_process is not None and _observer_process.poll() is None:
        _observer_process.terminate()
        try:
            _observer_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _observer_process.kill()
    _observer_process = None


def install(app) -> None:
    """Install read-only workspace routes and observer lifecycle once."""
    global _installed
    if _installed:
        return
    _installed = True
    router = getattr(app, "router", None)
    if router is None or not hasattr(router, "add_event_handler"):
        raise RuntimeError("terminal FastAPI router does not support lifecycle handlers")
    router.add_event_handler("startup", ensure_observer_worker)
    router.add_event_handler("startup", drain_telegram_on_startup)
    router.add_event_handler("shutdown", stop_observer_worker)
    app.add_api_route(
        "/api/broker-observer",
        observer_payload,
        methods=["GET"],
        name="broker_observer_status",
    )
    app.add_api_route(
        "/api/command-center-workspace",
        command_center_workspace,
        methods=["GET"],
        name="command_center_workspace",
    )
    app.add_api_route(
        "/api/scanner-workspace/{mode}",
        scanner_workspace,
        methods=["GET"],
        name="scanner_workspace",
    )
    app.add_api_route(
        "/api/radar-home",
        radar_home_workspace,
        methods=["GET"],
        name="radar_home_workspace",
    )
    app.add_api_route(
        "/api/recommendations-workspace",
        recommendations_workspace,
        methods=["GET"],
        name="recommendations_workspace",
    )
    app.add_api_route(
        "/api/market-reports-workspace",
        market_reports_workspace,
        methods=["GET"],
        name="market_reports_workspace",
    )
    app.add_api_route(
        "/api/compare",
        compare_workspace,
        methods=["GET"],
        name="compare_workspace",
    )
    app.add_api_route(
        "/api/watchlist",
        watchlist_workspace,
        methods=["GET"],
        name="watchlist_workspace",
    )
    app.add_api_route(
        "/api/watchlist",
        watchlist_add,
        methods=["POST"],
        name="watchlist_add",
    )
    app.add_api_route(
        "/api/watchlist/{row_id}",
        watchlist_remove,
        methods=["DELETE"],
        name="watchlist_remove",
    )
    install_data_routes(app)
