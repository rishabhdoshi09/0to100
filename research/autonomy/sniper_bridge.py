"""Breakout sniper bridge for the autonomy supervisor.

Keeps the Kite WebSocket sniper armed from the latest product scan payload —
the same path `run_quantterm.sh` already starts via `main.py autonomy`.
Does not enable legacy Streamlit daemons.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def records_from_payload(payload: Mapping[str, Any] | None) -> list[dict]:
    if not payload:
        return []
    rows = payload.get("records")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return []


def ensure_breakout_sniper(payload: Mapping[str, Any] | None = None) -> dict:
    """Start (idempotent) and refresh the sniper watch map.

    Returns a small status dict for job metadata / logs. Never raises to callers.
    """
    try:
        from product.scan_store import load_scan
        from scan.breakout_sniper import refresh_watch, start_sniper
    except Exception as exc:
        return {"ok": False, "error": f"import_failed:{exc}", "watching": 0}

    try:
        from research.autonomy import schedules as SCH
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if not SCH.market_is_open(now):
            return {"ok": False, "error": "market_closed", "watching": 0}
    except Exception:
        # If clock helper unavailable, still try — start_sniper needs Kite.
        pass

    data = payload if payload is not None else load_scan()
    records = records_from_payload(data)
    try:
        started = bool(start_sniper())
    except Exception as exc:
        return {"ok": False, "error": f"start_failed:{exc}", "watching": 0, "started": False}

    if not started:
        return {
            "ok": False,
            "error": "kite_unavailable",
            "watching": 0,
            "started": False,
            "hint": "Run python main.py login so the sniper can subscribe to ticks",
        }

    try:
        n = int(refresh_watch(records))
    except Exception as exc:
        return {"ok": False, "error": f"refresh_failed:{exc}", "watching": 0, "started": True}

    return {"ok": True, "started": True, "watching": n, "records": len(records)}


def diagnose_breakout_alerts() -> dict:
    """One-shot operator diagnostic for missing Telegram breakout alerts."""
    out: dict[str, Any] = {}
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    except Exception:
        pass
    try:
        from alerts.telegram_alerts import AlertEngine
        eng = AlertEngine()
        out["telegram_configured"] = bool(eng.is_configured())
    except Exception as exc:
        out["telegram_configured"] = False
        out["telegram_error"] = str(exc)
    try:
        from execution.trade_executor import kite_ready
        out["kite_ready"] = bool(kite_ready())
    except Exception as exc:
        out["kite_ready"] = False
        out["kite_error"] = str(exc)
    try:
        from product.autonomy_status import read_autonomy_status
        st = read_autonomy_status()
        out["autonomy_running"] = bool(st.get("running"))
        out["autonomy_heartbeat"] = st.get("heartbeat_ist")
    except Exception as exc:
        out["autonomy_running"] = False
        out["autonomy_error"] = str(exc)
    try:
        from product.scan_store import load_scan
        payload = load_scan() or {}
        recs = records_from_payload(payload)
        pre = [
            r for r in recs
            if "PRE_BREAKOUT" in [str(x).upper() for x in (r.get("signals") or [])]
            or str(r.get("status") or "") == "Watch for breakout"
        ]
        out["scan_present"] = bool(payload)
        out["scan_records"] = len(recs)
        out["pre_breakout_candidates"] = len(pre)
        out["scanned_at"] = payload.get("scanned_at")
        out["sample_pre"] = [
            {"symbol": r.get("symbol"), "entry": r.get("entry"), "avg_vol20": r.get("avg_vol20")}
            for r in pre[:5]
        ]
    except Exception as exc:
        out["scan_present"] = False
        out["scan_error"] = str(exc)
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from research.autonomy import schedules as SCH
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        out["ist_now"] = str(now)
        out["market_open"] = bool(SCH.market_is_open(now))
        out["scan_slot"] = SCH.scan_slot(now)
    except Exception as exc:
        out["clock_error"] = str(exc)
    sniper = ensure_breakout_sniper()
    out["sniper"] = sniper
    blockers = []
    if not out.get("telegram_configured"):
        blockers.append("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing or unread from .env")
    if not out.get("kite_ready"):
        blockers.append("Kite not ready — run: python main.py login")
    if not out.get("autonomy_running"):
        blockers.append("Autonomy supervisor not running — restart bash scripts/run_quantterm.sh")
    if not out.get("scan_present"):
        blockers.append("No scan payload yet — wait for MARKET_SCAN after 09:30 IST")
    elif int(out.get("pre_breakout_candidates") or 0) == 0:
        blockers.append("Scan has zero pre-breakout candidates right now")
    if sniper.get("error") == "kite_unavailable":
        blockers.append("Sniper cannot start WebSocket without Kite")
    if sniper.get("error") == "market_closed":
        blockers.append("Market closed — sniper arms only in session")
    out["blockers"] = blockers
    out["ok"] = not blockers
    return out
