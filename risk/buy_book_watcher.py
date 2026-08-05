"""Active Buys watcher — Telegram warnings when buys weaken.

Once per symbol+warning-code per day. Research warnings only; never places orders.
"""
from __future__ import annotations

import threading
from datetime import datetime

from core.market_clock import IST
from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_alerted: dict[str, set[str]] = {}


def check_buy_book() -> list[dict]:
    """Return fresh critical/warn events for active buys."""
    try:
        from product.buy_health import evaluate_book

        payload = evaluate_book()
    except Exception as exc:
        log.debug("buy_book_eval_failed", error=str(exc))
        return []

    events: list[dict] = []
    for row in payload.get("items") or []:
        health = row.get("health") or {}
        if health.get("severity") not in {"critical", "warn"}:
            continue
        sym = str(row.get("symbol") or "")
        for warning in health.get("warnings") or []:
            if warning.get("severity") not in {"critical", "warn"}:
                continue
            code = str(warning.get("code") or "WARN")
            events.append(
                {
                    "symbol": sym,
                    "event": code,
                    "severity": warning.get("severity"),
                    "message": (
                        f"{'🔴' if warning.get('severity') == 'critical' else '🟠'} "
                        f"<b>{sym}</b> ₹{health.get('price') or '—'} — {warning.get('text')}"
                    ),
                }
            )
    return events


def push_buy_book_alerts() -> int:
    """Telegram push — once per symbol:event per day."""
    try:
        from alerts.telegram_alerts import AlertEngine

        engine = AlertEngine()
        if not engine.is_configured():
            return 0
        events = check_buy_book()
        if not events:
            return 0
        today = datetime.now(IST).strftime("%Y-%m-%d")
        with _lock:
            _alerted.setdefault(today, set())
            for key in list(_alerted):
                if key != today:
                    del _alerted[key]
            fresh = [
                e for e in events
                if f"{e['symbol']}:{e['event']}" not in _alerted[today]
            ]
            if not fresh:
                return 0
            # Prefer critical first, cap message size.
            fresh.sort(key=lambda e: 0 if e.get("severity") == "critical" else 1)
            lines = [
                "<b>Active Buys — health warning</b>",
                "<i>Research only · not a sell order</i>",
            ]
            lines += [f"\n{e['message']}" for e in fresh[:8]]
            if engine.send("\n".join(lines)):
                _alerted[today].update(f"{e['symbol']}:{e['event']}" for e in fresh)
                log.info("buy_book_alerts_pushed", count=len(fresh))
                return len(fresh)
    except Exception as exc:
        log.debug("buy_book_alerts_skip", error=str(exc))
    return 0
