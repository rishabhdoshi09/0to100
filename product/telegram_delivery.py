"""Read-only Telegram delivery status + throttled last-scan replay.

Scan/breakout watches are supervisor-owned. This module never starts a second
sniper websocket and never places an order. Desk/API callers only replay the
saved scan when Telegram is configured and today's setup/pre keys are empty.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.autonomy import default_root
from research.autonomy.telegram_notifications import TelegramNotifier, sniper_symbols


def _live_feed_health(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads((root / "status.json").read_text(encoding="utf-8"))
        feed = dict(payload.get("live_feed") or {})
        feed["autonomy_running"] = bool(payload.get("process_running"))
        feed["autonomy_state"] = str(payload.get("state") or "")
        return feed
    except Exception:
        return {}


def delivery_status(root: Path | None = None) -> dict[str, Any]:
    """Desk snapshot: configured vs last scan pushed vs why sniper is silent."""
    base = Path(root) if root else default_root()
    notifier = TelegramNotifier(base)
    configured = False
    try:
        configured = bool(notifier.configured())
    except Exception:
        configured = False
    delivery = dict(notifier.state.get("delivery") or {})
    scan = dict(delivery.get("scan") or {})
    sniper = dict(delivery.get("sniper") or {})
    day = notifier._day()
    sent_today = list(notifier.state.get("sent", {}).get(day, []) or [])
    scan_pushed = any(str(k).startswith(("setup:", "pre:")) for k in sent_today)
    watch = 0
    try:
        from product.scan_store import load_scan
        payload = load_scan() or {}
        watch = len(sniper_symbols(payload))
        if not watch:
            watch = int(scan.get("sniper_watch") or 0)
    except Exception:
        watch = int(scan.get("sniper_watch") or 0)
    live = _live_feed_health(base)
    ticking = int(live.get("symbols_ticking") or sniper.get("fresh") or 0)
    live_ok = bool(live.get("connected")) and ticking > 0
    scan_reason = str(scan.get("reason") or "")
    sniper_reason = str(sniper.get("reason") or "")
    last_error = str(scan.get("last_error") or sniper.get("last_error") or "")

    if not configured:
        headline = "Telegram off"
        detail = "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
        state = "off"
    elif scan_pushed or scan_reason == "sent":
        if live_ok:
            headline = "Scan alerts sent · sniper live"
            detail = "Confirmed breakouts send when LTP holds 8s above trigger"
            state = "live"
        else:
            headline = "Scan alerts sent"
            detail = (
                "SNIPER BREAKOUT CONFIRMED needs Zerodha live ticks "
                "(login + autonomy) during market hours"
            )
            state = "scan_sent"
    elif scan_reason == "send_failed":
        headline = "Telegram connected · last scan failed to send"
        detail = last_error or "Retrying last-scan setups and breakout watches"
        state = "send_failed"
    elif scan_reason in {"not_configured", ""} and watch:
        headline = "Telegram connected · last scan not pushed"
        detail = "Pushing last-scan setups and near-breakout watches"
        state = "pending"
    elif scan_reason == "already_sent":
        headline = "Scan alerts already sent today"
        detail = (
            "SNIPER BREAKOUT CONFIRMED still needs fresh Zerodha LTP"
            if not live_ok else "Sniper is watching live ticks"
        )
        state = "scan_sent"
    elif scan_reason == "no_candidates":
        headline = "Telegram connected"
        detail = "Last scan had no setup or near-breakout rows to push"
        state = "idle"
    else:
        headline = "Telegram connected"
        detail = "Setups and breakout watches send after each market scan"
        state = "ready"

    return {
        "configured": configured,
        "state": state,
        "headline": headline,
        "detail": detail,
        "scan_reason": scan_reason,
        "sniper_reason": sniper_reason,
        "scan_setups": int(scan.get("setup") or 0),
        "scan_prebreakout": int(scan.get("prebreakout") or 0),
        "sniper_watch": watch,
        "sniper_watching": int(sniper.get("watching") or 0),
        "sniper_fresh": int(sniper.get("fresh") or ticking),
        "scan_pushed": scan_pushed,
        "live_ticks": live_ok,
        "last_error": last_error,
        "updated_at": str(scan.get("updated_at") or sniper.get("updated_at") or ""),
    }


def drain_scan_alerts(*, min_interval_s: float = 45.0, root: Path | None = None) -> dict[str, Any]:
    """Replay last saved scan to Telegram when connected and today's keys are empty."""
    base = Path(root) if root else default_root()
    notifier = TelegramNotifier(base)
    if not notifier.configured():
        return {"setup": 0, "prebreakout": 0, "reason": "not_configured"}
    sent = notifier.drain_last_scan(min_interval_s=min_interval_s) or {}
    return sent
