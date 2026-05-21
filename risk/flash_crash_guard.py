"""
Flash Crash Guard — The Chitauri Defense.

Detects sudden intraday market moves (>1.5% Nifty in <5 min) and automatically:
  1. Pauses new trade entries for the session
  2. Widens stop-loss recommendations by 1.5×
  3. Fires Telegram alert
  4. Logs the event with timestamp for post-analysis

Runs as a lightweight check called before every trade in the engine.
Also exposes is_flash_crash_active() for UI status bar.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_STATE_FILE = Path("logs/flash_crash_state.json")
_LOCK = threading.Lock()

# Thresholds
_NIFTY_MOVE_THRESHOLD_PCT = 1.5   # % move in _WINDOW_SECONDS = flash crash
_WINDOW_SECONDS = 300             # 5 minutes
_COOLDOWN_MINUTES = 30            # pause new trades for 30 min after event
_VIX_SPIKE_THRESHOLD = 2.5       # VIX jump in one reading = also a flash signal


def _load_state() -> dict:
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text())
    except Exception:
        pass
    return {"active": False, "triggered_at": None, "trigger_reason": None, "today": None}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state))
    except Exception as e:
        log.warning("flash_crash_state_save_failed", error=str(e))


def is_flash_crash_active() -> tuple[bool, Optional[str]]:
    """
    Returns (is_active, reason_string).
    Auto-clears if cooldown has elapsed or it's a new day.
    """
    with _LOCK:
        state = _load_state()
        today = date.today().isoformat()

        # Reset at start of new trading day
        if state.get("today") != today:
            state = {"active": False, "triggered_at": None, "trigger_reason": None, "today": today}
            _save_state(state)
            return False, None

        if not state.get("active"):
            return False, None

        # Check cooldown
        triggered_at = state.get("triggered_at")
        if triggered_at:
            try:
                t = datetime.fromisoformat(triggered_at)
                elapsed = (datetime.now() - t).total_seconds() / 60
                if elapsed >= _COOLDOWN_MINUTES:
                    state["active"] = False
                    _save_state(state)
                    log.info("flash_crash_cooldown_cleared", elapsed_min=round(elapsed, 1))
                    return False, None
            except Exception:
                pass

        return True, state.get("trigger_reason", "Flash crash detected")


def check_and_trigger(
    nifty_price_now: float,
    nifty_price_5min_ago: float,
    vix_now: float,
    vix_prev: float,
) -> Optional[str]:
    """
    Call every 60s from MarketMonitor. Returns trigger reason if flash crash detected, else None.
    Automatically fires Telegram alert on trigger.
    """
    reason = None

    if nifty_price_5min_ago and nifty_price_5min_ago > 0:
        move_pct = abs(nifty_price_now - nifty_price_5min_ago) / nifty_price_5min_ago * 100
        if move_pct >= _NIFTY_MOVE_THRESHOLD_PCT:
            direction = "crashed" if nifty_price_now < nifty_price_5min_ago else "surged"
            reason = f"Nifty {direction} {move_pct:.1f}% in 5 minutes"

    if vix_now and vix_prev and (vix_now - vix_prev) >= _VIX_SPIKE_THRESHOLD:
        reason = reason or f"VIX spiked +{vix_now - vix_prev:.1f} points ({vix_prev:.1f}→{vix_now:.1f})"

    if reason:
        with _LOCK:
            state = _load_state()
            if not state.get("active"):  # don't re-trigger if already active
                state.update({
                    "active": True,
                    "triggered_at": datetime.now().isoformat(),
                    "trigger_reason": reason,
                    "today": date.today().isoformat(),
                })
                _save_state(state)
                log.warning("flash_crash_triggered", reason=reason)
                _send_alert(reason)

    return reason


def stop_loss_multiplier() -> float:
    """Return 1.5 during flash crash (widen stops), else 1.0."""
    active, _ = is_flash_crash_active()
    return 1.5 if active else 1.0


def _send_alert(reason: str) -> None:
    try:
        from config import settings
        import requests
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return
        requests.post(
            f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
            json={
                "chat_id": settings.telegram_chat_id,
                "text": f"🚨 FLASH CRASH GUARD TRIGGERED\n{reason}\n\nNew trade entries PAUSED for {_COOLDOWN_MINUTES} minutes.\nStop losses widened 1.5×.",
                "parse_mode": "HTML",
            },
            timeout=5,
        )
    except Exception:
        pass


def render_status_badge() -> str:
    """Returns HTML badge for UI status bar. Empty string if inactive."""
    active, reason = is_flash_crash_active()
    if not active:
        return ""
    return (
        f"<span style='background:#ff4b4b22;border:1px solid #ff4b4b55;border-radius:6px;"
        f"padding:.2rem .6rem;font-size:.7rem;color:#ff4b4b;font-weight:700'>"
        f"🚨 FLASH CRASH GUARD ACTIVE — {reason}</span>"
    )
