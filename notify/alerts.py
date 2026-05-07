"""
Alert Manager + Background Signal Monitor.

Channels:
  Primary  : Telegram (requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)
  Fallback : structured log entry (always executed)

Background monitor (python main.py alerts --start):
  - Polls EnsembleSignalGenerator every 5 minutes during NSE market hours.
  - Sends alert when signal changes state (HOLD → BUY/SELL) for watchlist symbols.
  - Sends alert when RegimeDetector regime changes.
  - Sends a daily summary once at ~15:30 IST.
"""

from __future__ import annotations

import time
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional

import pytz
import requests

from config import settings
from logger import get_logger

log = get_logger(__name__)

_IST = pytz.timezone("Asia/Kolkata")
_MARKET_OPEN = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 30)
_POLL_INTERVAL_SECONDS = 300  # 5 minutes

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

_TYPE_EMOJI = {
    "signal":        "🔔",
    "risk":          "⚠️",
    "regime_change": "📊",
    "error":         "🚨",
    "summary":       "📈",
    "info":          "ℹ️",
}


class AlertManager:
    """
    Sends alerts via Telegram (primary) with structured-log fallback.

    Usage
    -----
    am = AlertManager()
    am.send_alert("RELIANCE turned BUY", alert_type="signal")
    """

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        self._token = telegram_token or getattr(settings, "telegram_bot_token", "")
        self._chat_id = telegram_chat_id or getattr(settings, "telegram_chat_id", "")
        self._telegram_ok = bool(self._token and self._chat_id)

        if not self._telegram_ok:
            log.warning("telegram_not_configured", hint="Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")

    def send_alert(self, message: str, alert_type: str = "info") -> bool:
        """
        Send an alert via Telegram (or log if not configured).

        Returns True if the alert was delivered successfully.
        """
        emoji = _TYPE_EMOJI.get(alert_type, "ℹ️")
        full_message = f"{emoji} [{alert_type.upper()}] {message}"

        # Always log
        log.info("alert_dispatched", type=alert_type, message=message)

        if self._telegram_ok:
            ok = self._send_telegram(full_message)
            if ok:
                return True
            log.warning("telegram_send_failed_falling_back_to_log", message=message)

        # Fallback: already logged above
        return False

    # ── Telegram ───────────────────────────────────────────────────────────

    def _send_telegram(self, text: str) -> bool:
        url = _TELEGRAM_API.format(token=self._token)
        try:
            resp = requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code == 200 and resp.json().get("ok"):
                log.debug("telegram_sent")
                return True
            log.warning("telegram_api_error", status=resp.status_code, body=resp.text[:200])
        except Exception as exc:
            log.warning("telegram_request_exception", error=str(exc))
        return False


# ── Background Monitor ─────────────────────────────────────────────────────────


class SignalMonitor:
    """
    Polls signals every 5 minutes during NSE market hours.
    Run via: python main.py alerts --start
    """

    def __init__(self) -> None:
        self._alerts = AlertManager()
        self._last_signals: Dict[str, str] = {}       # symbol → last action
        self._last_regime: Optional[str] = None
        self._daily_summary_sent: Optional[datetime] = None
        self._daily_signal_count: int = 0

    def run(self) -> None:
        """Blocking monitor loop. Runs until Ctrl+C."""
        log.info("signal_monitor_started", watchlist=settings.alert_watchlist_list)
        print(f"[AlertMonitor] Watching: {', '.join(settings.alert_watchlist_list)}")
        print("[AlertMonitor] Press Ctrl+C to stop.\n")

        try:
            while True:
                now_ist = datetime.now(_IST)
                now_time = now_ist.time()

                if _MARKET_OPEN <= now_time <= _MARKET_CLOSE:
                    self._tick(now_ist)

                    # Daily summary at ~15:30
                    if now_time >= dtime(15, 28) and self._should_send_daily_summary():
                        self._send_daily_summary()

                time.sleep(_POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log.info("signal_monitor_stopped")
            print("\n[AlertMonitor] Stopped.")

    def _tick(self, now: datetime) -> None:
        """Run one polling cycle."""
        self._check_signals()
        self._check_regime()

    def _check_signals(self) -> None:
        try:
            from ml.ensemble_signal import EnsembleSignalGenerator
            from data.kite_client import KiteClient
            from data.instruments import InstrumentManager
            from data.historical import HistoricalDataFetcher

            gen = EnsembleSignalGenerator()
            kite = KiteClient()
            instruments = InstrumentManager()
            fetcher = HistoricalDataFetcher(kite, instruments)

            to_d = datetime.now().strftime("%Y-%m-%d")
            from_d = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            for symbol in settings.alert_watchlist_list:
                try:
                    df = fetcher.fetch(symbol, from_d, to_d, "day")
                    if df is None or len(df) < 20:
                        continue
                    sig = gen.generate_signal(df, symbol)
                    action = sig.get("action", "HOLD")
                    confidence = sig.get("confidence", 0.5)

                    prev = self._last_signals.get(symbol, "HOLD")
                    if action != prev and action in ("BUY", "SELL"):
                        msg = (
                            f"Signal: {symbol} turned <b>{action}</b> "
                            f"(confidence {confidence:.0%})"
                        )
                        self._alerts.send_alert(msg, alert_type="signal")
                        self._daily_signal_count += 1

                    self._last_signals[symbol] = action
                except Exception as exc:
                    log.warning("monitor_symbol_check_failed", symbol=symbol, error=str(exc))

        except Exception as exc:
            log.error("monitor_signal_check_error", error=str(exc))

    def _check_regime(self) -> None:
        try:
            from analysis.regime_detector import RegimeDetector
            rd = RegimeDetector()
            result = rd.detect()
            current = result.get("regime", "UNKNOWN")

            if self._last_regime is not None and current != self._last_regime:
                msg = f"Regime changed: <b>{self._last_regime}</b> → <b>{current}</b>"
                self._alerts.send_alert(msg, alert_type="regime_change")

            self._last_regime = current
        except Exception as exc:
            log.warning("monitor_regime_check_failed", error=str(exc))

    def _should_send_daily_summary(self) -> bool:
        today = datetime.now().date()
        if self._daily_summary_sent is None:
            return True
        return self._daily_summary_sent.date() < today

    def _send_daily_summary(self) -> None:
        regime = self._last_regime or "UNKNOWN"
        msg = (
            f"Daily summary: {self._daily_signal_count} signal(s) generated. "
            f"Current regime: <b>{regime}</b>."
        )
        self._alerts.send_alert(msg, alert_type="summary")
        self._daily_summary_sent = datetime.now()
        self._daily_signal_count = 0


# Avoid circular import at module level — timedelta imported lazily inside methods
from datetime import timedelta  # noqa: E402
