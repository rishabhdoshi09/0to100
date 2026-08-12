"""
Market Monitor — runs as background daemon during trading hours.
Sends Telegram alerts for: regime changes, price alerts firing, VIX spikes.
Can be started as a thread from app.py or run standalone.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, date, time as dtime
from typing import Optional

import pytz

from logger import get_logger

log = get_logger(__name__)

_IST = pytz.timezone("Asia/Kolkata")
_MONITOR_OPEN  = dtime(9, 10)
_MONITOR_CLOSE = dtime(15, 35)
_PRE_OPEN_MINS = 30  # minutes before market open to allow start


class MarketMonitor:
    """
    Background daemon that polls market data during NSE hours and sends
    Telegram notifications proactively.
    """

    def __init__(self, poll_interval_secs: int = 60) -> None:
        self._poll_interval = poll_interval_secs
        self._stop_event    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # State tracking
        self._last_regime: Optional[str] = None
        self._last_vix: Optional[float]  = None

        # Once-per-day flags
        self._morning_brief_date: Optional[date] = None
        self._eod_summary_date:   Optional[date]  = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the daemon thread. No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            log.info("market_monitor_already_running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="MarketMonitor",
            daemon=True,
        )
        self._thread.start()
        log.info("market_monitor_started", poll_interval_secs=self._poll_interval)

    def stop(self) -> None:
        """Signal the polling thread to stop."""
        self._stop_event.set()
        log.info("market_monitor_stop_requested")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Polling loop — runs until stop() is called."""
        while not self._stop_event.is_set():
            try:
                now_ist  = datetime.now(_IST)
                now_time = now_ist.time()
                today    = now_ist.date()

                # Morning brief at 9:10 AM (once per day)
                if (
                    dtime(9, 9) <= now_time <= dtime(9, 12)
                    and self._morning_brief_date != today
                ):
                    try:
                        self._send_morning_brief()
                        self._morning_brief_date = today
                    except Exception as exc:
                        log.warning("morning_brief_error", error=str(exc))

                # Intraday polling — only within market hours
                if self._is_market_hours():
                    try:
                        self._check_regime()
                    except Exception as exc:
                        log.warning("regime_check_error", error=str(exc))

                    try:
                        self._check_vix()
                    except Exception as exc:
                        log.warning("vix_check_error", error=str(exc))

                    try:
                        self._check_price_alerts()
                    except Exception as exc:
                        log.warning("price_alert_check_error", error=str(exc))

                # EOD summary at 3:32 PM (once per day)
                if (
                    dtime(15, 31) <= now_time <= dtime(15, 35)
                    and self._eod_summary_date != today
                ):
                    try:
                        self._send_eod_summary()
                        self._eod_summary_date = today
                    except Exception as exc:
                        log.warning("eod_summary_error", error=str(exc))

            except Exception as exc:
                log.error("market_monitor_loop_error", error=str(exc))

            self._stop_event.wait(timeout=self._poll_interval)

        log.info("market_monitor_stopped")

    # ── Market hours check ────────────────────────────────────────────────────

    def _is_market_hours(self) -> bool:
        """Return True if it is currently NSE market hours (Mon–Fri 9:10–15:35 IST)."""
        now_ist  = datetime.now(_IST)
        now_time = now_ist.time()
        weekday  = now_ist.weekday()
        if weekday >= 5:  # Saturday or Sunday
            return False
        return _MONITOR_OPEN <= now_time <= _MONITOR_CLOSE

    # ── Regime check ──────────────────────────────────────────────────────────

    def _check_regime(self) -> None:
        from core.regime_engine import compute_regime
        from notify.alerts import AlertManager

        rs = compute_regime()
        current_regime = rs.market_regime

        if self._last_regime is not None and current_regime != self._last_regime:
            am = AlertManager()
            am.send_regime_change(
                from_regime=self._last_regime,
                to_regime=current_regime,
                score=rs.regime_score,
                confidence=rs.regime_confidence_label,
            )
            log.info(
                "regime_change_detected",
                from_regime=self._last_regime,
                to_regime=current_regime,
            )

        self._last_regime = current_regime

        # Also update VIX tracking from regime state
        if hasattr(rs, "vix") and rs.vix > 0:
            if self._last_vix is None:
                self._last_vix = rs.vix
            elif abs(rs.vix - self._last_vix) > 1.5:
                am = AlertManager()
                am.send_vix_spike(vix_prev=self._last_vix, vix_now=rs.vix)
                log.info("vix_spike_detected", prev=self._last_vix, now=rs.vix)
                self._last_vix = rs.vix
            else:
                self._last_vix = rs.vix

    # ── VIX check (standalone, in case regime engine not used) ───────────────

    def _check_vix(self) -> None:
        # VIX is checked inside _check_regime via regime state.
        # This method is a no-op placeholder to keep the loop structure clean.
        pass

    # ── Price alert check ─────────────────────────────────────────────────────

    def _check_price_alerts(self) -> None:
        from ai.jarvis_memory import JarvisMemory
        from notify.alerts import AlertManager

        try:
            from data.kite_client import KiteClient
            kite = KiteClient()
        except Exception:
            return  # KiteClient not available — skip silently

        # Collect symbols from active price alerts
        mem = JarvisMemory()
        try:
            active_alerts = mem.get_price_alerts()
        except Exception:
            return

        if not active_alerts:
            return

        symbols = list({a["symbol"] for a in active_alerts if "symbol" in a})
        if not symbols:
            return

        try:
            live_prices: dict[str, float] = kite.get_ltp(symbols)
        except Exception:
            return

        if not live_prices:
            return

        try:
            triggered = mem.check_price_alerts(live_prices)
        except Exception:
            return

        am = AlertManager()
        for alert in triggered:
            try:
                am.send_price_alert_fired(
                    symbol=alert.get("symbol", ""),
                    target_price=float(alert.get("price", 0)),
                    ltp=live_prices.get(alert.get("symbol", ""), 0.0),
                    direction=alert.get("direction", ""),
                )
            except Exception as exc:
                log.warning("price_alert_send_error", error=str(exc))

    # ── Morning brief ─────────────────────────────────────────────────────────

    def _send_morning_brief(self) -> None:
        from core.regime_engine import compute_regime
        from core.intelligence_hub import compute_opportunity_score
        from notify.alerts import AlertManager

        rs  = compute_regime()
        opp = compute_opportunity_score(regime_state=rs)

        regime_str    = rs.market_regime
        opp_score     = opp.total
        opp_grade     = opp.grade
        trading_rules = opp.recommended_aggression

        # Build a brief top-setups list from regime notes
        top_setups: list[str] = []
        for note in [opp.regime_note, opp.breadth_note, opp.volatility_note, opp.setup_note]:
            if note and note.strip():
                top_setups.append(note.strip())

        am = AlertManager()
        am.send_morning_brief(
            regime_str=regime_str,
            opp_score=opp_score,
            opp_grade=opp_grade,
            top_setups=top_setups,
            trading_rules=trading_rules,
        )
        log.info("morning_brief_sent", regime=regime_str, opp_score=opp_score)

    # ── EOD summary ───────────────────────────────────────────────────────────

    def _send_eod_summary(self) -> None:
        from paper_trading import get_trading_summary, get_open_positions
        from notify.alerts import AlertManager

        summary      = get_trading_summary()
        open_pos_df  = get_open_positions()
        open_count   = len(open_pos_df) if open_pos_df is not None and not open_pos_df.empty else 0

        am = AlertManager()
        am.send_eod_summary(
            total_trades=summary.get("num_trades", 0),
            win_rate=summary.get("win_rate", 0.0),
            total_pnl=summary.get("total_pnl", 0.0),
            open_positions=open_count,
        )
        log.info(
            "eod_summary_sent",
            num_trades=summary.get("num_trades"),
            total_pnl=summary.get("total_pnl"),
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_monitor_instance: Optional[MarketMonitor] = None
_monitor_lock = threading.Lock()


def get_monitor() -> MarketMonitor:
    """Return (or create) the module-level MarketMonitor singleton."""
    global _monitor_instance
    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = MarketMonitor()
    return _monitor_instance


def start_if_market_hours() -> None:
    """
    Convenience function: start the market monitor only if we are currently
    in a market session or within 30 minutes of the 9:15 AM open.
    """
    now_ist  = datetime.now(_IST)
    now_time = now_ist.time()
    weekday  = now_ist.weekday()

    if weekday >= 5:
        log.info("market_monitor_not_started_weekend")
        return

    # 8:45 AM → 15:35 PM window
    pre_open = dtime(8, 45)
    if not (pre_open <= now_time <= _MONITOR_CLOSE):
        log.info("market_monitor_not_started_outside_window", now=str(now_time))
        return

    monitor = get_monitor()
    monitor.start()
