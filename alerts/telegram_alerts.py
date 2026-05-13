"""
Telegram alert engine — price/RSI/breakout alerts via Telegram bot.

Setup: user sets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
Bot creation: https://t.me/BotFather → /newbot → get token
Chat ID: message the bot, then GET https://api.telegram.org/bot<token>/getUpdates
"""
from __future__ import annotations

import os
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import requests

logger = logging.getLogger("quantterm.telegram_alerts")

_LOGS_DIR = Path(os.environ.get("DEVBLOOM_LOG_DIR", "logs"))
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _LOGS_DIR / "alerts.db"

AlertType = Literal["PRICE_CROSS", "RSI_CROSS", "BREAKOUT"]


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertRule:
    rule_id: int
    symbol: str
    alert_type: AlertType
    threshold: float
    triggered: bool
    created_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Telegram engine
# ─────────────────────────────────────────────────────────────────────────────

class AlertEngine:
    """Sends Telegram messages via the Bot API."""

    _TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self) -> None:
        self._token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.enabled  = bool(self._token and self._chat_id)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if both token and chat_id are present in env."""
        return self.enabled

    def send(self, message: str) -> bool:
        """
        POST *message* to Telegram sendMessage.
        Returns True on success, False on any error (silent fail).
        """
        if not self.enabled:
            return False
        url = self._TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id":    self._chat_id,
            "text":       message,
            "parse_mode": "HTML",
        }
        try:
            resp = requests.post(url, json=payload, timeout=8)
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Formatted alert senders
    # ------------------------------------------------------------------

    def send_signal_alert(
        self,
        symbol: str,
        signal: str,
        price: float,
        score: float,
        rsi: float,
        vol_ratio: float,
    ) -> bool:
        """Send a BUY/SELL/WATCH signal alert."""
        direction_icon = "📈" if signal == "BUY" else ("📉" if signal == "SELL" else "📊")
        now_str = datetime.now().strftime("%H:%M") + " IST"

        # Build a human-readable summary line from the composite score.
        if score >= 75 and vol_ratio >= 2.0:
            summary = "Momentum + Volume surge confirmed"
        elif score >= 65:
            summary = "Momentum breakout signal"
        elif rsi < 35:
            summary = "Oversold RSI bounce setup"
        elif vol_ratio >= 2.0:
            summary = "Volume surge detected"
        else:
            summary = "Technical setup aligned"

        msg = (
            f"🚨 <b>QUANTTERM SIGNAL</b>\n\n"
            f"{direction_icon} <b>{signal} — {symbol}</b>\n"
            f"Price: ₹{price:,.2f}\n"
            f"Score: {score:.1f} | RSI: {rsi:.1f} | Vol: {vol_ratio:.1f}×\n\n"
            f"Signal: {summary}\n"
            f"Time: {now_str}"
        )
        return self.send(msg)

    def send_price_alert(
        self,
        symbol: str,
        price: float,
        target_price: float,
        alert_type: str,
    ) -> bool:
        """Notify that a price level was crossed."""
        direction = "crossed above" if price >= target_price else "crossed below"
        msg = (
            f"🔔 <b>Price Alert — {symbol}</b>\n\n"
            f"<b>{symbol}</b> {direction} ₹{target_price:,.2f}\n"
            f"Current price: ₹{price:,.2f}\n"
            f"Alert type: {alert_type}\n"
            f"Time: {datetime.now().strftime('%H:%M IST')}"
        )
        return self.send(msg)

    def send_breakout_alert(
        self,
        symbol: str,
        price: float,
        breakout_type: str,
        confidence: float,
    ) -> bool:
        """Notify about a technical breakout pattern."""
        _icons = {
            "52W_HIGH":         "🔴",
            "GOLDEN_CROSS":     "🟢",
            "VOL_SQUEEZE":      "🔵",
            "RESISTANCE_BREAK": "🟡",
            "CUP_HANDLE":       "⭐",
        }
        icon  = _icons.get(breakout_type, "⚡")
        label = breakout_type.replace("_", " ").title()
        msg = (
            f"💥 <b>Breakout Alert — {symbol}</b>\n\n"
            f"{icon} Pattern: <b>{label}</b>\n"
            f"Price: ₹{price:,.2f}\n"
            f"Confidence: {confidence:.0f}%\n"
            f"Time: {datetime.now().strftime('%H:%M IST')}"
        )
        return self.send(msg)

    def send_test(self) -> bool:
        """Send a connectivity test message."""
        msg = (
            "✅ <b>QUANTTERM connected!</b>\n\n"
            "Telegram alerts are working correctly.\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"
        )
        return self.send(msg)


# ─────────────────────────────────────────────────────────────────────────────
# SQLite-backed rule manager
# ─────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """
    Persist alert rules in SQLite (logs/alerts.db).
    Fires Telegram messages via AlertEngine when rules trigger.
    """

    def __init__(self, db_path: str | Path = _DB_PATH) -> None:
        self._db = Path(db_path)
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._engine = AlertEngine()
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol     TEXT    NOT NULL,
                    alert_type TEXT    NOT NULL,
                    threshold  REAL    NOT NULL,
                    triggered  INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT    NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS alert_fires (
                    fire_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id    INTEGER NOT NULL,
                    symbol     TEXT    NOT NULL,
                    alert_type TEXT    NOT NULL,
                    threshold  REAL    NOT NULL,
                    fired_at   TEXT    NOT NULL,
                    price      REAL,
                    rsi        REAL
                )
            """)
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, check_same_thread=False)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_rule(
        self,
        symbol: str,
        alert_type: AlertType,
        threshold: float,
    ) -> int:
        """Insert a new alert rule and return its rule_id."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO alert_rules (symbol, alert_type, threshold, triggered, created_at) "
                "VALUES (?, ?, ?, 0, ?)",
                (symbol.upper(), alert_type, threshold, now),
            )
            con.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_rules(self) -> list[AlertRule]:
        """Return all stored alert rules."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT rule_id, symbol, alert_type, threshold, triggered, created_at "
                "FROM alert_rules ORDER BY rule_id DESC"
            ).fetchall()
        return [
            AlertRule(
                rule_id=r[0],
                symbol=r[1],
                alert_type=r[2],  # type: ignore[arg-type]
                threshold=r[3],
                triggered=bool(r[4]),
                created_at=r[5],
            )
            for r in rows
        ]

    def delete_rule(self, rule_id: int) -> None:
        """Remove an alert rule by ID."""
        with self._connect() as con:
            con.execute("DELETE FROM alert_rules WHERE rule_id = ?", (rule_id,))
            con.commit()

    def get_recent_fires(self, limit: int = 20) -> list[dict]:
        """Return the last *limit* fired-alert records."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT fire_id, rule_id, symbol, alert_type, threshold, fired_at, price, rsi "
                "FROM alert_fires ORDER BY fire_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "fire_id":    r[0],
                "rule_id":    r[1],
                "symbol":     r[2],
                "alert_type": r[3],
                "threshold":  r[4],
                "fired_at":   r[5],
                "price":      r[6],
                "rsi":        r[7],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Runtime check
    # ------------------------------------------------------------------

    def check_and_fire(
        self,
        symbol: str,
        current_price: float,
        current_rsi: float,
    ) -> None:
        """
        Evaluate all non-triggered rules for *symbol*.
        Fire a Telegram alert and mark the rule as triggered when the
        condition is met.
        """
        symbol = symbol.upper()
        with self._connect() as con:
            rows = con.execute(
                "SELECT rule_id, alert_type, threshold "
                "FROM alert_rules "
                "WHERE symbol = ? AND triggered = 0",
                (symbol,),
            ).fetchall()

        for rule_id, alert_type, threshold in rows:
            triggered = False

            if alert_type == "PRICE_CROSS" and current_price >= threshold:
                triggered = True
                self._engine.send_price_alert(
                    symbol, current_price, threshold, alert_type
                )

            elif alert_type == "PRICE_CROSS_BELOW" and current_price <= threshold:
                triggered = True
                self._engine.send_price_alert(
                    symbol, current_price, threshold, alert_type
                )

            elif alert_type == "RSI_CROSS" and current_rsi >= threshold:
                triggered = True
                self._engine.send_price_alert(
                    symbol, current_price, threshold,
                    f"RSI crossed {threshold:.0f}"
                )

            elif alert_type == "BREAKOUT":
                # Breakout rules use threshold as a confidence floor (0–100).
                # They are treated as fired immediately for demonstration;
                # real callers should pass confidence via threshold.
                triggered = True
                self._engine.send_breakout_alert(
                    symbol, current_price, "RESISTANCE_BREAK", threshold
                )

            if triggered:
                now = datetime.now().isoformat(timespec="seconds")
                with self._connect() as con:
                    con.execute(
                        "UPDATE alert_rules SET triggered = 1 WHERE rule_id = ?",
                        (rule_id,),
                    )
                    con.execute(
                        "INSERT INTO alert_fires "
                        "(rule_id, symbol, alert_type, threshold, fired_at, price, rsi) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (rule_id, symbol, alert_type, threshold, now,
                         current_price, current_rsi),
                    )
                    con.commit()
                logger.info(
                    "alert_fired rule_id=%s symbol=%s type=%s threshold=%s price=%s rsi=%s",
                    rule_id, symbol, alert_type, threshold, current_price, current_rsi,
                )
