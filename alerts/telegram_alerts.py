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
        # Prefer settings (.env via pydantic). Fall back to os.environ after an
        # explicit dotenv load so WebSocket sniper threads still work when the
        # process did not export TELEGRAM_* into the environment.
        token = ""
        chat_id = ""
        try:
            from alerts.telegram_status import telegram_credentials, usable_telegram_secret

            token, chat_id = telegram_credentials()
            self._token = token
            self._chat_id = chat_id
            self.enabled = usable_telegram_secret(token) and usable_telegram_secret(chat_id)
        except Exception:
            try:
                from config import settings
                token = str(getattr(settings, "telegram_bot_token", "") or "").strip()
                chat_id = str(getattr(settings, "telegram_chat_id", "") or "").strip()
            except Exception:
                pass
            if not (token and chat_id):
                try:
                    from dotenv import load_dotenv
                    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
                except Exception:
                    pass
                token = (os.environ.get("TELEGRAM_BOT_TOKEN", "") or token).strip()
                chat_id = (os.environ.get("TELEGRAM_CHAT_ID", "") or chat_id).strip()
            self._token = token
            self._chat_id = chat_id
            self.enabled = bool(self._token and self._chat_id)
        if not self.enabled:
            logger.warning(
                "telegram_not_configured — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env"
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if both token and chat_id are present in env."""
        return self.enabled

    _MAX_LEN = 3800   # Telegram hard cap is 4096; headroom for HTML entities

    @staticmethod
    def _split_message(message: str, max_len: int) -> list[str]:
        """Split on blank-line boundaries so no chunk exceeds the cap —
        a too-long message must degrade to N messages, never to silence."""
        if len(message) <= max_len:
            return [message]
        chunks, current = [], ""
        for block in message.split("\n\n"):
            while len(block) > max_len:          # single huge block: hard cut
                chunks.append(block[:max_len])
                block = block[max_len:]
            if len(current) + len(block) + 2 > max_len:
                if current:
                    chunks.append(current)
                current = block
            else:
                current = f"{current}\n\n{block}" if current else block
        if current:
            chunks.append(current)
        return chunks

    def send(self, message: str, reply_markup: dict | None = None) -> bool:
        """
        POST *message* to Telegram sendMessage. Optional reply_markup
        attaches inline action buttons (see alerts/telegram_actions.py).
        Long messages auto-split at the 4096-char API cap (buttons ride
        on the last chunk). Returns True only if EVERY chunk delivered.
        """
        if not self.enabled:
            try:
                from alerts.telegram_status import record_send
                record_send(False, "not_configured")
            except Exception:
                pass
            return False
        url = self._TELEGRAM_API.format(token=self._token)
        chunks = self._split_message(message, self._MAX_LEN)
        ok = True
        for i, chunk in enumerate(chunks):
            payload = {
                "chat_id":    self._chat_id,
                "text":       chunk,
                "parse_mode": "HTML",
            }
            if reply_markup and i == len(chunks) - 1:
                payload["reply_markup"] = reply_markup
            try:
                resp = requests.post(url, json=payload, timeout=8)
                if resp.status_code == 400 and payload.get("parse_mode"):
                    plain = dict(payload)
                    plain.pop("parse_mode", None)
                    resp = requests.post(url, json=plain, timeout=8)
                resp.raise_for_status()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                try:
                    from alerts.telegram_status import classify_error, record_send
                    record_send(False, classify_error(exc, status))
                except Exception:
                    pass
                logger.warning("Telegram send failed (chunk %d/%d): %s",
                               i + 1, len(chunks), exc)
                ok = False
        try:
            from alerts.telegram_status import record_send
            if ok:
                record_send(True)
        except Exception:
            pass
        return ok

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
