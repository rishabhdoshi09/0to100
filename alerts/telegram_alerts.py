"""
Telegram alert engine — price/RSI/breakout alerts via Telegram bot.

Setup: user sets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
Bot creation: https://t.me/BotFather → /newbot → get token
Chat ID: message the bot, then GET https://api.telegram.org/bot<token>/getUpdates

Credentials resolve from process env, pydantic Settings (.env), then the
on-disk ``.env`` file — uvicorn stack scripts often start without sourcing
``.env``, so AlertEngine must not rely on process env alone.
"""
from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import requests

logger = logging.getLogger("quantterm.telegram_alerts")

_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR = Path(os.environ.get("DEVBLOOM_LOG_DIR", "logs"))
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _LOGS_DIR / "alerts.db"

AlertType = Literal["PRICE_CROSS", "RSI_CROSS", "BREAKOUT"]


def _read_dotenv_value(name: str) -> str:
    """Read one key from repo-root ``.env`` without mutating process env."""
    path = _ROOT / ".env"
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == name:
                return val.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def escape_html(text: object) -> str:
    """Escape dynamic content for Telegram HTML parse_mode."""
    return html.escape(str(text or ""), quote=False)


def strip_telegram_html(message: str) -> str:
    """Plain-text fallback — never leave raw <b> tags visible in Telegram."""
    text = re.sub(r"<br\s*/?>", "\n", message or "", flags=re.I)
    text = re.sub(r"</?(b|i|strong|em|u|code|pre)>", "", text, flags=re.I)
    return html.unescape(text)


def resolve_telegram_credentials() -> tuple[str, str, str]:
    """Return ``(token, chat_id, source)`` from env / Settings / ``.env`` file."""
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    source = "process_env" if token and chat else ""

    if not token or not chat:
        try:
            from config import settings

            token = token or (settings.telegram_bot_token or "").strip()
            chat = chat or (settings.telegram_chat_id or "").strip()
            if token and chat and not source:
                source = "settings"
        except Exception:
            pass

    if not token or not chat:
        try:
            from dotenv import load_dotenv

            load_dotenv(_ROOT / ".env", override=False)
            token = token or (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
            chat = chat or (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
            if token and chat and not source:
                source = "dotenv_load"
        except Exception:
            pass

    if not token or not chat:
        file_token = _read_dotenv_value("TELEGRAM_BOT_TOKEN")
        file_chat = _read_dotenv_value("TELEGRAM_CHAT_ID")
        token = token or file_token
        chat = chat or file_chat
        if token and chat and not source:
            source = "dotenv_file"

    return token, chat, source


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
        self._token, self._chat_id, self.cred_source = resolve_telegram_credentials()
        self.enabled = bool(self._token and self._chat_id)
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if both token and chat_id are resolvable."""
        return self.enabled

    def connection_status(self) -> dict:
        """Diagnose Telegram config for UI / API error surfaces."""
        return {
            "configured": self.enabled,
            "token_present": bool(self._token),
            "chat_id_present": bool(self._chat_id),
            "source": self.cred_source if self.enabled else "",
            "last_error": self.last_error,
            "message": (
                "Telegram ready"
                if self.enabled
                else "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env, then restart or retry send."
            ),
        }

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

    @classmethod
    def strip_html(cls, message: str) -> str:
        """Plain-text fallback — never leave raw <b> tags visible in Telegram."""
        return strip_telegram_html(message)

    @staticmethod
    def escape(text: str) -> str:
        """Escape dynamic content for Telegram HTML parse_mode."""
        return escape_html(text)

    def _post_chunk(self, url: str, payload: dict) -> tuple[bool, str | None]:
        try:
            resp = requests.post(url, json=payload, timeout=12)
        except Exception as exc:
            return False, str(exc)
        description = ""
        try:
            data = resp.json() if resp.content else {}
        except Exception:
            data = {}
        if isinstance(data, dict):
            description = str(data.get("description") or "").strip()
            if data.get("ok") is True:
                return True, None
            if data.get("ok") is False:
                return False, description or f"Telegram API error (HTTP {resp.status_code})"
        if resp.ok:
            return True, None
        return False, description or f"HTTP {resp.status_code}: {(resp.text or '')[:180]}"

    def send(self, message: str, reply_markup: dict | None = None) -> bool:
        """
        POST *message* to Telegram sendMessage. Optional reply_markup
        attaches inline action buttons (see alerts/telegram_actions.py).
        Long messages auto-split at the 4096-char API cap (buttons ride
        on the last chunk). Returns True only if EVERY chunk delivered.
        On HTML parse failures, retries as plain text with tags stripped
        so users never see literal ``<b>`` markers.
        """
        self.last_error = None
        if not self.enabled:
            self.last_error = "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)"
            return False
        url = self._TELEGRAM_API.format(token=self._token)
        chunks = self._split_message(message, self._MAX_LEN)
        ok = True
        for i, chunk in enumerate(chunks):
            payload = {
                "chat_id": self._chat_id,
                "text": chunk,
                "parse_mode": "HTML",
            }
            if reply_markup and i == len(chunks) - 1:
                payload["reply_markup"] = reply_markup
            delivered, err = self._post_chunk(url, payload)
            if not delivered:
                # Always fall back to cleaned plain text — never show raw tags.
                plain = dict(payload)
                plain.pop("parse_mode", None)
                plain["text"] = self.strip_html(chunk)
                delivered, err = self._post_chunk(url, plain)
            if not delivered:
                self.last_error = err or f"Telegram send failed (chunk {i + 1}/{len(chunks)})"
                logger.warning(
                    "Telegram send failed (chunk %d/%d): %s",
                    i + 1,
                    len(chunks),
                    self.last_error,
                )
                ok = False
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
