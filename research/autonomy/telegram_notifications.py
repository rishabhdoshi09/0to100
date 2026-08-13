"""Supervisor-owned Telegram notifications for QuantTerm.

This module restores the useful phone alerts from the legacy scanner without restoring any legacy
scheduler or broker-capable execution path.  The autonomy supervisor is the only caller.  Alerts are
read-only observations plus PAPER ledger notifications; they never submit an order.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = None


class TelegramNotifier:
    """Durable, restart-safe Telegram notifier owned by the autonomy process."""

    def __init__(
        self,
        root: str | Path,
        *,
        engine_factory: Callable[[], Any] | None = None,
        now_fn: Callable[[], datetime] | None = None,
        epoch_fn: Callable[[], float] = time.time,
        breakout_confirmation_s: float = 8.0,
        breakout_buffer_bps: float = 10.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "telegram_notifications.json"
        self._engine_factory = engine_factory
        self._now_fn = now_fn or (lambda: datetime.now(_IST) if _IST else datetime.now())
        self._epoch = epoch_fn
        self.breakout_confirmation_s = float(breakout_confirmation_s)
        self.breakout_buffer_bps = float(breakout_buffer_bps)
        self.state = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("sent", {})
                data.setdefault("arms", {})
                return data
        except Exception:
            pass
        return {"sent": {}, "arms": {}}

    def _save(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            pass

    def _engine(self):
        if self._engine_factory is not None:
            return self._engine_factory()
        try:
            from dotenv import load_dotenv
            load_dotenv(".env", override=False)
        except Exception:
            pass
        from alerts.telegram_alerts import AlertEngine
        return AlertEngine()

    def configured(self) -> bool:
        try:
            return bool(self._engine().is_configured())
        except Exception:
            return False

    def _day(self) -> str:
        return self._now_fn().strftime("%Y-%m-%d")

    def _prune(self, day: str) -> None:
        sent = self.state.setdefault("sent", {})
        for old in list(sent):
            if old != day:
                del sent[old]
        arms = self.state.setdefault("arms", {})
        for sym, arm in list(arms.items()):
            if str(arm.get("day", "")) != day:
                del arms[sym]

    def _was_sent(self, key: str, day: str | None = None) -> bool:
        day = day or self._day()
        return key in set(self.state.setdefault("sent", {}).setdefault(day, []))

    def _mark_sent(self, keys: list[str], day: str | None = None) -> None:
        day = day or self._day()
        self._prune(day)
        bucket = set(self.state.setdefault("sent", {}).setdefault(day, []))
        bucket.update(keys)
        self.state["sent"][day] = sorted(bucket)
        self._save()

    def _send_once(self, key: str, message: str) -> bool:
        day = self._day()
        self._prune(day)
        if self._was_sent(key, day):
            return False
        try:
            engine = self._engine()
            if not engine.is_configured() or not engine.send(message):
                return False
        except Exception:
            return False
        self._mark_sent([key], day)
        return True

    @staticmethod
    def _f(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _esc(value: Any) -> str:
        return html.escape(str(value or ""), quote=False)

    def notify_online(self) -> bool:
        return self._send_once(
            "autonomy_online",
            "🟢 <b>QuantTerm autonomy online</b>\n\n"
            "Single supervisor active · Telegram alerts enabled · LIVE orders locked.",
        )

    def notify_scan(self, payload: Mapping[str, Any] | None, *, phase: str = "") -> dict:
        """Send new setup, pre-breakout, premarket and EOD scan alerts once per symbol/day."""
        payload = dict(payload or {})
        records = [dict(r) for r in payload.get("records", []) if isinstance(r, Mapping)]
        summary = dict(payload.get("summary", {}) or {})
        day = self._day()
        self._prune(day)
        sent = {"setup": 0, "prebreakout": 0, "briefing": 0, "eod": 0}

        if phase == "premarket" and not self._was_sent("morning_brief", day):
            try:
                from core.brain import briefing_telegram
                if self._engine().send(briefing_telegram("IN")):
                    self._mark_sent(["morning_brief"], day)
                    sent["briefing"] = 1
            except Exception:
                pass

        ready = [r for r in records
                 if str(r.get("verdict", "")).upper() in ("BUY", "STRONG BUY")
                 and not bool(r.get("chase_risk"))
                 and self._f(r.get("entry")) > 0
                 and not self._was_sent(f"setup:{str(r.get('symbol', '')).upper()}", day)]
        ready.sort(key=lambda r: (-self._f(r.get("score")), str(r.get("symbol", ""))))
        ready = ready[:5]

        ready_symbols = {str(r.get("symbol", "")).upper() for r in ready}
        pre = [r for r in records
               if (str(r.get("status", "")) == "Watch for breakout"
                   or "PRE_BREAKOUT" in [str(x).upper() for x in (r.get("signals") or [])])
               and self._f(r.get("entry")) > 0
               and str(r.get("symbol", "")).upper() not in ready_symbols
               and not self._was_sent(f"pre:{str(r.get('symbol', '')).upper()}", day)]
        pre.sort(key=lambda r: (-self._f(r.get("score")), str(r.get("symbol", ""))))
        pre = pre[:4]

        lines: list[str] = []
        keys: list[str] = []
        if ready:
            lines.append("🎯 <b>QuantTerm — naye qualified setups</b>")
            for r in ready:
                sym = str(r.get("symbol", "")).upper()
                verdict = str(r.get("verdict", "BUY")).upper()
                emoji = "🔥" if verdict == "STRONG BUY" else "⚡"
                reasons = [self._esc(x) for x in (r.get("reasons") or []) if str(x).strip()]
                why = "\n".join(f"   ✓ {x}" for x in reasons[:3]) or "   ✓ Scanner gates passed"
                lines.append(
                    f"\n{emoji} <b>{self._esc(sym)}</b> ₹{self._f(r.get('price')):,.2f} — {self._esc(verdict)}\n"
                    f"{why}\n"
                    f"   Score {self._f(r.get('score')):.0f} · RSI {self._f(r.get('rsi')):.0f} · "
                    f"Vol {self._f(r.get('volume_ratio')):.1f}×\n"
                    f"   Entry ₹{self._f(r.get('entry')):,.2f} · Stop ₹{self._f(r.get('stop')):,.2f} · "
                    f"Target ₹{self._f(r.get('target')):,.2f}"
                )
                keys.append(f"setup:{sym}")
        if pre:
            lines.append("\n⏳ <b>Breakout ke kareeb — watch</b>")
            for r in pre:
                sym = str(r.get("symbol", "")).upper()
                gap = 0.0
                entry = self._f(r.get("entry")); price = self._f(r.get("price"))
                if entry > 0:
                    gap = max(0.0, (entry - price) / entry * 100.0)
                reason = self._esc((r.get("reasons") or ["Structure near trigger"])[0])
                lines.append(
                    f"\n👀 <b>{self._esc(sym)}</b> ₹{price:,.2f} · trigger ₹{entry:,.2f} "
                    f"({gap:.1f}% neeche)\n"
                    f"   {reason}\n"
                    f"   Confirm hone par alert · stop ₹{self._f(r.get('stop')):,.2f}"
                )
                keys.append(f"pre:{sym}")

        if lines:
            try:
                engine = self._engine()
                if engine.is_configured() and engine.send("\n".join(lines)):
                    self._mark_sent(keys, day)
                    sent["setup"] = len(ready)
                    sent["prebreakout"] = len(pre)
            except Exception:
                pass

        if phase == "eod" and not self._was_sent("eod_scan_summary", day):
            msg = (
                "📊 <b>QuantTerm EOD scan complete</b>\n\n"
                f"Universe: {int(payload.get('universe_size', 0) or 0):,}\n"
                f"Qualified setups: {int(summary.get('with_any_setup', 0) or 0)}\n"
                f"Ready to trade: {int(summary.get('ready_to_trade', 0) or 0)}\n"
                f"Near breakout: {int(summary.get('near_breakout', 0) or 0)}\n"
                f"Momentum: {int(summary.get('momentum', 0) or 0)}"
            )
            if self._send_once("eod_scan_summary", msg):
                sent["eod"] = 1
        return sent

    def notify_paper_cycle(self, result: Mapping[str, Any] | None, *, book=None) -> dict:
        """Notify simulated opens and closes.  This method cannot reach a broker order API."""
        result = dict(result or {})
        day = self._day()
        self._prune(day)
        opened = list(result.get("positions_opened", []) or [])
        closed = list(result.get("positions_closed", []) or [])
        counts = {"opened": 0, "closed": 0}

        open_by = {}
        closed_rows = []
        if book is not None:
            try:
                open_by = {(p.strategy_id, p.symbol): p for p in book.open.values()}
                closed_rows = list(book.closed)
            except Exception:
                open_by, closed_rows = {}, []

        for item in opened:
            if len(item) < 2:
                continue
            sid, sym = str(item[0]), str(item[1]).upper()
            key = f"paper_open:{sid}:{sym}:{day}"
            if self._was_sent(key, day):
                continue
            pos = open_by.get((sid, sym))
            detail = ""
            if pos is not None:
                detail = (f"\nQty: {int(pos.qty)} · Entry ₹{float(pos.entry_price):,.2f}"
                          f"\nStop ₹{float(pos.stop_price):,.2f} · Target ₹{float(pos.target_price):,.2f}")
            msg = (f"📝 <b>PAPER position opened — {self._esc(sym)}</b>\n\n"
                   f"Strategy: {self._esc(sid)}{detail}\n\nLIVE broker order: <b>NO</b>")
            if self._send_once(key, msg):
                counts["opened"] += 1

        for item in closed:
            if len(item) < 3:
                continue
            sid, sym, reason = str(item[0]), str(item[1]).upper(), str(item[2])
            trade = next((t for t in reversed(closed_rows)
                          if t.strategy_id == sid and t.symbol == sym and t.exit_reason == reason), None)
            exit_day = str(getattr(trade, "exit_date", day) or day)
            key = f"paper_close:{sid}:{sym}:{exit_day}:{reason}"
            if self._was_sent(key, day):
                continue
            detail = ""
            if trade is not None:
                detail = (f"\nEntry ₹{float(trade.entry_price):,.2f} → Exit ₹{float(trade.exit_price):,.2f}"
                          f"\nP&L ₹{float(trade.pnl):+,.2f} · {float(trade.realized_R):+.2f}R")
            icon = "✅" if trade is not None and float(trade.pnl) > 0 else "🛑"
            msg = (f"{icon} <b>PAPER position closed — {self._esc(sym)}</b>\n\n"
                   f"Reason: {self._esc(reason)} · Strategy: {self._esc(sid)}{detail}")
            if self._send_once(key, msg):
                counts["closed"] += 1
        return counts

    def observe_live_breakouts(self, payload: Mapping[str, Any] | None, live_feed) -> dict:
        """Confirm fresh Kite price crossings from the supervisor-owned live feed.

        A symbol must remain above its trigger for ``breakout_confirmation_s`` and has a 10 bps
        default buffer.  This avoids alerting on a one-tick touch and replaces the legacy sniper's
        separate WebSocket owner.
        """
        payload = dict(payload or {})
        records = [dict(r) for r in payload.get("records", []) if isinstance(r, Mapping)]
        day = self._day(); self._prune(day)
        arms = self.state.setdefault("arms", {})
        now = float(self._epoch())
        confirmed: list[tuple[dict, float]] = []

        try:
            from product.breakout_quality import gate_breakout_quality
        except Exception:
            gate_breakout_quality = None  # type: ignore

        for r in records:
            sym = str(r.get("symbol", "")).upper()
            entry = self._f(r.get("entry"))
            if not sym or entry <= 0:
                continue
            signals = [str(x).upper() for x in (r.get("signals") or [])]
            relevant = (str(r.get("status", "")) in ("Watch for breakout", "Ready to trade")
                        or "PRE_BREAKOUT" in signals)
            if not relevant:
                continue
            # Same hard gates as sniper/best — never alert 0.1× / blow-off / AVOID.
            if gate_breakout_quality is not None:
                ok, reasons, _ = gate_breakout_quality(r)
                if not ok:
                    arms.pop(sym, None)
                    continue
            elif self._f(r.get("volume_ratio")) < 1.0:
                arms.pop(sym, None)
                continue
            try:
                fresh = bool(live_feed.entry_allowed(sym))
                price = self._f(live_feed.price(sym))
            except Exception:
                fresh, price = False, 0.0
            trigger = entry * (1.0 + self.breakout_buffer_bps / 10000.0)
            if not fresh or price < trigger:
                arms.pop(sym, None)
                continue
            key = f"breakout:{sym}"
            if self._was_sent(key, day):
                continue
            arm = arms.get(sym)
            if not arm or abs(self._f(arm.get("entry")) - entry) > 1e-6:
                arms[sym] = {"day": day, "first_seen": now, "entry": entry, "last_price": price}
                continue
            arm["last_price"] = price
            if now - self._f(arm.get("first_seen")) >= self.breakout_confirmation_s:
                confirmed.append((r, price))

        self._save()
        if not confirmed:
            return {"confirmed": 0}
        confirmed.sort(key=lambda x: (-self._f(x[0].get("score")), str(x[0].get("symbol", ""))))
        confirmed = confirmed[:5]
        lines = ["🚨 <b>BREAKOUT CONFIRMED — fresh Kite ticks</b>"]
        keys = []
        for r, price in confirmed:
            sym = str(r.get("symbol", "")).upper()
            lines.append(
                f"\n⚡ <b>{self._esc(sym)}</b> crossed ₹{self._f(r.get('entry')):,.2f} "
                f"and held above it\n"
                f"   LTP ₹{price:,.2f} · Score {self._f(r.get('score')):.0f} · "
                f"Volume {self._f(r.get('volume_ratio')):.1f}× (≥1.0×)\n"
                f"   RSI {self._f(r.get('rsi')):.0f} · "
                f"{self._esc(str(r.get('classification') or 'fundamentals n/a'))}\n"
                f"   PAPER plan: stop ₹{self._f(r.get('stop')):,.2f} · "
                f"target ₹{self._f(r.get('target')):,.2f}"
            )
            keys.append(f"breakout:{sym}")
        try:
            engine = self._engine()
            if engine.is_configured() and engine.send("\n".join(lines)):
                self._mark_sent(keys, day)
                for r, _ in confirmed:
                    arms.pop(str(r.get("symbol", "")).upper(), None)
                self._save()
                return {"confirmed": len(confirmed)}
        except Exception:
            pass
        return {"confirmed": 0}

    def notify_long_term(self, payload: Mapping[str, Any] | None) -> dict:
        """Send the current weekly long-term shortlist once per symbol/day."""
        payload = dict(payload or {})
        records = [dict(r) for r in payload.get("records", []) if isinstance(r, Mapping)]
        eligible = [
            r for r in records
            if str(r.get("classification", "")) in
            ("QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE")
            and self._f(r.get("fundamental_coverage")) >= 0.50
        ]
        eligible.sort(key=lambda r: (-self._f(r.get("combined_score")),
                                     str(r.get("symbol", ""))))
        eligible = [r for r in eligible
                    if not self._was_sent(f"longterm:{str(r.get('symbol', '')).upper()}")][:6]
        if not eligible:
            return {"sent": 0}
        lines = ["💎 <b>QuantTerm — current long-term shortlist</b>",
                 "<i>Current fundamentals + official price history; not a historical backtest.</i>"]
        keys = []
        for row in eligible:
            sym = str(row.get("symbol", "")).upper()
            cls = str(row.get("classification", "")).replace("_", " ").title()
            factors = [self._esc(x) for x in (row.get("quality_factors") or []) if str(x).strip()]
            risks = [self._esc(x) for x in (row.get("risk_flags") or []) if str(x).strip()]
            lines.append(
                f"\n<b>{self._esc(sym)}</b> — {self._esc(cls)} · score "
                f"{self._f(row.get('combined_score')):.0f}\n"
                f"   Technical {self._f(row.get('technical_score')):.0f} · "
                f"Fundamental {self._f(row.get('fundamental_score')):.0f} · "
                f"Coverage {self._f(row.get('fundamental_coverage'))*100:.0f}%\n"
                f"   {self._esc(row.get('timing', '')).replace('_', ' ').title()}"
                + (f"\n   ✓ {'; '.join(factors[:2])}" if factors else "")
                + (f"\n   ⚠ {'; '.join(risks[:2])}" if risks else "")
            )
            keys.append(f"longterm:{sym}")
        try:
            engine = self._engine()
            if engine.is_configured() and engine.send("\n".join(lines)):
                self._mark_sent(keys)
                return {"sent": len(eligible)}
        except Exception:
            pass
        return {"sent": 0}

    def notify_incident(self, code: str, message: str) -> bool:
        important = {
            "CRITICAL_OVERDUE", "HANDLER_EXCEPTION", "AUTH_EXPIRED", "TOKEN_MISSING",
            "PROVIDER_UNAVAILABLE", "SCAN_ERROR", "CYCLE_ERROR", "OUTCOME_ERROR",
            "EVENT_STORE_FAILED", "BLOCKED",
        }
        code = str(code or "UNKNOWN").upper()
        if code not in important:
            return False
        digest = hashlib.sha1(str(message).encode("utf-8")).hexdigest()[:10]
        return self._send_once(
            f"incident:{code}:{digest}",
            f"⚠️ <b>QuantTerm needs attention</b>\n\n"
            f"Code: <b>{self._esc(code)}</b>\n{self._esc(message)[:1200]}",
        )
