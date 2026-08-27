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
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

_DRAIN_LOCK = threading.Lock()

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = None

# Same quality floor as scan/breakout_sniper: do not Telegram a chase or blow-off.
_SNIPER_RSI_BLOWOFF = 82.0


def is_sniper_watch(row: Mapping[str, Any] | None) -> bool:
    """True if this scan row is a live sniper candidate for Telegram."""
    if not row:
        return False
    if bool(row.get("chase_risk")):
        return False
    try:
        rsi = float(row.get("rsi") or 0.0)
    except (TypeError, ValueError):
        rsi = 0.0
    if rsi >= _SNIPER_RSI_BLOWOFF:
        return False
    try:
        entry = float(row.get("entry") or 0.0)
    except (TypeError, ValueError):
        entry = 0.0
    if entry <= 0:
        return False
    status = str(row.get("status") or "")
    signals = [str(x).upper() for x in (row.get("signals") or [])]
    categories = [str(x) for x in (row.get("categories") or [])]
    if status in ("Watch for breakout", "Ready to trade"):
        return True
    if "PRE_BREAKOUT" in signals:
        return True
    if "PreBreakout" in categories:
        try:
            dist = float(row.get("pivot_distance_pct") or 99.0)
        except (TypeError, ValueError):
            dist = 99.0
        if 0 < dist <= 2.5:
            return True
    return False


def sniper_symbols(payload: Mapping[str, Any] | None) -> set[str]:
    out: set[str] = set()
    for row in (payload or {}).get("records") or []:
        if not isinstance(row, Mapping) or not is_sniper_watch(row):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if symbol:
            out.add(symbol)
    return out


def live_sniper_symbols(payload: Mapping[str, Any] | None, *, limit: int = 40) -> list[str]:
    """Names the live feed should actually tick for confirmed-breakout Telegram.

    ``is_sniper_watch`` is a wide net (every Watch-for-breakout row). Subscribing
    all of them starved Kite LTP (REST overlay only fetches 80 names) so confirmed
    sniper messages never fired. Prefer radar-quality sniper candidates, score order.
    """
    records = [r for r in (payload or {}).get("records") or [] if isinstance(r, Mapping)]
    preferred: list[Mapping[str, Any]] = []
    try:
        from product.radar_workspace import is_sniper_breakout_candidate
        preferred = [r for r in records if is_sniper_breakout_candidate(r) and is_sniper_watch(r)]
    except Exception:
        preferred = []
    pool = preferred or [r for r in records if is_sniper_watch(r)]

    def _score(row: Mapping[str, Any]) -> float:
        try:
            return float(row.get("score") or row.get("breakout_conviction") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    pool.sort(key=lambda r: (-_score(r), str(r.get("symbol") or "")))
    out: list[str] = []
    seen: set[str] = set()
    cap = max(1, min(int(limit or 40), 80))
    for row in pool:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
        if len(out) >= cap:
            break
    return out


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
                data.setdefault("delivery", {})
                data.setdefault("drain", {})
                return data
        except Exception:
            pass
        return {"sent": {}, "arms": {}, "delivery": {}, "drain": {}}

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
            from alerts.telegram_alerts import _telegram_cred
            token = _telegram_cred("TELEGRAM_BOT_TOKEN")
            chat = _telegram_cred("TELEGRAM_CHAT_ID")
            if token:
                os.environ["TELEGRAM_BOT_TOKEN"] = token
            if chat:
                os.environ["TELEGRAM_CHAT_ID"] = chat
        except Exception:
            try:
                from dotenv import load_dotenv
                repo = Path(__file__).resolve().parents[2]
                load_dotenv(repo / ".env", override=True)
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

    def _engine_error(self, engine: Any) -> str:
        return str(getattr(engine, "last_error", "") or "").strip()

    def _record_delivery(self, kind: str, payload: Mapping[str, Any]) -> None:
        bucket = self.state.setdefault("delivery", {})
        bucket[kind] = {
            "day": self._day(),
            "updated_at": self._now_fn().isoformat(),
            **{str(k): v for k, v in dict(payload).items()},
        }
        self._save()

    def _send_status(self, key: str, message: str) -> str:
        """Deliver ``message`` once per day. Returns sent / already_sent / failed."""
        day = self._day()
        self._prune(day)
        if self._was_sent(key, day):
            return "already_sent"
        try:
            engine = self._engine()
            if not engine.is_configured():
                return "failed"
            if not engine.send(message):
                err = self._engine_error(engine)
                if "429" in err:
                    time.sleep(2.0)
                    if engine.send(message):
                        self._mark_sent([key], day)
                        return "sent"
                    err = self._engine_error(engine) or err
                print(f"[TELEGRAM] send failed for {key}" + (f" · {err}" if err else ""), flush=True)
                return "failed"
        except Exception as exc:
            print(f"[TELEGRAM] send error for {key}: {type(exc).__name__}: {exc}", flush=True)
            return "failed"
        self._mark_sent([key], day)
        return "sent"

    def _send_once(self, key: str, message: str) -> bool:
        return self._send_status(key, message) == "sent"

    def scan_keys_pending(self, payload: Mapping[str, Any] | None) -> bool:
        """True when last-scan setups or near-breakouts have not been marked sent today."""
        day = self._day()
        records = [r for r in (payload or {}).get("records") or [] if isinstance(r, Mapping)]
        if not records:
            return False
        for row in records:
            sym = str(row.get("symbol", "")).upper()
            if not sym:
                continue
            verdict = str(row.get("verdict", "")).upper()
            if (
                verdict in ("BUY", "STRONG BUY")
                and not bool(row.get("chase_risk"))
                and self._f(row.get("entry")) > 0
                and not self._was_sent(f"setup:{sym}", day)
            ):
                return True
            pre = (
                str(row.get("status", "")) == "Watch for breakout"
                or "PRE_BREAKOUT" in [str(x).upper() for x in (row.get("signals") or [])]
            )
            if pre and self._f(row.get("entry")) > 0 and not self._was_sent(f"pre:{sym}", day):
                return True
        return False

    def drain_last_scan(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        min_interval_s: float = 45.0,
    ) -> dict:
        """Retry last-scan setup / breakout watches if Telegram is on and today's keys are empty."""
        if not _DRAIN_LOCK.acquire(blocking=False):
            return {"setup": 0, "prebreakout": 0, "reason": "in_progress"}
        try:
            drain = self.state.setdefault("drain", {})
            now = float(self._epoch())
            last = float(drain.get("last_epoch") or 0.0)
            last_reason = str(drain.get("last_reason") or "")
            if (
                last_reason == "send_failed"
                and min_interval_s > 0
                and (now - last) < float(min_interval_s)
            ):
                return {"setup": 0, "prebreakout": 0, "reason": "retry_wait"}
            if payload is None:
                try:
                    from product.scan_store import load_scan
                    payload = load_scan() or {}
                except Exception:
                    payload = {}
            day = self._day()
            sent_keys = list(self.state.get("sent", {}).get(day, []) or [])
            if any(str(k).startswith(("setup:", "pre:")) for k in sent_keys):
                out = {"setup": 0, "prebreakout": 0, "reason": "already_sent"}
                drain["last_epoch"] = now
                drain["last_reason"] = "already_sent"
                self._save()
                return out
            sent = self.notify_scan(payload, phase="intraday") or {}
            drain["last_epoch"] = now
            drain["last_reason"] = str(sent.get("reason") or "")
            self._save()
            return sent
        finally:
            _DRAIN_LOCK.release()

    def notify_sniper_waiting(self, watching: int, reason: str) -> bool:
        """One-shot honesty notice: connected Telegram is not the same as a live confirm."""
        if int(watching or 0) <= 0:
            return False
        if reason not in {"no_live_ticks", "no_fresh_cross"}:
            return False
        return self._send_once(
            "sniper_waiting",
            "🎯 <b>QuantTerm sniper armed</b>\n\n"
            f"Watching {int(watching)} names from the last scan.\n"
            "Setup and near-breakout watches send after each market scan.\n"
            "<b>SNIPER BREAKOUT CONFIRMED</b> needs fresh Zerodha LTP "
            "(login + autonomy running) during 09:15–15:30 IST, "
            "and price must hold 8s above the trigger.",
        )

    @staticmethod
    def _f(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _esc(value: Any) -> str:
        return html.escape(str(value or ""), quote=False)

    def notify_online(self) -> str:
        return self._send_status(
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
        pre_all = [r for r in records
               if (str(r.get("status", "")) == "Watch for breakout"
                   or "PRE_BREAKOUT" in [str(x).upper() for x in (r.get("signals") or [])])
               and self._f(r.get("entry")) > 0
               and str(r.get("symbol", "")).upper() not in ready_symbols
               and not self._was_sent(f"pre:{str(r.get('symbol', '')).upper()}", day)]
        pre_all.sort(key=lambda r: (-self._f(r.get("score")), str(r.get("symbol", ""))))
        sniper_pre = [r for r in pre_all if is_sniper_watch(r)]
        chase_pre = [r for r in pre_all if not is_sniper_watch(r)]
        pre = (sniper_pre + chase_pre)[:4]

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
                sniper_note = ""
                if not is_sniper_watch(r):
                    sniper_note = "\n   Sniper will not confirm — chase/extended or RSI blow-off"
                lines.append(
                    f"\n{emoji} <b>{self._esc(sym)}</b> ₹{self._f(r.get('price')):,.2f} — {self._esc(verdict)}\n"
                    f"{why}\n"
                    f"   Score {self._f(r.get('score')):.0f} · RSI {self._f(r.get('rsi')):.0f} · "
                    f"Vol {self._f(r.get('volume_ratio')):.1f}×\n"
                    f"   Entry ₹{self._f(r.get('entry')):,.2f} · Stop ₹{self._f(r.get('stop')):,.2f} · "
                    f"Target ₹{self._f(r.get('target')):,.2f}"
                    f"{sniper_note}"
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
                if is_sniper_watch(r):
                    follow = "Sniper watching — confirm alert if LTP holds 8s above trigger"
                else:
                    follow = "Sniper will not confirm — chase/extended. Pullback ka wait"
                lines.append(
                    f"\n👀 <b>{self._esc(sym)}</b> ₹{price:,.2f} · trigger ₹{entry:,.2f} "
                    f"({gap:.1f}% neeche)\n"
                    f"   {reason}\n"
                    f"   {follow} · stop ₹{self._f(r.get('stop')):,.2f}"
                )
                keys.append(f"pre:{sym}")

        watch = sorted(sniper_symbols(payload))
        if watch and lines:
            extra = f" +{len(watch) - 8}" if len(watch) > 8 else ""
            lines.append(
                "\n🎯 <b>Sniper live watch</b> (not chase): "
                + ", ".join(self._esc(s) for s in watch[:8])
                + extra
                + "\nConfirm Telegram only if LTP holds 8s above entry · 09:15–15:30 IST"
            )

        if lines:
            try:
                engine = self._engine()
                if not engine.is_configured():
                    sent["reason"] = "not_configured"
                    sent["last_error"] = "not_configured"
                elif engine.send("\n".join(lines)):
                    self._mark_sent(keys, day)
                    sent["setup"] = len(ready)
                    sent["prebreakout"] = len(pre)
                    sent["reason"] = "sent"
                else:
                    sent["reason"] = "send_failed"
                    sent["last_error"] = self._engine_error(engine) or "send_failed"
                    print(
                        "[TELEGRAM] scan/breakout send failed — check token, chat id, bot start"
                        + (f" · {sent['last_error']}" if sent.get("last_error") else ""),
                        flush=True,
                    )
            except Exception as exc:
                sent["reason"] = "send_failed"
                sent["last_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[TELEGRAM] scan/breakout send error: {type(exc).__name__}: {exc}", flush=True)
        else:
            had_setup = any(
                str(r.get("verdict", "")).upper() in ("BUY", "STRONG BUY")
                and not bool(r.get("chase_risk"))
                and self._f(r.get("entry")) > 0
                for r in records
            )
            had_pre = any(
                (str(r.get("status", "")) == "Watch for breakout"
                 or "PRE_BREAKOUT" in [str(x).upper() for x in (r.get("signals") or [])])
                and self._f(r.get("entry")) > 0
                for r in records
            )
            sent["reason"] = "already_sent" if (had_setup or had_pre) else "no_candidates"

        sent["sniper_watch"] = len(sniper_symbols(payload))
        self._record_delivery("scan", sent)

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
        watching = 0
        fresh_n = 0

        for r in records:
            if not is_sniper_watch(r):
                continue
            watching += 1
            sym = str(r.get("symbol", "")).upper()
            entry = self._f(r.get("entry"))
            if not sym or entry <= 0:
                continue
            try:
                fresh = bool(live_feed.entry_allowed(sym))
                price = self._f(live_feed.price(sym))
            except Exception:
                fresh, price = False, 0.0
            if fresh:
                fresh_n += 1
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
            reason = "holding_for_confirm" if arms else "no_fresh_cross"
            if watching == 0:
                reason = "no_sniper_candidates"
            elif fresh_n == 0:
                reason = "no_live_ticks"
            waiting = False
            if reason == "no_live_ticks" and watching > 0:
                waiting = self.notify_sniper_waiting(watching, reason)
            out = {"confirmed": 0, "watching": watching, "armed": len(arms),
                    "fresh": fresh_n, "reason": reason, "waiting_notice": waiting}
            self._record_delivery("sniper", out)
            return out
        confirmed.sort(key=lambda x: (-self._f(x[0].get("score")), str(x[0].get("symbol", ""))))
        confirmed = confirmed[:5]
        lines = ["🚨 <b>SNIPER BREAKOUT CONFIRMED — live ticks</b>"]
        keys = []
        for r, price in confirmed:
            sym = str(r.get("symbol", "")).upper()
            lines.append(
                f"\n⚡ <b>{self._esc(sym)}</b> crossed ₹{self._f(r.get('entry')):,.2f} "
                f"and held above it\n"
                f"   LTP ₹{price:,.2f} · Score {self._f(r.get('score')):.0f} · "
                f"Scan volume {self._f(r.get('volume_ratio')):.1f}×\n"
                f"   PAPER plan: stop ₹{self._f(r.get('stop')):,.2f} · "
                f"target ₹{self._f(r.get('target')):,.2f}"
            )
            keys.append(f"breakout:{sym}")
        last_error = ""
        try:
            engine = self._engine()
            if engine.is_configured() and engine.send("\n".join(lines)):
                self._mark_sent(keys, day)
                for r, _ in confirmed:
                    arms.pop(str(r.get("symbol", "")).upper(), None)
                self._save()
                out = {"confirmed": len(confirmed), "watching": watching,
                        "armed": 0, "fresh": fresh_n, "reason": "sent"}
                self._record_delivery("sniper", out)
                return out
            last_error = self._engine_error(engine)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[TELEGRAM] sniper send error: {last_error}", flush=True)
        out = {"confirmed": 0, "watching": watching, "armed": len(arms),
                "fresh": fresh_n, "reason": "send_failed", "last_error": last_error}
        self._record_delivery("sniper", out)
        return out

    def notify_long_term(self, payload: Mapping[str, Any] | None) -> dict:
        """Send the current weekly long-term shortlist once per symbol/day."""
        payload = dict(payload or {})
        records = [dict(r) for r in payload.get("records", []) if isinstance(r, Mapping)]
        eligible = [r for r in records if str(r.get("classification", "")) in
                    ("QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE")]
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

    def notify_recommendations(self, workspace: Mapping[str, Any] | None) -> dict[str, Any]:
        """Send the recommendations desk after a market scan. Not a second scan."""
        payload = dict(workspace or {})
        if not payload:
            return {"sent": False, "reason": "no_workspace"}
        now = self._now_fn()
        scan_at = str(payload.get("scan_scanned_at") or payload.get("scan_at") or "")[:19]
        kind = f"reco_desk:{scan_at}" if scan_at else "reco_desk"
        if self._was_sent(kind):
            return {"sent": False, "reason": "already_sent"}
        high, good = self._reco_tiers(payload)
        if not high and not good:
            empty_kind = f"reco_empty:{scan_at}" if scan_at else "reco_empty"
            if self._was_sent(empty_kind):
                return {"sent": False, "reason": "empty"}
            text = (
                f"<b>Recommendations — {now.strftime('%d %b %Y %H:%M')} IST</b>\n"
                "No high-conviction names from this scan.\n"
                "<i>Not a broker order. Empty is empty.</i>"
            )
            try:
                engine = self._engine()
                if engine.is_configured() and engine.send(text):
                    self._mark_sent([empty_kind])
                    return {"sent": True, "kind": "recommendations_empty"}
            except Exception:
                pass
            return {"sent": False, "reason": "send_failed"}
        lines = [
            f"<b>Recommendations — {now.strftime('%d %b %Y %H:%M')} IST</b>",
            "<i>Same market scan as the desk. Not a broker order.</i>",
            "",
            "<b>High conviction</b>",
        ]
        if not high:
            lines.append("None.")
        for card in high[:8]:
            lines.append(self._reco_line(card))
        if good:
            lines.extend(["", "<b>Good setups</b>"])
            for card in good[:8]:
                lines.append(self._reco_line(card))
        try:
            engine = self._engine()
            if engine.is_configured() and engine.send("\n".join(lines)):
                self._mark_sent([kind])
                return {
                    "sent": True,
                    "kind": "recommendations",
                    "high_conviction": len(high),
                    "good_setups": len(good),
                }
        except Exception:
            pass
        return {"sent": False, "reason": "send_failed"}

    def notify_market_report_if_due(self, pulse: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Send today's saved market pulse after 15:30 IST. Does not crawl."""
        now = self._now_fn()
        if now.weekday() >= 5:
            return {"sent": False, "reason": "weekend"}
        if now.hour < 15 or (now.hour == 15 and now.minute < 30):
            return {"sent": False, "reason": "before_close"}
        if self._was_sent("market_report"):
            return {"sent": False, "reason": "already_sent"}
        payload = dict(pulse) if isinstance(pulse, Mapping) and pulse else None
        if payload is None:
            try:
                from product.recommendations_workspace import load_today_pulse
                payload = load_today_pulse()
            except Exception:
                payload = None
        if not payload:
            return {"sent": False, "reason": "no_pulse"}
        try:
            from reports.street_pulse import pulse_to_telegram
            body = str(pulse_to_telegram(dict(payload)) or "").strip()
        except Exception as exc:
            return {"sent": False, "reason": "format_failed", "error": str(exc)[:200]}
        if not body:
            return {"sent": False, "reason": "empty"}
        text = (
            f"📊 <b>Market report — after 15:30 IST</b>\n"
            f"{body}"
        )
        try:
            engine = self._engine()
            if engine.is_configured() and engine.send(text):
                self._mark_sent(["market_report"])
                return {"sent": True, "kind": "market_report"}
        except Exception:
            pass
        return {"sent": False, "reason": "send_failed"}

    def drain_desk_alerts(self) -> dict[str, Any]:
        """Retry saved recommendations and the after-close market report."""
        recos: dict[str, Any] = {"sent": False, "reason": "skipped"}
        report: dict[str, Any] = {"sent": False, "reason": "skipped"}
        try:
            from product.recommendations_store import load_recommendations
            recos = self.notify_recommendations(load_recommendations())
        except Exception as exc:
            recos = {"sent": False, "reason": "error", "error": str(exc)[:200]}
        try:
            report = self.notify_market_report_if_due()
        except Exception as exc:
            report = {"sent": False, "reason": "error", "error": str(exc)[:200]}
        return {"recommendations": recos, "market_report": report}

    @staticmethod
    def _reco_tiers(workspace: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        cards: list[dict[str, Any]] = []
        seen: set[str] = set()
        buckets = list(workspace.get("categories") or [])
        lifecycle = dict(workspace.get("lifecycle") or {})
        buckets.append({"cards": list(lifecycle.get("active") or [])})
        for cat in buckets:
            if not isinstance(cat, Mapping):
                continue
            for raw in cat.get("cards") or []:
                if not isinstance(raw, Mapping):
                    continue
                symbol = str(raw.get("symbol") or "").upper()
                if not symbol or symbol in seen:
                    continue
                seen.add(symbol)
                cards.append(dict(raw))
        high = [c for c in cards if str(c.get("reco_tier") or "") == "high_conviction"]
        good = [c for c in cards if str(c.get("reco_tier") or "") == "good_setup"]
        return high, good

    @classmethod
    def _reco_line(cls, card: Mapping[str, Any]) -> str:
        symbol = cls._esc(card.get("symbol") or "?")
        name = cls._esc(card.get("company") or card.get("name") or "")
        action = cls._esc(card.get("action_badge") or card.get("action") or "")
        try:
            score_txt = f"{int(round(float(card.get('score'))))}/100"
        except (TypeError, ValueError):
            score_txt = "n/a"
        why = cls._esc(
            str(card.get("reason") or card.get("primary_thesis") or card.get("why") or "").strip()
        )
        head = f"• <b>{symbol}</b>"
        if name:
            head += f" {name}"
        head += f" — {score_txt}"
        if action:
            head += f" · {action}"
        if why:
            return f"{head}\n  {why[:220]}"
        return head

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
