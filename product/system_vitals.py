"""System vitals: catch silent failures within minutes, not weeks.

On 2026-11-11 a single inspection of the owner's Mac found, among others:
BUY outcomes not logged since 2026-07-30, corporate actions not refreshed
since 2026-09-18, the index store frozen at 2026-10-04, the F&O desk blocked
all session by a clock bug, and the paper bot unable to run a single cycle
after a late login. Each failure was invisible because nothing asked "did the
thing that should have happened today actually happen?".

collect() answers that from durable artifacts only (cheap reads, no network):
each vital is OK / WARN / FAIL with a plain reason. run_vitals_watch() checks
every 15 minutes during NSE sessions (09:45-16:30 IST), persists
logs/product/system_vitals.json for the desk, and sends one Telegram alert per
vital per day when it newly fails or warns.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

OK, WARN, FAIL = "OK", "WARN", "FAIL"
POLL_S = 15 * 60


def _logs() -> Path:
    from core.runtime_paths import logs_dir
    return Path(logs_dir())


def _age_hours(path: Path, now: float) -> float | None:
    try:
        return (now - path.stat().st_mtime) / 3600.0
    except OSError:
        return None


def _vital(key: str, label: str, status: str, detail: str) -> dict[str, str]:
    return {"key": key, "label": label, "status": status, "detail": detail}


def _ro(path: Path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)


def collect(*, now: datetime | None = None, logs: Path | None = None,
            session_day: bool | None = None) -> list[dict[str, str]]:
    from core.market_clock import now_ist

    clock = now or now_ist()
    stamp = time.time() if now is None else clock.timestamp()
    root = Path(logs) if logs else _logs()
    today = clock.date().isoformat()
    if session_day is None:
        try:
            from research.intelligence.data.nse_calendar import is_session, load_holidays
            session_day = is_session(clock.date(), load_holidays())
        except Exception:
            session_day = clock.weekday() < 5
    in_session = bool(session_day) and (clock.hour, clock.minute) >= (10, 0) and (clock.hour, clock.minute) <= (15, 30)
    vitals: list[dict[str, str]] = []

    # 1. Every published BUY is outcome-tracked (Invariant 7).
    try:
        con = _ro(root / "signal_outcomes.db")
        n = con.execute("SELECT count(*) FROM signal_log WHERE logged_at >= ?", (today,)).fetchone()[0]
        last = con.execute("SELECT max(logged_at) FROM signal_log").fetchone()[0]
        con.close()
        status = FAIL if in_session and n == 0 else OK
        vitals.append(_vital("outcome_logging", "BUY outcome logging", status,
                             f"{n} BUY signal(s) logged today; last {last or 'never'}"))
    except Exception as exc:
        vitals.append(_vital("outcome_logging", "BUY outcome logging", WARN, f"unreadable: {type(exc).__name__}"))

    # 2. The paper bot actually completes cycles during the session.
    try:
        con = _ro(root / "autonomy" / "jobs.db")
        start = datetime(clock.year, clock.month, clock.day).timestamp()
        rows = con.execute(
            "SELECT status, count(*) FROM jobs WHERE job_type='paper_cycle' AND created_at >= ? GROUP BY status",
            (start,)).fetchall()
        con.close()
        done = dict(rows).get("SUCCEEDED", 0)
        status = FAIL if in_session and clock.hour >= 11 and done == 0 else OK
        vitals.append(_vital("paper_cycles", "Paper cycles", status,
                             f"{done} paper cycle(s) completed today; by status {dict(rows) or {}}"))
    except Exception as exc:
        vitals.append(_vital("paper_cycles", "Paper cycles", WARN, f"unreadable: {type(exc).__name__}"))

    # 3. Scan freshness during the session.
    age = _age_hours(root / "product" / "latest_momentum_scan.json", stamp)
    if age is None:
        vitals.append(_vital("scan_fresh", "Market scan", FAIL if in_session else WARN, "no saved scan"))
    else:
        status = FAIL if in_session and age > 1.0 else OK
        vitals.append(_vital("scan_fresh", "Market scan", status, f"latest scan saved {age * 60:.0f} min ago"))

    # 4. F&O desk refreshes during the session.
    try:
        con = _ro(root / "market_ops" / "jobs.db")
        last_ok = con.execute(
            "SELECT max(finished_at) FROM operations WHERE kind='FNO_REFRESH' AND status='SUCCEEDED'").fetchone()[0]
        con.close()
        fno_age = (stamp - float(last_ok)) / 3600.0 if last_ok else None
        status = FAIL if in_session and (fno_age is None or fno_age > 1.0) else OK
        vitals.append(_vital("fno_refresh", "F&O desk refresh", status,
                             "never" if fno_age is None else f"last success {fno_age * 60:.0f} min ago"))
    except Exception as exc:
        vitals.append(_vital("fno_refresh", "F&O desk refresh", WARN, f"unreadable: {type(exc).__name__}"))

    # 5. Reference data that silently froze before.
    for key, label, path, max_h in (
        ("corporate_actions", "Corporate actions (split/bonus adjustment)", root / "ca_events.json", 72),
        ("index_store", "Official index store (regime/benchmark)", root / "indices", 96),
        ("industry_map", "NSE industry map (sectors)", root / "nse_industry_map.json", 24 * 14),
        ("fundamentals", "Official filings warehouse", root / "product" / "pit_warehouse.db", 24 * 7),
    ):
        age = _age_hours(path, stamp)
        if age is None:
            vitals.append(_vital(key, label, WARN, "missing"))
        else:
            status = WARN if age > max_h else OK
            vitals.append(_vital(key, label, status, f"updated {age / 24:.1f} day(s) ago (limit {max_h / 24:.0f})"))
    return vitals


def worst(vitals: list[dict[str, str]]) -> str:
    order = {OK: 0, WARN: 1, FAIL: 2}
    return max((v["status"] for v in vitals), key=lambda s: order.get(s, 0), default=OK)


def persist(vitals: list[dict[str, str]], *, logs: Path | None = None) -> None:
    try:
        from core.atomic_io import atomic_write_text
        root = Path(logs) if logs else _logs()
        atomic_write_text(root / "product" / "system_vitals.json", json.dumps({
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "status": worst(vitals),
            "vitals": vitals,
        }, indent=1))
    except Exception:
        pass


class VitalsAlerter:
    """One alert per vital per day when it newly turns WARN/FAIL."""

    def __init__(self, send: Callable[[str], Any]):
        self.send = send
        self.sent: set[tuple[str, str]] = set()

    def observe(self, vitals: list[dict[str, str]], day: str) -> list[str]:
        bad = [v for v in vitals if v["status"] in (WARN, FAIL) and (day, v["key"]) not in self.sent]
        if not bad:
            return []
        lines = [f"{'🔴' if v['status'] == FAIL else '🟠'} {v['label']}: {v['detail']}" for v in bad]
        message = "⚠️ QuantTerm vitals\n" + "\n".join(lines)
        try:
            self.send(message)
        except Exception:
            pass
        for v in bad:
            self.sent.add((day, v["key"]))
        return lines


def _telegram(message: str) -> None:
    from alerts.telegram_alerts import AlertEngine
    AlertEngine().send(message)


def run_vitals_watch(stop: threading.Event, *, send: Callable[[str], Any] | None = None,
                     poll_s: float = POLL_S) -> None:
    from core.market_clock import now_ist

    alerter = VitalsAlerter(send or _telegram)
    while not stop.is_set():
        try:
            clock = now_ist()
            if (9, 45) <= (clock.hour, clock.minute) <= (16, 30):
                vitals = collect(now=clock)
                persist(vitals)
                alerter.observe(vitals, clock.date().isoformat())
        except Exception:
            pass
        stop.wait(poll_s)


def start_vitals_watch(stop: threading.Event) -> threading.Thread:
    thread = threading.Thread(target=run_vitals_watch, args=(stop,), name="quantterm-vitals", daemon=True)
    thread.start()
    return thread
