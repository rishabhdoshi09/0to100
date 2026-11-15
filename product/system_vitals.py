"""System vitals: catch silent failures within minutes, not weeks.

On 2026-11-11 a single inspection of the owner's Mac found, among others:
BUY outcomes not logged since 2026-07-30, corporate actions not refreshed
since 2026-09-18, the index store frozen at 2026-10-04, the F&O desk blocked
all session by a clock bug, and the paper bot unable to run a single cycle
after a late login. Each failure was invisible because nothing asked "did the
thing that should have happened today actually happen?".

collect() answers that from durable artifacts only (cheap reads, no network):
each vital is OK / WARN / FAIL with a plain reason. run_vitals_watch() checks
every 15 minutes during NSE sessions (09:45-16:30 IST) and again in the
evening after the EOD window (20:15-23:00 IST), persists
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


def _session_boundary(clock: datetime, session: str, hour: int, minute: int) -> float | None:
    """Epoch of ``session`` at hour:minute in the clock's own timezone."""
    try:
        day = datetime.fromisoformat(str(session)[:10]).date()
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=clock.tzinfo).timestamp()
    except Exception:
        return None


def _last_completed_session(clock: datetime) -> str:
    try:
        from research.autonomy.schedules import last_completed_session_date
        return str(last_completed_session_date(clock) or "")
    except Exception:
        return ""


def _official_history_vital(clock: datetime, stamp: float, history: dict[str, Any]) -> dict[str, str]:
    """Is the last completed session's official archive on disk?

    A missing archive is normal between the close and its EOD publication,
    so it only warns once that session's EOD window (to 20:00 IST) is over:
    then the host missed the window (asleep, offline) and nothing will fetch
    it until the next window unless someone notices.
    """
    label = "Official NSE history (bhavcopy)"
    reason = str(history.get("reason_code") or "")
    available = str(history.get("available_session") or "unknown")
    completed = str(history.get("completed_session") or history.get("expected_latest_completed_session") or "")
    if reason == "HISTORY_CURRENT":
        return _vital("official_history", label, OK, f"current through {available}")
    if reason == "HISTORY_PUBLICATION_PENDING":
        due = _session_boundary(clock, completed, 20, 0)
        if due is not None and stamp > due:
            return _vital("official_history", label, WARN,
                          f"{completed} archive missing after its EOD window; latest {available}")
        return _vital("official_history", label, OK, f"{completed} archive publishing; latest {available}")
    return _vital("official_history", label, FAIL,
                  f"{reason or 'UNKNOWN'}: latest {available}, expected {completed or 'unknown'}")


def collect(*, now: datetime | None = None, logs: Path | None = None,
            session_day: bool | None = None,
            history: dict[str, Any] | None = None) -> list[dict[str, str]]:
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

    # 3. Scan freshness. In session: saved within the hour. Off-session: the
    # scan must postdate the last completed session's close once that
    # session's EOD rescan window is over -- a 3-day-old scan after a missed
    # Friday is stale, not "OK" (Invariant 2).
    age = _age_hours(root / "product" / "latest_momentum_scan.json", stamp)
    if age is None:
        vitals.append(_vital("scan_fresh", "Market scan", FAIL if in_session else WARN, "no saved scan"))
    else:
        status = FAIL if in_session and age > 1.0 else OK
        detail = f"latest scan saved {age * 60:.0f} min ago"
        if not in_session:
            last = _last_completed_session(clock)
            closed_at = _session_boundary(clock, last, 15, 30) if last else None
            eod_over = _session_boundary(clock, last, 20, 0) if last else None
            saved_at = stamp - age * 3600.0
            if closed_at is not None and eod_over is not None and stamp > eod_over and saved_at < closed_at:
                status = WARN
                detail += f"; predates the {last} close"
        vitals.append(_vital("scan_fresh", "Market scan", status, detail))

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
        ("signal_calibration", "Scanner signal calibration (walk-forward backtest)",
         root / "signal_backtest.json", 24 * 14),
    ):
        age = _age_hours(path, stamp)
        if age is None:
            vitals.append(_vital(key, label, WARN, "missing"))
        else:
            status = WARN if age > max_h else OK
            vitals.append(_vital(key, label, status, f"updated {age / 24:.1f} day(s) ago (limit {max_h / 24:.0f})"))

    # 6. Official completed-session archive. Read from the canonical runtime;
    # a caller inspecting another logs root must pass ``history`` itself.
    if history is None and logs is None:
        try:
            from data.bhavcopy_runtime import official_history_freshness
            history = official_history_freshness(now=clock, load_cache=False, require_store=False)
        except Exception as exc:
            vitals.append(_vital("official_history", "Official NSE history (bhavcopy)", WARN,
                                 f"unreadable: {type(exc).__name__}"))
            history = None
    if history is not None:
        vitals.append(_official_history_vital(clock, stamp, dict(history)))
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


def watch_due(clock: datetime) -> bool:
    """Session hours, plus an evening pass after the EOD window.

    The evening pass is what reports a missed EOD (official archive, EOD
    rescan, reference upkeep) the same night -- or on the first evening the
    host is awake again after sleeping through one.
    """
    hm = (clock.hour, clock.minute)
    return (9, 45) <= hm <= (16, 30) or (20, 15) <= hm <= (23, 0)


def run_vitals_watch(stop: threading.Event, *, send: Callable[[str], Any] | None = None,
                     poll_s: float = POLL_S) -> None:
    from core.market_clock import now_ist

    alerter = VitalsAlerter(send or _telegram)
    while not stop.is_set():
        try:
            clock = now_ist()
            if watch_due(clock):
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
