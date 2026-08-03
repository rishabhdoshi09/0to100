"""Dedicated market-operations worker for QuantTerm.

User-requested scans and market-data jobs run here, independently of PAPER autonomy.
The worker has isolated lanes so a news refresh or paper cycle cannot delay a market
scan. Every operation writes durable progress to SQLite and visible console output.

At startup the worker also inspects persisted product state and queues only the work
that is missing or stale. This makes the terminal self-preparing without creating a
second data source or hiding provider failures.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback
from typing import Any

from operations.store import BLOCKED, FAILED, SUCCEEDED, OperationStore

# macOS default RLIMIT_NOFILE is often 256. Concurrent lanes + progress SQLite
# opens can exhaust that even after connection.close() if other libs hold FDs.
try:
    import resource as _resource

    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _target = 8192
    if _soft < _target:
        _resource.setrlimit(
            _resource.RLIMIT_NOFILE,
            (min(_target, _hard if _hard > 0 else _target), _hard),
        )
except Exception:
    pass

MARKET_SCAN = "MARKET_SCAN"
LONG_TERM_SCAN = "LONG_TERM_SCAN"
LONG_TERM_REFRESH = "LONG_TERM_REFRESH"
NEWS_REFRESH = "NEWS_REFRESH"
FNO_REFRESH = "FNO_REFRESH"
DATA_PREPARE = "DATA_PREPARE"
FULL_UNIVERSE_BACKTEST = "FULL_UNIVERSE_BACKTEST"
US_DATA_PREPARE = "US_DATA_PREPARE"
US_MARKET_SCAN = "US_MARKET_SCAN"
SNIPER_BOARD_EVAL = "SNIPER_BOARD_EVAL"

LANES = {
    MARKET_SCAN: "market_scan",
    LONG_TERM_SCAN: "long_term",
    LONG_TERM_REFRESH: "long_term",
    NEWS_REFRESH: "news",
    FNO_REFRESH: "data",
    DATA_PREPARE: "data",
    FULL_UNIVERSE_BACKTEST: "research",
    # Separate US lane so NSE scans are never blocked by Yahoo US prepare.
    US_DATA_PREPARE: "us_market",
    US_MARKET_SCAN: "us_market",
    # Focused rank of confirmed sniper hits — market_scan lane (uses scan + LT join).
    SNIPER_BOARD_EVAL: "market_scan",
}

ROOT = Path(__file__).resolve().parents[1]
OPS_ROOT = ROOT / "logs" / "market_ops"
RUNTIME_PATH = OPS_ROOT / "runtime.json"
LOCK_PATH = OPS_ROOT / "worker.lock"

NEWS_FRESH_S = 20 * 60
FNO_FRESH_S = 24 * 60 * 60
SCAN_FRESH_S = 6 * 60 * 60
LONG_TERM_FRESH_S = 3 * 24 * 60 * 60
HISTORY_DAYS = 500


class OperationBlocked(RuntimeError):
    def __init__(self, message: str, *, code: str = "BLOCKED", result: dict | None = None):
        super().__init__(message)
        self.code = code
        self.result = result or {}


class SingleWorkerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None

    def holder_pid(self) -> int | None:
        try:
            text = self.path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            return int(text) if text else None
        except Exception:
            return None

    def pid_alive(self, pid: int | None) -> bool:
        if not pid or int(pid) <= 0:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False

    def reclaim_if_dead(self) -> bool:
        """Clear a lock file whose owner PID is gone. Flock itself dies with the process,
        but a leftover lock file can confuse diagnostics; unlink when safe."""
        pid = self.holder_pid()
        if self.pid_alive(pid):
            return False
        try:
            self.path.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def terminate_holder(self, *, reason: str = "") -> bool:
        """Best-effort SIGTERM/SIGKILL against the lock-file PID.

        Used when a previous market-ops worker is alive but no longer heartbeating
        (common after stop scripts kill only the API port and leave an orphan).
        """
        pid = self.holder_pid()
        if not self.pid_alive(pid):
            return self.reclaim_if_dead()
        _emit("WARN", f"terminating stale market-ops lock holder pid={pid} {reason}".strip())
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            return self.reclaim_if_dead()
        deadline = time.time() + 2.0
        while time.time() < deadline and self.pid_alive(pid):
            time.sleep(0.1)
        if self.pid_alive(pid):
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.1)
        return self.reclaim_if_dead() or not self.pid_alive(pid)

    def acquire(self) -> bool:
        try:
            import fcntl
            self.reclaim_if_dead()
            self._handle = self.path.open("w")
            try:
                fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._handle.close()
                self._handle = None
                return False
            self._handle.seek(0)
            self._handle.truncate()
            self._handle.write(str(os.getpid()))
            self._handle.flush()
            return True
        except Exception:
            return False

    def release(self) -> None:
        try:
            if self._handle is not None:
                import fcntl
                fcntl.flock(self._handle, fcntl.LOCK_UN)
                self._handle.close()
            self.path.unlink(missing_ok=True)
        except Exception:
            pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _emit(kind: str, message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] MARKET OPS {kind:<9} {message}", flush=True)


def _operation_result(report: Any) -> dict[str, Any]:
    if hasattr(report, "as_dict"):
        return dict(report.as_dict())
    if isinstance(report, dict):
        return dict(report)
    return {"value": str(report)}


def _notify_market_scan_telegram(payload: dict[str, Any]) -> dict[str, Any]:
    """Push setup / near-breakout alerts after a terminal MARKET_SCAN.

    Uses the same durable notifier as autonomy so dedupe survives restarts.
    Never raises into the scan lane — Telegram failure must not fail the scan.
    """
    try:
        from research.autonomy import default_root
        from research.autonomy import schedules as SCH
        from research.autonomy.telegram_notifications import TelegramNotifier

        holidays = set()
        try:
            from research.intelligence.data import nse_calendar as CAL

            holidays = CAL.load_holidays() or set()
        except Exception:
            holidays = set()
        try:
            from core.market_clock import now_ist

            now = now_ist()
        except Exception:
            from datetime import datetime

            try:
                from zoneinfo import ZoneInfo

                now = datetime.now(ZoneInfo("Asia/Kolkata"))
            except Exception:
                now = datetime.now()
        phase = SCH.session_phase(now, holidays)
        notifier = TelegramNotifier(default_root())
        if not notifier.configured():
            _emit("WARN", "Telegram not configured — scan saved, no phone alert")
            return {"skipped": "not_configured"}
        sent = notifier.notify_scan(payload, phase=phase) or {}
        _emit(
            "INFO",
            "Telegram scan notify · "
            f"setup={sent.get('setup', 0)} prebreakout={sent.get('prebreakout', 0)} "
            f"briefing={sent.get('briefing', 0)} eod={sent.get('eod', 0)} · phase={phase}",
        )
        return dict(sent)
    except Exception as exc:
        _emit("WARN", f"Telegram scan notify failed: {type(exc).__name__}: {exc}")
        return {"error": str(exc)}


def _stale(path: Path, max_age_s: float, *, now: float | None = None) -> bool:
    if not path.exists():
        return True
    try:
        age = float(now if now is not None else time.time()) - path.stat().st_mtime
        return age < 0 or age > float(max_age_s)
    except Exception:
        return True


def _write_instrument_cache(rows: list[dict[str, Any]]) -> Path:
    path = ROOT / "logs" / "instruments_cache.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    os.replace(tmp, path)
    return path


def _persist_fno_report(report) -> Path:
    path = ROOT / "logs" / "product" / "fno_universe.json"
    payload = {
        "generated_at": time.time(),
        "source": report.source,
        "total_instrument_rows": report.total_instrument_rows,
        "total_future_contracts": report.total_future_contracts,
        "index_future_contracts": report.index_future_contracts,
        "unique_stock_underlyings": report.unique_stock_underlyings,
        "mapped_underlyings": report.mapped_underlyings,
        "underlyings": [item.__dict__ for item in report.underlyings],
        "exclusions": [item.__dict__ for item in report.exclusions],
    }
    _atomic_json(path, payload)
    return path


class MarketOperationsWorker:
    def __init__(self, store: OperationStore | None = None) -> None:
        self.store = store or OperationStore(OPS_ROOT / "jobs.db")
        self.stop_event = threading.Event()
        self.lock = SingleWorkerLock(LOCK_PATH)
        self._active_lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}
        self._threads: list[threading.Thread] = []
        self._history_lock = threading.Lock()

    def _set_active(self, lane: str, operation: dict[str, Any] | None) -> None:
        with self._active_lock:
            if operation:
                self._active[lane] = {
                    "operation_id": operation.get("operation_id"),
                    "kind": operation.get("kind"),
                    "started_at": operation.get("started_at") or time.time(),
                    "attempt": operation.get("attempt"),
                }
            else:
                self._active.pop(lane, None)

    def _runtime_payload(self, *, running: bool) -> dict[str, Any]:
        with self._active_lock:
            active = {lane: dict(value) for lane, value in self._active.items()}
        return {
            "process_running": bool(running),
            "worker_pid": os.getpid(),
            "heartbeat_epoch": time.time(),
            "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "lanes": sorted(set(LANES.values())),
            "active": active,
        }

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(2.0):
            _atomic_json(RUNTIME_PATH, self._runtime_payload(running=True))

    def _progress(self, operation_id: str, stage: str, message: str,
                  current: int | None = None, total: int | None = None) -> None:
        self.store.progress(
            operation_id,
            stage=stage,
            message=message,
            current=current,
            total=total,
        )
        if total and current is not None:
            pct = (current / total) * 100 if total else 0
            _emit("PROGRESS", f"{operation_id[:8]} · {stage} · {current}/{total} ({pct:.0f}%) · {message}")
        else:
            _emit("PROGRESS", f"{operation_id[:8]} · {stage} · {message}")

    def _ensure_history(self, operation_id: str, *, days: int = HISTORY_DAYS) -> dict[str, Any]:
        # Parallel lanes share one history prepare. Announce waits so the UI
        # never looks frozen on ACCEPTED while blocked on this lock.
        waited_s = 0.0
        while not self._history_lock.acquire(timeout=0.5):
            waited_s += 0.5
            self._progress(
                operation_id,
                "WAITING_HISTORY",
                f"Waiting for shared NSE history prepare · {waited_s:.0f}s "
                "(another lane is loading bhavcopy)",
            )
        try:
            from data.bhavcopy_runtime import status as history_status
            current = history_status(load_cache=True)
            if current.get("ready") and int(current.get("sessions", 0) or 0) >= 60:
                self._progress(
                    operation_id,
                    "HISTORY_READY",
                    f"Official history ready · {current.get('sessions', 0)} sessions · {current.get('symbols', 0)} symbols",
                )
                return current
            self._progress(operation_id, "PREPARING_HISTORY", f"Preparing {days}-session official NSE history")
            from data.bhavcopy_store import build_store

            def progress(current_count: int, total: int) -> None:
                self._progress(
                    operation_id,
                    "PREPARING_HISTORY",
                    f"Downloading/loading official NSE bhavcopy · {int(current_count)}/{int(total)} sessions",
                    current_count,
                    total,
                )

            build_store(days=days, progress=progress)
            current = history_status(load_cache=True)
            if not current.get("ready"):
                raise OperationBlocked(
                    "Official NSE bhavcopy history could not be prepared",
                    code="BHAVCOPY_NOT_READY",
                    result=current,
                )
            return current
        finally:
            self._history_lock.release()

    def _run_market_scan(self, operation: dict[str, Any]) -> dict[str, Any]:
        operation_id = str(operation["operation_id"])
        history = self._ensure_history(operation_id)
        self._progress(operation_id, "LOADING_UNIVERSE", "Loading approved NSE cash universe")
        from scan.market_scan_service import run_whole_market_scan

        def prepared_prefetch(symbols, *, progress=None):
            # History already prepared by _ensure_history. Mark the in-process
            # bhav cache warm so UnifiedScanner does not re-enter build_store /
            # apply_live (that looked like a frozen "Scanning 0/N").
            try:
                from scan import bulk_fetcher as bf

                with bf._lock:
                    bf._bhav_ok = True
            except Exception:
                pass
            total = len(list(symbols) or [])
            if callable(progress) and total:
                try:
                    progress(0, total)
                except Exception:
                    pass
            return total

        def progress_callback(current: int, total: int) -> None:
            self._progress(
                operation_id,
                "SCANNING",
                f"Scanning market · {int(current):,}/{int(total):,} stocks",
                int(current),
                int(total),
            )

        universe_guess = max(1, int(history.get("symbols") or 1))
        self._progress(
            operation_id,
            "SCANNING",
            f"Starting whole-market evaluation · {history.get('sessions', 0)} official sessions · ~{universe_guess:,} symbols",
            0,
            universe_guess,
        )
        report = run_whole_market_scan(
            prefetch_fn=prepared_prefetch,
            progress_callback=progress_callback,
            save=True,
            skip_scanner_prefetch=True,
        )
        result = _operation_result(report)
        if not getattr(report, "ok", False):
            code = str(getattr(report, "error_code", "SCAN_FAILED") or "SCAN_FAILED")
            message = str(getattr(report, "error_message", "Whole-market scan failed") or "Whole-market scan failed")
            if str(getattr(report, "status", "")) == "DATA_UNAVAILABLE":
                raise OperationBlocked(message, code=code, result=result)
            raise RuntimeError(f"{code}: {message}")
        payload = dict(getattr(report, "payload", {}) or {})
        summary = dict(payload.get("summary", {}) or {})
        result["summary"] = summary
        result["records"] = len(payload.get("records", []) or [])
        result["history"] = history
        # Terminal/UI scans used to save silently with no phone alerts — only the
        # autonomy job path called Telegram. Notify here so Scan now reaches Telegram.
        result["telegram"] = _notify_market_scan_telegram(payload)
        return result

    def _run_long_term(self, operation: dict[str, Any], *, refresh: bool) -> dict[str, Any]:
        operation_id = str(operation["operation_id"])
        history = self._ensure_history(operation_id)
        self._progress(
            operation_id,
            "TECHNICAL_SCREEN",
            f"Running long-term screen across {history.get('symbols', 0):,} history symbols",
            0,
            max(1, int(history.get("symbols") or 1)),
        )
        from scan.long_term_service import run_long_term_scan

        def progress(current: int, total: int, message: str) -> None:
            text = str(message or "")
            lower = text.lower()
            if "fundamental" in lower:
                stage = "FUNDAMENTALS"
            elif "sav" in lower:
                stage = "SAVING"
            elif "histor" in lower or "bhav" in lower:
                stage = "PREPARING_HISTORY"
            else:
                stage = "TECHNICAL_SCREEN"
            self._progress(operation_id, stage, text, current, total)

        report = run_long_term_scan(refresh_fundamentals=refresh, save=True, progress=progress)
        result = _operation_result(report)
        status = str(getattr(report, "status", ""))
        if hasattr(report, "ok") and not report.ok and status != "NO_CANDIDATES":
            code = str(getattr(report, "error_code", "LONG_TERM_FAILED") or "LONG_TERM_FAILED")
            message = str(getattr(report, "error_message", "Long-term scan failed") or "Long-term scan failed")
            if status == "DATA_UNAVAILABLE":
                raise OperationBlocked(message, code=code, result=result)
            raise RuntimeError(f"{code}: {message}")
        payload = dict(getattr(report, "payload", {}) or {})
        result["summary"] = dict(payload.get("summary", {}) or {})
        result["records"] = len(payload.get("records", []) or [])
        result["history"] = history
        return result

    def _run_news(self, operation: dict[str, Any]) -> dict[str, Any]:
        operation_id = str(operation["operation_id"])
        self._progress(operation_id, "FETCHING_SOURCES", "Fetching official and editorial market-news sources")
        from news.curator import NewsCurator
        report = NewsCurator().refresh()
        result = report.as_dict()
        self._progress(
            operation_id,
            "CURATED",
            f"Curated {result.get('curated_articles', 0)} articles from {result.get('sources_ok', 0)} healthy sources",
        )
        if int(result.get("sources_ok", 0) or 0) <= 0:
            raise OperationBlocked(
                "No configured news source returned usable data",
                code="NEWS_SOURCES_UNAVAILABLE",
                result=result,
            )
        return result

    def _run_fno(self, operation: dict[str, Any]) -> dict[str, Any]:
        operation_id = str(operation["operation_id"])
        self._progress(operation_id, "LOADING_INSTRUMENTS", "Refreshing NSE/NFO instrument master")
        from data.fno_universe import build_fno_universe, current_fno_universe
        report = None
        live_error = ""
        try:
            from research.intelligence.data.kite_activation import KiteDataClient
            client = KiteDataClient.from_config()
            rows = [dict(row) for row in (list(client.instruments("NSE")) + list(client.instruments("NFO")))]
            if rows:
                _write_instrument_cache(rows)
                report = build_fno_universe(rows, as_of=None, source="zerodha_kite")
        except Exception as exc:
            live_error = str(exc)
        if report is None:
            report = current_fno_universe()
        result = {
            "source": report.source,
            "live_refresh_error": live_error,
            "total_instrument_rows": report.total_instrument_rows,
            "total_future_contracts": report.total_future_contracts,
            "index_future_contracts": report.index_future_contracts,
            "unique_stock_underlyings": report.unique_stock_underlyings,
            "mapped_underlyings": report.mapped_underlyings,
            "exclusions": len(report.exclusions),
        }
        _persist_fno_report(report)
        if report.mapped_underlyings <= 0:
            raise OperationBlocked(
                "No current stock F&O underlyings could be mapped; Zerodha login or instrument cache is required",
                code="FNO_UNIVERSE_UNAVAILABLE",
                result=result,
            )
        return result

    def _run_data_prepare(self, operation: dict[str, Any]) -> dict[str, Any]:
        operation_id = str(operation["operation_id"])
        history = self._ensure_history(operation_id)
        fno = self._run_fno(operation)
        return {"history": history, "fno": fno}

    def _run_us_data_prepare(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Warm US Yahoo EOD cache for the retail default scope (S&P 500)."""
        import os

        operation_id = str(operation["operation_id"])
        scope = str(
            (operation.get("payload") or {}).get("scope")
            or os.getenv("QT_US_SCAN_SCOPE", "S&P 500")
        ).strip() or "S&P 500"
        self._progress(
            operation_id,
            "US_UNIVERSE",
            f"Loading US universe scope · {scope}",
        )
        from scan.us_market_scan_service import _scope_universe
        from data import us_history_store as hist

        names, scope_label = _scope_universe(scope)
        symbols = sorted(names.keys())
        self._progress(
            operation_id,
            "US_HISTORY",
            f"Downloading Yahoo daily bars for {len(symbols):,} US symbols · {scope_label}",
            0,
            max(1, len(symbols)),
        )

        def progress(current: int, total: int) -> None:
            self._progress(
                operation_id,
                "US_HISTORY",
                f"US history cache · {int(current):,}/{int(total):,} symbols",
                int(current),
                int(total),
            )

        result = hist.prepare_history(symbols, progress=progress, scope=scope_label)
        result["market"] = "US"
        result["places_orders"] = False
        if not result.get("ready"):
            raise OperationBlocked(
                "US Yahoo history cache is still incomplete",
                code="US_HISTORY_NOT_READY",
                result=result,
            )
        return result

    def _run_us_market_scan(self, operation: dict[str, Any]) -> dict[str, Any]:
        import os

        operation_id = str(operation["operation_id"])
        scope = str(
            (operation.get("payload") or {}).get("scope")
            or os.getenv("QT_US_SCAN_SCOPE", "S&P 500")
        ).strip() or "S&P 500"
        self._progress(
            operation_id,
            "US_SCANNING",
            f"Scanning US market · scope {scope}",
            0,
            1,
        )
        from scan.us_market_scan_service import run_us_market_scan

        def progress(current: int, total: int) -> None:
            self._progress(
                operation_id,
                "US_SCANNING",
                f"Scanning US · {int(current):,}/{int(total):,} stocks",
                int(current),
                int(total),
            )

        report = run_us_market_scan(scope=scope, progress_callback=progress, save=True)
        result = _operation_result(report)
        if not report.ok:
            code = str(report.error_code or "US_SCAN_FAILED")
            message = str(report.error_message or "US market scan failed")
            if report.status == "DATA_UNAVAILABLE":
                raise OperationBlocked(message, code=code, result=result)
            raise RuntimeError(f"{code}: {message}")
        payload = dict(report.payload or {})
        result["summary"] = dict(payload.get("summary") or {})
        result["records"] = len(payload.get("records") or [])
        result["scope"] = report.scope
        result["market"] = "US"
        result["places_orders"] = False
        return result

    def _run_sniper_board_eval(self, operation: dict[str, Any]) -> dict[str, Any]:
        operation_id = str(operation["operation_id"])
        self._progress(
            operation_id,
            "LOADING_BOARD",
            "Loading confirmed sniper breakouts for focused evaluation",
        )
        from product.sniper_board import board_symbols, evaluate_board, load_board

        board = load_board()
        symbols = board_symbols(board)
        if not symbols:
            self._progress(
                operation_id,
                "EMPTY_BOARD",
                "No confirmed sniper hits yet — waiting for live breakout confirms",
            )
            evaluation = evaluate_board(save=True)
            return {
                "records": 0,
                "summary": dict(evaluation.get("summary") or {}),
                "evaluated_at": evaluation.get("evaluated_at"),
                "places_orders": False,
                "live_locked": True,
                "message": "Sniper board is empty — no symbols to rank",
            }

        self._progress(
            operation_id,
            "RANKING",
            f"Ranking {len(symbols)} confirmed breakout symbols · momentum · fundamentals · measured edge",
            0,
            len(symbols),
        )

        def progress(current: int, total: int, message: str) -> None:
            text = str(message or "")
            lower = text.lower()
            if "fundamental" in lower or "long-term" in lower or "long term" in lower:
                stage = "FUNDAMENTALS"
            elif "market scan" in lower or "loading" in lower:
                stage = "LOADING_CONTEXT"
            else:
                stage = "RANKING"
            self._progress(operation_id, stage, text, current, total)

        evaluation = evaluate_board(progress=progress, save=True)
        summary = dict(evaluation.get("summary") or {})
        records = list(evaluation.get("records") or [])
        self._progress(
            operation_id,
            "SAVING",
            f"Saved sniper-board ranking · {len(records)} symbols · "
            f"priority={summary.get('priority', 0)} · candidate={summary.get('candidate', 0)}",
            len(records),
            max(1, len(records)),
        )
        return {
            "records": len(records),
            "summary": summary,
            "evaluated_at": evaluation.get("evaluated_at"),
            "symbols": list(evaluation.get("symbols") or []),
            "places_orders": False,
            "live_locked": True,
            "honesty": evaluation.get("honesty"),
        }

    def _run_full_universe_backtest(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Walk-forward signal backtest on 100% of bhav EQ symbols. Never places orders."""
        operation_id = str(operation["operation_id"])
        history = self._ensure_history(operation_id)
        self._progress(
            operation_id,
            "FULL_UNIVERSE_BACKTEST",
            f"Backtesting scanner signals across {history.get('symbols', 0):,} official EQ symbols",
        )
        from product.full_universe_backtest import run_full_universe_backtest
        from scan.signal_backtest import get_state

        stop_poll = threading.Event()

        def _poll_progress() -> None:
            while not stop_poll.wait(2.0):
                try:
                    st = get_state()
                    if st.get("running") and int(st.get("total") or 0) > 0:
                        self._progress(
                            operation_id,
                            "FULL_UNIVERSE_BACKTEST",
                            f"Measuring signal edge · {st.get('progress', 0)}/{st.get('total', 0)} symbols",
                            int(st.get("progress") or 0),
                            int(st.get("total") or 0),
                        )
                except Exception:
                    pass

        poller = threading.Thread(target=_poll_progress, name="bt-progress", daemon=True)
        poller.start()

        def progress(current: int, total: int, message: str) -> None:
            self._progress(operation_id, "FULL_UNIVERSE_BACKTEST", message, current, total)

        try:
            report = run_full_universe_backtest(scope="full", progress=progress)
        finally:
            stop_poll.set()

        if not report.get("ok", True) and report.get("error_code"):
            raise OperationBlocked(
                str(report.get("message") or "Full-universe backtest unavailable"),
                code=str(report.get("error_code") or "BACKTEST_BLOCKED"),
                result=report,
            )
        result = {
            "generated_at": report.get("generated_at"),
            "symbols": report.get("symbols"),
            "horizon_days": report.get("horizon_days"),
            "signal_count": len(report.get("signals") or {}),
            "universe": report.get("universe") or {},
            "elapsed_s": report.get("elapsed_s"),
            "places_orders": False,
            "live_locked": True,
            "history": history,
            "recommended_target_pct": report.get("recommended_target_pct"),
        }
        uni = result["universe"]
        self._progress(
            operation_id,
            "BACKTEST_DONE",
            f"Measured {result['signal_count']} signal types on {uni.get('run', result['symbols'])} symbols",
        )
        return result

    def _execute(self, operation: dict[str, Any]) -> dict[str, Any]:
        kind = str(operation.get("kind", ""))
        if kind == MARKET_SCAN:
            return self._run_market_scan(operation)
        if kind == LONG_TERM_SCAN:
            return self._run_long_term(operation, refresh=False)
        if kind == LONG_TERM_REFRESH:
            return self._run_long_term(operation, refresh=True)
        if kind == NEWS_REFRESH:
            return self._run_news(operation)
        if kind == FNO_REFRESH:
            return self._run_fno(operation)
        if kind == DATA_PREPARE:
            return self._run_data_prepare(operation)
        if kind == US_DATA_PREPARE:
            return self._run_us_data_prepare(operation)
        if kind == US_MARKET_SCAN:
            return self._run_us_market_scan(operation)
        if kind == FULL_UNIVERSE_BACKTEST:
            return self._run_full_universe_backtest(operation)
        if kind == SNIPER_BOARD_EVAL:
            return self._run_sniper_board_eval(operation)
        raise RuntimeError(f"No market-operations handler for {kind}")

    def _lane_loop(self, lane: str) -> None:
        low_power = str(os.getenv("QT_LOW_POWER", "") or "").strip() in {"1", "true", "TRUE", "yes"}
        idle_wait_s = 1.5 if low_power else 0.5
        idle_cap_s = 8.0 if low_power else 3.0
        while not self.stop_event.is_set():
            try:
                operation = self.store.lease_next(lane, worker_pid=os.getpid())
            except Exception as exc:
                _emit("WARN", f"lane={lane} lease failed: {type(exc).__name__}: {exc}")
                self.stop_event.wait(min(idle_cap_s, idle_wait_s * 2))
                idle_wait_s = min(idle_cap_s, idle_wait_s * 1.5)
                continue
            if operation is None:
                # Back off while idle so six lanes do not spam SQLite opens.
                self.stop_event.wait(idle_wait_s)
                idle_wait_s = min(idle_cap_s, idle_wait_s + (0.5 if low_power else 0.25))
                continue
            idle_wait_s = 1.5 if low_power else 0.5
            self._set_active(lane, operation)
            operation_id = str(operation["operation_id"])
            kind = str(operation["kind"])
            started = time.monotonic()
            _emit("START", f"{kind} · id={operation_id} · lane={lane} · attempt={operation.get('attempt')}")
            try:
                result = self._execute(operation)
                elapsed = time.monotonic() - started
                message = f"{kind} completed in {elapsed:.1f}s"
                self.store.finish(operation_id, status=SUCCEEDED, message=message, result=result)
                _emit("DONE", f"{kind} · id={operation_id} · {elapsed:.1f}s · {result}")
            except OperationBlocked as exc:
                elapsed = time.monotonic() - started
                try:
                    self.store.finish(
                        operation_id,
                        status=BLOCKED,
                        message=str(exc),
                        result=exc.result,
                        error_code=exc.code,
                        error_message=str(exc),
                    )
                except Exception as finish_exc:
                    _emit("WARN", f"finish BLOCKED failed for {operation_id}: {finish_exc}")
                _emit("BLOCKED", f"{kind} · id={operation_id} · {elapsed:.1f}s · {exc.code}: {exc}")
            except Exception as exc:
                elapsed = time.monotonic() - started
                try:
                    self.store.finish(
                        operation_id,
                        status=FAILED,
                        message=f"{kind} failed after {elapsed:.1f}s",
                        error_code=type(exc).__name__,
                        error_message=str(exc),
                    )
                except Exception as finish_exc:
                    _emit("WARN", f"finish FAILED failed for {operation_id}: {finish_exc}")
                _emit("FAILED", f"{kind} · id={operation_id} · {elapsed:.1f}s · {type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                self._set_active(lane, None)
                try:
                    _atomic_json(RUNTIME_PATH, self._runtime_payload(running=True))
                except Exception as runtime_exc:
                    _emit("WARN", f"runtime write failed: {runtime_exc}")

    def _bootstrap(self) -> list[str]:
        """Queue missing product inputs without blocking worker bring-up.

        Intentionally avoids ``load_cache=True`` here — loading the full bhav
        pickle can take minutes and used to delay the first heartbeat, leaving
        MARKET_SCAN PENDING while the UI said the worker was offline.
        Heavy history load happens later inside each job via ``_ensure_history``.
        """
        queued: list[str] = []
        try:
            from data.bhavcopy_runtime import status as history_status

            history = history_status(load_cache=False)
        except Exception:
            history = {"ready": False, "sessions": 0, "cache_exists": False, "csv_files": 0}
        history_present = bool(
            history.get("ready")
            or history.get("cache_exists")
            or int(history.get("csv_files") or 0) >= 60
            or int(history.get("sessions") or 0) >= 60
        )
        if not history_present:
            _item, created = self.store.enqueue(DATA_PREPARE, lane=LANES[DATA_PREPARE], requested_by="bootstrap")
            if created:
                queued.append(DATA_PREPARE)
        elif _stale(ROOT / "logs" / "product" / "fno_universe.json", FNO_FRESH_S):
            _item, created = self.store.enqueue(FNO_REFRESH, lane=LANES[FNO_REFRESH], requested_by="bootstrap")
            if created:
                queued.append(FNO_REFRESH)
        if _stale(ROOT / "logs" / "news_curator.sqlite3", NEWS_FRESH_S):
            _item, created = self.store.enqueue(NEWS_REFRESH, lane=LANES[NEWS_REFRESH], requested_by="bootstrap")
            if created:
                queued.append(NEWS_REFRESH)
        # Canonical scan artifact is latest_momentum_scan.json (product/scan_store).
        low_power = str(os.getenv("QT_LOW_POWER", "") or "").strip() in {"1", "true", "TRUE", "yes"}
        # Low-power / old Macs: do not auto-queue heavy scans at boot — user runs Scan now.
        if (
            not low_power
            and str(os.getenv("QT_DISABLE_AUTO_MARKET_SCAN", "") or "").strip() not in {"1", "true", "TRUE", "yes"}
            and _stale(ROOT / "logs" / "product" / "latest_momentum_scan.json", SCAN_FRESH_S)
        ):
            _item, created = self.store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="bootstrap")
            if created:
                queued.append(MARKET_SCAN)
        if (
            not low_power
            and str(os.getenv("QT_DISABLE_AUTO_LONG_TERM", "") or "").strip() not in {"1", "true", "TRUE", "yes"}
            and _stale(ROOT / "logs" / "product" / "latest_long_term_scan.json", LONG_TERM_FRESH_S)
        ):
            _item, created = self.store.enqueue(LONG_TERM_SCAN, lane=LANES[LONG_TERM_SCAN], requested_by="bootstrap")
            if created:
                queued.append(LONG_TERM_SCAN)
        # US retail plane — separate lane; default liquid S&P 500 scope.
        disable_us = low_power or str(os.getenv("QT_DISABLE_US_BOOTSTRAP", "") or "").strip() in {
            "1", "true", "TRUE", "yes",
        }
        if not disable_us:
            try:
                from data import us_history_store as us_hist

                us_ready = bool(us_hist.status().get("ready"))
            except Exception:
                us_ready = False
            if not us_ready:
                _item, created = self.store.enqueue(
                    US_DATA_PREPARE, lane=LANES[US_DATA_PREPARE], requested_by="bootstrap",
                )
                if created:
                    queued.append(US_DATA_PREPARE)
            elif _stale(ROOT / "logs" / "product" / "latest_us_scan.json", SCAN_FRESH_S):
                _item, created = self.store.enqueue(
                    US_MARKET_SCAN, lane=LANES[US_MARKET_SCAN], requested_by="bootstrap",
                )
                if created:
                    queued.append(US_MARKET_SCAN)
        return queued

    def run(self) -> int:
        if not self.lock.acquire():
            # Another live process owns the lock. If heartbeat is dead, reclaim once.
            self.lock.reclaim_if_dead()
            if not self.lock.acquire():
                holder = self.lock.holder_pid()
                _emit(
                    "INFO",
                    f"another market-operations worker already owns the lock"
                    + (f" (pid={holder})" if holder else ""),
                )
                return 1
        recovered = self.store.recover_orphans()
        # Come ONLINE before bootstrap so PENDING market scans can lease immediately.
        # A slow bootstrap used to leave scans queued for many minutes with no heartbeat.
        _atomic_json(RUNTIME_PATH, self._runtime_payload(running=True))
        heartbeat = threading.Thread(target=self._heartbeat_loop, name="market-ops-heartbeat", daemon=True)
        heartbeat.start()
        self._threads = [heartbeat]
        for lane in sorted(set(LANES.values())):
            thread = threading.Thread(
                target=self._lane_loop,
                args=(lane,),
                name=f"market-ops-{lane}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        bootstrap = self._bootstrap()
        _emit(
            "ONLINE",
            f"pid={os.getpid()} · lanes={','.join(sorted(set(LANES.values())))} · "
            f"recovered={recovered} · bootstrap={','.join(bootstrap) or 'nothing_due'}",
        )
        _atomic_json(RUNTIME_PATH, self._runtime_payload(running=True))
        try:
            while not self.stop_event.wait(1.0):
                pass
        finally:
            self.stop_event.set()
            for thread in self._threads:
                thread.join(timeout=3.0)
            _atomic_json(RUNTIME_PATH, self._runtime_payload(running=False))
            self.lock.release()
            _emit("OFFLINE", "market-operations worker stopped")
        return 0

    def stop(self, *_args) -> None:
        self.stop_event.set()


def run_worker() -> int:
    worker = MarketOperationsWorker()
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(run_worker())
