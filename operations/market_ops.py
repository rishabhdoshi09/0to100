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

MARKET_SCAN = "MARKET_SCAN"
LONG_TERM_SCAN = "LONG_TERM_SCAN"
LONG_TERM_REFRESH = "LONG_TERM_REFRESH"
NEWS_REFRESH = "NEWS_REFRESH"
FNO_REFRESH = "FNO_REFRESH"
DATA_PREPARE = "DATA_PREPARE"
FULL_UNIVERSE_BACKTEST = "FULL_UNIVERSE_BACKTEST"

LANES = {
    MARKET_SCAN: "market_scan",
    LONG_TERM_SCAN: "long_term",
    LONG_TERM_REFRESH: "long_term",
    NEWS_REFRESH: "news",
    FNO_REFRESH: "data",
    DATA_PREPARE: "data",
    FULL_UNIVERSE_BACKTEST: "research",
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

    def acquire(self) -> bool:
        try:
            import fcntl
            self._handle = self.path.open("w")
            try:
                fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._handle.close()
                self._handle = None
                return False
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
        with self._history_lock:
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
                    "Downloading/loading missing official NSE bhavcopy sessions",
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

    def _run_market_scan(self, operation: dict[str, Any]) -> dict[str, Any]:
        operation_id = str(operation["operation_id"])
        history = self._ensure_history(operation_id)
        self._progress(operation_id, "LOADING_UNIVERSE", "Loading approved NSE cash universe")
        from scan.market_scan_service import run_whole_market_scan

        def prepared_prefetch(symbols, *, progress=None):
            return len(symbols)

        self._progress(
            operation_id,
            "SCANNING",
            f"Evaluating whole NSE universe with {history.get('sessions', 0)} official sessions",
        )
        report = run_whole_market_scan(prefetch_fn=prepared_prefetch, save=True)
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
        return result

    def _run_long_term(self, operation: dict[str, Any], *, refresh: bool) -> dict[str, Any]:
        operation_id = str(operation["operation_id"])
        history = self._ensure_history(operation_id)
        self._progress(
            operation_id,
            "TECHNICAL_SCREEN",
            f"Running long-term screen across {history.get('symbols', 0):,} history symbols",
        )
        from scan.long_term_service import run_long_term_scan
        report = run_long_term_scan(refresh_fundamentals=refresh, save=True)
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
        if kind == FULL_UNIVERSE_BACKTEST:
            return self._run_full_universe_backtest(operation)
        raise RuntimeError(f"No market-operations handler for {kind}")

    def _lane_loop(self, lane: str) -> None:
        while not self.stop_event.is_set():
            operation = self.store.lease_next(lane, worker_pid=os.getpid())
            if operation is None:
                self.stop_event.wait(0.5)
                continue
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
                self.store.finish(
                    operation_id,
                    status=BLOCKED,
                    message=str(exc),
                    result=exc.result,
                    error_code=exc.code,
                    error_message=str(exc),
                )
                _emit("BLOCKED", f"{kind} · id={operation_id} · {elapsed:.1f}s · {exc.code}: {exc}")
            except Exception as exc:
                elapsed = time.monotonic() - started
                self.store.finish(
                    operation_id,
                    status=FAILED,
                    message=f"{kind} failed after {elapsed:.1f}s",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
                _emit("FAILED", f"{kind} · id={operation_id} · {elapsed:.1f}s · {type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                self._set_active(lane, None)
                _atomic_json(RUNTIME_PATH, self._runtime_payload(running=True))

    def _bootstrap(self) -> list[str]:
        queued: list[str] = []
        try:
            from data.bhavcopy_runtime import status as history_status
            history = history_status(load_cache=True)
        except Exception:
            history = {"ready": False, "sessions": 0}
        history_missing = not history.get("ready") or int(history.get("sessions", 0) or 0) < 60
        if history_missing:
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
        if _stale(ROOT / "logs" / "product" / "latest_scan.json", SCAN_FRESH_S):
            _item, created = self.store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="bootstrap")
            if created:
                queued.append(MARKET_SCAN)
        if _stale(ROOT / "logs" / "product" / "latest_long_term_scan.json", LONG_TERM_FRESH_S):
            _item, created = self.store.enqueue(LONG_TERM_SCAN, lane=LANES[LONG_TERM_SCAN], requested_by="bootstrap")
            if created:
                queued.append(LONG_TERM_SCAN)
        return queued

    def run(self) -> int:
        if not self.lock.acquire():
            _emit("INFO", "another market-operations worker already owns the lock")
            return 1
        recovered = self.store.recover_orphans()
        bootstrap = self._bootstrap()
        _emit(
            "ONLINE",
            f"pid={os.getpid()} · lanes={','.join(sorted(set(LANES.values())))} · "
            f"recovered={recovered} · bootstrap={','.join(bootstrap) or 'nothing_due'}",
        )
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
