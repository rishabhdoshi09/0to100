"""Scheduled read-only Zerodha observation worker.

This process captures broker state, reconciles durable OMS state, and verifies protection.
It cannot place, modify or cancel an order or GTT. A single-process lock prevents duplicate
observers when the terminal is started more than once.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import threading
import time
from datetime import datetime, time as clock_time
from typing import Any
from zoneinfo import ZoneInfo

from execution.oms.store import OmsStore
from execution.protection.store import ProtectionStore
from execution.reconciliation.snapshot_store import BrokerSnapshotStore
from execution.reconciliation.store import ReconciliationReportStore
from execution.reconciliation.zerodha_cycle import run_zerodha_observation_cycle

ROOT = Path(__file__).resolve().parents[1]
OBSERVER_ROOT = ROOT / "logs" / "reconciliation"
RUNTIME_PATH = OBSERVER_ROOT / "observer_runtime.json"
STATE_PATH = OBSERVER_ROOT / "observer_schedule.json"
LOCK_PATH = OBSERVER_ROOT / "observer.lock"
OMS_DB = ROOT / "logs" / "oms" / "orders.db"
PROTECTION_DB = ROOT / "logs" / "protection" / "plans.db"
SNAPSHOT_DB = OBSERVER_ROOT / "broker_snapshots.db"
REPORT_DB = OBSERVER_ROOT / "reports.db"
IST = ZoneInfo("Asia/Kolkata")
BROKER_MUTATIONS_ENABLED = False


class SingleObserverLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None

    def acquire(self) -> bool:
        try:
            import fcntl

            self._handle = self.path.open("w", encoding="utf-8")
            fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._handle.write(str(os.getpid()))
            self._handle.flush()
            return True
        except Exception:
            try:
                if self._handle is not None:
                    self._handle.close()
            finally:
                self._handle = None
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
        finally:
            self._handle = None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else dict(default)
    except Exception:
        return dict(default)


def observation_slot(now: datetime, completed_slots: set[str] | None = None) -> tuple[str, str] | None:
    """Return the due durable schedule slot as ``(slot_id, phase)``.

    The observer may run on an exchange holiday; that is harmless because it is read-only and
    never grants trading permission. Exchange-calendar authority remains a separate live gate.
    """
    completed_slots = completed_slots or set()
    local = now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)
    if local.weekday() >= 5:
        return None
    day = local.date().isoformat()
    current = local.time().replace(tzinfo=None)

    candidates: list[tuple[str, str]] = []
    if clock_time(8, 45) <= current < clock_time(9, 15):
        candidates.append((f"{day}:premarket", "PREMARKET"))
    if clock_time(9, 15) <= current <= clock_time(15, 30):
        minutes = (local.hour * 60 + local.minute) - (9 * 60 + 15)
        bucket = max(0, minutes // 15)
        candidates.append((f"{day}:intraday:{bucket:02d}", "INTRADAY"))
    if current >= clock_time(15, 35):
        candidates.append((f"{day}:eod", "EOD"))

    for slot_id, phase in candidates:
        if slot_id not in completed_slots:
            return slot_id, phase
    return None


class ZerodhaObserverWorker:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.lock = SingleObserverLock(LOCK_PATH)
        self.oms = OmsStore(OMS_DB)
        self.protection = ProtectionStore(PROTECTION_DB)
        self.snapshots = BrokerSnapshotStore(SNAPSHOT_DB)
        self.reports = ReconciliationReportStore(REPORT_DB)
        self._last_result: dict[str, Any] = {}
        self._last_error = ""

    def _completed_slots(self) -> set[str]:
        raw = _read_json(STATE_PATH, {"completed_slots": []})
        return {str(item) for item in raw.get("completed_slots", []) if str(item)}

    def _mark_complete(self, slot_id: str) -> None:
        completed = self._completed_slots()
        completed.add(slot_id)
        ordered = sorted(completed)[-160:]
        _atomic_json(STATE_PATH, {"completed_slots": ordered, "updated_at": datetime.now(IST).isoformat()})

    def _runtime(self, *, running: bool, phase: str = "IDLE") -> dict[str, Any]:
        return {
            "process_running": bool(running),
            "running": bool(running),
            "worker_pid": os.getpid(),
            "heartbeat_epoch": time.time(),
            "heartbeat": datetime.now(IST).isoformat(),
            "phase": phase,
            "broker_mutations_enabled": BROKER_MUTATIONS_ENABLED,
            "last_result": dict(self._last_result),
            "last_error": self._last_error,
            "snapshot_db": str(SNAPSHOT_DB),
            "report_db": str(REPORT_DB),
        }

    @staticmethod
    def _connected_client():
        from data.kite_client import KiteClient

        client = KiteClient()
        return client if client.is_connected() else None

    def _run_slot(self, slot_id: str, phase: str) -> bool:
        client = self._connected_client()
        if client is None:
            self._last_error = "KITE_ACCESS_TOKEN_MISSING"
            _atomic_json(RUNTIME_PATH, self._runtime(running=True, phase="CREDENTIALS_MISSING"))
            return False
        _atomic_json(RUNTIME_PATH, self._runtime(running=True, phase=f"RUNNING_{phase}"))
        try:
            result = run_zerodha_observation_cycle(
                client=client,
                oms_store=self.oms,
                protection_store=self.protection,
                snapshot_store=self.snapshots,
                report_store=self.reports,
                observed_at=datetime.now(IST),
                apply_repairs=True,
            )
            self._last_result = {
                "slot_id": slot_id,
                "phase": phase,
                "snapshot_id": result.snapshot_id,
                "snapshot_complete": result.snapshot_complete,
                "entries_allowed": result.entries_allowed,
                "blockers": list(result.blockers),
                "reconciliation_status": result.reconciliation.report.status,
                "reconciliation_report_id": result.reconciliation.report.report_id,
                "protection_entry_freeze": result.protection.entry_freeze_required,
                "finished_at": datetime.now(IST).isoformat(),
            }
            self._last_error = ""
            self._mark_complete(slot_id)
            _atomic_json(RUNTIME_PATH, self._runtime(running=True, phase="IDLE"))
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            _atomic_json(RUNTIME_PATH, self._runtime(running=True, phase="FAILED_SAFE"))
            return False

    def run(self) -> int:
        if not self.lock.acquire():
            return 1
        try:
            _atomic_json(RUNTIME_PATH, self._runtime(running=True, phase="STARTING"))
            startup_id = f"{datetime.now(IST).date().isoformat()}:startup:{os.getpid()}"
            self._run_slot(startup_id, "STARTUP")
            while not self.stop_event.wait(10.0):
                _atomic_json(RUNTIME_PATH, self._runtime(running=True, phase="IDLE"))
                due = observation_slot(datetime.now(IST), self._completed_slots())
                if due is not None:
                    slot_id, phase = due
                    self._run_slot(slot_id, phase)
            return 0
        finally:
            _atomic_json(RUNTIME_PATH, self._runtime(running=False, phase="STOPPED"))
            self.lock.release()

    def stop(self, *_args) -> None:
        self.stop_event.set()


def run_worker() -> int:
    worker = ZerodhaObserverWorker()
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(run_worker())
