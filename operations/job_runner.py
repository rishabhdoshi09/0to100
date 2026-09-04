"""Run one leased market operation in a killable child process.

The parent lane waits on this process and can SIGTERM/SIGKILL the group when
the current attempt deadline expires. The child never finish()es the SQLite
row — the parent marks the operation terminal only after this process exits.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def _write_result(path: str, payload: dict) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    os.replace(tmp, dest)


def _run_test_scan(store, operation_id: str) -> dict:
    from product.scan_progress import finish_progress, write_progress

    total = 4
    for current in range(1, total + 1):
        write_progress(current=current, total=total, stage="SCANNING", source="market_ops")
        store.progress(
            operation_id,
            stage="SCANNING",
            message=f"Scanning {current}/{total} stocks",
            current=current,
            total=total,
        )
        time.sleep(0.05)
    scanned_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    artifact = Path(
        os.environ.get("QT_SCAN_PATH")
        or (Path(__file__).resolve().parents[1] / "logs" / "product" / "latest_momentum_scan.json")
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": 1,
        "scanned_at": scanned_at,
        "as_of_session": os.environ.get("QT_JOB_TEST_SESSION") or scanned_at[:10],
        "records": [{"symbol": "TEST"}],
        "source": "job_runner_test_scan",
    }
    artifact.write_text(json.dumps(body), encoding="utf-8")
    finish_progress(records=1, setups=0)
    return {
        "records": 1,
        "scanned_at": scanned_at,
        "as_of_session": body["as_of_session"],
        "test_scan": True,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    operation_id = str(os.environ.get("QT_JOB_OPERATION_ID") or (argv[0] if argv else "")).strip()
    db_path = str(os.environ.get("QT_JOB_DB") or "").strip()
    result_path = str(os.environ.get("QT_JOB_RESULT") or "").strip()
    if not operation_id or not db_path or not result_path:
        print("job_runner requires QT_JOB_OPERATION_ID, QT_JOB_DB, QT_JOB_RESULT", file=sys.stderr)
        return 3

    sleep_raw = str(os.environ.get("QT_JOB_TEST_SLEEP") or "").strip()
    if sleep_raw:
        time.sleep(float(sleep_raw))
        _write_result(result_path, {"ok": True, "status": "SUCCEEDED", "result": {"slept": float(sleep_raw)}})
        return 0

    from operations.store import OperationStore

    store = OperationStore(db_path)
    operation = store.get(operation_id)
    if operation is None:
        _write_result(result_path, {"ok": False, "status": "FAILED", "error": "operation not found"})
        return 1

    if str(os.environ.get("QT_JOB_TEST_SCAN") or "").strip() in {"1", "true", "yes"}:
        result = _run_test_scan(store, operation_id)
        _write_result(result_path, {"ok": True, "status": "SUCCEEDED", "result": result})
        return 0

    from operations.market_ops import MarketOperationsWorker, OperationBlocked

    worker = MarketOperationsWorker(store)
    try:
        result = worker._execute(operation)
        _write_result(result_path, {"ok": True, "status": "SUCCEEDED", "result": result})
        return 0
    except OperationBlocked as exc:
        _write_result(
            result_path,
            {
                "ok": False,
                "status": "BLOCKED",
                "error": str(exc),
                "error_code": exc.code,
                "result": exc.result,
            },
        )
        return 2
    except Exception as exc:
        traceback.print_exc()
        _write_result(
            result_path,
            {
                "ok": False,
                "status": "FAILED",
                "error": str(exc),
                "error_code": type(exc).__name__,
                "result": getattr(exc, "result", None) if isinstance(getattr(exc, "result", None), dict) else {},
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
