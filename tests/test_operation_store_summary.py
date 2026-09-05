from __future__ import annotations

import json
import os
import sqlite3
import time

import operations.store as store_module
from operations.store import FAILED, PENDING, RUNNING, OperationStore, SUCCEEDED


def _seed_large_results(store: OperationStore, *, rows: int = 120, blob_size: int = 800_000) -> list[str]:
    now = time.time()
    ids: list[str] = []
    payload_json = json.dumps({"source": "test"})
    result_json = json.dumps({"records": "x" * blob_size, "summary": {"qualified": 3}})
    with sqlite3.connect(str(store.path)) as con:
        for index in range(rows):
            operation_id = f"scan-{index:04d}"
            ids.append(operation_id)
            con.execute(
                """
                INSERT INTO operations (
                    operation_id, kind, lane, status, requested_by,
                    requested_at, started_at, finished_at, updated_at,
                    attempt, worker_pid, stage, message,
                    progress_current, progress_total,
                    payload_json, result_json, error_code, error_message, priority
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    operation_id,
                    "MARKET_SCAN",
                    "market_scan",
                    SUCCEEDED,
                    "test",
                    now + index,
                    now + index,
                    now + index + 1,
                    now + index + 1,
                    1,
                    None,
                    SUCCEEDED,
                    "complete",
                    3371,
                    3371,
                    payload_json,
                    result_json,
                    "",
                    "",
                    0,
                ),
            )
    return ids


def _set_running(
    store: OperationStore,
    operation_id: str,
    *,
    worker_pid: int | None,
    started_at: float,
    updated_at: float,
) -> None:
    with sqlite3.connect(str(store.path)) as con:
        con.execute(
            """
            UPDATE operations
            SET status=?, worker_pid=?, started_at=?, finished_at=NULL, updated_at=?,
                stage='RUNNING', message='test running'
            WHERE operation_id=?
            """,
            (RUNNING, worker_pid, started_at, updated_at, operation_id),
        )


def test_recent_is_metadata_only_and_never_decodes_large_result_json(tmp_path, monkeypatch):
    store = OperationStore(tmp_path / "jobs.db")
    _seed_large_results(store)

    def forbidden_json_loads(*_args, **_kwargs):
        raise AssertionError("latency-sensitive recent() must not decode payload/result JSON")

    monkeypatch.setattr(store_module.json, "loads", forbidden_json_loads)

    rows = store.recent(100)

    assert len(rows) == 100
    assert rows[0]["kind"] == "MARKET_SCAN"
    assert "payload" not in rows[0]
    assert "result" not in rows[0]
    assert rows[0]["progress_pct"] == 100.0


def test_full_operation_detail_remains_available_by_id(tmp_path):
    store = OperationStore(tmp_path / "jobs.db")
    ids = _seed_large_results(store, rows=2, blob_size=25_000)

    item = store.get(ids[-1])

    assert item is not None
    assert item["payload"] == {"source": "test"}
    assert item["result"]["summary"]["qualified"] == 3
    assert len(item["result"]["records"]) == 25_000


def test_recent_full_is_explicit_escape_hatch_for_full_history(tmp_path):
    store = OperationStore(tmp_path / "jobs.db")
    _seed_large_results(store, rows=3, blob_size=10_000)

    rows = store.recent_full(2)

    assert len(rows) == 2
    assert "payload" in rows[0]
    assert "result" in rows[0]
    assert rows[0]["result"]["summary"]["qualified"] == 3


def test_requested_at_desc_index_exists_for_recent_status_reads(tmp_path):
    store = OperationStore(tmp_path / "jobs.db")

    with sqlite3.connect(str(store.path)) as con:
        names = {row[1] for row in con.execute("PRAGMA index_list(operations)")}

    assert "idx_operations_requested_at" in names


def test_successful_market_scan_result_is_compacted_to_canonical_artifact(tmp_path):
    store = OperationStore(tmp_path / "jobs.db")
    operation, created = store.enqueue(
        "MARKET_SCAN",
        lane="market_scan",
        requested_by="test",
        deduplicate=False,
    )
    assert created is True
    operation_id = str(operation["operation_id"])
    huge_payload = {
        "records": [{"symbol": "TEST", "blob": "x" * 800_000}],
        "summary": {"qualified": 1},
    }

    store.finish(
        operation_id,
        status=SUCCEEDED,
        message="scan complete",
        result={
            "payload": huge_payload,
            "scanned_at": "2026-09-05T04:00:00+00:00",
            "requested_universe": 3371,
            "approved_universe": 3371,
        },
    )

    with sqlite3.connect(str(store.path)) as con:
        raw = con.execute(
            "SELECT result_json FROM operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()[0]
    assert len(raw) < 5_000

    item = store.get(operation_id)
    assert item is not None
    assert "payload" not in item["result"]
    assert item["result"]["result_compacted"] is True
    assert item["result"]["artifact"] == {
        "type": "canonical_scan_store",
        "path": "logs/product/latest_momentum_scan.json",
    }
    assert item["result"]["requested_universe"] == 3371


def test_non_market_scan_result_is_not_compacted(tmp_path):
    store = OperationStore(tmp_path / "jobs.db")
    operation, _ = store.enqueue(
        "NEWS_REFRESH",
        lane="news",
        requested_by="test",
        deduplicate=False,
    )
    operation_id = str(operation["operation_id"])
    result = {"payload": {"articles": [1, 2, 3]}, "source": "test"}

    store.finish(operation_id, status=SUCCEEDED, message="done", result=result)

    item = store.get(operation_id)
    assert item is not None
    assert item["result"] == result


def test_recovered_dead_operation_gets_a_fresh_start_clock_when_released(tmp_path):
    store = OperationStore(tmp_path / "jobs.db")
    operation, _ = store.enqueue(
        "MARKET_SCAN",
        lane="market_scan",
        requested_by="test",
        deduplicate=False,
    )
    operation_id = str(operation["operation_id"])
    old = time.time() - 10_000
    _set_running(
        store,
        operation_id,
        worker_pid=999_999_999,
        started_at=old,
        updated_at=old,
    )

    assert store.recover_dead_running() == 1
    recovered = store.get(operation_id)
    assert recovered is not None
    assert recovered["status"] == PENDING
    assert recovered["started_at"] is None
    assert recovered["finished_at"] is None
    assert recovered["worker_pid"] is None

    before = time.time()
    leased = store.lease_next("market_scan", worker_pid=os.getpid())
    after = time.time()
    assert leased is not None
    assert leased["operation_id"] == operation_id
    assert leased["status"] == RUNNING
    assert before <= float(leased["started_at"]) <= after


def test_live_worker_operation_is_never_failed_by_generic_stale_recovery(tmp_path):
    store = OperationStore(tmp_path / "jobs.db")
    operation, _ = store.enqueue(
        "MARKET_SCAN",
        lane="market_scan",
        requested_by="test",
        deduplicate=False,
    )
    operation_id = str(operation["operation_id"])
    old = time.time() - 10_000
    _set_running(
        store,
        operation_id,
        worker_pid=os.getpid(),
        started_at=old,
        updated_at=old,
    )

    changed = store.recover_stale_running(
        now=time.time(),
        deadlines={"MARKET_SCAN": 1.0},
    )

    assert changed == 0
    item = store.get(operation_id)
    assert item is not None
    assert item["status"] == RUNNING


def test_abandoned_unowned_operation_can_still_expire(tmp_path):
    store = OperationStore(tmp_path / "jobs.db")
    operation, _ = store.enqueue(
        "MARKET_SCAN",
        lane="market_scan",
        requested_by="test",
        deduplicate=False,
    )
    operation_id = str(operation["operation_id"])
    old = time.time() - 100
    _set_running(
        store,
        operation_id,
        worker_pid=None,
        started_at=old,
        updated_at=old,
    )

    changed = store.recover_stale_running(
        now=time.time(),
        deadlines={"MARKET_SCAN": 1.0},
    )

    assert changed == 1
    item = store.get(operation_id)
    assert item is not None
    assert item["status"] == FAILED
    assert item["error_code"] == "DEADLINE_EXCEEDED"
