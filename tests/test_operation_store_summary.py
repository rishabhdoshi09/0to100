from __future__ import annotations

import json
import sqlite3
import time

import operations.store as store_module
from operations.store import OperationStore, SUCCEEDED


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
