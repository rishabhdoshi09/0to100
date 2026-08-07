from __future__ import annotations

from datetime import datetime, timezone

from execution.reconciliation.snapshot_store import (
    BrokerSnapshotStore,
    capture_and_store_zerodha_snapshot,
)


class CompleteClient:
    def orders(self):
        return []

    def trades(self):
        return []

    def positions(self):
        return {"net": [], "day": []}

    def margins(self):
        return {"equity": {"available": {"cash": 100_000}, "net": 100_000}}

    def get_gtts(self):
        return []


class IncompleteProtectionClient(CompleteClient):
    def get_gtts(self):
        raise TimeoutError("gtt timeout")


def test_complete_snapshot_is_persisted_idempotently(tmp_path):
    store = BrokerSnapshotStore(tmp_path / "snapshots.db")
    observed = datetime(2026, 8, 1, 4, 30, tzinfo=timezone.utc)

    first = capture_and_store_zerodha_snapshot(
        store, CompleteClient(), observed_at=observed
    )
    second = capture_and_store_zerodha_snapshot(
        store, CompleteClient(), observed_at=observed
    )

    assert first.account.snapshot_id == second.account.snapshot_id
    assert first.complete is True
    assert store.summary()["snapshots"] == 1
    assert store.summary()["complete_snapshots"] == 1
    assert store.latest()["account"]["snapshot_id"] == first.account.snapshot_id
    assert store.latest_complete()["account"]["snapshot_id"] == first.account.snapshot_id


def test_incomplete_snapshot_is_preserved_but_not_promoted_to_complete(tmp_path):
    store = BrokerSnapshotStore(tmp_path / "snapshots.db")
    complete = capture_and_store_zerodha_snapshot(
        store,
        CompleteClient(),
        observed_at="2026-08-01T04:30:00+00:00",
    )
    incomplete = capture_and_store_zerodha_snapshot(
        store,
        IncompleteProtectionClient(),
        observed_at="2026-08-01T04:31:00+00:00",
    )

    assert incomplete.account.complete is True
    assert incomplete.protections_complete is False
    assert incomplete.complete is False
    assert store.summary()["snapshots"] == 2
    assert store.summary()["complete_snapshots"] == 1
    assert store.latest()["account"]["snapshot_id"] == incomplete.account.snapshot_id
    assert store.latest_complete()["account"]["snapshot_id"] == complete.account.snapshot_id
    assert store.latest_account_complete()["account"]["snapshot_id"] == incomplete.account.snapshot_id


def test_store_survives_restart_and_preserves_errors(tmp_path):
    path = tmp_path / "snapshots.db"
    store = BrokerSnapshotStore(path)
    bundle = capture_and_store_zerodha_snapshot(
        store,
        IncompleteProtectionClient(),
        observed_at="2026-08-01T04:31:00+00:00",
    )

    restarted = BrokerSnapshotStore(path)
    restored = restarted.get(bundle.account.snapshot_id)

    assert restored is not None
    assert restored["protections_complete"] is False
    assert any(error.startswith("gtts:TimeoutError") for error in restored["errors"])
    assert restarted.summary()["latest_complete_snapshot_id"] == ""


def test_store_does_not_capture_implicitly(tmp_path):
    path = tmp_path / "snapshots.db"
    store = BrokerSnapshotStore(path)

    assert store.summary()["snapshots"] == 0
    assert store.latest() is None
    assert store.latest_complete() is None
