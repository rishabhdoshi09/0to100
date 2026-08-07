from __future__ import annotations

import json

from execution.reconciliation.observation_cli import main


class EmptyHealthyClient:
    def orders(self):
        return []

    def trades(self):
        return []

    def positions(self):
        return {"net": []}

    def margins(self):
        return {"equity": {"available": {"cash": 100_000}, "net": 100_000}}

    def get_gtts(self):
        return []

    def place_order(self, **kwargs):
        raise AssertionError("observation command attempted broker mutation")


class IncompleteClient(EmptyHealthyClient):
    def positions(self):
        raise TimeoutError("positions unavailable")


class UnknownOrderClient(EmptyHealthyClient):
    def orders(self):
        return [
            {
                "order_id": "unknown-broker-order",
                "status": "OPEN",
                "tradingsymbol": "AAA",
                "transaction_type": "BUY",
                "quantity": 5,
                "filled_quantity": 0,
            }
        ]


def _args(tmp_path):
    return [
        "--oms-db",
        str(tmp_path / "oms.db"),
        "--protection-db",
        str(tmp_path / "protection.db"),
        "--snapshot-db",
        str(tmp_path / "snapshots.db"),
        "--report-db",
        str(tmp_path / "reports.db"),
        "--observed-at",
        "2026-08-01T04:30:00+00:00",
        "--internal-cash",
        "100000",
        "--internal-margin",
        "100000",
        "--require-entry-ready",
    ]


def test_empty_complete_account_is_entry_ready_and_read_only(tmp_path, capsys):
    code = main(_args(tmp_path), client=EmptyHealthyClient())
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["snapshot_complete"] is True
    assert payload["entries_allowed"] is True
    assert payload["blockers"] == []
    assert payload["reconciliation"]["status"] == "HEALTHY"
    assert payload["broker_mutations_enabled"] is False


def test_incomplete_snapshot_returns_nonzero_and_persists_report(tmp_path, capsys):
    code = main(_args(tmp_path), client=IncompleteClient())
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["entries_allowed"] is False
    assert "BROKER_ACCOUNT_SNAPSHOT_INCOMPLETE" in payload["blockers"]
    assert payload["reconciliation"]["status"] == "INCOMPLETE"
    assert (tmp_path / "snapshots.db").exists() is True
    assert (tmp_path / "reports.db").exists() is True


def test_unknown_broker_order_freezes_without_broker_mutation(tmp_path, capsys):
    code = main(_args(tmp_path), client=UnknownOrderClient())
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["entries_allowed"] is False
    assert "RECONCILIATION_ENTRY_FREEZE" in payload["blockers"]
    assert payload["reconciliation"]["status"] == "QUARANTINED"


def test_no_repairs_flag_is_accepted(tmp_path, capsys):
    args = _args(tmp_path)
    args.insert(-1, "--no-repairs")

    code = main(args, client=EmptyHealthyClient())
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["reconciliation"]["applied_repairs"] == []
