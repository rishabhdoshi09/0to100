from __future__ import annotations

import json

from execution.reconciliation.capture_cli import main


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

    def place_order(self, **kwargs):
        raise AssertionError("capture command attempted broker mutation")


class IncompleteClient(CompleteClient):
    def positions(self):
        raise TimeoutError("positions unavailable")


def test_capture_command_persists_and_reports_read_only_state(tmp_path, capsys):
    path = tmp_path / "snapshots.db"

    code = main(
        [
            "--db",
            str(path),
            "--observed-at",
            "2026-08-01T04:30:00+00:00",
            "--require-complete",
        ],
        client=CompleteClient(),
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["complete"] is True
    assert payload["mutations_enabled"] is False
    assert payload["store"]["snapshots"] == 1
    assert path.exists() is True


def test_require_complete_exits_nonzero_but_preserves_incomplete_evidence(tmp_path, capsys):
    path = tmp_path / "snapshots.db"

    code = main(
        [
            "--db",
            str(path),
            "--observed-at",
            "2026-08-01T04:30:00+00:00",
            "--require-complete",
        ],
        client=IncompleteClient(),
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["complete"] is False
    assert payload["account_complete"] is False
    assert payload["store"]["snapshots"] == 1
    assert payload["store"]["complete_snapshots"] == 0
    assert any(error.startswith("positions:TimeoutError") for error in payload["errors"])


def test_capture_without_require_complete_returns_success_for_observation(tmp_path, capsys):
    code = main(
        [
            "--db",
            str(tmp_path / "snapshots.db"),
            "--observed-at",
            "2026-08-01T04:30:00+00:00",
        ],
        client=IncompleteClient(),
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["complete"] is False
    assert payload["mutations_enabled"] is False
