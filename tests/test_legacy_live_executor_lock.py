from __future__ import annotations

import sqlite3


def _latest_status(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            "SELECT status FROM trades ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        connection.close()


def test_connected_legacy_live_path_is_locked_by_default(tmp_path, monkeypatch):
    import execution.trade_executor as executor

    db_path = tmp_path / "trades.db"
    monkeypatch.setattr(executor, "_DB", db_path)
    monkeypatch.setattr(executor, "kite_ready", lambda: True)
    monkeypatch.setattr(executor, "legacy_live_enabled", lambda: False)

    result = executor.place_trade(
        "HAL", qty=10, entry_type="LIMIT",
        entry_price=4500, stop=4300, target=4900,
    )

    assert result["ok"] is False
    assert result["mode"] == "LIVE"
    assert "locked" in result["message"].lower()
    assert _latest_status(db_path) == "BLOCKED_LEGACY_LIVE_LOCK"


def test_governance_unavailability_blocks_unsafe_override(tmp_path, monkeypatch):
    import core.governance as governance
    import execution.trade_executor as executor

    db_path = tmp_path / "trades.db"
    monkeypatch.setattr(executor, "_DB", db_path)
    monkeypatch.setattr(executor, "kite_ready", lambda: True)
    monkeypatch.setattr(executor, "legacy_live_enabled", lambda: True)

    def unavailable():
        raise RuntimeError("sentinel unavailable")

    monkeypatch.setattr(governance, "can_place_order", unavailable)

    result = executor.place_trade(
        "HAL", qty=10, entry_type="LIMIT",
        entry_price=4500, stop=4300, target=4900,
    )

    assert result["ok"] is False
    assert result["mode"] == "LIVE"
    assert "governance unavailable" in result["message"].lower()
    assert _latest_status(db_path) == "BLOCKED_GOVERNANCE_UNAVAILABLE"


def test_ambiguous_broker_submission_enters_recovery_required(tmp_path, monkeypatch):
    import core.governance as governance
    import data.kite_client as kite_client
    import execution.trade_executor as executor

    db_path = tmp_path / "trades.db"
    monkeypatch.setattr(executor, "_DB", db_path)
    monkeypatch.setattr(executor, "kite_ready", lambda: True)
    monkeypatch.setattr(executor, "legacy_live_enabled", lambda: True)
    monkeypatch.setattr(governance, "can_place_order", lambda: (True, 1.0, ""))
    monkeypatch.setattr(governance, "record_order_result", lambda **kwargs: None)

    class FakeKite:
        def place_order(self, **kwargs):
            raise TimeoutError("broker response lost")

    monkeypatch.setattr(kite_client, "KiteClient", FakeKite)

    result = executor.place_trade(
        "HAL", qty=10, entry_type="LIMIT",
        entry_price=4500, stop=4300, target=4900,
    )

    assert result["ok"] is False
    assert "uncertain" in result["message"].lower()
    assert _latest_status(db_path) == "RECOVERY_REQUIRED"
