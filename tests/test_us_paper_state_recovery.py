"""Fail-closed US paper autonomy when its persisted control state is not trustworthy.

These tests use only disposable files. No broker, trade journal, or host runtime
is touched; unknown state must never turn into fresh PAPER_AUTO permission.
"""
from __future__ import annotations

import json

import pytest

from execution import us_autopilot as US


@pytest.fixture(autouse=True)
def isolate_us_state(tmp_path, monkeypatch):
    monkeypatch.setattr(US, "_STATE_FILE", tmp_path / "us_autopilot.json")
    monkeypatch.setattr(US, "_state", {})
    monkeypatch.setattr(US, "_state_load_error", "")
    monkeypatch.setattr(US, "_has_us_trade_history", lambda: False)


@pytest.mark.parametrize("bad", [
    "{invalid json",
    "[]",
    "{}",
    '{"armed": "false", "allocation": 100000}',
    '{"armed": true, "allocation": 100000, "realized_pnl": "unknown"}',
])
def test_invalid_saved_state_disarms_without_overwriting(tmp_path, bad):
    path = tmp_path / "us_autopilot.json"
    path.write_text(bad, encoding="utf-8")
    loaded = US._load()
    assert loaded["armed"] is False
    assert loaded["disarmed_reason"] == "STATE_RECOVERY_REQUIRED"
    assert US._state_load_error
    assert US.arm()[0] is False
    US._save()
    assert path.read_text(encoding="utf-8") == bad


def test_missing_state_with_existing_us_trade_history_blocks_entries(monkeypatch):
    monkeypatch.setattr(US, "_has_us_trade_history", lambda: True)
    assert US._load()["armed"] is False
    assert US._state_load_error
    called = []
    monkeypatch.setattr(US, "consider", lambda **kwargs: called.append(kwargs))
    US.on_setups([{"symbol": "AAPL", "verdict": "STRONG BUY", "entry": 200, "stop": 190}])
    assert called == []


def test_unreadable_history_probe_blocks_first_run(monkeypatch):
    def corrupted_history():
        raise OSError("paper history unavailable")
    monkeypatch.setattr(US, "_has_us_trade_history", corrupted_history)
    assert US._load()["armed"] is False
    assert US._state_load_error


def test_new_install_without_any_history_retains_paper_default():
    assert US._load()["armed"] is True
    assert US._state_load_error == ""


def test_valid_disarmed_saved_state_remains_disarmed(tmp_path):
    state = dict(US._DEFAULTS)
    state.update(armed=False, disarmed_reason="owner", realized_pnl=120.0)
    (tmp_path / "us_autopilot.json").write_text(json.dumps(state), encoding="utf-8")
    loaded = US._load()
    assert loaded["armed"] is False
    assert loaded["disarmed_reason"] == "owner"
    assert loaded["realized_pnl"] == 120.0
    assert US._state_load_error == ""


def test_corrupt_state_never_reaccounts_closed_trades(tmp_path, monkeypatch):
    (tmp_path / "us_autopilot.json").write_text("{", encoding="utf-8")
    assert US._load()["armed"] is False
    calls = []
    monkeypatch.setattr(US, "_trades", lambda *args, **kwargs: calls.append("ledger") or [])
    monkeypatch.setattr(US, "_notify", lambda *args, **kwargs: calls.append("notify"))
    US._account_closed()
    assert calls == []
