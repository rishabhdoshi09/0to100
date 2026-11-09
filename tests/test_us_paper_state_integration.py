"""Combined #320 + #322 US paper state contract: one recovery authority.

The two PRs each made the US paper lane fail closed on lost/corrupt state, with
different mechanisms. The integrated module keeps a single in-memory authority
(``state_recovery_required`` / ``state_recovery_error``) and adds the cases
neither PR covered on its own:

* a damaged exact-ID settlement ledger (would re-book every closed trade);
* a detached installed-host runtime volume (indistinguishable from a fresh
  install by file existence alone);
* a journal file without a ``trades`` table (a fresh install, not history);
* config/preset mutators that reported success while nothing was saved.

Everything runs on disposable files. No broker, network or host runtime.
"""
from __future__ import annotations

import json
import sqlite3

import pytest


def _setup(tmp_path, monkeypatch, *, state=None):
    import execution.trade_executor as te
    import execution.us_autopilot as us

    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(us, "_STATE_FILE", tmp_path / "us_autopilot.json")
    monkeypatch.setattr(us, "_state", {})
    sent: list[str] = []
    monkeypatch.setattr(us, "_notify", sent.append)
    if state is not None:
        us._STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    return us, te, sent


def _journal_with_closed_trade(te):
    conn = te.connect()
    try:
        te.ensure_schema(conn)
        conn.execute(
            """INSERT INTO trades
                (placed_at, mode, symbol, qty, entry_type, entry_price,
                 stop_price, target_price, status, exit_price, note)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("2026-11-06T10:00:00", "PAPER", "DONE", 10, "MARKET", 100.0, 95.0,
             110.0, "PAPER_WIN", 110.0, "US_AUTOPILOT:test|session=2026-11-06"),
        )
        conn.commit()
    finally:
        conn.close()


def _valid_state(us, **overrides):
    state = dict(us._DEFAULTS)
    for key in us._RUNTIME_ONLY_KEYS:
        state.pop(key, None)
    state.update(overrides)
    return state


@pytest.mark.parametrize("damage", [
    {"accounted_close_ids": "1,2,3"},
    {"accounted_close_ids": [1, "2"]},
    {"accounted_close_ids": [True]},
    {"accounted_close_ids": [0]},
    {"accounted_ids": {"1": 1}},
    {"accounting_ids_migrated": "yes"},
    {"accounting_reconciliation_required": 1},
])
def test_damaged_settlement_ledger_fails_closed(tmp_path, monkeypatch, damage):
    import execution.us_autopilot as us
    state = _valid_state(us, accounting_ids_migrated=True, accounted_close_ids=[1])
    state.update(damage)
    us, te, sent = _setup(tmp_path, monkeypatch, state=state)
    before = us._STATE_FILE.read_bytes()

    loaded = us._load()
    assert loaded["state_recovery_required"] is True
    assert loaded["armed"] is False
    assert us._save() is False
    assert us._STATE_FILE.read_bytes() == before
    assert not sent


def test_migrated_state_that_lost_its_close_ids_never_rebooks_pnl(tmp_path, monkeypatch):
    """Without validation this state replays every closed trade into the pool."""
    import execution.us_autopilot as us
    state = _valid_state(us, accounting_ids_migrated=True, realized_pnl=97.5)
    state.pop("accounted_close_ids")
    us, te, sent = _setup(tmp_path, monkeypatch, state=state)
    _journal_with_closed_trade(te)

    assert us._load()["state_recovery_required"] is True
    with pytest.raises(RuntimeError, match="US_PAPER_STATE_RECOVERY_REQUIRED"):
        us._account_closed()
    with pytest.raises(RuntimeError, match="US_PAPER_STATE_RECOVERY_REQUIRED"):
        us.review_cycle()
    on_disk = json.loads(us._STATE_FILE.read_text(encoding="utf-8"))
    assert on_disk["realized_pnl"] == 97.5
    assert not sent


def test_detached_installed_runtime_is_not_a_fresh_install(tmp_path, monkeypatch):
    """No state file AND no journal: on an installed host that is a lost volume."""
    us, te, sent = _setup(tmp_path, monkeypatch)
    vanished = tmp_path / "Volumes" / "QuantTermRuntime"
    monkeypatch.setattr(us, "_STATE_FILE", vanished / "logs" / "us_autopilot.json")
    monkeypatch.setattr(te, "_DB", vanished / "logs" / "trades.db")
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(vanished))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")

    st = us._load()
    assert st["state_recovery_required"] is True
    assert "runtime root is missing" in st["state_recovery_error"]
    assert st["armed"] is False and st["allocation"] == 0.0
    assert us.arm()[0] is False
    assert us._save() is False
    assert not vanished.exists(), "a vanished volume path must never be recreated"


def test_save_refuses_to_recreate_a_volume_that_vanished_after_load(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    (root / "logs").mkdir(parents=True)
    us, te, sent = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(us, "_STATE_FILE", root / "logs" / "us_autopilot.json")
    monkeypatch.setattr(te, "_DB", root / "logs" / "trades.db")
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    assert us._load()["state_recovery_required"] is False
    assert us._save() is True

    for child in (root / "logs").iterdir():
        child.unlink()
    (root / "logs").rmdir()
    root.rmdir()
    assert us._save() is False
    assert not root.exists()


def test_journal_without_trades_table_is_a_fresh_install(tmp_path, monkeypatch):
    us, te, sent = _setup(tmp_path, monkeypatch)
    sqlite3.connect(te._DB).close()        # file exists, schema never created
    st = us._load()
    assert st["state_recovery_required"] is False
    assert st["armed"] is True


def test_unreadable_journal_bytes_block_a_missing_state(tmp_path, monkeypatch):
    us, te, sent = _setup(tmp_path, monkeypatch)
    te._DB.write_bytes(b"this is not a sqlite database at all" * 32)
    st = us._load()
    assert st["state_recovery_required"] is True
    assert st["armed"] is False


def test_runtime_only_recovery_flags_are_never_persisted_or_trusted(tmp_path, monkeypatch):
    import execution.us_autopilot as us
    state = _valid_state(us, realized_pnl=12.5)
    state["state_recovery_required"] = True       # cannot come from our writer
    state["state_recovery_error"] = "stale"
    us, te, sent = _setup(tmp_path, monkeypatch, state=state)
    st = us._load()
    assert st["state_recovery_required"] is False
    assert st["state_recovery_error"] == ""
    assert st["realized_pnl"] == 12.5
    assert us._save() is True
    on_disk = json.loads(us._STATE_FILE.read_text(encoding="utf-8"))
    assert "state_recovery_required" not in on_disk
    assert "state_recovery_error" not in on_disk


def test_status_reports_the_recovery_reason(tmp_path, monkeypatch):
    us, te, sent = _setup(tmp_path, monkeypatch)
    us._STATE_FILE.write_text("{torn", encoding="utf-8")
    status = us.get_status()
    assert status["state_recovery_required"] is True
    assert status["state_recovery_error"].startswith("JSONDecodeError")
    assert status["pool"] == 0.0


def test_status_recovery_error_is_none_when_healthy(tmp_path, monkeypatch):
    us, te, sent = _setup(tmp_path, monkeypatch)
    status = us.get_status()
    assert status["state_recovery_required"] is False
    assert status["state_recovery_error"] is None


def test_config_and_preset_report_failure_when_state_is_unrestored(tmp_path, monkeypatch):
    us, te, sent = _setup(tmp_path, monkeypatch)
    us._STATE_FILE.write_text("[]", encoding="utf-8")
    before = us._STATE_FILE.read_bytes()
    assert us.set_config(min_score=70.0) is False
    ok, msg = us.apply_preset("Conservative")
    assert ok is False and "recovery required" in msg
    assert us._load()["preset"] == "Balanced"
    assert us._STATE_FILE.read_bytes() == before


def test_preset_is_reported_unapplied_when_its_write_fails(tmp_path, monkeypatch):
    us, te, sent = _setup(tmp_path, monkeypatch)
    us._load()
    monkeypatch.setattr(us, "_save", lambda: False)
    ok, msg = us.apply_preset("Aggressive")
    assert ok is False and "not applied" in msg


def test_preset_name_and_thresholds_land_in_one_durable_write(tmp_path, monkeypatch):
    us, te, sent = _setup(tmp_path, monkeypatch)
    us._load()
    ok, _ = us.apply_preset("Conservative")
    assert ok is True
    on_disk = json.loads(us._STATE_FILE.read_text(encoding="utf-8"))
    assert on_disk["preset"] == "Conservative"
    assert on_disk["min_score"] == 70.0


def test_closed_trade_alert_uses_a_real_line_break(tmp_path, monkeypatch):
    import execution.us_autopilot as us
    us, te, sent = _setup(
        tmp_path, monkeypatch,
        state=_valid_state(us, accounting_ids_migrated=True, accounted_close_ids=[]),
    )
    _journal_with_closed_trade(te)
    us._load()
    us._account_closed()
    assert len(sent) == 1
    assert "\n" in sent[0] and "\\n" not in sent[0]


def test_home_projection_reports_unreadable_us_paper_as_blocked(monkeypatch):
    import execution.us_autopilot as us
    from product import us_market_status

    def broken():
        raise OSError("journal unavailable")

    monkeypatch.setattr(us, "get_status", broken)
    paper = us_market_status.status()["paper"]
    assert paper["armed"] is False
    assert paper["state_recovery_required"] is True
    assert paper["state_recovery_error"] == "US paper status unavailable"


def test_home_projection_carries_backend_recovery_truth(tmp_path, monkeypatch):
    us, te, sent = _setup(tmp_path, monkeypatch)
    us._STATE_FILE.write_text("{torn", encoding="utf-8")
    from product import us_market_status
    paper = us_market_status.status()["paper"]
    assert paper["state_recovery_required"] is True
    assert paper["state_recovery_error"].startswith("JSONDecodeError")
    assert paper["disarmed_reason"] == "US_PAPER_STATE_RECOVERY_REQUIRED"
    assert paper["accounting_reconciliation_required"] is True
