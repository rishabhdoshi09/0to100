"""US paper realized-P&L settlement must use exact close IDs, not a high-water mark.

These tests exercise the real SQLite journal with out-of-order exits, durable
restart, legacy migration, and accounting checkpoint failure. No market data or
broker credentials are used.
"""
from __future__ import annotations

import json

import pytest


def _setup(tmp_path, monkeypatch):
    import execution.trade_executor as te
    import execution.us_autopilot as us

    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(us, "_STATE_FILE", tmp_path / "us_autopilot.json")
    monkeypatch.setattr(us, "_state", dict(us._DEFAULTS))
    sent = []
    monkeypatch.setattr(us, "_notify", sent.append)
    conn = te.connect()
    te.ensure_schema(conn)
    conn.close()
    return us, te, sent


def _insert(te, symbol, status, exit_price=None):
    conn = te.connect()
    try:
        cursor = conn.execute(
            """INSERT INTO trades
                (placed_at, mode, symbol, qty, entry_type, entry_price,
                 stop_price, target_price, status, exit_price, note)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "2026-11-06T10:00:00", "PAPER", symbol, 10,
                "MARKET", 100.0, 95.0, 110.0, status,
                exit_price, "US_AUTOPILOT:test|session=2026-11-06",
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _close(te, trade_id, status, exit_price):
    conn = te.connect()
    try:
        conn.execute(
            "UPDATE trades SET status=?,exit_price=? WHERE id=?",
            (status, exit_price, trade_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_late_closing_older_us_position_is_accounted_once_and_survives_restart(
    tmp_path, monkeypatch,
):
    us, te, sent = _setup(tmp_path, monkeypatch)
    older = _insert(te, "OLDER", "PAPER_OPEN")
    newer = _insert(te, "NEWER", "PAPER_OPEN")
    assert older < newer

    _close(te, newer, "PAPER_WIN", 110.0)
    us._account_closed()
    newer_pnl = round(us._net_pnl(next(
        t for t in us._trades(us._WIN) if t["id"] == newer
    )), 2)
    assert us._load()["realized_pnl"] == newer_pnl
    assert us._load()["accounted_close_ids"] == [newer]

    _close(te, older, "PAPER_LOSS", 94.0)
    us._account_closed()
    older_pnl = round(us._net_pnl(next(
        t for t in us._trades(us._LOSS) if t["id"] == older
    )), 2)
    expected = round(newer_pnl + older_pnl, 2)
    assert us._load()["realized_pnl"] == expected
    assert us._load()["accounted_close_ids"] == [older, newer]
    assert len(sent) == 2

    monkeypatch.setattr(us, "_state", {})
    us._account_closed()
    assert us._load()["realized_pnl"] == expected
    assert len(sent) == 2  # replay/restart must not double-book or notify


def test_legacy_watermark_migration_never_forgets_an_older_open_trade(
    tmp_path, monkeypatch,
):
    us, te, sent = _setup(tmp_path, monkeypatch)
    old_open = _insert(te, "WAITING", "PAPER_OPEN")
    recent_closed = _insert(te, "RECENT", "PAPER_WIN", 110.0)
    prior_pnl = round(us._net_pnl(next(t for t in us._trades(us._WIN))), 2)

    legacy = dict(us._DEFAULTS)
    legacy["realized_pnl"] = prior_pnl
    legacy["max_accounted_id"] = recent_closed
    for key in ("accounted_close_ids", "accounting_ids_migrated",
                "accounting_reconciliation_required"):
        legacy.pop(key)
    us._STATE_FILE.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(us, "_state", {})

    us._account_closed()
    assert us._load()["realized_pnl"] == prior_pnl
    assert us._load()["accounted_close_ids"] == [recent_closed]
    assert us._load()["accounting_reconciliation_required"] is True
    assert sent == []

    _close(te, old_open, "PAPER_LOSS", 93.0)
    us._account_closed()
    loss = round(us._net_pnl(next(t for t in us._trades(us._LOSS))), 2)
    assert us._load()["realized_pnl"] == round(prior_pnl + loss, 2)
    assert us._load()["accounted_close_ids"] == [old_open, recent_closed]
    assert len(sent) == 1


def test_us_accounting_does_not_announce_or_advance_on_failed_checkpoint(
    tmp_path, monkeypatch,
):
    us, te, sent = _setup(tmp_path, monkeypatch)
    _insert(te, "FAILWRITE", "PAPER_WIN", 110.0)
    monkeypatch.setattr(us, "_save", lambda: False)

    with pytest.raises(OSError, match="checkpoint was not persisted"):
        us._account_closed()
    assert us._load()["realized_pnl"] == 0.0
    assert us._load()["accounted_close_ids"] == []
    assert us._load()["accounting_ids_migrated"] is False
    assert not sent
