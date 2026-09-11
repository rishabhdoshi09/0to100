from datetime import datetime


def _logged_decision(dj, symbol: str) -> None:
    dj.log_decision(
        symbol,
        "TAKEN",
        "",
        "regression",
        entry_ref=100.0,
        stop_ref=95.0,
        score=80.0,
    )


def test_update_outcomes_includes_entire_cutoff_calendar_day(tmp_path, monkeypatch):
    """A row earlier on the 40-day cutoff date must remain eligible all day."""
    import core.decision_journal as dj
    import core.outcome_resolver as resolver

    monkeypatch.setattr(dj, "_DB_PATH", str(tmp_path / "decisions.db"))
    monkeypatch.setattr(dj, "_now", lambda: datetime(2026, 9, 10, 18, 0, 0))

    _logged_decision(dj, "HAL")
    c = dj._conn()
    c.execute("UPDATE decisions SET decided_at='2026-08-01T10:00:00'")
    c.commit()
    c.close()

    calls = []

    def _resolve(symbol, day):
        calls.append((symbol, day))
        return 105.0, 5.0

    monkeypatch.setattr(resolver, "session_close_return", _resolve)

    assert dj.update_outcomes(lookback_days=40) == 1
    assert calls == [("HAL", "2026-08-01")]

    c = dj._conn()
    row = c.execute(
        "SELECT outcome_price, outcome_pct FROM decisions WHERE symbol='HAL'"
    ).fetchone()
    c.close()
    assert row["outcome_price"] == 105.0
    assert row["outcome_pct"] == 5.0


def test_update_outcomes_excludes_day_before_cutoff(tmp_path, monkeypatch):
    """Calendar retention must not silently widen to include the prior date."""
    import core.decision_journal as dj
    import core.outcome_resolver as resolver

    monkeypatch.setattr(dj, "_DB_PATH", str(tmp_path / "decisions.db"))
    monkeypatch.setattr(dj, "_now", lambda: datetime(2026, 9, 10, 18, 0, 0))

    _logged_decision(dj, "OLD")
    c = dj._conn()
    c.execute("UPDATE decisions SET decided_at='2026-07-31T23:59:59'")
    c.commit()
    c.close()

    calls = []

    def _resolve(symbol, day):
        calls.append((symbol, day))
        return 105.0, 5.0

    monkeypatch.setattr(resolver, "session_close_return", _resolve)

    assert dj.update_outcomes(lookback_days=40) == 0
    assert calls == []

    c = dj._conn()
    row = c.execute(
        "SELECT outcome_price, outcome_pct FROM decisions WHERE symbol='OLD'"
    ).fetchone()
    c.close()
    assert row["outcome_price"] is None
    assert row["outcome_pct"] is None
