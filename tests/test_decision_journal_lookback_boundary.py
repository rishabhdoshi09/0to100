from datetime import datetime

import pandas as pd


def test_update_outcomes_includes_entire_cutoff_calendar_day(tmp_path, monkeypatch):
    """A row earlier on the 40-day cutoff date must remain eligible all day."""
    import core.decision_journal as dj

    monkeypatch.setattr(dj, "_DB_PATH", str(tmp_path / "decisions.db"))
    monkeypatch.setattr(dj, "_now", lambda: datetime(2026, 9, 10, 18, 0, 0))

    dj.log_decision(
        "HAL",
        "TAKEN",
        "",
        "regression",
        entry_ref=100.0,
        stop_ref=95.0,
        score=80.0,
    )
    c = dj._conn()
    c.execute("UPDATE decisions SET decided_at='2026-08-01T10:00:00'")
    c.commit()
    c.close()

    idx = pd.bdate_range("2026-08-01", periods=8)
    closes = [100.0] * 5 + [105.0, 9999.0, 9999.0]
    bars = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1,
        },
        index=idx,
    )
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda _symbol: bars)

    assert dj.update_outcomes(lookback_days=40) == 1

    c = dj._conn()
    row = c.execute(
        "SELECT outcome_price, outcome_pct FROM decisions WHERE symbol='HAL'"
    ).fetchone()
    c.close()
    assert row["outcome_price"] == 105.0
    assert row["outcome_pct"] == 5.0
