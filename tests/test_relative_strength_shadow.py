from datetime import date, datetime
import json
import sqlite3

import pandas as pd
import pytest

from product import relative_strength_shadow as shadow


def test_strategy_level_forecast_is_frozen_and_normalized():
    estimate = shadow.historical_forecast()
    assert estimate["resolved_fills"] == 69
    assert sum(estimate["exit_probability"].values()) == pytest.approx(1)
    assert estimate["exit_probability"]["target"] == pytest.approx(19.5/70.5)
    assert shadow._spec()["portfolio"]["max_equal_weight_slots"] == 5


def test_registered_shadow_is_durable_and_late_decisions_are_missed(tmp_path, monkeypatch):
    root = tmp_path / "product"
    root.mkdir()
    monkeypatch.setattr(shadow, "logs_path", lambda *parts: tmp_path.joinpath(*parts))
    assert shadow.nightly() == {"status": "DISABLED"}
    reg = {
        "first_decision_session": "2026-12-01", "mode": "SHADOW_ONLY",
        "registered_at": "2026-11-30T19:00:00+05:30", "spec_sha256": shadow.SPEC_SHA,
        "implementation_sha256": shadow.implementation_sha(),
    }
    path = root / shadow.REGISTRATION
    path.write_text(json.dumps(reg))
    current = [datetime(2026, 12, 1, 19, tzinfo=shadow.IST)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return current[0]

    monkeypatch.setattr(shadow, "datetime", Clock)
    monkeypatch.setattr(shadow, "_dates",
                         lambda a, b: list(pd.bdate_range(a, b).date))
    calls = []

    def shortlist(day):
        calls.append(day)
        return {"session": day.isoformat(), "eligible": ["AAA", "BBB"],
                "selected": ["AAA"], "spec_sha": shadow.SPEC_SHA, "history_sha": "observed",
                "decision_archive_sha": "first"}

    def archives(days, symbols):
        flat = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.},
                            index=pd.DatetimeIndex(days))
        return {s: flat for s in symbols}, {
            "market_wide_excluded_sessions": {}, "archive_sha256": {days[0].isoformat(): "first"},
        }, "outcome"

    monkeypatch.setattr(shadow, "shortlist", shortlist)
    monkeypatch.setattr(shadow, "_inputs", archives)
    assert shadow.nightly()["mature_cycles"] == 0
    assert shadow.nightly()["mature_cycles"] == 0
    assert calls == [date(2026, 12, 1)]
    current[0] = datetime(2026, 12, 16, 19, tzinfo=shadow.IST)
    status = shadow.nightly()
    assert status["mature_cycles"] == 1
    assert status["missed_cycles"] == 1
    assert status["metrics"]["matched_market_excess_pct_per_cycle"] == pytest.approx(0)
    assert status["metrics"]["net_r_per_resolved"] == pytest.approx(-.0032/.03)
    assert status["metrics"]["prospective_brier_using_frozen_historical_forecast"] is not None
    with sqlite3.connect(root / "relative_strength_shadow.db") as db:
        pred = json.loads(db.execute("SELECT prediction FROM cycles WHERE session='2026-12-01'").fetchone()[0])
    assert len(pred["forward_sessions"]) == 10
    assert len(pred["random"]) == len(pred["selected"])
    assert "payload_sha256" in pred

    reg["first_decision_session"] = "2026-12-02"
    path.write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="Registration changed"):
        shadow.nightly()
