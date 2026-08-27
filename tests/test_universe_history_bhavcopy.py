"""Official NSE bhavcopy membership fills point-in-time universe history."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor


def _frame_ending(last: date, sessions: int) -> pd.DataFrame:
    idx = pd.bdate_range(end=last, periods=sessions)
    return pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000},
        index=idx,
    )


def test_membership_marks_names_missing_from_latest_session(monkeypatch):
    from data import bhavcopy_store as BS

    last = date(2026, 8, 27)
    live = _frame_ending(last, 80)
    gone = _frame_ending(date(2026, 3, 31), 40)
    monkeypatch.setattr(BS, "_store", {"RELIANCE": live, "OLDTATA": gone}, raising=False)
    monkeypatch.setattr(BS, "_store_last_day", last, raising=False)
    monkeypatch.setattr(BS, "_store_sessions", 80, raising=False)

    rows, meta = BS.membership_rows()
    by_sym = {row["symbol"]: row for row in rows}
    assert meta["sessions"] == 80
    assert "delisted" not in by_sym["RELIANCE"]
    assert by_sym["OLDTATA"]["delisted"] == (
        pd.Timestamp(gone.index.max()).normalize() + pd.Timedelta(days=1)
    ).date().isoformat()
    assert by_sym["RELIANCE"]["source"] == "official_nse_bhavcopy"


def test_refresh_universe_history_writes_official_table(tmp_path, monkeypatch):
    from data import bhavcopy_store as BS
    from data import nse_universe as U

    path = tmp_path / "universe_history.json"
    monkeypatch.setenv("QT_UNIVERSE_HISTORY_FILE", str(path))
    last = date(2026, 8, 27)
    live = _frame_ending(last, 80)
    monkeypatch.setattr(BS, "_store", {"INFY": live}, raising=False)
    monkeypatch.setattr(BS, "_store_last_day", last, raising=False)
    monkeypatch.setattr(BS, "_store_sessions", 80, raising=False)

    info = U.refresh_universe_history(as_of=last, force=True)
    assert info["survivorship_complete"] is True
    assert info["source"] == "official_nse_bhavcopy"
    assert "INFY" in info["symbols"]
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written[0]["symbol"] == "INFY"
    assert written[0]["listed"] == pd.Timestamp(live.index.min()).date().isoformat()
    assert "delisted" not in written[0]


def test_refresh_universe_history_stays_incomplete_when_store_is_shallow(tmp_path, monkeypatch):
    from data import bhavcopy_store as BS
    from data import nse_universe as U

    path = tmp_path / "universe_history.json"
    monkeypatch.setenv("QT_UNIVERSE_HISTORY_FILE", str(path))
    monkeypatch.setattr(BS, "_store", {"TINY": _frame_ending(date(2026, 8, 7), 5)}, raising=False)
    monkeypatch.setattr(BS, "_store_last_day", date(2026, 8, 7), raising=False)
    monkeypatch.setattr(BS, "_store_sessions", 5, raising=False)

    info = U.refresh_universe_history(as_of=date(2026, 8, 7), force=True)
    assert info["survivorship_complete"] is False
    assert not path.exists()


def test_universe_job_clears_when_bhavcopy_membership_is_ready():
    class Deps:
        def universe_history_status(self):
            return {"survivorship_complete": True, "symbols": ["RELIANCE", "INFY"]}

    result = JOBS.run_universe_history(JOBS._Ctx(Deps()))
    assert result.status == JS.SUCCEEDED
    assert H.UNIVERSE_INCOMPLETE in result.clears


def test_bhavcopy_update_writes_universe_and_unblocks():
    class Deps:
        def update_bhavcopy(self):
            return {"symbols": 1800, "ready": True, "source": "official_nse"}

        def universe_history_status(self):
            return {"survivorship_complete": True, "symbols": ["RELIANCE"]}

    result = JOBS.run_bhavcopy_update(JOBS._Ctx(Deps()))
    assert result.status == JS.SUCCEEDED
    assert H.UNIVERSE_INCOMPLETE in result.clears
    assert JOBS.DEP_UNIVERSE_SOURCE in result.unblocks


def test_supervisor_start_requeues_blocked_ca_and_universe_jobs(tmp_path):
    root = tmp_path / "auto"
    root.mkdir()

    class Deps:
        def now_ist(self):
            return date.today()

        def holidays(self):
            return set()

        def notify_online(self):
            return "already_sent"

    sup = Supervisor(root, deps=Deps())
    ca = sup.jobs.enqueue(SCH.CORPORATE_ACTIONS, idempotency_key="corporate_actions:2026-08-27")
    uni = sup.jobs.enqueue(SCH.UNIVERSE_HISTORY, idempotency_key="universe_history:2026-08-27")
    sup.jobs.block(ca.job_id, dependency=JOBS.DEP_CA_SOURCE, reason="missing file")
    sup.jobs.block(uni.job_id, dependency=JOBS.DEP_UNIVERSE_SOURCE, reason="missing file")
    assert sup.jobs.get(ca.job_id).status == JS.BLOCKED
    assert sup.start() is True
    assert sup.jobs.get(ca.job_id).status == JS.PENDING
    assert sup.jobs.get(uni.job_id).status == JS.PENDING
