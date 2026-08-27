"""Official NSE corporate-action parse + refresh. Network-free except the live probe."""
from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
from research.autonomy.supervisor import Supervisor


def test_parse_bonus_and_split_subjects():
    from data.corporate_actions import parse_action_subject

    assert parse_action_subject("Bonus 1:1") == ("bonus", 2.0)
    assert parse_action_subject("Bonus 1:5") == ("bonus", 1.2)
    assert parse_action_subject("Bonus 2:1") == ("bonus", 3.0)
    assert parse_action_subject(
        "Face Value Split (Sub-Division) - From Rs 10/- To Re 1/-"
    ) == ("split", 10.0)
    assert parse_action_subject(
        "Face Value Split (Sub-Division) - From Rs 10 To Rs 2"
    ) == ("split", 5.0)
    assert parse_action_subject(
        "Consolidation of Face Value From Re 1 To Rs 10"
    ) == ("consolidation", 0.1)


def test_parse_skips_dividends_rights_and_ncrps():
    from data.corporate_actions import parse_action_subject

    assert parse_action_subject("Interim Dividend - Rs 5 Per Share") is None
    assert parse_action_subject("Rights 1:4") is None
    assert parse_action_subject("Bonus 1:1 on NCRPS") is None
    assert parse_action_subject("") is None


def test_refresh_events_writes_equity_bonus_and_split_only(tmp_path, monkeypatch):
    from data import corporate_actions as CA

    path = tmp_path / "ca_events.json"
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(path))
    payload = [
        {"symbol": "RELIANCE", "series": "EQ", "subject": "Bonus 1:1",
         "exDate": "2024-10-28"},
        {"symbol": "HDFCBANK", "series": "EQ",
         "subject": "Face Value Split (Sub-Division) - From Rs 10 To Re 1",
         "exDate": "2019-09-19"},
        {"symbol": "INFY", "series": "EQ", "subject": "Interim Dividend - Rs 20",
         "exDate": "2024-10-28"},
        {"symbol": "SKIPME", "series": "N1", "subject": "Bonus 1:1",
         "exDate": "2024-10-28"},
        {"symbol": "NCRPSCO", "series": "EQ", "subject": "Bonus 1:1 on NCRPS",
         "exDate": "2024-10-28"},
    ]

    class Session:
        def get(self, url, params=None, timeout=None):
            return SimpleNamespace(status_code=200, json=lambda: payload)

    info = CA.refresh_events(force=True, years=1, budget_s=8.0, session=Session())
    assert info["available"] is True
    assert info["events"] == 2
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_sym = {row["symbol"]: row for row in rows}
    assert by_sym["RELIANCE"]["type"] == "bonus"
    assert by_sym["RELIANCE"]["factor"] == 2.0
    assert by_sym["HDFCBANK"]["type"] == "split"
    assert by_sym["HDFCBANK"]["factor"] == 10.0
    assert "INFY" not in by_sym and "SKIPME" not in by_sym


def test_refresh_events_reuses_fresh_cache(tmp_path, monkeypatch):
    from data import corporate_actions as CA

    path = tmp_path / "ca_events.json"
    path.write_text(json.dumps([{
        "symbol": "TCS", "ex_date": "2024-01-04", "factor": 2.0, "type": "bonus",
    }]), encoding="utf-8")
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(path))
    calls = []

    class Session:
        def get(self, url, params=None, timeout=None):
            calls.append(url)
            raise AssertionError("fresh cache must not hit NSE")

    info = CA.refresh_events(force=False, session=Session())
    assert info["available"] is True
    assert info["source"] == "nse_corporate_actions_cached"
    assert info["fetched"] == 0
    assert calls == []


def test_refresh_events_keeps_existing_rows_when_nse_errors(tmp_path, monkeypatch):
    from data import corporate_actions as CA

    path = tmp_path / "ca_events.json"
    path.write_text(json.dumps([{
        "symbol": "TCS", "ex_date": "2024-01-04", "factor": 2.0, "type": "bonus",
    }]), encoding="utf-8")
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(path))
    monkeypatch.setattr(CA, "_STALE_S", 0)

    class Session:
        def get(self, url, params=None, timeout=None):
            return SimpleNamespace(status_code=403, json=lambda: [])

    info = CA.refresh_events(force=True, years=1, budget_s=8.0, session=Session())
    assert info["available"] is True
    assert info["events"] == 1
    assert info["errors"]


def test_corporate_actions_job_clears_when_ledger_available():
    class Deps:
        def corporate_actions_status(self):
            return {"available": True, "symbols": 12, "events": 40}

    result = JOBS.run_corporate_actions(JOBS._Ctx(Deps()))
    assert result.status == JS.SUCCEEDED
    assert H.CA_INCOMPLETE in result.clears
    assert H.OPTIONS_HISTORY_INCOMPLETE in result.clears


def test_corporate_actions_job_blocks_when_table_missing():
    class Deps:
        def corporate_actions_status(self):
            return {"available": False, "errors": ["NSE corporate-actions HTTP 403"]}

    result = JOBS.run_corporate_actions(JOBS._Ctx(Deps()))
    assert result.status == JS.BLOCKED
    assert H.CA_INCOMPLETE in result.failures
    assert result.blocked_on == JOBS.DEP_CA_SOURCE


def test_canonicalize_drops_stale_options_history():
    cleaned = H.canonicalize_failures({
        H.CA_INCOMPLETE, H.OPTIONS_HISTORY_INCOMPLETE, "not_a_real_code",
    })
    assert cleaned == {H.CA_INCOMPLETE}
    caps = H.capabilities({H.OPTIONS_HISTORY_INCOMPLETE})
    assert H.OPTIONS_HISTORY_INCOMPLETE not in caps["active_failures"]
    assert caps["new_paper_entries"] == H.ALLOWED
    assert caps["research"] == H.ALLOWED


def test_supervisor_rewrites_stale_options_history_failure(tmp_path):
    root = tmp_path / "auto"
    root.mkdir()
    (root / "failures.json").write_text(json.dumps([
        "corporate_actions_incomplete", "options_history_incomplete", "bogus",
    ]), encoding="utf-8")

    class Deps:
        def now_ist(self):
            return date.today()

        def holidays(self):
            return set()

    sup = Supervisor(root, deps=Deps())
    assert H.OPTIONS_HISTORY_INCOMPLETE not in sup.failures
    assert H.CA_INCOMPLETE in sup.failures
    saved = json.loads((root / "failures.json").read_text(encoding="utf-8"))
    assert "options_history_incomplete" not in saved
    assert "corporate_actions_incomplete" in saved
    assert "bogus" not in saved


def test_read_status_hides_stale_options_history(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({
        "state": "OBSERVING",
        "heartbeat_ist": "2099-01-01T00:00:00",
        "active_failures": ["options_history_incomplete", "corporate_actions_incomplete"],
        "process_running": True,
    }), encoding="utf-8")
    raw = H.read_status(state_path=path)
    assert "options_history_incomplete" not in raw["active_failures"]
    assert "corporate_actions_incomplete" in raw["active_failures"]


def test_rows_from_nse_date_window_helper_exists():
    # Keep a cheap sanity check that the live window helper still takes dates.
    from data.corporate_actions import _fetch_nse_window

    class Session:
        def get(self, url, params=None, timeout=None):
            assert params["from_date"] == "01-08-2026"
            assert params["to_date"] == "27-08-2026"
            return SimpleNamespace(status_code=200, json=lambda: [])

    assert _fetch_nse_window(Session(), date(2026, 8, 1), date(2026, 8, 27)) == []
    assert timedelta(days=180).days == 180
