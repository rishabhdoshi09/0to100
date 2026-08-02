"""Corporate-action ledger + PIT universe history gap-fill tests."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest


def test_ca_write_merge_and_object_schema(tmp_path, monkeypatch):
    from data import corporate_actions as CA

    dest = tmp_path / "ca_events.json"
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(dest))

    status = CA.write_events(
        [
            {"symbol": "reliance", "ex_date": "2024-01-04", "factor": 2.0, "type": "bonus"},
            {"symbol": "BAD", "ex_date": "2024-01-04", "factor": 1.0, "type": "bonus"},
        ],
        path=dest,
        source="unit_test",
    )
    assert status["available"] is True
    assert status["symbols"] == 1
    assert status["events"] == 1
    assert status["research_grade"] is True

    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source"] == "unit_test"
    assert isinstance(payload["events"], list)

    merged = CA.merge_events(
        [{"symbol": "TCS", "ex_date": "2023-06-01", "factor": 5.0, "type": "split"}],
        path=dest,
    )
    assert merged["symbols"] == 2
    assert merged["events"] == 2
    ev = CA.load_events(dest)
    assert set(ev) == {"RELIANCE", "TCS"}


def test_ca_ingest_from_csv(tmp_path, monkeypatch):
    from data import corporate_actions as CA

    dest = tmp_path / "ca_events.json"
    src = tmp_path / "incoming.csv"
    src.write_text(
        "symbol,ex_date,factor,type\nINFY,2022-09-01,2,bonus\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(dest))
    report = CA.ingest_from_path(src, dest=dest)
    assert report["events"] == 1
    assert "INFY" in CA.load_events(dest)


def test_ca_job_clears_only_with_real_events(tmp_path, monkeypatch):
    from research.autonomy import health as H
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as JOBS

    dest = tmp_path / "ca_events.json"
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(dest))

    class Deps:
        def corporate_actions_status(self):
            from data import corporate_actions as CA

            return CA.ledger_status()

        def ensure_corporate_actions(self):
            from data import corporate_actions as CA

            return CA.ledger_status()

    blocked = JOBS.run_corporate_actions(JOBS._Ctx(Deps()))
    assert blocked.status == JS.BLOCKED
    assert H.CA_INCOMPLETE in blocked.failures

    from data import corporate_actions as CA

    CA.write_events(
        [{"symbol": "GAIL", "ex_date": "2021-07-15", "factor": 1.5, "type": "bonus"}],
        path=dest,
    )
    ok = JOBS.run_corporate_actions(JOBS._Ctx(Deps()))
    assert ok.status == JS.SUCCEEDED
    assert H.CA_INCOMPLETE in ok.clears


def test_universe_history_bhav_bootstrap(tmp_path, monkeypatch):
    from data import bhavcopy_store as BS
    from data import universe_history as UH
    from data import nse_universe as U

    dest = tmp_path / "universe_history.json"
    monkeypatch.setenv("QT_UNIVERSE_HISTORY_FILE", str(dest))

    idx_alive = pd.date_range("2024-01-01", periods=40, freq="B")
    idx_dead = pd.date_range("2023-01-01", periods=20, freq="B")
    alive = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000.0},
        index=idx_alive,
    )
    dead = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000.0},
        index=idx_dead,
    )
    monkeypatch.setattr(BS, "_store", {"ALIVE": alive, "DEADCO": dead}, raising=False)
    monkeypatch.setattr(BS, "_store_last_day", idx_alive[-1].date(), raising=False)

    report = UH.build_from_bhav(path=dest, inactive_after_days=30, min_sessions=5, force=True)
    assert report["built"] is True
    assert report["survivorship_complete"] is True
    assert report["source"] == "bhav_inferred"
    assert report["rows"] == 2

    pit = U.point_in_time_universe(idx_alive[-1].date(), path=dest)
    assert pit["survivorship_complete"] is True
    assert "ALIVE" in pit["symbols"]
    assert "DEADCO" not in pit["symbols"]
    assert pit["source"] == "bhav_inferred"
    assert pit["research_grade"] is False


def test_universe_history_job_builds_and_clears(tmp_path, monkeypatch):
    from data import bhavcopy_store as BS
    from research.autonomy import health as H
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as JOBS

    dest = tmp_path / "universe_history.json"
    monkeypatch.setenv("QT_UNIVERSE_HISTORY_FILE", str(dest))
    idx = pd.date_range("2024-06-01", periods=30, freq="B")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0},
        index=idx,
    )
    monkeypatch.setattr(BS, "_store", {"AAA": frame}, raising=False)
    monkeypatch.setattr(BS, "_store_last_day", idx[-1].date(), raising=False)

    class Deps:
        def now_ist(self):
            return idx[-1].to_pydatetime()

        def universe_history_status(self):
            from data.nse_universe import point_in_time_universe

            return point_in_time_universe(self.now_ist().date())

        def ensure_universe_history(self):
            from data import universe_history as UH

            return UH.build_from_bhav(force=False)

    result = JOBS.run_universe_history(JOBS._Ctx(Deps()))
    assert result.status == JS.SUCCEEDED
    assert H.UNIVERSE_INCOMPLETE in result.clears
    assert dest.exists()


def test_adjustment_policy_raw_vs_adjusted(tmp_path, monkeypatch):
    from research.momentum_breakout.dataset import BhavDataProvider

    dest = tmp_path / "ca_events.json"
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(dest))

    class _P:
        adjustment_policy = BhavDataProvider.adjustment_policy

    raw = _P().adjustment_policy()
    assert raw["corporate_actions"] == "RAW"
    assert "RAW" in json.dumps(raw)

    from data import corporate_actions as CA

    CA.write_events(
        [{"symbol": "X", "ex_date": "2020-01-01", "factor": 2.0, "type": "split"}],
        path=dest,
    )
    adj = _P().adjustment_policy()
    assert adj["corporate_actions"] == "ADJUSTED"
    assert "RAW" not in json.dumps(adj)


def test_snapshot_from_bhav_store_certifies(tmp_path, monkeypatch):
    from data import bhavcopy_store as BS
    from research.intelligence.data.from_bhav import snapshot_from_bhav_store
    from research.intelligence.data.snapshot_store import SnapshotStore

    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    frame = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 1000.0},
        index=idx,
    )
    monkeypatch.setattr(BS, "_store", {"GAIL": frame}, raising=False)
    monkeypatch.setattr("data.bhavcopy_runtime.ensure_loaded", lambda **kwargs: True)

    store = SnapshotStore(tmp_path / "snapshots")
    sid, report = snapshot_from_bhav_store(store, activate=True, actor="test")
    assert sid
    assert report["result"] == "committed"
    assert report["activated"] is True
    assert report["accepted"] == 5
    assert store.get_active_snapshot() == sid
    snap = store.open_snapshot(sid)
    assert snap.manifest.get("source") == "bhav_store"
    assert snap.manifest.get("last_trading_date") == idx[-1].date().isoformat()


def test_options_eod_store_roundtrip(tmp_path):
    from options.eod_store import history, load_chain, save_chain_snapshot, store_status

    db = tmp_path / "eod.sqlite3"
    saved = save_chain_snapshot(
        "NIFTY",
        as_of="2026-07-31",
        expiry="2026-08-07",
        rows=[{"strike": 25000, "ce_oi": 10, "pe_oi": 20, "ce_iv": 12.0, "pe_iv": 14.0}],
        source="unit_test",
        pcr=2.0,
        max_pain=25000.0,
        atm_iv=13.0,
        spot=24980.0,
        path=db,
    )
    assert saved["strike_count"] == 1
    payload = load_chain("NIFTY", as_of="2026-07-31", path=db)
    assert payload is not None
    assert payload["pcr"] == 2.0
    assert len(payload["rows"]) == 1
    hist = history("NIFTY", days=10, path=db)
    assert hist and hist[0]["as_of"] == "2026-07-31"
    status = store_status(path=db)
    assert status["available"] is True
    assert status["symbols"] == 1
    assert status["snapshots"] == 1


def test_options_eod_job_clears_on_save(tmp_path, monkeypatch):
    from research.autonomy import health as H
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as JOBS

    db = tmp_path / "eod.sqlite3"
    monkeypatch.setattr("options.eod_store._DEFAULT_DB", db)

    class Deps:
        def capture_options_eod(self, symbols=None):
            from options.eod_store import save_chain_snapshot, store_status

            save_chain_snapshot(
                "BANKNIFTY",
                as_of="2026-07-31",
                expiry="2026-08-06",
                rows=[{"strike": 52000, "ce_oi": 1, "pe_oi": 1}],
                source="unit_test",
                path=db,
            )
            status = store_status(path=db)
            return {"requested": 1, "saved": 1, "failed": 0, **status}

        def options_eod_status(self):
            from options.eod_store import store_status

            return store_status(path=db)

    result = JOBS.run_options_eod(JOBS._Ctx(Deps()))
    assert result.status == JS.SUCCEEDED
    assert H.OPTIONS_HISTORY_INCOMPLETE in result.clears


def test_product_readiness_includes_snapshot_and_options_lanes():
    from datetime import datetime, timezone

    from product.product_readiness import build_product_readiness

    now = datetime(2026, 8, 1, 5, 0, tzinfo=timezone.utc)
    payload = build_product_readiness(
        market={"available": True, "summary": "ok"},
        scan={"available": True, "scanned_at": "2026-08-01T03:00:00+00:00", "records": [{"symbol": "A"}]},
        long_term={"available": True, "scanned_at": "2026-07-31T03:00:00+00:00", "records": [{"symbol": "A"}],
                   "summary": {"coverage_pct": 50}},
        news={"available": True, "articles": [{"published_at": "2026-08-01T01:00:00+00:00"}], "stats": {"total": 1}},
        fno={"available": True, "mapped_underlyings": 10, "generated_at": "2026-08-01T00:00:00+00:00"},
        data={
            "bhavcopy": {"ready": True, "sessions": 500, "symbols": 1800, "latest_date": "2026-07-31"},
            "snapshot": {"ready": True, "snapshot_id": "abc", "latest_date": "2026-07-31", "source": "bhav_store"},
            "options_eod": {"available": True, "snapshots": 3, "symbols": 3, "latest_as_of": "2026-07-31"},
        },
        operations={"running": True, "heartbeat": "2026-08-01T04:59:55+00:00", "active": []},
        now=now,
    )
    keys = {item["key"] for item in payload["lanes"]}
    assert "snapshot" in keys
    assert "options_eod" in keys
    assert payload["state"] == "READY"
    assert payload["score"] >= 90
