from __future__ import annotations

import pandas as pd

from product import historical_replay as HR


class _Versions:
    def as_dict(self):
        return {"test": "v1"}


def _patch_common(monkeypatch):
    monkeypatch.setattr("data.bhavcopy_runtime.ensure_loaded", lambda **_k: {"ready": True})
    monkeypatch.setattr(
        "data.nse_universe.refresh_universe_history",
        lambda **_k: {
            "available": True,
            "survivorship_complete": True,
            "symbols": ["INFY"],
        },
    )
    monkeypatch.setattr("product.pit_versions.current_versions", lambda: _Versions())
    monkeypatch.setattr("product.pit_warehouse.warehouse_fingerprint", lambda: "warehouse-test")
    monkeypatch.setattr(HR, "evaluate_outcomes", lambda rows: list(rows))


def test_empty_replay_cannot_report_succeeded(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    sessions = ["2026-09-08", "2026-09-09", "2026-09-10"]
    monkeypatch.setattr(
        HR,
        "universe_as_of",
        lambda *_a, **_k: {
            "symbols": [],
            "survivorship_complete": False,
        },
    )
    monkeypatch.setattr(
        HR,
        "scan_session",
        lambda *_a, **_k: {
            "scanned": 0,
            "records": [],
            "rejected_candidates": [],
            "errors": [],
        },
    )
    monkeypatch.setattr(HR, "decide_session", lambda *_a, **_k: [])

    result = HR.run_historical_replay(
        sessions=2,
        universe_limit=40,
        force=True,
        directory=tmp_path,
        dates_fn=lambda: sessions,
    )

    assert result["status"] == "DEGRADED"
    assert result["evidence_ready"] is False
    assert result["blocker_reason"] == "NO_HISTORICAL_MARKET_OBSERVATIONS"
    assert result["universe_observations"] == 0
    assert result["stocks_evaluated"] == 0


def test_real_evaluated_no_setup_session_is_valid_success(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    sessions = ["2026-09-08", "2026-09-09", "2026-09-10"]
    monkeypatch.setattr(
        HR,
        "universe_as_of",
        lambda *_a, **_k: {
            "symbols": ["INFY"],
            "survivorship_complete": True,
        },
    )
    monkeypatch.setattr(
        HR,
        "scan_session",
        lambda *_a, **_k: {
            "scanned": 1,
            "records": [],
            "rejected_candidates": [],
            "errors": [],
        },
    )
    monkeypatch.setattr(HR, "decide_session", lambda *_a, **_k: [])

    result = HR.run_historical_replay(
        sessions=2,
        universe_limit=40,
        force=True,
        directory=tmp_path,
        dates_fn=lambda: sessions,
    )

    assert result["status"] == "SUCCEEDED"
    assert result["evidence_ready"] is True
    assert result["blocker_reason"] == ""
    assert result["universe_observations"] == 2
    assert result["stocks_evaluated"] == 2
    assert result["decisions_tested"] == 0


def test_survivorship_incomplete_replay_is_degraded_even_with_observations(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    sessions = ["2026-09-08", "2026-09-09", "2026-09-10"]
    monkeypatch.setattr(
        HR,
        "universe_as_of",
        lambda *_a, **_k: {
            "symbols": ["INFY"],
            "survivorship_complete": False,
        },
    )
    monkeypatch.setattr(
        HR,
        "scan_session",
        lambda *_a, **_k: {
            "scanned": 1,
            "records": [],
            "rejected_candidates": [],
            "errors": [],
        },
    )
    monkeypatch.setattr(HR, "decide_session", lambda *_a, **_k: [])

    result = HR.run_historical_replay(
        sessions=2,
        universe_limit=40,
        force=True,
        directory=tmp_path,
        dates_fn=lambda: sessions,
    )

    assert result["status"] == "DEGRADED"
    assert result["evidence_ready"] is False
    assert result["blocker_reason"] == "UNIVERSE_HISTORY_INCOMPLETE"


def test_explicit_symbol_scope_does_not_require_whole_market_membership(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    sessions = ["2026-09-08", "2026-09-09", "2026-09-10"]
    monkeypatch.setattr(
        "data.nse_universe.refresh_universe_history",
        lambda **_k: (_ for _ in ()).throw(AssertionError("whole-market universe refresh must not run")),
    )
    monkeypatch.setattr(
        HR,
        "universe_as_of",
        lambda *_a, **_k: {
            "symbols": ["INFY"],
            "survivorship_complete": False,
        },
    )
    monkeypatch.setattr(
        HR,
        "scan_session",
        lambda *_a, **_k: {
            "scanned": 1,
            "records": [],
            "rejected_candidates": [],
            "errors": [],
        },
    )
    monkeypatch.setattr(HR, "decide_session", lambda *_a, **_k: [])

    first = HR.run_historical_replay(
        sessions=2,
        universe_limit=1,
        symbols=["INFY"],
        force=True,
        directory=tmp_path,
        dates_fn=lambda: sessions,
    )
    second = HR.run_historical_replay(
        sessions=2,
        universe_limit=1,
        symbols=["INFY"],
        force=False,
        directory=tmp_path,
        dates_fn=lambda: sessions,
    )

    assert first["status"] == "SUCCEEDED"
    assert first["universe_scope"] == "EXPLICIT_SYMBOLS"
    assert first["evidence_ready"] is True
    assert second["cache_hit"] is True
    assert second["run_id"] == first["run_id"]



def test_universe_as_of_requires_exact_session_bar(monkeypatch):
    as_of = "2026-09-21"
    dates = pd.bdate_range(end=as_of, periods=70)
    current = pd.DataFrame(
        {"close": range(70), "high": range(70), "low": range(70), "volume": [1000] * 70},
        index=dates,
    )
    stale = current.iloc[:-1].copy()
    frames = {"CURRENT": current, "STALE": stale}
    monkeypatch.setattr(
        "data.nse_universe.point_in_time_universe",
        lambda _day: {
            "survivorship_complete": True,
            "symbols": ["CURRENT", "STALE"],
        },
    )

    out = HR.universe_as_of(
        as_of,
        ohlcv_fn=lambda symbol: frames[symbol],
        limit=10,
    )

    assert out["symbols"] == ["CURRENT"]
    assert out["survivorship_complete"] is True
    assert any("STALE: no official bar on 2026-09-21" in x for x in out["degraded"])


def test_pit_universe_snapshot_reused_on_forced_replay_when_data_identity_unchanged(
    tmp_path, monkeypatch
):
    _patch_common(monkeypatch)
    sessions = ["2026-09-18", "2026-09-21", "2026-09-22"]
    calls = {"universe": 0}

    def universe(day, **_kwargs):
        calls["universe"] += 1
        return {
            "as_of": day,
            "symbols": ["INFY"],
            "requested": 1,
            "survivorship_complete": True,
            "degraded": [],
            "pit": {"survivorship_complete": True, "n": 1},
        }

    monkeypatch.setattr(HR, "universe_as_of", universe)
    monkeypatch.setattr(
        HR,
        "scan_session",
        lambda *a, **k: {
            "scanned": 1,
            "records": [],
            "rejected_candidates": [],
            "errors": [],
        },
    )
    monkeypatch.setattr(HR, "decide_session", lambda *a, **k: [])

    first = HR.run_historical_replay(
        sessions=2,
        universe_limit=1,
        force=True,
        directory=tmp_path,
        dates_fn=lambda: sessions,
        calibration_snapshot={"snapshot_id": "cal-test"},
    )
    first_calls = calls["universe"]
    second = HR.run_historical_replay(
        sessions=2,
        universe_limit=1,
        force=True,
        directory=tmp_path,
        dates_fn=lambda: sessions,
        calibration_snapshot={"snapshot_id": "cal-test"},
    )

    assert first_calls == 2
    assert calls["universe"] == first_calls
    assert first["universe_snapshot"]["cache_misses"] == 2
    assert second["universe_snapshot"]["cache_hits"] == 2
    assert second["universe_snapshot"]["cache_misses"] == 0
    assert second["universe_snapshot"]["cache_id"] == first["universe_snapshot"]["cache_id"]


def test_pit_universe_snapshot_identity_changes_with_data_fingerprint(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    sessions = ["2026-09-18", "2026-09-21", "2026-09-22"]
    fingerprints = iter(["warehouse-a", "warehouse-b"])
    monkeypatch.setattr(
        "product.pit_warehouse.warehouse_fingerprint",
        lambda: next(fingerprints),
    )

    # Identity helper alone proves stale universe membership cannot be reused
    # across a changed immutable market dataset.
    a = HR._universe_cache_identity(
        data_fingerprint="warehouse-a",
        versions={"v": 1},
        symbols=None,
        universe_limit=40,
    )
    b = HR._universe_cache_identity(
        data_fingerprint="warehouse-b",
        versions={"v": 1},
        symbols=None,
        universe_limit=40,
    )
    assert a != b
