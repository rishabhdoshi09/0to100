from __future__ import annotations

from scan import calibration_snapshot as CS
from scan import signal_registry as SR
from scan.unified_scanner import UnifiedScanner
from product import historical_replay as HR


def _reports():
    backtest = {
        "signals": {
            "MOMENTUM": {"trades": 40, "expectancy_r": 0.35},
            "VCP": {"trades": 40, "expectancy_r": -0.2},
        }
    }
    live = {
        "signals": {
            "MOMENTUM": {"n": 35, "expectancy_r": -0.2},
            "VCP": {"n": 35, "expectancy_r": 0.4},
        }
    }
    return backtest, live


def test_snapshot_identity_is_stable_for_identical_inputs():
    backtest, live = _reports()
    a = CS.build_snapshot(
        backtest_report=backtest,
        live_profile=live,
        data_identity="official_nse:2026-09-21",
        thesis_hash="thesis-a",
    )
    b = CS.build_snapshot(
        backtest_report=backtest,
        live_profile=live,
        data_identity="official_nse:2026-09-21",
        thesis_hash="thesis-a",
    )

    assert a["snapshot_id"] == b["snapshot_id"]
    assert a["identities"] == b["identities"]
    assert a["snapshot_id"].startswith("cal_")


def test_snapshot_changes_when_evidence_or_thesis_changes():
    backtest, live = _reports()
    a = CS.build_snapshot(
        backtest_report=backtest,
        live_profile=live,
        data_identity="official_nse:2026-09-21",
        thesis_hash="thesis-a",
    )
    live2 = {"signals": {**live["signals"]}}
    live2["signals"]["MOMENTUM"] = {"n": 36, "expectancy_r": -0.2}
    b = CS.build_snapshot(
        backtest_report=backtest,
        live_profile=live2,
        data_identity="official_nse:2026-09-21",
        thesis_hash="thesis-a",
    )
    c = CS.build_snapshot(
        backtest_report=backtest,
        live_profile=live,
        data_identity="official_nse:2026-09-21",
        thesis_hash="thesis-b",
    )

    assert a["snapshot_id"] != b["snapshot_id"]
    assert a["snapshot_id"] != c["snapshot_id"]


def test_effective_multiplier_is_conservative_minimum():
    backtest, live = _reports()
    out = CS.effective_multipliers(backtest, live)

    # MOMENTUM: backtest 1.25, forward 0.45 -> 0.45.
    assert out["MOMENTUM"] == 0.45
    # VCP: backtest 0.45, forward 1.25 -> remains 0.45.
    assert out["VCP"] == 0.45


def test_get_or_create_reuses_immutable_snapshot(tmp_path):
    backtest, live = _reports()
    first = CS.get_or_create_snapshot(
        backtest_report=backtest,
        live_profile=live,
        data_identity="official_nse:2026-09-21",
        thesis_hash="thesis-a",
        directory=tmp_path,
    )
    second = CS.get_or_create_snapshot(
        backtest_report=backtest,
        live_profile=live,
        data_identity="official_nse:2026-09-21",
        thesis_hash="thesis-a",
        directory=tmp_path,
    )

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert (tmp_path / f"{first['snapshot_id']}.json").exists()


def test_scanner_uses_supplied_frozen_snapshot_without_recalibration():
    registry = SR.build_registry(backtest_report={}, live_profile={})
    frozen = {
        "snapshot_id": "cal_test",
        "multipliers": {"MOMENTUM": 0.45, "LEGACY_MAGIC": 9.0},
        "signal_registry": registry,
        "cache_hit": True,
    }

    scanner = UnifiedScanner(max_workers=1, calibration_snapshot=frozen)

    assert scanner._calibration_snapshot_id == "cal_test"
    assert scanner._calib == {"MOMENTUM": 0.45}
    assert scanner._signal_registry["registry_version"] == SR.registry_version()


def test_replay_identity_includes_calibration_snapshot():
    days = ["2026-09-17", "2026-09-18", "2026-09-21"]
    a = HR.replay_identity(
        sessions=2,
        universe_limit=10,
        symbols=["TCS"],
        dates_fn=lambda: days,
        calibration_snapshot={"snapshot_id": "cal_a"},
    )
    b = HR.replay_identity(
        sessions=2,
        universe_limit=10,
        symbols=["TCS"],
        dates_fn=lambda: days,
        calibration_snapshot={"snapshot_id": "cal_b"},
    )

    assert a["calibration_snapshot_id"] == "cal_a"
    assert b["calibration_snapshot_id"] == "cal_b"
    assert a["run_id"] != b["run_id"]
