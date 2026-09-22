from __future__ import annotations

from scan import live_edge as LE
from scan import signal_registry as SR
from scan import unified_scanner as US


def _reports(missing_forward="DELIVERY_SPIKE"):
    backtest = {
        "signals": {
            sig: {"trades": 40, "expectancy_r": 0.2}
            for sig in SR.signal_ids()
        }
    }
    live = {
        "signals": {
            sig: {"n": (0 if sig == missing_forward else 35), "expectancy_r": 0.2}
            for sig in SR.signal_ids()
        }
    }
    return backtest, live


def test_registry_is_the_canonical_17_signal_catalog():
    assert len(SR.signal_ids()) == 17
    assert set(US.SIGNAL_META) == set(SR.signal_ids())
    for signal_id, meta in US.SIGNAL_META.items():
        row = SR.SIGNAL_DEFINITIONS[signal_id]
        assert meta == (row["label"], row["category"], row["base_score"])


def test_registry_explains_17_scanner_vs_16_forward_calibrated():
    backtest, live = _reports()
    snapshot = SR.build_registry(backtest_report=backtest, live_profile=live)
    summary = snapshot["summary"]

    assert summary["scanner_catalog"] == 17
    assert summary["backtest_calibrated"] == 17
    assert summary["forward_calibrated"] == 16
    assert summary["effective_calibrated"] == 17
    assert summary["count_difference_explained"] is True
    missing = summary["scanner_without_forward_calibration"]
    assert missing == [{
        "signal_id": "DELIVERY_SPIKE",
        "forward_samples": 0,
        "reason": "NO_FORWARD_OUTCOMES",
    }]


def test_registry_reports_sample_shortfall_reason():
    backtest, live = _reports()
    live["signals"]["DELIVERY_SPIKE"] = {"n": 12, "expectancy_r": 0.1}
    snapshot = SR.build_registry(backtest_report=backtest, live_profile=live)
    row = next(r for r in snapshot["signals"] if r["signal_id"] == "DELIVERY_SPIKE")
    assert row["forward_calibration_eligible"] is False
    assert row["forward_exclusion_reason"] == "FORWARD_SAMPLE_12_LT_30"


def test_unknown_legacy_signal_is_visible_but_not_registry_eligible():
    backtest, live = _reports()
    backtest["signals"]["LEGACY_MAGIC"] = {"trades": 999, "expectancy_r": 9.0}
    live["signals"]["OLD_SIGNAL"] = {"n": 999, "expectancy_r": 9.0}
    snapshot = SR.build_registry(backtest_report=backtest, live_profile=live)
    assert snapshot["unknown_backtest_signal_ids"] == ["LEGACY_MAGIC"]
    assert snapshot["unknown_forward_signal_ids"] == ["OLD_SIGNAL"]
    assert all(r["signal_id"] != "LEGACY_MAGIC" for r in snapshot["signals"])


def test_unified_backtest_calibration_filters_unknown_ids(monkeypatch):
    monkeypatch.setattr(
        "scan.signal_backtest.load_report",
        lambda: {
            "signals": {
                "MOMENTUM": {"trades": 30, "expectancy_r": 0.2},
                "LEGACY_MAGIC": {"trades": 1000, "expectancy_r": 2.0},
            }
        },
    )
    out = US._load_calibration()
    assert out["MOMENTUM"] == 1.0
    assert "LEGACY_MAGIC" not in out


def test_live_calibration_filters_unknown_ids(monkeypatch):
    monkeypatch.setattr(
        LE,
        "profile_edge",
        lambda: {
            "signals": {
                "MOMENTUM": {"n": 35, "expectancy_r": 0.2},
                "LEGACY_MAGIC": {"n": 1000, "expectancy_r": 2.0},
            }
        },
    )
    out = LE.live_calibration(min_n=30)
    assert out == {"MOMENTUM": 1.0}


def test_registry_snapshot_persists_version_and_reasons(tmp_path):
    backtest, live = _reports()
    path = tmp_path / "signal_registry.json"
    expected = SR.build_registry(backtest_report=backtest, live_profile=live)
    SR.save_registry(expected, path=path)
    loaded = SR.load_registry(path=path)
    assert loaded["registry_version"] == SR.registry_version()
    assert loaded["summary"]["forward_calibrated"] == 16
