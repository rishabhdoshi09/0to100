"""Deterministic acceptance tests for the F&O exposure/selection-bias
tracker (product/fno_exposure_tracking.py) -- diagnostic only, never feeds
ranking or gates a trade.
"""
from __future__ import annotations

from product.fno_exposure_tracking import (
    HIGH_SELECTION_RATE,
    LOW_SELECTION_RATE_DESPITE_TOP5,
    MIN_SCANS_FOR_A_READING,
    exposure_report,
    record_scan_exposure,
)


def _setup(direction="LONG", sector_strength=2.5, nifty_alignment=3.0):
    return {
        "score": 80.0, "direction": direction, "atr_pct": 2.0, "breakout_distance_pct": 1.5,
        "components": {"nifty_alignment": nifty_alignment, "sector_strength": sector_strength},
    }


def _row(symbol: str, **overrides):
    base = {"symbol": symbol, "setup": _setup(**{k: v for k, v in overrides.items() if k in ("direction", "sector_strength", "nifty_alignment")})}
    return base


def test_empty_store_reports_zero_contexts(tmp_path, monkeypatch):
    path = tmp_path / "exposure.json"
    monkeypatch.setenv("QT_FNO_EXPOSURE_TRACKING", str(path))
    report = exposure_report()
    assert report["contexts_tracked"] == 0
    assert report["possibly_self_reinforcing"] == []
    assert report["possibly_exploration_starved"] == []


def test_row_without_direction_is_skipped_not_recorded(tmp_path, monkeypatch):
    path = tmp_path / "exposure.json"
    monkeypatch.setenv("QT_FNO_EXPOSURE_TRACKING", str(path))
    row = {"symbol": "NODIR", "setup": {"direction": "SIDEWAYS"}}
    record_scan_exposure([row], opened_underlyings=set())
    report = exposure_report()
    assert report["contexts_tracked"] == 0


def test_consistently_selected_context_flags_possible_self_reinforcement(tmp_path, monkeypatch):
    path = tmp_path / "exposure.json"
    monkeypatch.setenv("QT_FNO_EXPOSURE_TRACKING", str(path))
    row = _row("ALWAYSPICKED")
    for _ in range(MIN_SCANS_FOR_A_READING):
        record_scan_exposure([row], opened_underlyings={"ALWAYSPICKED"})

    report = exposure_report()
    assert report["contexts_tracked"] == 1
    flagged = report["possibly_self_reinforcing"]
    assert flagged and flagged[0]["selection_rate"] >= HIGH_SELECTION_RATE
    assert flagged[0]["scans_seen"] == MIN_SCANS_FOR_A_READING
    assert flagged[0]["times_selected"] == MIN_SCANS_FOR_A_READING


def test_frequently_top5_but_rarely_selected_context_flags_exploration_starvation(tmp_path, monkeypatch):
    path = tmp_path / "exposure.json"
    monkeypatch.setenv("QT_FNO_EXPOSURE_TRACKING", str(path))
    # Ranked in the top 5 (index 0) every cycle, but capacity/other caps mean
    # it is essentially never the one actually opened.
    row = _row("NEVERPICKED")
    for i in range(MIN_SCANS_FOR_A_READING):
        opened = {"NEVERPICKED"} if i == 0 else set()  # selected once only
        record_scan_exposure([row], opened_underlyings=opened)

    report = exposure_report()
    starved = report["possibly_exploration_starved"]
    assert starved and starved[0]["context_key"]
    assert starved[0]["selection_rate"] <= LOW_SELECTION_RATE_DESPITE_TOP5
    assert starved[0]["times_top5"] == MIN_SCANS_FOR_A_READING


def test_below_minimum_sample_never_flagged_either_way(tmp_path, monkeypatch):
    path = tmp_path / "exposure.json"
    monkeypatch.setenv("QT_FNO_EXPOSURE_TRACKING", str(path))
    row = _row("TOOFEW")
    for _ in range(MIN_SCANS_FOR_A_READING - 1):
        record_scan_exposure([row], opened_underlyings={"TOOFEW"})

    report = exposure_report()
    assert report["possibly_self_reinforcing"] == []
    assert report["possibly_exploration_starved"] == []
    assert report["contexts_tracked"] == 1


def test_rank_position_beyond_top5_never_counts_as_top5(tmp_path, monkeypatch):
    path = tmp_path / "exposure.json"
    monkeypatch.setenv("QT_FNO_EXPOSURE_TRACKING", str(path))
    filler = [_row(f"FILLER{i}", sector_strength=-9.0 + i) for i in range(5)]
    tail_row = _row("TAILEND", sector_strength=9.0)
    record_scan_exposure(filler + [tail_row], opened_underlyings=set())

    report = exposure_report()
    tail_context = [r for r in report["top_contexts"] if r["scans_seen"] == 1][0]
    assert tail_context["times_top5"] == 0


def test_report_is_read_only_and_never_mutates_store(tmp_path, monkeypatch):
    path = tmp_path / "exposure.json"
    monkeypatch.setenv("QT_FNO_EXPOSURE_TRACKING", str(path))
    row = _row("STABLE")
    record_scan_exposure([row], opened_underlyings={"STABLE"})
    before = path.read_text(encoding="utf-8")
    exposure_report()
    exposure_report()
    after = path.read_text(encoding="utf-8")
    assert before == after
