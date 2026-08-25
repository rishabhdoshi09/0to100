"""Autonomy self-feed: taken vs skipped, SEPA exam, cheap official-bar tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from product.paper_learning import build_paper_memory, select_paper_signal
from product.paper_self_feed import (
    NOT_A_SAMPLE,
    attach_to_paper_memory,
    build_report,
    ingest_paper_cycle,
    save_report,
    shadow_setup,
)


def _win_frame():
    idx = pd.to_datetime(["2026-08-02", "2026-08-03", "2026-08-04"])
    return pd.DataFrame(
        {"high": [101.0, 121.0, 122.0], "low": [99.0, 100.0, 101.0], "close": [100.0, 118.0, 119.0]},
        index=idx,
    )


def _loss_frame():
    idx = pd.to_datetime(["2026-08-02", "2026-08-03", "2026-08-04"])
    return pd.DataFrame(
        {"high": [101.0, 102.0, 103.0], "low": [99.0, 88.0, 90.0], "close": [100.0, 89.0, 91.0]},
        index=idx,
    )


def test_report_tags_taken_skipped_and_sepa_exam():
    cycle = {
        "as_of_date": "2026-08-01",
        "eligibility": "TRADED",
        "positions_opened": [("s1", "TAKEN")],
        "blocked_target_positions": [("s1", "SKIPPED", "PAPER_LESSON_COOLDOWN")],
    }
    sepa = [
        {"symbol": "TAKEN", "sepa_score": 80, "sepa_verdict": "Stage 2", "entry": 100, "stop": 90, "target": 120},
        {"symbol": "SEPA1", "sepa_score": 100, "sepa_verdict": "Stage 2", "entry": 100, "stop": 90, "target": 120},
        {"symbol": "SKIPPED", "sepa_score": 70, "sepa_verdict": "Stage 2", "entry": 100, "stop": 90, "target": 120},
    ]
    frames = {"TAKEN": _win_frame(), "SEPA1": _win_frame(), "SKIPPED": _loss_frame()}
    report = build_report(
        cycle,
        scan={"scanned_at": "2026-08-01T00:00:00+00:00", "records": sepa},
        sepa_cards=sepa,
        sepa_note="overlay",
        as_of="2026-08-01",
        slot="intraday",
        load_frame=lambda symbol: frames[symbol],
    )
    assert [row["symbol"] for row in report["taken"]] == ["TAKEN"]
    assert report["skipped"][0]["symbol"] == "SKIPPED"
    by_symbol = {row["symbol"]: row for row in report["sepa_best"]}
    assert by_symbol["TAKEN"]["paper_status"] == "taken"
    assert by_symbol["SEPA1"]["paper_status"] == "not_signaled"
    assert by_symbol["SKIPPED"]["paper_status"] == "skipped"
    assert by_symbol["SEPA1"]["not_a_buy"] is True
    tests = {row["symbol"]: row for row in report["candidate_tests"]}
    assert tests["SEPA1"]["outcome"] == "WIN"
    assert tests["SEPA1"]["n_forward_bars"] == 3
    assert tests["SKIPPED"]["outcome"] == "LOSS"
    assert "SEPA1" in report["shadow_prefer"]
    assert "SKIPPED" not in report["shadow_prefer"]
    assert report["live_locked"] is True
    assert NOT_A_SAMPLE in report["disclaimer"]


def test_shadow_without_levels_is_honest():
    got = shadow_setup("NONE", entry=0, stop=0, target=0, as_of="2026-08-01", load_frame=lambda _: None)
    assert got["outcome"] == "NO_LEVELS"
    assert got["n_forward_bars"] == 0
    assert got["r_multiple"] is None


def test_shadow_without_bars_is_honest():
    got = shadow_setup("NONE", entry=100, stop=90, target=120, as_of="2026-08-01", load_frame=lambda _: None)
    assert got["outcome"] == "NO_BARS"
    assert got["n_forward_bars"] == 0


def test_missed_sepa_win_is_preferred_only_among_existing_signals(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_PAPER_MEMORY", str(tmp_path / "paper_memory.json"))
    monkeypatch.setenv("QT_PAPER_SELF_FEED", str(tmp_path / "self_feed.json"))
    monkeypatch.setenv("QT_PAPER_SELF_FEED_DIR", str(tmp_path / "reports"))
    cycle = {
        "as_of_date": "2026-08-01",
        "positions_opened": [("s1", "AAA")],
        "blocked_target_positions": [],
    }
    sepa = [{"symbol": "SEPA1", "sepa_score": 100, "entry": 100, "stop": 90, "target": 120}]
    report = ingest_paper_cycle(
        cycle,
        as_of="2026-08-01",
        slot="intraday",
        scan={"records": sepa, "scanned_at": "2026-08-01T00:00:00+00:00"},
        sepa_cards=sepa,
        load_frame=lambda _: _win_frame(),
    )
    assert report["shadow_prefer"] == ["SEPA1"]
    memory = attach_to_paper_memory(build_paper_memory([], as_of="2026-08-01"), report)
    picked, skipped = select_paper_signal(
        [{"symbol": "AAA", "entry": 1}, {"symbol": "SEPA1", "entry": 1}],
        memory,
        as_of="2026-08-01",
    )
    assert picked["symbol"] == "SEPA1"
    assert skipped == ()
    none, _ = select_paper_signal([{"symbol": "AAA", "entry": 1}], memory, as_of="2026-08-01")
    assert none["symbol"] == "AAA"


def test_save_report_appends_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_PAPER_SELF_FEED", str(tmp_path / "latest.json"))
    monkeypatch.setenv("QT_PAPER_SELF_FEED_DIR", str(tmp_path / "reports"))
    path = save_report({"schema_version": 1, "as_of": "2026-08-01", "taken": [], "skipped": []})
    assert path.exists()
    log = tmp_path / "reports" / "2026-08-01.jsonl"
    assert log.exists()
    assert "2026-08-01" in log.read_text(encoding="utf-8")


def test_jobs_and_learning_wire_self_feed():
    root = Path(__file__).resolve().parents[1]
    jobs = (root / "research" / "autonomy" / "jobs.py").read_text(encoding="utf-8")
    loop = (root / "research" / "autonomy" / "research_loop.py").read_text(encoding="utf-8")
    feed = (root / "product" / "paper_self_feed.py").read_text(encoding="utf-8")
    assert "ingest_paper_cycle" in jobs
    assert "PYTEST_CURRENT_TEST" in jobs
    assert "fold_latest_into_memory" in loop
    assert "place_order" not in feed
    assert "place_order" not in jobs
