"""Cycle 2 (21 Nov 2026): QT-06 research-outcome identity, QT-07 GOOD_WAIT recovery, QT-08 sampling stability.

Each defect test fails on d0b668fe. Network-free; every store isolated.
"""
from __future__ import annotations

import pytest


def _qt06_world(tmp_path, monkeypatch, card_over=None):
    from product import scan_store as ss
    from research import feature_store
    from tests.test_paper_autopilot_money_path import _eligible_card, _workspace

    monkeypatch.setattr(feature_store, "_DB_PATH", tmp_path / "features.sqlite3")
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    card = _eligible_card(chase_risk=True, entry_state="extended", **(card_over or {}))
    ws = _workspace([card])
    monkeypatch.setattr(ss, "load_scan", lambda *a, **k: {
        "scanned_at": ws["scan_scanned_at"], "records": [{"symbol": "TCS", "verdict": "BUY"}]})
    return card, ws


def _paper_cycle(ws):
    from product import paper_autopilot as pa
    from research.auto_research.paper_book import PaperBook
    from tests.test_paper_autopilot_money_path import _now

    return pa.run_reco_paper_cycle(book=PaperBook(capital=100_000), workspace=ws, now=_now(),
                                   as_of="2026-09-01", enforce_history=False, persist_journal=False)


def _non_taken(cycle):
    return [r for g in ("rejections", "waits") for r in cycle.get(g) or []]


# ── QT-06 ────────────────────────────────────────────────────────────────────────

def test_qt06_a_rejection_outcome_labels_the_board_observation_it_came_from(tmp_path, monkeypatch):
    from product import decision_service as ds, forward_evidence as fe
    from research import feature_store
    from tests.test_learning_exactly_once import _settle

    _card, ws = _qt06_world(tmp_path, monkeypatch)
    board = ds.decision_board(workspace=ws)
    board_id = board["decisions"][0]["decision_id"]
    assert feature_store.get_observation(f"decision::{board_id}")["outcome"] is None
    cycle = _paper_cycle(ws)
    (row,) = _non_taken(cycle)
    assert row["feature_decision_id"] == board_id                 # one identity, board == paper
    fe.freeze_cycle({**cycle, "as_of": "2026-09-01", "cycle_id": "c1"})
    (fwd,) = fe.load_ledger()
    assert fwd["feature_decision_id"] == board_id and fwd["decision_id"] != board_id
    _settle()
    assert feature_store.get_observation(f"decision::{board_id}")["outcome"] == pytest.approx(-0.4)


def test_qt06_an_outcome_never_labels_another_symbols_observation(tmp_path, monkeypatch):
    from product import paper_learning_loop as PLL
    from product.counterfactual_learning import settle_rulebook
    from research import feature_store
    from tests.evidence_fixtures import rulebook_outcome

    monkeypatch.setattr(feature_store, "_DB_PATH", tmp_path / "features.sqlite3")
    feature_store.snapshot("decision::dec_other", "INFY", "DECISION", {}, ts="2026-09-01",
                           meta={"evidence_class": "PAPER_FORWARD"})
    settled = settle_rulebook({"symbol": "TCS", "as_of": "2026-09-01", "reason_code": "ENTRY_TOO_EXTENDED",
                               "evidence": {"feature_decision_id": "dec_other"}}, rulebook_outcome(-0.4, "LOSS"))
    out = PLL.ingest_counterfactual(settled)
    assert out["feature_link"] == "LINK_CONFLICT_SYMBOL"
    assert feature_store.get_observation("decision::dec_other")["outcome"] is None


def test_qt06_a_forward_outcome_never_labels_a_simulated_observation(tmp_path, monkeypatch):
    from product import paper_learning_loop as PLL
    from product.counterfactual_learning import settle_rulebook
    from research import feature_store
    from tests.evidence_fixtures import rulebook_outcome

    monkeypatch.setattr(feature_store, "_DB_PATH", tmp_path / "features.sqlite3")
    feature_store.snapshot("decision::dec_replay", "TCS", "DECISION", {}, ts="2026-09-01",
                           meta={"evidence_class": "HISTORICAL_REPLAY"})
    settled = settle_rulebook({"symbol": "TCS", "as_of": "2026-09-01", "reason_code": "ENTRY_TOO_EXTENDED",
                               "evidence": {"feature_decision_id": "dec_replay"}}, rulebook_outcome(-0.4, "LOSS"))
    assert PLL.ingest_counterfactual(settled)["feature_link"] == "LINK_CONFLICT_EVIDENCE_CLASS"
    assert feature_store.get_observation("decision::dec_replay")["outcome"] is None


def test_qt06_another_scans_observation_is_a_conflict_and_a_legacy_row_is_unlinked(tmp_path, monkeypatch):
    from product import paper_learning_loop as PLL
    from product.counterfactual_learning import settle_rulebook
    from research import feature_store
    from tests.evidence_fixtures import rulebook_outcome

    monkeypatch.setattr(feature_store, "_DB_PATH", tmp_path / "features.sqlite3")
    feature_store.snapshot("decision::dec_a", "TCS", "DECISION", {}, ts="2026-09-01",
                           meta={"evidence_class": "PAPER_FORWARD", "source_scan_id": "scan-A"})
    other_scan = settle_rulebook({"symbol": "TCS", "as_of": "2026-09-01", "reason_code": "R1",
                                  "evidence": {"feature_decision_id": "dec_a", "scan_scanned_at": "scan-B"}},
                                 rulebook_outcome(-0.4, "LOSS"))
    assert PLL.ingest_counterfactual(other_scan)["feature_link"] == "LINK_CONFLICT_SCAN"
    legacy = settle_rulebook({"symbol": "TCS", "as_of": "2026-09-02", "reason_code": "R1",
                              "forward_decision_id": "2026-09-02|TCS|REJECTED|R1|c9"},
                             rulebook_outcome(-0.4, "LOSS"))
    assert PLL.ingest_counterfactual(legacy)["feature_link"] == "NO_FEATURE_OBSERVATION"
    assert feature_store.get_observation("decision::dec_a")["outcome"] is None


def test_qt06_the_outbox_records_whether_the_research_label_landed(tmp_path, monkeypatch):
    from product import decision_service as ds, forward_evidence as fe, learning_observations as LO
    from tests.test_learning_exactly_once import _ledger, _settle

    _card, ws = _qt06_world(tmp_path, monkeypatch)
    ds.decision_board(workspace=ws)
    fe.freeze_cycle({**_paper_cycle(ws), "as_of": "2026-09-01", "cycle_id": "c1"})
    _settle()
    import sqlite3
    conn = sqlite3.connect(_ledger())
    (result,) = [r[0] for r in conn.execute("SELECT result FROM outbox")]
    conn.close()
    assert result == "applied;feature=VERIFIED:settled"


# ── QT-07 ────────────────────────────────────────────────────────────────────────

def _qt07_wait(monkeypatch, tmp_path, days):
    from core import session_window
    from product import autonomous_loop as loop
    from product import counterfactual_learning as cf

    monkeypatch.setattr(session_window, "official_dates", lambda: days)
    monkeypatch.setattr(loop, "LEARNING_PATH", tmp_path / "official-learning.jsonl")
    monkeypatch.setattr(loop.CL, "get", lambda *a, **k: None)
    cf.freeze_decision(symbol="TCS", decision="WAIT", reason_code="WAIT_FOR_PULLBACK",
                       entry=100, stop=94, target=115, as_of="2026-09-01")


def _cf_rows():
    import json
    from product import counterfactual_learning as cf
    return [json.loads(line) for line in cf.ledger_path().read_text().splitlines() if line.strip()]


def test_qt07_good_wait_replays_the_original_fill_after_archive_recovery(tmp_path, monkeypatch):
    from core import outcome_resolver
    from product import autonomous_loop as loop, paper_learning_loop as pll
    from tests.evidence_fixtures import rulebook_outcome
    from tests.test_learning_exactly_once import _n

    days: list[str] = []
    _qt07_wait(monkeypatch, tmp_path, days)
    assert pll.note_later_entry("TCS", as_of="2026-09-03")["updated"] == 0     # archive unavailable
    assert _n("ENTRY_QUALITY::GOOD_WAIT") == 0                                # never credited unverified
    days.extend(["2026-09-01", "2026-09-02", "2026-09-03"])
    monkeypatch.setattr(outcome_resolver, "counterfactual_outcome", lambda *a, **k: rulebook_outcome(-0.4, "LOSS"))
    loop.settle_official_outcomes("2026-09-18")
    (row,) = _cf_rows()
    assert row["classification"] == "GOOD_WAIT" and row["later_entry_verified"] == "2026-09-03"
    assert _n("ENTRY_QUALITY::GOOD_WAIT") == 1


def test_qt07_a_recovered_fill_outside_the_horizon_is_not_credited(tmp_path, monkeypatch):
    import pandas as pd
    from core import outcome_resolver
    from product import autonomous_loop as loop, paper_learning_loop as pll
    from tests.evidence_fixtures import rulebook_outcome
    from tests.test_learning_exactly_once import _n

    days: list[str] = []
    _qt07_wait(monkeypatch, tmp_path, days)
    pll.note_later_entry("TCS", as_of="2026-09-25")
    days.extend(d.date().isoformat() for d in pd.bdate_range("2026-09-01", "2026-09-30"))
    monkeypatch.setattr(outcome_resolver, "counterfactual_outcome", lambda *a, **k: rulebook_outcome(-0.4, "LOSS"))
    out = loop.settle_official_outcomes("2026-09-30")
    (row,) = _cf_rows()
    assert out["good_waits"]["expired"] == 1 and row["classification"] != "GOOD_WAIT"
    assert row["later_entry_expired"] == "2026-09-25" and _n("ENTRY_QUALITY::GOOD_WAIT") == 0


def test_qt07_a_still_unverifiable_fill_holds_the_wait_off_its_price_path(tmp_path, monkeypatch):
    from core import outcome_resolver
    from product import autonomous_loop as loop, paper_learning_loop as pll
    from tests.evidence_fixtures import rulebook_outcome

    _qt07_wait(monkeypatch, tmp_path, [])
    pll.note_later_entry("TCS", as_of="2026-09-03")
    monkeypatch.setattr(outcome_resolver, "counterfactual_outcome", lambda *a, **k: rulebook_outcome(-0.4, "LOSS"))
    out = loop.settle_official_outcomes("2026-09-18")
    (row,) = _cf_rows()
    assert not row.get("classification") and row["later_entry_pending"]["fill_session"] == "2026-09-03"
    assert any(p.get("reason") == "LATER_ENTRY_UNVERIFIED" for p in out["pending"])
    dropped = pll.retry_pending_good_waits(today="2026-11-30")            # 45+ days: dropped, not credited
    assert dropped["dropped_unverifiable"] == 1 and not _cf_rows()[0].get("classification")


# ── QT-08: an append-only history never changes the earlier measured sample ──────

def _grid_run(monkeypatch, frames, dates, *, grid, step=5, lookback=40):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from data import bhavcopy_store as BS
    from research.intelligence.data import nse_calendar as CAL
    from scan import calibration_snapshot as CS, signal_backtest as SB
    from scan.unified_scanner import UnifiedScanner

    holidays = set(json.loads(Path("data/nse_holidays.json").read_text()))
    plan = SimpleNamespace(signals=["VCP"], entry=100., stop=90., target=120., confidence=0.0)
    monkeypatch.setattr(CAL, "load_holidays", lambda: holidays)
    monkeypatch.setattr(BS, "store_symbols", lambda: sorted(frames))
    monkeypatch.setattr(BS, "_dates_on_disk", lambda: list(dates.date))
    monkeypatch.setattr(BS, "get_ohlcv", lambda s: frames.get(s))
    monkeypatch.setattr(CS, "get_or_create_snapshot", lambda: {"snapshot_id": "neutral"})
    monkeypatch.setattr(SB, "_nifty_regime_series", lambda: None)
    monkeypatch.setattr(SB, "_calibration_version", lambda: "test")
    monkeypatch.setattr(UnifiedScanner, "_analyze", lambda self, sym, hist: plan)
    trades = []
    SB.run_backtest(sample_step=step, lookback_sessions=lookback, max_symbols=len(frames),
                    on_trade=trades.append, decision_grid=grid)
    return {(t["symbol"], str(t["entry_datetime"])[:10]): t["net_R"] for t in trades}


def test_qt08_session_anchored_grid_keeps_every_earlier_decision_on_append(monkeypatch):
    import numpy as np
    import pandas as pd
    from scan import signal_backtest as SB
    from tests.test_stopped_symbol_censoring import _dates

    dates = _dates()
    rng = np.random.default_rng(3)
    full = {}
    for k in range(3):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        full[f"S{k}"] = pd.DataFrame({"open": c, "high": c * 1.03, "low": c * 0.97, "close": c,
                                      "volume": 1e6}, index=dates)
    cut = {s: df.iloc[:-3] for s, df in full.items()}                   # the same history, 3 sessions earlier
    anchored_cut = _grid_run(monkeypatch, cut, dates[:-3], grid=SB.GRID_SESSION_ANCHORED)
    anchored_full = _grid_run(monkeypatch, full, dates, grid=SB.GRID_SESSION_ANCHORED)
    lo = max(min(d for _s, d in anchored_cut), min(d for _s, d in anchored_full))
    kept = {k: v for k, v in anchored_cut.items() if k[1] >= lo}
    assert kept and all(anchored_full.get(k) == v for k, v in kept.items())  # nested, identical outcomes
    rolling_cut = _grid_run(monkeypatch, cut, dates[:-3], grid=SB.GRID_ROLLING)
    rolling_full = _grid_run(monkeypatch, full, dates, grid=SB.GRID_ROLLING)
    lo_r = max(min(d for _s, d in rolling_cut), min(d for _s, d in rolling_full))
    shared = {k for k in rolling_cut if k[1] >= lo_r} & set(rolling_full)
    assert len(shared) < len({k for k in rolling_cut if k[1] >= lo_r})      # the rolling grid moved


def test_qt08_the_production_program_is_the_anchored_grid_and_a_new_program():
    from scan import signal_backtest as SB

    assert SB.DECISION_GRID == SB.GRID_SESSION_ANCHORED
    assert SB.gate_program()["sampling"] == SB.SAMPLING != "even_stride"   # earlier attestations fail closed


def test_qt08_a_new_listing_displaces_at_most_one_sampled_symbol():
    from scan.signal_backtest import _spread_sample

    store = sorted(f"SYM{i:04d}" for i in range(3469))
    before = set(_spread_sample(store, 800))
    for extra in ("AAAA", "MMMM", "ZZZZ"):
        after = set(_spread_sample(sorted(store + [extra]), 800))
        assert len(before - after) <= 1 and len(after - before) <= 1
