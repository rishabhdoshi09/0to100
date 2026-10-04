"""Real store boundaries behind the October whole-system review findings."""
from datetime import datetime
import json
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

SCAN = "2026-10-04T08:00:00+00:00"


def _candidate_db(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("CREATE TABLE candidates (symbol TEXT, scan_run_id TEXT, updated_at TEXT, decision TEXT, decision_id TEXT, recommendation_id TEXT, reason TEXT)")
    con.execute("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("AAA", SCAN, SCAN, "BUY", f"d|{SCAN}", f"{SCAN}:AAA:high_conviction", "COMMITTEE_BUY"))
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return con


def test_current_verdict_changes_on_wal_commit_without_db_mtime_change(tmp_path, monkeypatch):
    from product import recommendation_truth as truth
    path = tmp_path / "candidates.db"
    monkeypatch.setattr(truth.CL, "DB_PATH", path)
    writer = _candidate_db(path)
    try:
        assert truth.candidates_for_current_scan(SCAN)["AAA"]["decision"] == "BUY"
        before = path.stat().st_mtime_ns
        writer.execute("UPDATE candidates SET decision='WAIT', reason='WAIT_EVIDENCE'")
        writer.commit()
        assert path.stat().st_mtime_ns == before
        assert truth.candidates_for_current_scan(SCAN)["AAA"]["decision"] == "WAIT"
    finally:
        writer.close()


def test_presentation_read_never_migrates_schema_or_waits_for_writer(tmp_path, monkeypatch):
    from product import recommendation_truth as truth
    path = tmp_path / "legacy.db"
    monkeypatch.setattr(truth.CL, "DB_PATH", path)
    writer = _candidate_db(path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        assert truth.candidates_for_current_scan(SCAN)["AAA"]["decision"] == "BUY"
        columns = {r[1] for r in writer.execute("PRAGMA table_info(candidates)")}
        assert "opportunity_id" not in columns
        writer.rollback()
    finally:
        writer.close()


def test_missing_committee_store_is_not_created(tmp_path, monkeypatch):
    from product import recommendation_truth as truth
    path = tmp_path / "missing" / "candidates.db"
    monkeypatch.setattr(truth.CL, "DB_PATH", path)
    assert truth.candidates_for_current_scan(SCAN) == {}
    assert not path.parent.exists()


def _workspace():
    return {"scan_scanned_at": SCAN, "categories": [{"cards": [
        {"symbol": "AAA", "reco_tier": "high_conviction", "action_badge": "Buy", "score": 88},
    ]}]}


def test_telegram_and_adapter_follow_current_wait_not_research_buy(tmp_path, monkeypatch):
    from product import recommendation_truth as truth
    from product.decision_adapter import decision_from_card
    from research.autonomy.telegram_notifications import TelegramNotifier
    path = tmp_path / "candidates.db"
    monkeypatch.setattr(truth.CL, "DB_PATH", path)
    monkeypatch.setattr(truth, "current_scan_run_id", lambda: SCAN)
    writer = _candidate_db(path)
    writer.execute("UPDATE candidates SET decision='WAIT', reason='WAIT_EVIDENCE'")
    writer.commit()
    messages = []
    engine = SimpleNamespace(is_configured=lambda: True, send=lambda text: messages.append(text) or True)
    notifier = TelegramNotifier(tmp_path, engine_factory=lambda: engine, now_fn=lambda: datetime(2026, 10, 4, 13))
    try:
        assert notifier.notify_recommendations(_workspace())["sent"]
        assert "· Wait" in messages[0] and "WAIT_EVIDENCE" in messages[0]
        assert "· Buy" not in messages[0]
        card = truth.project_workspace_truth(_workspace())["categories"][0]["cards"][0]
        assert decision_from_card(card, source_scan_id=SCAN).state == "WAIT"
        older = truth.project_workspace_truth({**_workspace(), "scan_scanned_at": "old"})
        assert older["categories"][0]["cards"][0]["canonical_decision"] == "NO_JUDGMENT"
    finally:
        writer.close()


def test_stale_cash_notifier_cannot_erase_fno_delivery_keys(tmp_path):
    from research.autonomy.telegram_notifications import TelegramNotifier
    now = lambda: datetime(2026, 10, 4, 13)
    cash = TelegramNotifier(tmp_path, now_fn=now)
    fno = TelegramNotifier(tmp_path, now_fn=now)
    fno._mark_sent(["fno_open:position1"])
    cash._mark_sent(["reco_desk:scan1"])
    restarted = TelegramNotifier(tmp_path, now_fn=now)
    assert restarted._was_sent("fno_open:position1")
    assert restarted._was_sent("reco_desk:scan1")


@pytest.fixture
def us_scan(tmp_path, monkeypatch):
    import scan.us_scanner as scanner
    import data.us_data as data
    import scan.unified_scanner as unified
    import execution.us_autopilot as autopilot
    monkeypatch.setattr(scanner, "_STORE", tmp_path / "us_scan.json")
    monkeypatch.setattr(scanner, "_scan_running", False)
    monkeypatch.setattr(scanner, "_index_universe", lambda index: (["AAA", "BBB"], "S&P 500"))
    monkeypatch.setattr(scanner, "_rank", lambda rows: [])
    monkeypatch.setattr(scanner, "_push_us_setups", lambda rows: None)
    monkeypatch.setattr(data, "sp500_return_30d", lambda: 0)
    monkeypatch.setattr(unified.UnifiedScanner, "_analyze", lambda self, sym, df: None)
    monkeypatch.setattr(autopilot, "review_cycle", lambda: None)
    monkeypatch.setattr(autopilot, "on_setups", lambda rows: None)
    return scanner, data, unified


@pytest.mark.parametrize("count, expected", [(0, "error"), (1, "partial"), (2, "ready")])
def test_us_scan_distinguishes_missing_data_from_complete_no_signal(us_scan, monkeypatch, count, expected):
    scanner, data, _ = us_scan
    frame = pd.DataFrame({"close": [100.] * 60})
    monkeypatch.setattr(data, "get_us_daily_batch", lambda syms: {sym: frame for sym in syms[:count]})
    assert scanner.scan_us(index="S&P 500") == []
    artifact = scanner.persisted_us_scan()
    assert artifact["status"] == expected
    assert artifact["coverage"]["requested"] == 2
    assert artifact["coverage"]["evaluated"] == count
    assert artifact["coverage"]["missing_history"] == 2 - count
    assert artifact["coverage"]["no_signal"] == count


def test_us_batch_failure_replaces_old_success_and_blocks_operation(us_scan, monkeypatch):
    scanner, data, _ = us_scan
    from operations.market_ops import MarketOperationsWorker, OperationBlocked
    scanner._persist_results([{"symbol": "OLD"}], scope="All")
    def unavailable(_symbols):
        raise TimeoutError("upstream unavailable")
    monkeypatch.setattr(data, "get_us_daily_batch", unavailable)
    worker = SimpleNamespace(_progress=lambda *a: None)
    with pytest.raises(OperationBlocked, match="usable new"):
        MarketOperationsWorker._run_us_market_scan(worker, {"operation_id": "test", "payload": {}})
    artifact = scanner.persisted_us_scan()
    assert artifact["status"] == "error" and artifact["records"] == []
    assert artifact["coverage"]["batch_failed"] == 2


def test_us_analysis_errors_are_visible_not_counted_as_evaluated(us_scan, monkeypatch):
    scanner, data, unified = us_scan
    frame = pd.DataFrame({"close": [100.] * 60})
    monkeypatch.setattr(data, "get_us_daily_batch", lambda syms: {sym: frame for sym in syms})
    def fail(self, sym, frame):
        raise ValueError("invalid OHLC")
    monkeypatch.setattr(unified.UnifiedScanner, "_analyze", fail)
    scanner.scan_us()
    coverage = scanner.persisted_us_scan()["coverage"]
    assert coverage["history_available"] == 2 and coverage["evaluated"] == 0
    assert coverage["analysis_failed"] == 2


def _net_row(i, **overrides):
    from product.us_learning import OUTCOME_BASIS
    return {"decision_id": f"d{i}", "settled": True, "action": "TAKE",
            "evidence_class": "US_PAPER_FORWARD", "outcome_basis": OUTCOME_BASIS,
            "not_pnl": False, "outcome_R": 1., "score_bucket": "80+",
            "conviction_bucket": "70+", **overrides}


@pytest.mark.parametrize("overrides", [
    {"action": "REJECT", "evidence_class": "US_FORWARD_COUNTERFACTUAL", "not_pnl": True},
    {"outcome_basis": "GROSS"}, {"evidence_class": "HISTORICAL_REPLAY"},
    {"outcome_R": float("nan")}, {"outcome_R": float("inf")},
])
def test_uncertified_and_counterfactual_outcomes_cannot_promote(tmp_path, monkeypatch, overrides):
    import product.us_learning as learning
    monkeypatch.setattr(learning, "MODEL", tmp_path / "model.json")
    model = learning.rebuild_model(rows=[_net_row(i, **overrides) for i in range(50)])
    assert model["ranking_eligible_decisions"] == 0
    assert not model["mature_cells"]


def test_duplicate_decisions_do_not_inflate_samples_and_conflicts_quarantine(tmp_path, monkeypatch):
    import product.us_learning as learning
    monkeypatch.setattr(learning, "MODEL", tmp_path / "model.json")
    model = learning.rebuild_model(rows=[_net_row(1)] * 50)
    assert model["ranking_eligible_decisions"] == 1 and not model["mature_cells"]
    assert learning._ranking_rows([_net_row(1), _net_row(1, outcome_R=-1.)]) == []


@pytest.mark.parametrize("journal_stop", [95., 105.])
def test_closed_trade_learning_uses_actual_entry_quantity_and_net_costs(monkeypatch, journal_stop):
    import execution.us_autopilot as autopilot
    import product.us_learning as learning
    from execution.cost_model import net_result
    trade = {"symbol": "AAA", "note": "session=2026-10-02|paper", "status": autopilot._WIN[0],
             "entry_price": 100., "exit_price": 100.01, "stop_price": journal_stop, "qty": 10}
    monkeypatch.setattr(autopilot, "_trades", lambda statuses: [trade])
    outcome = learning._closed_trade_outcome({"symbol": "AAA", "session_date": "2026-10-02", "action": "TAKE", "entry": 99., "stop": 95.})
    expected = net_result(100., 100.01, 10, market="US")["net"] / 50.
    assert outcome[0] == pytest.approx(expected) and outcome[0] < 0


def test_legacy_model_is_invalidated_and_runtime_ledger_path_is_dynamic(tmp_path, monkeypatch):
    import product.us_learning as learning
    monkeypatch.setattr(learning, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(learning, "MODEL", tmp_path / "model.json")
    learning._write_jsonl([_net_row(1)])
    assert learning._read_jsonl()[0]["decision_id"] == "d1"
    learning.MODEL.write_text(json.dumps({"schema_version": 1, "cells": {"score:80+": {"adjustment": 3}}}))
    model = learning.load_model()
    assert model["schema_version"] == 2 and not model["mature_cells"]


def test_fno_batch_reads_one_coherent_evidence_generation(monkeypatch):
    import product.conditional_evidence as conditional
    import product.fno_ranking as ranking
    loads, stores = [], []
    snapshot = {"generation": "before"}
    monkeypatch.setattr(conditional, "load", lambda path: loads.append(path) or dict(snapshot))
    def fuse(setup, **kwargs):
        stores.append(kwargs.get("store"))
        snapshot["generation"] = "after"
        return {"adjustment": 0}
    monkeypatch.setattr(ranking, "fuse_fno_ranking_evidence", fuse)
    ranking.rank_fno_candidates([{"setup": {"score": 80}}, {"setup": {"score": 75}}])
    assert len(loads) == 1
    assert stores == [{"generation": "before"}, {"generation": "before"}]
