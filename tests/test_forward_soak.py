"""Phase 12 — real forward paper-trading soak.

These tests prove the ledger, provenance split, settlement, soak health,
daily report, restart continuity, and promotion lock. They do not invent a
second scanner or trading path.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from product.forward_evidence import (
    BACKTEST,
    REAL_FORWARD_MARKET,
    TEST_FIXTURE,
    attach_settlement,
    freeze_cycle,
    freeze_observation,
    load_ledger,
    real_forward_only,
)
from product.forward_soak import (
    BLOCKED,
    COLLECTING,
    HEALTHY,
    NOT_STARTED,
    attach_closed_trades,
    build_runtime_journey,
    scoreboard,
    settle_and_report,
    settle_pending_from_market,
    soak_status,
    verify_persisted_soak,
    write_daily_report,
)
from product.paper_autopilot import DD_GATE_FAILED, STALE_RECOMMENDATION, run_reco_paper_cycle
from research.auto_research.paper_book import PaperBook


def _now():
    return datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _eligible_card(symbol="TCS", **over):
    card = {
        "symbol": symbol,
        "reco_tier": "high_conviction",
        "reco_tier_label": "High Conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "family_confirms": 3,
        "score": 82,
        "primary_thesis": "VCP + quality",
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [
            {"id": "tape", "status": "pass", "points": 90, "detail": "clean"},
            {"id": "sepa", "status": "pass", "points": 70, "detail": "base"},
            {"id": "funds", "status": "pass", "points": 80, "detail": "quality"},
            {"id": "trend", "status": "pass", "points": 85, "detail": "up"},
            {"id": "rs", "status": "pass", "points": 75, "detail": "leader"},
            {"id": "ev", "status": "unknown", "points": None, "detail": "n<30"},
            {"id": "conviction", "status": "pass", "points": 80, "detail": "ready"},
            {"id": "case", "status": "unknown", "points": None, "detail": "n<30"},
            {"id": "sector", "status": "pass", "points": 80, "detail": "leader"},
        ],
    }
    card.update(over)
    return card


def _workspace(cards, *, generated_at=None):
    stamp = generated_at or _now().isoformat()
    return {
        "schema_version": 4,
        "generated_at": stamp,
        "scan_scanned_at": stamp,
        "categories": [{"id": "wealth_builders", "count": len(cards), "cards": cards}],
    }


def _write_scan_reco(cards=None, *, as_of="2026-09-01"):
    cards = list(cards or [_eligible_card()])
    stamp = f"{as_of}T10:00:00+00:00"
    Path(os.environ["QT_SCAN_PATH"]).write_text(json.dumps({
        "available": True,
        "records": [{"symbol": c["symbol"], "score": c.get("score", 80)} for c in cards],
        "scanned_at": stamp,
    }), encoding="utf-8")
    Path(os.environ["QT_RECO_PATH"]).write_text(json.dumps({
        "schema_version": 4,
        "generated_at": stamp,
        "scan_scanned_at": stamp,
        "categories": [{"id": "wealth_builders", "count": len(cards), "cards": cards}],
    }), encoding="utf-8")


def _cycle(book, cards, **kwargs):
    kwargs.setdefault("now", _now())
    kwargs.setdefault("as_of", "2026-09-01")
    kwargs.setdefault("entries_allowed", True)
    kwargs.setdefault("paper_enabled", True)
    kwargs.setdefault("persist_journal", True)
    kwargs.setdefault("workspace", _workspace(cards))
    kwargs.setdefault("enforce_history", False)
    return run_reco_paper_cycle(book=book, cards=cards, **kwargs)


def test_money_path_freezes_forward_ledger_with_cycle_id():
    _write_scan_reco()
    book = PaperBook(capital=100_000)
    out = _cycle(book, [_eligible_card()])
    assert out["cycle_id"]
    assert out["taken"]
    rows = load_ledger()
    taken = [r for r in rows if r.get("entered")]
    assert taken
    assert taken[0]["symbol"] == "TCS"
    assert taken[0]["provenance"] == TEST_FIXTURE
    assert taken[0]["cycle_id"] == out["cycle_id"]
    assert taken[0]["pit_proof"]["future_data_used_for_decision"] is False
    assert taken[0]["later_outcome"] is None


def test_ledger_does_not_overwrite_pit_fields_on_duplicate_freeze():
    row = {
        "symbol": "INFY",
        "reason_code": "ELIGIBLE",
        "decision": "ENTER_NOW",
        "tier": "high_conviction",
        "entry": 100,
        "stop": 94,
        "target": 115,
        "setup_label": "VCP",
        "sector": "Technology",
        "regime": "RISK_ON",
    }
    first = freeze_observation(
        row, cycle_id="c1", as_of="2026-09-01", rules_hash="abc",
        group="taken", entered=True, provenance=REAL_FORWARD_MARKET,
    )
    mutated = dict(row)
    mutated["entry"] = 999
    mutated["setup_label"] = "CHANGED"
    second = freeze_observation(
        mutated, cycle_id="c1", as_of="2026-09-01", rules_hash="abc",
        group="taken", entered=True, provenance=REAL_FORWARD_MARKET,
    )
    assert first["decision_id"] == second["decision_id"]
    assert second["entry"] == 100
    assert second["setup"] == "VCP"
    assert len(load_ledger()) == 1


def test_test_fixture_rows_are_excluded_from_promotion_stats():
    freeze_observation(
        {"symbol": "TCS", "reason_code": "ELIGIBLE", "decision": "ENTER_NOW", "entry": 100, "stop": 94},
        cycle_id="t1", as_of="2026-09-01", group="taken", entered=True, provenance=TEST_FIXTURE,
    )
    freeze_observation(
        {"symbol": "INFY", "reason_code": "ELIGIBLE", "decision": "ENTER_NOW", "entry": 100, "stop": 94},
        cycle_id="r1", as_of="2026-09-01", group="taken", entered=True, provenance=REAL_FORWARD_MARKET,
    )
    board = scoreboard()
    assert board["provenance_filter"] == REAL_FORWARD_MARKET
    assert board["real_forward_observations"] == 1
    assert board["paper_trades_taken"] == 1
    assert board["insufficient_evidence"] is True
    assert board["evidence_label"] == "INSUFFICIENT EVIDENCE"
    assert board["gross_expectancy"] is None


def test_backtest_rows_cannot_contaminate_real_forward_stats():
    freeze_observation(
        {"symbol": "RELIANCE", "reason_code": "ELIGIBLE", "decision": "BUY", "entry": 100, "stop": 94},
        cycle_id="hist", as_of="2025-12-24", group="taken", entered=True, provenance=BACKTEST,
    )
    rows = load_ledger()
    assert any(r.get("provenance") == BACKTEST for r in rows)
    assert real_forward_only(rows) == []
    board = scoreboard()
    assert board["real_forward_observations"] == 0
    assert board["paper_trades_taken"] == 0
    assert board["insufficient_evidence"] is True


def test_no_judgment_is_excluded_from_forward_ledger():
    row = freeze_observation(
        {
            "symbol": "NOTAREALTICKERZZZ",
            "decision": "NO_JUDGMENT",
            "reason_code": "INVALID_SYMBOL",
            "entry": 100,
            "stop": 94,
        },
        cycle_id="bad",
        as_of="2026-09-01",
        group="rejected",
        entered=False,
        provenance=REAL_FORWARD_MARKET,
    )
    assert row is None
    assert load_ledger() == []


def test_rejected_candidates_settle_automatically_and_are_not_pnl():
    from product.counterfactual_learning import MISSED_WINNER

    freeze_cycle({
        "as_of": "2026-09-01",
        "cycle_id": "rej-1",
        "rules_hash": "hash",
        "rejections": [{
            "symbol": "WIPRO",
            "reason_code": "ENTRY_TOO_EXTENDED",
            "decision": "WAIT",
            "entry": 100,
            "stop": 94,
            "target": 115,
            "setup_label": "VCP",
            "regime": "RISK_ON",
        }],
    })
    first = settle_pending_from_market(return_fn=lambda symbol, _as_of: 14.8 if symbol == "WIPRO" else None)
    assert first["updated"] == 1
    assert first["classifications"][MISSED_WINNER] == 1
    second = settle_pending_from_market(return_fn=lambda symbol, _as_of: 14.8 if symbol == "WIPRO" else None)
    assert second["updated"] == 0
    row = load_ledger()[0]
    assert row["entered"] is False
    assert row["not_pnl"] is True
    assert row["later_outcome"]["not_pnl"] is True
    assert row["counterfactual_classification"] == MISSED_WINNER


def test_missed_winner_does_not_disable_hard_dd_or_chase_gates():
    freeze_cycle({
        "as_of": "2026-09-01",
        "cycle_id": "hard-1",
        "rejections": [{
            "symbol": "TCS",
            "reason_code": DD_GATE_FAILED,
            "decision": "NO_TRADE",
            "entry": 100,
            "stop": 94,
            "target": 115,
        }],
    })
    settle_pending_from_market(return_fn=lambda symbol, _as_of: 12.0)
    failed = dict(_eligible_card())
    failed["dd_verdict"] = "FAIL"
    blocked = _cycle(PaperBook(capital=100_000), [failed], persist_journal=False)
    assert not blocked["taken"]
    assert blocked["rejections"][0]["reason_code"] == DD_GATE_FAILED

    chased = dict(_eligible_card())
    chased["chase_risk"] = True
    chased["entry_state"] = "extended"
    waited = _cycle(PaperBook(capital=100_000), [chased], persist_journal=False)
    assert not waited["taken"]
    assert waited["waits"][0]["reason_code"] == "ENTRY_TOO_EXTENDED"


def test_closed_paper_trade_attaches_execution_fields_without_repricing_fill():
    _write_scan_reco()
    book = PaperBook(capital=100_000)
    out = _cycle(book, [_eligible_card()])
    pos = next(iter(book.open.values()))
    intended = pos.entry_price
    closed = book.mark({"TCS": (100.0, 116.0, 99.0, 115.0)}, "2026-09-02")
    assert closed
    attached = attach_closed_trades(book)
    assert attached["attached"] == 1
    again = attach_closed_trades(book)
    assert again["attached"] == 0
    row = [r for r in load_ledger() if r.get("entered")][0]
    assert row["gross_R"] is not None
    assert row["later_outcome"] is not None
    assert next(t.entry_price for t in book.closed) == intended


def test_execution_coverage_blocks_execution_sensitive_promotion():
    for i in range(5):
        freeze_observation(
            {"symbol": f"S{i}", "reason_code": "ELIGIBLE", "decision": "ENTER_NOW", "entry": 100, "stop": 94},
            cycle_id=f"cov-{i}", as_of="2026-09-01", group="taken", entered=True,
            provenance=REAL_FORWARD_MARKET,
        )
        attach_settlement(
            f"2026-09-01|S{i}|taken|ELIGIBLE|cov-{i}",
            classification="TARGET",
            gross_R=0.8,
            execution_adjusted_R=0.4 if i < 2 else None,
            execution_coverage=1.0 if i < 2 else 0.0,
            outcome_provenance=REAL_FORWARD_MARKET,
        )
    board = scoreboard()
    assert board["execution_adjusted_coverage_pct"] == 40.0
    assert board["evidence_label"] == "INSUFFICIENT EVIDENCE"
    components = {c["component"]: c for c in board["promotion_blockers"]["components"]}
    assert components["execution_reality_fills"]["decision"] == "KEEP_SHADOW"
    assert "EXECUTION_EVIDENCE_INCOMPLETE" in components["execution_reality_fills"]["blockers"]
    assert components["live_money"]["decision"] == "KEEP_SHADOW"
    assert components["regime_intelligence_2"]["decision"] == "KEEP_SHADOW"
    assert components["ml_challenger"]["decision"] == "KEEP_SHADOW"
    assert board["live_locked"] is True


def test_process_alive_is_not_healthy(monkeypatch):
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda *a, **k: {"running": True},
    )
    status = soak_status()
    assert status["process_alive_is_not_healthy"] is True
    assert status["status"] in {NOT_STARTED, BLOCKED, COLLECTING}
    assert status["status"] != HEALTHY


def test_journey_and_verifier_from_persisted_artifacts(monkeypatch):
    _write_scan_reco()
    book = PaperBook(capital=100_000)
    out = _cycle(book, [_eligible_card()])
    assert out["taken"]
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda *a, **k: {"running": True},
    )
    journey = build_runtime_journey()
    names = [s["name"] for s in journey["stages"]]
    assert names[0] == "DATA_REFRESH"
    assert names[-1] == "NEXT_CYCLE_POLICY_CONSUMPTION"
    for stage in journey["stages"]:
        assert stage["timestamp"]
        assert "status" in stage
        assert "input_artifact" in stage
        assert "output_artifact" in stage
        assert "reason_code" in stage
    assert journey["summary"]["MARKET_SCAN"] == "PASS"
    assert journey["summary"]["RECOMMENDATIONS"] == "PASS"
    assert journey["summary"]["SELECTION_AUTHORITY"] == "PASS"
    assert journey["summary"]["PAPER_EXECUTION"] == "PASS"
    verified = verify_persisted_soak()
    assert verified["lanes"]["SCAN"] == "PASS"
    assert verified["lanes"]["RECOMMENDATIONS"] == "PASS"
    assert verified["lanes"]["SELECTION"] == "PASS"
    assert verified["lanes"]["AUTOPILOT"] == "PASS"
    assert verified["lanes"]["PAPER EXECUTION"] == "PASS"
    assert verified["lanes"]["LIVE MONEY"] == "LOCKED"
    assert verified["live_locked"] is True


def test_valid_no_trade_day_is_not_a_failure(monkeypatch):
    today = datetime.now(timezone.utc).date().isoformat()
    clock = datetime.now(timezone.utc)
    failed = dict(_eligible_card())
    failed["dd_verdict"] = "FAIL"
    _write_scan_reco([failed], as_of=today)
    book = PaperBook(capital=100_000)
    out = _cycle(book, [failed], now=clock, as_of=today)
    assert not out["taken"]
    assert out["rejections"]
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda *a, **k: {"running": True},
    )
    verified = verify_persisted_soak()
    assert verified["valid_no_trade"] is True
    assert verified["lanes"]["PAPER EXECUTION"] == "NO_ELIGIBLE_TRADE"
    assert verified["lanes"]["LIVE MONEY"] == "LOCKED"
    status = soak_status()
    assert status["status"] in {HEALTHY, COLLECTING}



def test_fresh_zero_action_discovery_is_valid_no_trade_without_paper_cycle(monkeypatch):
    today = datetime.now(timezone.utc).date().isoformat()
    stamp = f"{today}T10:00:00+00:00"
    rejected = dict(_eligible_card())
    rejected["dd_verdict"] = "FAIL"
    _write_scan_reco([rejected], as_of=today)

    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {
            "discovery_ready": True,
            "scan_fresh": True,
            "scan_scanned_at": stamp,
            "actionable": 0,
            "decision_count": 1,
            "approved": True,
        },
    )
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda *a, **k: {"running": True},
    )

    journey = build_runtime_journey()
    assert journey["valid_no_trade"] is True
    assert journey["summary"]["SELECTION_AUTHORITY"] == "PASS"
    assert journey["summary"]["PAPER_EXECUTION"] == "PASS"
    selection = next(
        stage for stage in journey["stages"]
        if stage["name"] == "SELECTION_AUTHORITY"
    )
    assert selection["reason_code"] == "NO_ELIGIBLE_TRADE"
    assert selection["output_artifact"].endswith("startup_trade_discovery.json")

    verified = verify_persisted_soak()
    assert verified["lanes"]["SELECTION"] == "PASS"
    assert verified["lanes"]["AUTOPILOT"] == "PASS"
    assert verified["lanes"]["PAPER EXECUTION"] == "NO_ELIGIBLE_TRADE"
    assert soak_status()["status"] in {HEALTHY, COLLECTING}


def test_actionable_discovery_without_paper_cycle_stays_fail_closed(monkeypatch):
    today = datetime.now(timezone.utc).date().isoformat()
    stamp = f"{today}T10:00:00+00:00"
    _write_scan_reco([_eligible_card()], as_of=today)

    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {
            "discovery_ready": True,
            "scan_fresh": True,
            "scan_scanned_at": stamp,
            "actionable": 1,
            "decision_count": 1,
            "approved": True,
        },
    )

    journey = build_runtime_journey()
    assert journey["valid_no_trade"] is False
    assert journey["summary"]["SELECTION_AUTHORITY"] == "FAIL"
    assert journey["summary"]["PAPER_EXECUTION"] == "FAIL"


def test_daily_report_json_and_markdown_are_written():
    _write_scan_reco()
    book = PaperBook(capital=100_000)
    _cycle(book, [_eligible_card()])
    written = write_daily_report("2026-09-01", book=book)
    json_path = Path(written["json_path"])
    md_path = Path(written["md_path"])
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["market_date"] == "2026-09-01"
    assert payload["scan_count"] >= 1
    assert payload["recommendation_count"] >= 1
    assert payload["trades_entered"] == 1
    assert payload["rules_hash"]
    assert payload["live_locked"] is True
    text = md_path.read_text(encoding="utf-8")
    assert "rules_hash" in text
    assert "live_locked: True" in text
    assert "live_lock_verified: True" in text


def test_settle_and_report_is_idempotent_eod_hook():
    _write_scan_reco()
    book = PaperBook(capital=100_000)
    out = _cycle(book, [_eligible_card()])
    rejected = dict(_eligible_card("WIPRO"))
    rejected["dd_verdict"] = "FAIL"
    _cycle(book, [rejected], persist_journal=True)
    book.mark({"TCS": (100.0, 116.0, 99.0, 115.0)}, "2026-09-02")
    first = settle_and_report(
        "2026-09-02",
        book=book,
        forward_returns={"WIPRO": -6.5},
    )
    second = settle_and_report(
        "2026-09-02",
        book=book,
        forward_returns={"WIPRO": -6.5},
    )
    assert first["closed_attached"]["attached"] == 1
    assert second["closed_attached"]["attached"] == 0
    assert second["counterfactuals"]["updated"] == 0
    assert first["live_locked"] is True
    entered = [r for r in load_ledger() if r.get("entered")]
    assert len(entered) == 1


def test_stale_previous_day_reco_does_not_enter():
    old = (_now() - timedelta(hours=48)).isoformat()
    cards = [_eligible_card()]
    book = PaperBook(capital=100_000)
    out = _cycle(book, cards, workspace=_workspace(cards, generated_at=old), now=_now())
    assert not book.open
    assert STALE_RECOMMENDATION in out["cycle_reasons"]
    entered = [r for r in load_ledger() if r.get("entered")]
    assert entered == []


def test_duplicate_cycle_does_not_duplicate_position_or_ledger_row():
    _write_scan_reco()
    book = PaperBook(capital=100_000)
    first = _cycle(book, [_eligible_card()])
    second = _cycle(book, [_eligible_card()])
    assert len(book.open) == 1
    taken_rows = [r for r in load_ledger() if r.get("entered") and r.get("symbol") == "TCS"]
    assert len(taken_rows) == 1
    assert taken_rows[0]["cycle_id"] == first["cycle_id"]
    assert second["rejections"]


def test_restart_after_exit_before_learning_does_not_lose_or_duplicate_ingest():
    from product.paper_learning_loop import ingest_closed_book

    _write_scan_reco()
    book = PaperBook(capital=100_000)
    _cycle(book, [_eligible_card()])
    book.mark({"TCS": (100.0, 116.0, 99.0, 115.0)}, "2026-09-02")
    first = ingest_closed_book(book)
    second = ingest_closed_book(book)
    assert first["applied"] == 1
    assert second["applied"] == 0
    attach_closed_trades(book)
    attach_closed_trades(book)
    entered = [r for r in load_ledger() if r.get("entered")]
    assert len(entered) == 1
    assert entered[0]["gross_R"] is not None


def test_missing_later_bars_stay_pending():
    freeze_cycle({
        "as_of": "2026-09-01",
        "cycle_id": "pend-1",
        "rejections": [{
            "symbol": "NESTLEIND",
            "reason_code": "LOW_QUALITY_SETUP",
            "decision": "NO_TRADE",
            "entry": 100,
            "stop": 94,
            "target": 115,
        }],
    })
    out = settle_pending_from_market(return_fn=lambda symbol, _as_of: None)
    assert out["updated"] == 0
    assert out["pending"] == 1
    assert load_ledger()[0]["later_outcome"] is None
    assert load_ledger()[0]["counterfactual_classification"] is None


def test_live_adapter_stays_locked_from_verifier():
    from product.execution_adapter import LiveExecutionAdapter, LiveMoneyLocked
    from product.live_readiness import evaluate_live_readiness

    try:
        LiveExecutionAdapter().submit(object())
        raise AssertionError("live adapter must raise")
    except LiveMoneyLocked:
        pass
    ready = evaluate_live_readiness()
    assert ready["live_enabled"] is False
    assert verify_persisted_soak()["lanes"]["LIVE MONEY"] == "LOCKED"


def test_operator_script_prints_required_lanes(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_forward_soak.py"
    spec = importlib.util.spec_from_file_location("verify_forward_soak", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    failed = dict(_eligible_card())
    failed["dd_verdict"] = "FAIL"
    _write_scan_reco([failed])
    _cycle(PaperBook(capital=100_000), [failed])
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda *a, **k: {"running": True},
    )
    code = module.main()
    captured = capsys.readouterr().out
    assert "SCAN: PASS" in captured
    assert "RECOMMENDATIONS: PASS" in captured
    assert "SELECTION: PASS" in captured
    assert "AUTOPILOT: PASS" in captured
    assert "PAPER EXECUTION: NO_ELIGIBLE_TRADE" in captured
    assert "LIVE MONEY: LOCKED" in captured
    assert code == 0


def test_forward_soak_api_is_registered():
    import terminal_product_api_parallel as api
    paths = {getattr(route, "path", "") for route in api.app.routes}
    assert "/api/forward-soak" in paths


def test_completed_scan_with_zero_qualifiers_is_not_scan_failure(monkeypatch):
    stamp = _now().isoformat()
    Path(os.environ["QT_SCAN_PATH"]).write_text(
        json.dumps({
            "schema_version": 2,
            "scanned_at": stamp,
            "scanned": 598,
            "universe_size": 598,
            "records": [],
            "summary": {"qualified": 0, "with_any_setup": 0},
            "provenance": {
                "market_session_date": "2026-09-01",
                "data_current": True,
            },
        }),
        encoding="utf-8",
    )
    Path(os.environ["QT_RECO_PATH"]).write_text(
        json.dumps({
            "schema_version": 4,
            "generated_at": stamp,
            "scan_scanned_at": stamp,
            "categories": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda *a, **k: {"running": True},
    )

    journey = build_runtime_journey()

    assert journey["summary"]["MARKET_SCAN"] == "PASS"
    assert journey["summary"]["RECOMMENDATIONS"] == "PASS"
    verified = verify_persisted_soak()
    assert verified["lanes"]["SCAN"] == "PASS"


def test_rejection_gate_scorecard_keeps_counterfactuals_out_of_pnl():
    from product.forward_soak import rejection_gate_scorecard

    rows = [
        {
            "entered": False,
            "reason_code": "ENTRY_TOO_EXTENDED",
            "counterfactual_classification": "MISSED_WINNER",
            "entry": 100.0,
            "stop": 95.0,
            "later_outcome": {"forward_return_pct": 10.0},
        },
        {
            "entered": False,
            "reason_code": "ENTRY_TOO_EXTENDED",
            "counterfactual_classification": "CORRECT_REJECTION",
            "entry": 100.0,
            "stop": 95.0,
            "later_outcome": {"forward_return_pct": -5.0},
        },
    ]

    card = rejection_gate_scorecard(rows, min_n=2)["ENTRY_TOO_EXTENDED"]

    assert card["n"] == 2
    assert card["MISSED_WINNER"] == 1
    assert card["CORRECT_REJECTION"] == 1
    assert card["missed_winner_rate"] == 0.5
    assert card["protective_rejection_rate"] == 0.5
    assert card["sum_rejected_move_R"] == 1.0
    assert card["mean_rejected_move_R"] == 0.5
    assert card["r_coverage"] == 1.0
    assert card["evidence"] == "MEASURED"
    assert card["not_pnl"] is True


def test_forward_soak_concurrent_publication_uses_distinct_atomic_temps(tmp_path, monkeypatch):
    """Two writers must never share the old forward_soak_verify.json.tmp."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import product.forward_soak as forward

    target = tmp_path / "forward_soak_verify.json"
    barrier = threading.Barrier(2)
    originals = forward.os.replace
    seen = []
    guard = threading.Lock()

    def colliding_replace(source, destination):
        with guard:
            seen.append(str(source))
        barrier.wait(timeout=8)
        return originals(source, destination)

    monkeypatch.setattr(forward.os, "replace", colliding_replace)
    def writer(index):
        return forward._write_json(target, {"schema_version": 1, "writer": index})

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(writer, (1, 2)))

    assert results == [target, target]
    assert len(seen) == 2 and len(set(seen)) == 2
    assert json.loads(target.read_text(encoding="utf-8")) in (
        {"schema_version": 1, "writer": 1},
        {"schema_version": 1, "writer": 2},
    )
    assert not list(tmp_path.glob(".*.tmp"))


def test_forward_soak_failed_rename_preserves_previous_artifact(tmp_path, monkeypatch):
    import product.forward_soak as forward

    target = tmp_path / "forward_journey.json"
    forward._write_json(target, {"version": "previous"})

    def fail_replace(_a, _b):
        raise OSError("simulated atomic-rename error")

    monkeypatch.setattr(forward.os, "replace", fail_replace)
    import pytest
    with pytest.raises(OSError, match="simulated"):
        forward._write_json(target, {"version": "new"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": "previous"}
    assert not list(tmp_path.glob(".*.tmp"))


def test_forward_soak_unmounted_volume_fails_before_any_directory_created(tmp_path, monkeypatch):
    import pytest
    import product.forward_soak as forward

    root = tmp_path / "external" / "QuantTerm" / "runtime"
    root.mkdir(parents=True)
    target = root / "logs" / "product" / "forward_soak_verify.json"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")

    forward._write_json(target, {"status": "OLD"})
    disconnected = root.with_name("runtime.offline")
    root.rename(disconnected)

    with pytest.raises(RuntimeError, match="runtime root is missing"):
        forward._write_json(target, {"status": "NEW"})
    assert not root.exists()
    assert json.loads(
        (disconnected / "logs" / "product" / "forward_soak_verify.json").read_text()
    ) == {"status": "OLD"}
    disconnected.rename(root)
    forward._write_json(target, {"status": "RESTORED"})
    assert json.loads(target.read_text()) == {"status": "RESTORED"}


def test_forward_soak_strict_runtime_blocks_symlink_escape(tmp_path, monkeypatch):
    import pytest
    import product.forward_soak as forward

    root = tmp_path / "external" / "QuantTerm" / "runtime"
    root.mkdir(parents=True)
    outside = tmp_path / "untrusted"
    outside.mkdir()
    (root / "logs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")

    target = root / "logs" / "product" / "forward_journey.json"
    with pytest.raises(RuntimeError, match="outside the persistent runtime"):
        forward._write_json(target, {"sensitive": "never outside"})
    assert not (outside / "product").exists()


def test_concurrent_forward_ledger_freezes_preserve_all_unique_decisions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import product.forward_evidence as evidence

    ledger = tmp_path / "forward_evidence.jsonl"
    workers = 8
    start = Barrier(workers)

    def freeze(n):
        start.wait(timeout=8)
        return evidence.freeze_observation(
            {"symbol": f"TEST{n}", "decision": "ENTER_NOW",
             "reason_code": "ELIGIBLE", "entry": 100, "stop": 94},
            cycle_id=f"cycle-{n}", as_of="2026-10-09",
            group="TAKEN", entered=True, path=ledger,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(freeze, range(workers)))

    assert len(out) == workers
    rows = evidence.load_ledger(ledger)
    assert len(rows) == workers
    assert {r["symbol"] for r in rows} == {f"TEST{i}" for i in range(workers)}
    assert len({r["decision_id"] for r in rows}) == workers

    # Races for the same identity must deduplicate, never overwrite PIT data.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        duplicates = list(pool.map(freeze, range(workers)))
    assert len(duplicates) == workers
    assert len(evidence.load_ledger(ledger)) == workers


def test_freeze_cycle_batches_candidate_publication_into_one_atomic_write(tmp_path, monkeypatch):
    import product.forward_evidence as evidence

    ledger = tmp_path / "forward_evidence.jsonl"
    original = evidence._write_jsonl
    count = []

    def counted_write(path, rows):
        count.append(len(rows))
        return original(path, rows)

    monkeypatch.setattr(evidence, "_write_jsonl", counted_write)
    cycle = {
        "as_of": "2026-10-09", "cycle_id": "whole-market-1",
        "taken": [
            {"symbol": f"SYM{i}", "decision": "ENTER_NOW",
             "reason_code": "ELIGIBLE", "entry": 100, "stop": 94}
            for i in range(30)
        ],
        "rejections": [{"symbol": "WAITME", "decision": "REJECTED",
                        "reason_code": "CHASE_RISK"}],
    }
    first = evidence.freeze_cycle(cycle, path=ledger)
    assert first["written"] == 31
    assert first["skipped_duplicates"] == 0
    assert count == [31]
    assert len(evidence.load_ledger(ledger)) == 31

    second = evidence.freeze_cycle(cycle, path=ledger)
    assert second["written"] == 0
    assert second["skipped_duplicates"] == 31
    assert count == [31]  # No redundant write on a duplicate replay.


def test_settlement_and_concurrent_new_freeze_cannot_lose_one_another(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import product.forward_evidence as evidence

    ledger = tmp_path / "forward_evidence.jsonl"
    orig = evidence.freeze_observation(
        {"symbol": "INFY", "decision": "ENTER_NOW", "reason_code": "ELIGIBLE",
         "entry": 100, "stop": 94},
        cycle_id="orig", as_of="2026-10-09", group="TAKEN",
        entered=True, path=ledger,
    )
    start = Barrier(2)

    def settle():
        start.wait(timeout=8)
        return evidence.attach_settlement(
            orig["decision_id"], classification="TARGET", gross_R=1.4,
            outcome_provenance=REAL_FORWARD_MARKET, path=ledger,
        )

    def add():
        start.wait(timeout=8)
        return evidence.freeze_observation(
            {"symbol": "TCS", "decision": "ENTER_NOW", "reason_code": "ELIGIBLE",
             "entry": 200, "stop": 190},
            cycle_id="new", as_of="2026-10-09", group="TAKEN",
            entered=True, path=ledger,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(settle), pool.submit(add)]
        for f in futures:
            assert f.result(timeout=12) is not None

    rows = evidence.load_ledger(ledger)
    assert {row["symbol"] for row in rows} == {"INFY", "TCS"}
    settled = next(row for row in rows if row["symbol"] == "INFY")
    assert settled["entry"] == 100
    assert settled["later_outcome"] is not None
    assert settled["gross_R"] == 1.4
    assert len(rows) == 2


def test_corrupt_forward_ledger_cannot_be_silently_rewritten(tmp_path):
    import product.forward_evidence as evidence
    import pytest

    ledger = tmp_path / "forward_evidence.jsonl"
    first = evidence.freeze_observation(
        {"symbol": "INFY", "decision": "ENTER_NOW",
         "reason_code": "ELIGIBLE", "entry": 100, "stop": 94},
        cycle_id="orig", as_of="2026-10-09", group="TAKEN", path=ledger,
    )
    with ledger.open("a", encoding="utf-8") as file:
        file.write("{partially-written-old-decision\n")
    before = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="invalid JSON at line 2"):
        evidence.freeze_observation(
            {"symbol": "TCS", "decision": "ENTER_NOW",
             "reason_code": "ELIGIBLE", "entry": 200, "stop": 180},
            cycle_id="another", as_of="2026-10-09", group="TAKEN", path=ledger,
        )
    with pytest.raises(RuntimeError, match="invalid JSON at line 2"):
        evidence.attach_settlement(first["decision_id"], gross_R=1.5, path=ledger)
    with pytest.raises(RuntimeError, match="invalid JSON at line 2"):
        evidence.freeze_cycle({"as_of": "2026-10-09", "cycle_id": "batch",
                               "taken": [{"symbol": "RELIANCE"}]}, path=ledger)

    assert ledger.read_bytes() == before  # No silent permanent deletion.


def test_forward_ledger_missing_volume_refuses_write_and_recovers(tmp_path, monkeypatch):
    import product.forward_evidence as evidence
    import pytest

    root = tmp_path / "external" / "QuantTerm" / "runtime"
    root.mkdir(parents=True)
    ledger = root / "logs" / "product" / "forward_evidence.jsonl"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    first = evidence.freeze_observation(
        {"symbol": "INFY", "decision": "ENTER_NOW",
         "reason_code": "ELIGIBLE", "entry": 100, "stop": 94},
        cycle_id="orig", as_of="2026-10-09", group="TAKEN", path=ledger,
    )
    assert first is not None
    offline = root.with_name("runtime.offline")
    root.rename(offline)

    with pytest.raises(RuntimeError, match="runtime root is missing"):
        evidence.freeze_observation(
            {"symbol": "TCS", "decision": "ENTER_NOW", "reason_code": "ELIGIBLE"},
            cycle_id="new", as_of="2026-10-09", group="TAKEN", path=ledger,
        )
    assert not root.exists()
    assert len(evidence.load_ledger(offline / "logs" / "product" / "forward_evidence.jsonl")) == 1

    offline.rename(root)
    evidence.freeze_observation(
        {"symbol": "TCS", "decision": "ENTER_NOW", "reason_code": "ELIGIBLE"},
        cycle_id="new", as_of="2026-10-09", group="TAKEN", path=ledger,
    )
    assert len(evidence.load_ledger(ledger)) == 2


def test_large_forward_cycle_deduplicates_with_one_historical_ledger_pass(monkeypatch, tmp_path):
    """An 8 GB Mac must not walk every historical decision for every card."""
    import product.forward_evidence as evidence

    initial = [
        {"decision_id": f"legacy-{i}", "symbol": f"OLD{i}", "entry": i}
        for i in range(1000)
    ]

    class CountingRows(list):
        iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            return super().__iter__()

    rows = CountingRows(initial)
    published = []
    monkeypatch.setattr(evidence, "_read_jsonl", lambda *_a, **_kw: rows)
    monkeypatch.setattr(evidence, "_write_jsonl", lambda path, output: published.append(len(output)))

    cycle = {
        "cycle_id": "large-cycle", "as_of": "2026-10-09",
        "taken": [
            {"symbol": f"NEW{i}", "decision": "ENTER_NOW",
             "reason_code": "ELIGIBLE", "entry": 100 + i, "stop": 90}
            for i in range(300)
        ],
    }
    cycle["taken"].append(dict(cycle["taken"][0], entry=99999))

    result = evidence.freeze_cycle(cycle, path=tmp_path / "forward_evidence.jsonl")
    assert result["written"] == 300
    assert result["skipped_duplicates"] == 1
    assert published == [1300]
    assert rows.iter_calls == 1  # One historical index; no quadratic rescanning.
    new = [r for r in rows if r.get("symbol") == "NEW0"]
    assert len(new) == 1
    assert new[0]["entry"] == 100  # First immutable point-in-time record wins.


def test_forward_index_keeps_first_existing_duplicate_row(tmp_path):
    """Preserve legacy first-record-wins semantics when old JSONL has duplicates."""
    import product.forward_evidence as evidence

    target = tmp_path / "forward_evidence.jsonl"
    candidate = {"symbol": "INFY", "decision": "ENTER_NOW",
                 "reason_code": "ELIGIBLE", "entry": 999, "stop": 900}
    did = evidence.decision_id(
        as_of="2026-10-09", symbol="INFY", group="TAKEN",
        reason_code="ELIGIBLE", cycle_id="repeat",
    )
    target.write_text(
        json.dumps({"decision_id": did, "symbol": "INFY", "entry": 100}) + "\n" +
        json.dumps({"decision_id": did, "symbol": "INFY", "entry": 200}) + "\n",
        encoding="utf-8",
    )
    result = evidence.freeze_cycle(
        {"cycle_id": "repeat", "as_of": "2026-10-09",
         "taken": [candidate]},
        path=target,
    )
    assert result["written"] == 0
    assert result["skipped_duplicates"] == 1
    assert len(evidence.load_ledger(target)) == 2
    record = evidence.freeze_observation(
        candidate, cycle_id="repeat", as_of="2026-10-09",
        group="TAKEN", path=target,
    )
    assert record["entry"] == 100


def test_forward_ledger_parses_streaming_without_loading_entire_text(tmp_path, monkeypatch):
    """Large on-disk forward ledgers must not materialize read_text().splitlines()."""
    import product.forward_evidence as evidence

    ledger = tmp_path / "forward_evidence.jsonl"
    with ledger.open("w", encoding="utf-8") as output:
        for i in range(1200):
            output.write(json.dumps({"decision_id": f"d{i}", "symbol": f"S{i}"}) + "\n")

    original_read_text = Path.read_text

    def disallow_full_read(self, *args, **kwargs):
        if self == ledger:
            raise AssertionError("Full ledger read_text() would cause avoidable peak memory")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", disallow_full_read)
    rows = evidence.load_ledger(ledger)
    assert len(rows) == 1200
    assert rows[0]["decision_id"] == "d0"
    assert rows[-1]["decision_id"] == "d1199"

    with ledger.open("a", encoding="utf-8") as output:
        output.write("{corrupt-record\n")
    import pytest
    with pytest.raises(RuntimeError, match="invalid JSON at line 1201"):
        evidence._read_jsonl(ledger, strict=True)
    assert len(evidence.load_ledger(ledger)) == 1200


def test_duplicate_forward_settlement_does_not_rewrite_entire_historical_ledger(
    tmp_path, monkeypatch,
):
    """A repeated paper settle check is a true no-op, including on disk."""
    import product.forward_evidence as evidence

    target = tmp_path / "forward_evidence.jsonl"
    decision = evidence.freeze_observation(
        {"symbol": "INFY", "decision": "ENTER_NOW", "reason_code": "ELIGIBLE",
         "entry": 100, "stop": 94},
        cycle_id="repeat-settle", as_of="2026-10-09",
        group="TAKEN", entered=True, path=target,
    )
    assert decision is not None

    original_write = evidence._write_jsonl
    publications = []

    def counted_write(path, rows):
        publications.append(len(rows))
        return original_write(path, rows)

    monkeypatch.setattr(evidence, "_write_jsonl", counted_write)
    first = evidence.attach_settlement(
        decision["decision_id"], classification="TARGET", gross_R=1.5,
        outcome_provenance=REAL_FORWARD_MARKET, path=target,
    )
    first_bytes = target.read_bytes()
    second = evidence.attach_settlement(
        decision["decision_id"], classification="STOP", gross_R=-1.0,
        outcome_provenance=REAL_FORWARD_MARKET, path=target,
    )
    assert first["gross_R"] == second["gross_R"] == 1.5
    assert publications == [1]
    assert target.read_bytes() == first_bytes
    assert len(evidence.load_ledger(target)) == 1


def test_forward_paths_follow_runtime_root_after_import(tmp_path, monkeypatch):
    """Store functions must not keep module-import-time runtime locations."""
    import product.forward_evidence as evidence
    import product.forward_soak as soak

    for name in ("QT_FORWARD_LEDGER", "QT_FORWARD_JOURNEY",
                 "QT_FORWARD_DAILY", "QT_FORWARD_SOAK_VERIFY"):
        monkeypatch.delenv(name, raising=False)

    for name in ("first-runtime", "second-runtime"):
        root = tmp_path / name
        monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
        assert evidence.ledger_path() == root / "logs" / "product" / "forward_evidence.jsonl"
        assert soak.journey_path() == root / "logs" / "product" / "forward_journey.json"
        assert soak.daily_dir() == root / "logs" / "product" / "forward_daily"
        assert soak.verify_state_path() == root / "logs" / "product" / "forward_soak_verify.json"

    # No writer should have created a default checkout-relative logs folder.
    assert not (tmp_path / "first-runtime").exists()
    assert not (tmp_path / "second-runtime").exists()


def test_corrupted_forward_ledger_cannot_produce_measured_or_healthy_report(tmp_path, monkeypatch):
    import pytest
    import product.forward_evidence as evidence
    import product.forward_soak as forward

    ledger = tmp_path / "forward_evidence.jsonl"
    valid = [
        {"decision_id": f"real-{i}", "provenance": REAL_FORWARD_MARKET,
         "symbol": f"STOCK{i}", "entered": True, "gross_R": 0.8}
        for i in range(25)
    ]
    with ledger.open("w", encoding="utf-8") as out:
        for record in valid:
            out.write(json.dumps(record) + "\n")
        out.write("{broken point-in-time evidence\n")

    monkeypatch.setenv("QT_FORWARD_LEDGER", str(ledger))
    # Only explicit diagnostic reads may omit the broken row.
    assert len(evidence.load_ledger(strict=False)) == 25
    with pytest.raises(RuntimeError, match="invalid JSON at line 26"):
        evidence.load_ledger(strict=True)

    # An incomplete ledger must never earn MEASURED, HEALTHY or
    # a verification PASS by silently ignoring the damaged row.
    with pytest.raises(RuntimeError, match="invalid JSON at line 26"):
        forward.scoreboard()
    with pytest.raises(RuntimeError, match="invalid JSON at line 26"):
        forward.soak_status()
    with pytest.raises(RuntimeError, match="invalid JSON at line 26"):
        forward.settle_pending_from_market(return_fn=lambda _symbol, _day: None)
