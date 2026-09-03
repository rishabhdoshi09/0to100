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
    assert "live_locked: true" in text


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
