"""Overnight GPT cross-review (2026-11-21) of f6081b26..8281810c: one base-failing test per verified finding.

E-xx: evidence / BUY admission / censoring / corporate-action coverage.
L-xx: settlement → learning exactly-once.
Network-free; every test isolates its own stores.
"""
from __future__ import annotations

import pytest


# ── E-02: a scan replaced mid-cycle must not let a production entry open ─────────

def test_e02_replaced_scan_blocks_production_entry(monkeypatch):
    from product import paper_autopilot as pa, scan_store as ss
    from research.auto_research.paper_book import PaperBook
    from tests.test_paper_autopilot_money_path import _eligible_card, _now, _workspace

    ws = _workspace([_eligible_card()])
    scan = {"scanned_at": ws["scan_scanned_at"], "records": [{"symbol": "TCS", "verdict": "BUY"}]}
    monkeypatch.setattr(ss, "load_scan", lambda: scan)
    freeze = pa._freeze_taken_decision

    def freeze_then_publish(*args, **kwargs):
        frozen = freeze(*args, **kwargs)
        scan.update(scanned_at="2026-09-01T10:01:00+00:00", records=[])
        return frozen

    monkeypatch.setattr(pa, "_freeze_taken_decision", freeze_then_publish)
    book = PaperBook(capital=100_000)
    out = pa.run_reco_paper_cycle(book=book, workspace=ws, now=_now(), as_of="2026-09-01",
                                  enforce_history=False, persist_journal=False)
    assert scan["scanned_at"] != ws["scan_scanned_at"]
    assert not book.open
    assert any(r.get("reason_code") == pa.EVIDENCE_WITHDRAWN for r in out.get("rejections", []))


def test_e02_a_still_certified_scan_still_opens(monkeypatch):
    from product import paper_autopilot as pa, scan_store as ss
    from research.auto_research.paper_book import PaperBook
    from tests.test_paper_autopilot_money_path import _eligible_card, _now, _workspace

    ws = _workspace([_eligible_card()])
    scan = {"scanned_at": ws["scan_scanned_at"], "records": [{"symbol": "TCS", "verdict": "BUY"}]}
    monkeypatch.setattr(ss, "load_scan", lambda: scan)
    book = PaperBook(capital=100_000)
    pa.run_reco_paper_cycle(book=book, workspace=ws, now=_now(), as_of="2026-09-01",
                            enforce_history=False, persist_journal=False)
    assert [p.symbol for p in book.open.values()] == ["TCS"]


# ── E-01: a cached board is re-certified on every response ──────────────────────

def test_e01_cached_board_withdraws_buy_on_first_request(monkeypatch, tmp_path):
    import terminal_product_api as api
    from product import decision_service as ds, recommendations_store as rs, scan_store as ss
    from product.projection_cache import ProjectionCache
    from tests.test_decision_service import _card, _workspace

    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))
    ws = _workspace(_card("TCS"))
    verdict = ["BUY"]
    monkeypatch.setattr(rs, "load_recommendations", lambda: ws)
    monkeypatch.setattr(ss, "load_scan", lambda *a, **k: {
        "scanned_at": ws["scan_scanned_at"], "records": [{"symbol": "TCS", "verdict": verdict[0]}]})
    monkeypatch.setattr(ss, "evidence_authority_key", lambda: tuple(verdict))
    monkeypatch.setattr(api, "_market_regime", lambda: "")
    monkeypatch.setattr(ds, "_best_trades_from_production_thesis",
                        lambda *a, **k: ([{"symbol": "TCS", "production_execution_slot": True}], []))
    cache = ProjectionCache("test-board")
    monkeypatch.setattr(cache, "_spawn", lambda *a: None)  # hold the rebuild in flight
    monkeypatch.setattr(api, "DECISION_BOARD_CACHE", cache)

    first = api.decisions_board()
    assert first["decisions"][0]["state"] == "BUY" and first["paper_actionable"] == 1
    verdict[0] = "WATCH"
    assert not ss.current_buy_certificate(ws["scan_scanned_at"])
    served = api.decisions_board()
    assert served["projection_meta"]["status"] == "STALE_REFRESHING"
    row = served["decisions"][0]
    assert row["state"] == "WATCH" and row["demoted_by"] == "BUY_NOT_CERTIFIED_AT_READ"
    assert served["best_trades"] == [] and served["paper_actionable"] == 0
    assert served["actionable"] == 0 and served["counts"].get("BUY", 0) == 0
    assert served["counts"]["WATCH"] == first["counts"].get("WATCH", 0) + 1


def test_e01_recertify_counts_buys_beyond_the_display_limit(monkeypatch):
    from product import decision_service as ds

    board = {"available": True, "scan_scanned_at": "S", "decisions": [{"symbol": "A", "state": "BUY"}],
             "counts": {"BUY": 2}, "actionable": 2, "research_actionable": 2, "historically_demoted": 0,
             "buy_symbols": ["A", "B"], "best_trades": []}
    monkeypatch.setattr("product.scan_store.current_buy_certificate", lambda _at: {"A"})
    out = ds.recertify_board(board)
    assert out["counts"] == {"BUY": 1, "WATCH": 1} and out["actionable"] == 1
    assert out["decisions"][0]["state"] == "BUY" and out["buy_symbols"] == ["A"]
    assert board["counts"] == {"BUY": 2}                       # the cached value is not mutated


# ── E-03: "Why this decision" and the board never present a withdrawn BUY ───────

def test_e03_why_withdraws_current_committee_buy(monkeypatch, tmp_path):
    from product import decision_service as ds, recommendation_truth as rt, scan_store as ss
    from tests.test_decision_service import _card, _workspace

    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))
    ws = _workspace(_card("TCS"))
    stamp = ws["scan_scanned_at"]
    monkeypatch.setattr(ss, "load_scan", lambda *a, **k: {
        "scanned_at": stamp, "records": [{"symbol": "TCS", "verdict": "WATCH"}]})
    monkeypatch.setattr(rt, "current_scan_run_id", lambda: stamp)
    monkeypatch.setattr(rt, "candidates_for_current_scan", lambda _: {"TCS": {
        "scan_run_id": stamp, "decision": "BUY",
        "decision_id": f"TCS|{stamp}", "recommendation_id": f"{stamp}:TCS:fixture"}})

    assert not ss.current_buy_certificate(stamp)
    why = ds.decision_why("TCS", workspace=ws)
    assert why["state"] != "BUY" and why["committee"]["decision"] != "BUY"
    assert why["committee"]["status"] == rt.BUY_WITHDRAWN_AT_READ
    assert why["committee"]["decision_id"] == f"TCS|{stamp}"   # the published judgment stays auditable


def test_e03_a_certified_committee_buy_is_still_shown(monkeypatch, tmp_path):
    from product import decision_service as ds, recommendation_truth as rt, scan_store as ss
    from tests.test_decision_service import _card, _workspace

    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))
    ws = _workspace(_card("TCS"))
    stamp = ws["scan_scanned_at"]
    monkeypatch.setattr(ss, "load_scan", lambda *a, **k: {
        "scanned_at": stamp, "records": [{"symbol": "TCS", "verdict": "BUY"}]})
    monkeypatch.setattr(rt, "current_scan_run_id", lambda: stamp)
    monkeypatch.setattr(rt, "candidates_for_current_scan", lambda _: {"TCS": {
        "scan_run_id": stamp, "decision": "BUY",
        "decision_id": f"TCS|{stamp}", "recommendation_id": f"{stamp}:TCS:fixture"}})
    why = ds.decision_why("TCS", workspace=ws)
    assert why["state"] == "BUY" and why["committee"]["decision"] == "BUY"


def test_e03_board_never_freezes_an_uncertified_buy(monkeypatch, tmp_path):
    from product import decision_service as ds, evidence_intelligence as ei, scan_store as ss
    from tests.test_decision_service import _card, _workspace

    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))
    ws = _workspace(_card("TCS"))
    monkeypatch.setattr(ss, "load_scan", lambda *a, **k: {
        "scanned_at": ws["scan_scanned_at"], "records": [{"symbol": "TCS", "verdict": "WATCH"}]})
    frozen = []
    monkeypatch.setattr(ei, "freeze_decision", lambda d, **k: frozen.append(d.state) or {})
    board = ds.decision_board(workspace=ws)
    assert frozen == ["WATCH"]
    row = board["decisions"][0]
    assert row["state"] == "WATCH" and row["engine_state"] == "BUY"
    assert row["demoted_by"] == "BUY_NOT_CERTIFIED_AT_READ" and board["historically_demoted"] == 1


def test_e03_a_withdrawn_card_cannot_keep_a_committee_buy():
    from product.decision_adapter import _state
    from product.scan_store import _withdraw_uncertified

    card = {"symbol": "TCS", "reco_tier": "high_conviction", "canonical_decision": "BUY"}
    out = _withdraw_uncertified(dict(card), set())
    assert _state(out) == "WATCH" and out["published_canonical_decision"] == "BUY"
