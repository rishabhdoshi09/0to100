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


# ── E-04: a snapshot's edges are derived from its ledger-bound admission rows ────

def _e04_world(tmp_path, monkeypatch, family_significant):
    import json
    from product import scan_store as ss
    from research import alpha_ledger as al
    from scan import calibration_snapshot as cs, signal_backtest as sb
    from tests.evidence_fixtures import proven_provenance

    p = proven_provenance()
    p["alpha_accounting"] = al.reserve(sb.gate_program(), "snapshot-edge-test", family_size=p["family_size"],
                                       members_sha256=p["members_sha256"],
                                       binding={k: p.get(k) for k in al.BINDING_FIELDS})
    p.pop("result_sha256_recomputed", None)
    report = {"evidence_provenance": p, "signals": {"VCP": {
        "trades": 80, "expectancy_r": 0.5, "lower_r": 0.2, "independent_sessions": 25,
        "inference": "OVERLAP_ROBUST", "family_significant": family_significant}}}
    p["result_sha256"] = sb.result_digest(report)
    assert al.finalize(p["alpha_accounting"]["look_id"], p["result_sha256"])
    snap = cs.build_snapshot(backtest_report=report, data_identity="test", thesis_hash="test")
    sid = snap["snapshot_id"]
    target = tmp_path / f"{sid}.json"
    target.write_text(json.dumps(snap))
    (tmp_path / "current.json").write_text(json.dumps({"snapshot_id": sid}))
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({
        "schema_version": 2, "provenance": {}, "evidence_identity": {"id": sid},
        "records": [{"symbol": "TCS", "verdict": "BUY", "signals": ["VCP"], "edge_evidence_id": sid}]}))
    monkeypatch.setattr(cs, "DEFAULT_DIR", tmp_path)
    monkeypatch.setattr(ss, "_AUTHORITY_CACHE", {"key": None, "snapshot": None})
    monkeypatch.setattr(ss, "scan_source_status", lambda _: (True, ""))
    return snap, target, scan


def test_e04_active_snapshot_edges_must_match_finalized_report(tmp_path, monkeypatch):
    import json
    from product import scan_store as ss

    snap, target, scan = _e04_world(tmp_path, monkeypatch, family_significant=False)
    assert ss.load_scan(scan)["records"][0]["verdict"] == "WATCH"
    snap["backtest_edges"]["VCP"]["family"] = True                 # forge the derived edge only
    target.write_text(json.dumps(snap))
    assert ss.load_scan(scan)["records"][0]["verdict"] == "WATCH"


def test_e04_a_forged_admission_row_breaks_the_digest(tmp_path, monkeypatch):
    import json
    from product import scan_store as ss

    snap, target, scan = _e04_world(tmp_path, monkeypatch, family_significant=False)
    snap["backtest_admission"]["signals"]["VCP"]["family_significant"] = True
    snap["backtest_edges"]["VCP"]["family"] = True
    target.write_text(json.dumps(snap))
    out = ss.load_scan(scan)["records"][0]
    assert out["verdict"] == "WATCH"


def test_e04_a_genuinely_proven_snapshot_still_certifies(tmp_path, monkeypatch):
    from product import scan_store as ss

    _snap, _target, scan = _e04_world(tmp_path, monkeypatch, family_significant=True)
    assert ss.load_scan(scan)["records"][0]["verdict"] == "BUY"


# ── E-07: a stop observed before a suspension survives later bars ────────────────

def test_e07_observed_stop_survives_a_later_resume(monkeypatch):
    import pandas as pd
    from tests.test_stopped_symbol_censoring import _dates, _flat, _run

    dates = _dates()
    before = _flat(dates[:90])
    before.loc[dates[87], "low"] = 89.0
    first, _ = _run(monkeypatch, {"DYING": before}, 86, dates)
    assert [t["exit_reason"] for t in first] == ["LOSS"]
    resumed = pd.concat([before, _flat(dates[91:105])])
    second, prov = _run(monkeypatch, {"DYING": resumed}, 86, dates)
    assert [t["exit_reason"] for t in second] == ["LOSS"]
    assert prov["stopped_symbol_trades_resolved"] == 1


def test_e07_a_trade_open_at_a_suspension_is_censored_even_if_bars_resume(monkeypatch):
    import pandas as pd
    from tests.test_stopped_symbol_censoring import _dates, _flat, _run

    dates = _dates()
    resumed = pd.concat([_flat(dates[:90]), _flat(dates[91:])])     # still trading after the gap
    resumed.loc[dates[95], "low"] = 89.0                            # a stop AFTER the gap is not observable
    trades, prov = _run(monkeypatch, {"GAPPED": resumed}, 86, dates)
    assert trades == []
    assert prov["stopped_symbol_trades_censored"] == 1 and prov["stopped_symbol_trades_resolved"] == 0


def test_e07_a_recent_window_of_a_trading_symbol_stays_pending_even_when_resolved(monkeypatch):
    from tests.test_stopped_symbol_censoring import _dates, _flat, _run

    dates = _dates()
    alive = _flat(dates)
    alive.loc[dates[-3], "low"] = 89.0                               # resolved, but the window is unfinished
    trades, prov = _run(monkeypatch, {"ALIVE": alive}, len(dates) - 4, dates)
    assert trades == []
    assert prov["stopped_symbol_trades_resolved"] == 0 and prov["stopped_symbol_trades_censored"] == 0


# ── E-05: a CA retry never vouches for events lost in a crash ────────────────────

def test_e05_ca_retry_never_binds_lost_historical_events(tmp_path, monkeypatch):
    import json
    from datetime import date
    from data import bhavcopy_store as bs, corporate_actions as ca
    from data import corporate_actions_resilient as car
    from tests.test_corporate_actions_resilient import _event, _refresh

    events = tmp_path / "ca_events.json"
    coverage = tmp_path / "ca_coverage.json"
    events.write_text(json.dumps([_event(symbol="OTHER")]))
    monkeypatch.setattr(bs, "reload_corporate_actions", lambda: None)
    with monkeypatch.context() as crash:
        with pytest.raises(OSError):
            _refresh(tmp_path, crash_on_events=True, monkeypatch=crash)
    today = date(2026, 8, 28)
    args = dict(years=1, today=today, events_path=events, coverage_path=coverage)
    interrupted = car.coverage_status(**args)
    _refresh(tmp_path)
    retry = car.coverage_status(**args)
    oldest = car.build_windows(today=today, years=1)[0].start.isoformat()
    recovered = {str(e["ex_date"])[:10] for e in ca.load_events(events).get("ABC", [])}
    assert retry["events_binding"] != "BOUND" or oldest in recovered
    assert not interrupted["coverage_complete"]


def test_e05_a_crash_after_one_committed_window_resumes_with_every_event(tmp_path, monkeypatch):
    from datetime import date
    from data import bhavcopy_store as bs, corporate_actions as ca
    from data import corporate_actions_resilient as car
    from tests.test_corporate_actions_resilient import _refresh

    events = tmp_path / "ca_events.json"
    monkeypatch.setattr(bs, "reload_corporate_actions", lambda: None)
    real, writes = car._atomic_json, []

    def second_events_write_fails(path, payload):
        if path == events:
            writes.append(1)
            if len(writes) == 2:
                raise OSError("power cut on the second window's events write")
        return real(path, payload)

    with monkeypatch.context() as crash:
        crash.setattr(car, "_atomic_json", second_events_write_fails)
        with pytest.raises(OSError):
            _refresh(tmp_path)
    today = date(2026, 8, 28)
    windows = car.build_windows(today=today, years=1)
    cov_path = tmp_path / "ca_coverage.json"
    import json
    assert set(car.trusted_windows(json.loads(cov_path.read_text()), events)) == {windows[0].window_id}
    _refresh(tmp_path)
    status = car.coverage_status(years=1, today=today, events_path=events, coverage_path=cov_path)
    recovered = {str(e["ex_date"])[:10] for e in ca.load_events(events).get("ABC", [])}
    assert status["events_binding"] == "BOUND" and status["coverage_complete"]
    assert {w.start.isoformat() for w in windows} <= recovered


def test_e05_a_pre_protocol_bound_declaration_is_refetched(tmp_path, monkeypatch):
    import json
    from datetime import date
    from data import bhavcopy_store as bs, corporate_actions_resilient as car
    from tests.test_corporate_actions_resilient import _refresh

    monkeypatch.setattr(bs, "reload_corporate_actions", lambda: None)
    events, coverage = _refresh(tmp_path)
    cov = json.loads(coverage.read_text())
    cov.pop("commit_protocol")
    for rec in cov["windows"].values():
        rec.pop("events_committed")
    coverage.write_text(json.dumps(cov))
    args = dict(years=1, today=date(2026, 8, 28), events_path=events, coverage_path=coverage)
    assert car.coverage_binding(cov, events) == "BOUND" and not car.coverage_status(**args)["coverage_complete"]
    fetched = []
    car.refresh_events_resilient(
        years=1, today=date(2026, 8, 28), events_path=events, coverage_path=coverage,
        nse_json_fetcher=lambda s, e: fetched.append(s) or [], nse_csv_fetcher=lambda s, e: [],
        bse_fetcher=lambda s, e: [], sleep_fn=lambda _s: None, budget_s=30)
    assert len(fetched) == len(car.build_windows(today=date(2026, 8, 28), years=1))
    assert car.coverage_status(**args)["coverage_complete"]


def test_e05_a_corrupt_events_file_is_never_overwritten(tmp_path, monkeypatch):
    from datetime import date
    from data import bhavcopy_store as bs, corporate_actions_resilient as car
    from tests.test_corporate_actions_resilient import _refresh

    monkeypatch.setattr(bs, "reload_corporate_actions", lambda: None)
    events = tmp_path / "ca_events.json"
    events.write_text("{not json")
    _refresh(tmp_path)
    assert events.read_text() == "{not json"
    status = car.coverage_status(years=1, today=date(2026, 8, 28), events_path=events,
                                 coverage_path=tmp_path / "ca_coverage.json")
    assert not status["coverage_complete"] and status["events_binding"] != "BOUND"


# ── E-06: the PIT loader binds coverage to the very bytes it applies ─────────────

def test_e06_ca_binding_checks_the_bytes_used_for_adjustment(tmp_path, monkeypatch):
    import json
    import pickle

    import pandas as pd
    from data import corporate_actions as ca, corporate_actions_resilient as car
    from research import edge_diagnosis as ed
    from tests.test_corporate_actions_resilient import _event

    logs = tmp_path / "logs"
    (logs / "bhav").mkdir(parents=True)
    frame = pd.DataFrame({"close": [100.]}, index=pd.to_datetime(["2026-01-14"]))
    (logs / "bhav" / "store_cache.pkl").write_bytes(pickle.dumps({"store": {"ABC": frame}}))
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    events, coverage = logs / "ca_events.json", logs / "ca_coverage.json"

    def publish(factor):
        events.write_text(json.dumps([_event(factor=factor)]))
        coverage.write_text(json.dumps({
            "events_sha256": car.events_digest(events), "commit_protocol": car.COMMIT_PROTOCOL,
            "windows": {"w": {"success": True, "covered_from": "2025-01-01", "covered_to": "2026-08-28",
                              "events_committed": True}}}))

    publish(2.0)
    assert ed.load_copy(tmp_path)["coverage_binding"] == "BOUND"
    read = ca.load_events_with_digest

    def read_then_replace(path=None):
        loaded = read(path)
        publish(4.0)
        return loaded

    monkeypatch.setattr(ca, "load_events_with_digest", read_then_replace)
    loaded = ed.load_copy(tmp_path)
    assert loaded["frames"]["ABC"]["close"].iloc[0] == pytest.approx(50.0)   # generation A applied
    assert loaded["coverage"] == {} and loaded["coverage_binding"] == "MISMATCH"


# ── L: settlement → learning exactly once ────────────────────────────────────────

def _review_freeze(*, cycle="review", decision_id="", reason="ENTRY_TOO_EXTENDED", target=115):
    from product import forward_evidence as FE
    return FE.freeze_observation(
        {"symbol": "TCS", "decision": "BLOCK", "decision_id": decision_id,
         "reason_code": reason, "entry": 100, "stop": 94, "target": target, "setup_label": "VCP"},
        cycle_id=cycle, as_of="2026-09-01", group="REJECTED", provenance=FE.REAL_FORWARD_MARKET,
    )


def test_l01_official_settler_cannot_bypass_learning(monkeypatch):
    from core import outcome_resolver
    from product.autonomous_loop import settle_official_outcomes
    from tests.evidence_fixtures import rulebook_outcome
    from tests.test_forward_settlement_batch import _open_ids
    from tests.test_learning_exactly_once import _n, _settle

    _review_freeze()
    monkeypatch.setattr(outcome_resolver, "counterfactual_outcome", lambda *a, **k: rulebook_outcome(-0.4, "LOSS"))
    assert settle_official_outcomes("2026-09-18")["n_settled"] == 1
    assert _open_ids() == []
    _settle()
    assert _n() == 1


def test_l02_replay_rejects_corrupt_body_with_matching_id(monkeypatch):
    import json
    import sqlite3
    from product import forward_evidence as FE, forward_soak as FS, learning_observations as LO
    from product import paper_learning_loop as PLL
    from tests.test_learning_exactly_once import _ledger, _n, _settle

    _review_freeze()

    def fail(*a, **k):
        raise OSError("stop after ledger write")

    with monkeypatch.context() as patch:
        patch.setattr(PLL, "ingest_counterfactual", fail)
        with pytest.raises(RuntimeError):
            _settle()
    (event,) = LO.pending_events(_ledger())
    bad = {k: event["payload"][k] for k in ("forward_decision_id", "eval_spec")}
    bad["classification"] = "AVOIDED_LOSER"
    conn = sqlite3.connect(_ledger())
    with conn:
        conn.execute("UPDATE outbox SET payload=? WHERE event_id=?", (json.dumps(bad), event["event_id"]))
    conn.close()
    FS.replay_learning_outbox(settled_ids=FS._settled_decision_ids(FE.load_ledger()))
    assert (LO.event_state(_ledger(), event["event_id"]), _n("REJECT::REJECTED")) == (LO.QUARANTINED, 0)


def test_l02_replay_waits_for_uncommitted_settlement(monkeypatch):
    from product import forward_soak as FS
    from tests.test_learning_exactly_once import _n, _settle

    _review_freeze()

    def fail(*a, **k):
        raise OSError("stop before ledger write")

    with monkeypatch.context() as patch:
        patch.setattr(FS, "attach_settlements", fail)
        with pytest.raises(OSError):
            _settle()
    result = FS.replay_learning_outbox()
    assert (result["waiting"], _n()) == (1, 0)


def test_l02_a_retry_with_a_different_outcome_learns_the_committed_one(monkeypatch):
    from product import forward_soak as FS, learning_observations as LO
    from tests.evidence_fixtures import rulebook_outcome
    from tests.test_learning_exactly_once import _ledger

    _review_freeze()

    def fail(*a, **k):
        raise OSError("stop before ledger write")

    with monkeypatch.context() as patch:                      # queued: a LOSS, never committed
        patch.setattr(FS, "attach_settlements", fail)
        with pytest.raises(OSError):
            FS.settle_pending_from_market(outcome_fn=lambda *a: rulebook_outcome(-0.4, "LOSS"))
    from product import paper_learning_loop as PLL

    def ingest_fails(*a, **k):
        raise OSError("stop after the ledger write")

    with monkeypatch.context() as patch:                      # committed: a WIN; learning crashes
        patch.setattr(PLL, "ingest_counterfactual", ingest_fails)
        with pytest.raises(RuntimeError):
            FS.settle_pending_from_market(outcome_fn=lambda *a: rulebook_outcome(0.6, "WIN"))
    FS.replay_learning_outbox()                               # the replay must learn the WIN
    (row,) = [r for r in __import__("product.forward_evidence", fromlist=["x"]).load_ledger()]
    assert row["later_outcome"]["counterfactual_R"] == pytest.approx(0.6)
    assert LO.outbox_counts(_ledger()) == {LO.DONE: 1}
    from product import learning_policy_store as LPS
    pol = next(p for p in LPS.load_policies()["policies"] if p["policy_id"] == "REJECT::ENTRY_TOO_EXTENDED")
    assert pol["sample_size"] == 1 and pol["classification"] == row["counterfactual_classification"]


def test_l03_resume_uses_the_original_unit_payload(monkeypatch):
    from product import forward_soak as FS, learning_observations as LO
    from tests.evidence_fixtures import rulebook_outcome
    from tests.test_learning_exactly_once import _n

    reason = "EVIDENCE_POLICY_BLOCK"
    first = _review_freeze(cycle="first", reason=reason)
    _review_freeze(cycle="second", reason=reason, target=103)
    real_mark = LO.mark_applied
    failed = False

    def mark(path, scope, key):
        nonlocal failed
        if scope == "COUNTERFACTUAL" and not failed:
            failed = True
            raise OSError("stop before the first unit's final mark")
        return real_mark(path, scope, key)

    def outcome(_symbol, _day, row):
        if row["decision_id"] == first["decision_id"]:
            return rulebook_outcome(-0.4, "LOSS")
        return rulebook_outcome(0.4, "WIN")

    monkeypatch.setattr(LO, "mark_applied", mark)
    with pytest.raises(RuntimeError):
        FS.settle_pending_from_market(outcome_fn=outcome)
    FS.settle_pending_from_market(outcome_fn=outcome)
    assert (_n(f"REJECT::{reason}"), _n("SETUP::VCP")) == (1, 0)


def test_l05_recovery_rejects_claim_for_another_policy():
    from product import learning_observations as LO, learning_policy_store as LPS
    from tests.test_learning_exactly_once import _ledger, _n

    body = dict(policy_id="SETUP::B", dimension="setup", bucket="B", realized_R=1.0,
                source="paper_forward", extra={})
    LPS.record_measured_outcome(**body, observation_key="same", session="2026-09-01")
    assert LO.claim(_ledger(), "SETUP::A", "same", "2026-09-01", payload=body)
    LPS.recover_claims()
    assert (_n("SETUP::B"), LO.claim_state(_ledger(), "SETUP::A", "same")) == (1, LO.CLAIMED)


def test_l05_a_malformed_claim_never_blocks_recovery_of_the_others():
    from product import learning_observations as LO, learning_policy_store as LPS
    from tests.test_learning_exactly_once import _ledger, _n

    good = dict(policy_id="SETUP::G", dimension="setup", bucket="G", realized_R=1.0,
                source="paper_forward", extra={})
    assert LO.claim(_ledger(), "SETUP::M", "k1", "2026-09-01", payload={"policy_id": "SETUP::M"})
    assert LO.claim(_ledger(), "SETUP::G", "k2", "2026-09-01", payload=good)
    out = LPS.recover_claims()
    assert out["corrupt"] == 1 and out["replayed"] == 1 and _n("SETUP::G") == 1


def test_l06_committed_counterfactual_labels_linked_features(tmp_path, monkeypatch):
    from research import feature_store
    from tests.test_learning_exactly_once import _settle

    monkeypatch.setattr(feature_store, "_DB_PATH", tmp_path / "features.sqlite3")
    feature_id = "decision::d1"
    assert feature_store.snapshot(feature_id, "TCS", "DECISION", {}, ts="2026-09-01")["status"] == "frozen"
    _review_freeze(decision_id="d1")
    _settle()
    assert feature_store.get_observation(feature_id)["outcome"] == pytest.approx(-0.4)


def test_l06_a_failed_feature_label_is_retried_not_dropped(tmp_path, monkeypatch):
    from product import evidence_intelligence as EI, learning_observations as LO
    from research import feature_store
    from tests.test_learning_exactly_once import _ledger, _n, _settle

    monkeypatch.setattr(feature_store, "_DB_PATH", tmp_path / "features.sqlite3")
    assert feature_store.snapshot("decision::d1", "TCS", "DECISION", {}, ts="2026-09-01")["status"] == "frozen"
    _review_freeze(decision_id="d1")
    with monkeypatch.context() as patch:
        patch.setattr(EI, "settle_decision", lambda *a, **k: {"status": "error", "reason": "locked"})
        with pytest.raises(RuntimeError):
            _settle()
    assert feature_store.get_observation("decision::d1")["outcome"] is None and _n() == 1
    _settle()                                                  # the replay retries only what did not land
    assert feature_store.get_observation("decision::d1")["outcome"] == pytest.approx(-0.4) and _n() == 1
    assert LO.outbox_counts(_ledger()) == {LO.DONE: 1}


def test_l09_existing_schema_migrates_with_two_connections(tmp_path, monkeypatch):
    import sqlite3
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from product import learning_observations as LO

    db = tmp_path / "legacy.sqlite3"
    connect = sqlite3.connect
    conn = connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(LO._SCHEMA)
    conn.close()
    both_inspected = threading.Barrier(2)

    class Connection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            result = super().execute(sql, parameters)
            if sql == "PRAGMA table_info(observations)":
                rows = result.fetchall()
                try:
                    both_inspected.wait(timeout=3)
                except threading.BrokenBarrierError:
                    pass
                return rows
            return result

    monkeypatch.setattr(LO.sqlite3, "connect", lambda *a, **k: connect(*a, factory=Connection, **k))

    def open_db():
        LO._connect(db).close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [pool.submit(open_db) for _ in range(2)]
        for worker in workers:
            worker.result(timeout=20)


def test_l04_history_publication_preserves_interleaved_learning(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from product import autonomous_evolution as history, learning_policy_store as LPS
    from tests.test_learning_exactly_once import _n

    staged, applied = threading.Event(), threading.Event()
    real_save = LPS.save_policies
    obs = dict(policy_id="REJECT::ENTRY_TOO_EXTENDED", dimension="reason_code",
               bucket="ENTRY_TOO_EXTENDED", realized_R=0.4, observation_key="obs")

    def save(payload, path=None):
        result = real_save(payload, path)
        if ".history-stage-" in str(path):
            staged.set()
            applied.wait(3)
        return result

    def learn_after_snapshot():
        assert staged.wait(5)
        try:
            return LPS.record_measured_outcome(**obs)
        finally:
            applied.set()

    monkeypatch.setattr(LPS, "save_policies", save)
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(learn_after_snapshot)
        history._publish_history_policies({}, {}, generation_fingerprint="review")
        worker.result(timeout=5)
    LPS.record_measured_outcome(**obs)
    assert _n() == 1


def test_l04_history_publication_still_replaces_the_historical_projection():
    from product import autonomous_evolution as history, learning_policy_store as LPS

    LPS.upsert_policy(policy_id="HIST_SETUP::OLD", dimension="setup", bucket="OLD", sample_size=3,
                      expectancy_R=0.1, source="backtest_reproduced")
    LPS.record_measured_outcome(policy_id="REJECT::X", dimension="reason_code", bucket="X",
                                realized_R=0.4, observation_key="k")
    history._publish_history_policies({"NEW": {"n": 5, "mean_R": 0.2}}, {}, generation_fingerprint="g2")
    ids = {p["policy_id"] for p in LPS.load_policies()["policies"]}
    assert "HIST_SETUP::NEW" in ids and "HIST_SETUP::OLD" not in ids and "REJECT::X" in ids


def test_l07_replayed_close_calibrates_once(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from product import decision_calibration, learning_policy_store as LPS, paper_learning_loop as PLL
    from research import feature_store

    monkeypatch.setattr(feature_store, "_DB_PATH", tmp_path / "features.sqlite3")
    assert feature_store.snapshot("decision::d1", "TCS", "DECISION", {}, ts="2026-09-01",
                                  meta={"predicted_p": 0.7})["status"] == "frozen"
    book = SimpleNamespace(closed=[{
        "decision_id": "d1", "paper_intent_id": "i1", "context_key": "ctx", "symbol": "TCS",
        "setup_label": "VCP", "entry_date": "2026-09-01", "exit_date": "2026-09-02",
        "exit_reason": "TARGET", "realized_R": 1.0}])
    real_apply = LPS._apply_measured

    def apply(**kw):
        if kw["policy_id"] == "SETUP::VCP":
            raise OSError("stop after calibration, before book checkpoint")
        return real_apply(**kw)

    with monkeypatch.context() as patch:
        patch.setattr(LPS, "_apply_measured", apply)
        with pytest.raises(OSError):
            PLL.ingest_closed_book(book)
    PLL.ingest_closed_book(book)
    assert len(decision_calibration.load_store()["observations"]) == 1


def test_l07_a_crash_between_cell_and_ladder_still_reaches_the_ladder(monkeypatch):
    from product import conditional_evidence as CE, learning_policy_store as LPS
    from product.decision_chain import Outcome
    from product.evidence_class import PAPER_FORWARD
    from tests.test_learning_exactly_once import _n

    outcome = Outcome(position_id="p1", paper_order_id="i1", paper_intent_id="i1", decision_id="d1",
                      symbol="TCS", exit_reason="TARGET", realized_R=1.0, entry_price=100, exit_price=110,
                      entry_session="2026-09-01", exit_session="2026-09-02", evidence_class=PAPER_FORWARD)
    with monkeypatch.context() as patch:
        patch.setattr(LPS, "record_measured_outcome",
                      lambda **k: (_ for _ in ()).throw(OSError("crash after the cell save")))
        CE.record_outcome(outcome, context_key="ctx", evidence_class=PAPER_FORWARD)
    assert _n("cond::ctx") == 0
    CE.record_outcome(outcome, context_key="ctx", evidence_class=PAPER_FORWARD)
    CE.record_outcome(outcome, context_key="ctx", evidence_class=PAPER_FORWARD)
    assert _n("cond::ctx") == 1


def test_l08_good_wait_cannot_credit_session_eleven(monkeypatch):
    from core import session_window
    from product import paper_learning_loop as PLL
    from product.counterfactual_learning import freeze_decision
    from tests.test_learning_exactly_once import _n

    dates = [f"2026-09-{day}" for day in ("01", "02", "03", "04", "07", "08", "09", "10", "11", "16", "17")]
    monkeypatch.setattr(session_window, "official_dates", lambda: dates)
    freeze_decision(symbol="TCS", decision="WAIT", reason_code="WAIT_FOR_PULLBACK",
                    entry=100, stop=94, target=115, as_of="2026-09-01")
    result = PLL.note_later_entry("TCS", as_of="2026-09-18")
    assert (result["updated"], _n("ENTRY_QUALITY::GOOD_WAIT")) == (0, 0)


def test_l08_a_fill_inside_the_horizon_is_still_a_good_wait(monkeypatch):
    from core import session_window
    from product import paper_learning_loop as PLL
    from product.counterfactual_learning import freeze_decision
    from tests.test_learning_exactly_once import _n

    dates = [f"2026-09-{day}" for day in ("01", "02", "03", "04", "07")]
    monkeypatch.setattr(session_window, "official_dates", lambda: dates)
    freeze_decision(symbol="TCS", decision="WAIT", reason_code="WAIT_FOR_PULLBACK",
                    entry=100, stop=94, target=115, as_of="2026-09-01")
    result = PLL.note_later_entry("TCS", as_of="2026-09-08")      # session 5, intraday
    assert (result["updated"], _n("ENTRY_QUALITY::GOOD_WAIT")) == (1, 1)


def test_l08_no_archive_means_the_credit_stays_pending(monkeypatch):
    from core import session_window
    from product import paper_learning_loop as PLL
    from product.counterfactual_learning import freeze_decision

    monkeypatch.setattr(session_window, "official_dates", lambda: [])
    freeze_decision(symbol="TCS", decision="WAIT", reason_code="WAIT_FOR_PULLBACK",
                    entry=100, stop=94, target=115, as_of="2026-09-01")
    assert PLL.note_later_entry("TCS", as_of="2026-09-03")["updated"] == 0


# ── Verification pass (GPT, 38429d98..f0ccf35a): QT-01..QT-05 ───────────────────

def test_qt01_ca_restart_preserves_unresolved_factor_conflict(tmp_path, monkeypatch):
    import json
    from datetime import date
    from data import bhavcopy_store as bs, corporate_actions_resilient as ca
    from tests.test_corporate_actions_resilient import _event

    today = date(2026, 8, 28)
    events, coverage = tmp_path / "events.json", tmp_path / "coverage.json"
    start = ca.build_windows(today=today, years=1)[0].start
    events.write_text(json.dumps([_event(ex_date=str(start), factor=2)]))
    monkeypatch.setattr(bs, "reload_corporate_actions", lambda: None)

    def fetch(lo, hi):
        if lo == start:
            return [_event(ex_date=str(lo), factor=3)]
        return [_event(symbol="OTHER", ex_date=str(lo))]

    args = dict(years=1, today=today, events_path=events, coverage_path=coverage)
    run = dict(nse_json_fetcher=fetch, nse_csv_fetcher=lambda s, e: [], bse_fetcher=lambda s, e: [],
               sleep_fn=lambda _s: None, budget_s=30)
    atomic, writes = ca._atomic_json, []

    def interrupt(path, value):
        if path == events:
            writes.append(1)
            if len(writes) == 2:
                raise OSError("interrupted after first window commit")
        return atomic(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(ca, "_atomic_json", interrupt)
        with pytest.raises(OSError):
            ca.refresh_events_resilient(**args, **run)
    assert ca.coverage_status(**args)["unresolved_conflicts"] == 1
    ca.refresh_events_resilient(**args, **run)
    status = ca.coverage_status(**args)
    # (Readiness gates on conflicts in the installed equity-filter wrapper.)
    assert status["unresolved_conflicts"] == 1
    assert "ABC" not in json.dumps(json.loads(events.read_text()))      # neither factor is applied


def test_qt01_a_later_single_factor_fetch_does_not_resolve_a_conflict():
    from data import corporate_actions_resilient as ca
    from tests.test_corporate_actions_resilient import _event

    prior = [{"symbol": "ABC", "ex_date": "2026-01-15", "type": "bonus", "factors": [2.0, 3.0]}]
    merged, conflicts = ca.merge_verified_events([], [_event(factor=3.0)], prior)
    assert merged == [] and len(conflicts) == 1 and conflicts[0]["factors"] == [2.0, 3.0]


def test_qt02_two_settlers_never_interleave(monkeypatch):
    import threading
    from product import forward_evidence as fe, forward_soak as fs, learning_observations as lo
    from tests.evidence_fixtures import rulebook_outcome
    from tests.test_learning_exactly_once import _ledger, _n

    _review_freeze()
    a_started, b_inside = threading.Event(), threading.Event()
    results = {}

    def b_outcome(*args):
        b_inside.set()                         # B holds the settlement lock, mid-measurement
        a_started.wait(2)
        return rulebook_outcome(0.6, "WIN")

    def settler_a():
        a_started.set()
        results["a"] = fs.settle_pending_from_market(outcome_fn=lambda *a: rulebook_outcome(-0.4, "LOSS"))

    b = threading.Thread(target=lambda: results.setdefault("b", fs.settle_pending_from_market(outcome_fn=b_outcome)))
    b.start()
    assert b_inside.wait(5)
    a = threading.Thread(target=settler_a)
    a.start()
    b.join(10)
    a.join(10)
    (row,) = fe.load_ledger()
    assert row["later_outcome"]["counterfactual_R"] == pytest.approx(0.6)     # B committed; A saw it settled
    assert results["a"]["updated"] == 0
    assert (_n(), lo.outbox_counts(_ledger())) == (1, {lo.DONE: 1})


def test_qt03_legacy_unit_claim_cannot_resume_from_another_decision(monkeypatch):
    from product import learning_observations as lo, paper_learning_loop as pll
    from product.counterfactual_learning import settle_rulebook
    from tests.evidence_fixtures import rulebook_outcome
    from tests.test_learning_exactly_once import _n

    reason = "EVIDENCE_POLICY_BLOCK"
    base = dict(symbol="TCS", as_of="2026-09-01", reason_code=reason, evidence={"setup_label": "VCP"})
    loss = settle_rulebook(dict(base, decision_id="first"), rulebook_outcome(-0.4, "LOSS"))
    win = settle_rulebook(dict(base, decision_id="second"), rulebook_outcome(0.6, "WIN"))
    claim, mark = lo.claim, lo.mark_applied

    def legacy_claim(path, scope, key, day, payload=None):
        if scope == "COUNTERFACTUAL":
            payload = {"kind": "counterfactual_ingest", "decision_id": "first"}
        return claim(path, scope, key, day, payload=payload)

    def interrupt(path, scope, key):
        if scope == "COUNTERFACTUAL":
            raise OSError("stopped before the unit mark")
        return mark(path, scope, key)

    with monkeypatch.context() as patch:
        patch.setattr(lo, "claim", legacy_claim)
        patch.setattr(lo, "mark_applied", interrupt)
        with pytest.raises(OSError):
            pll.ingest_counterfactual(loss)
    out = pll.ingest_counterfactual(win)
    assert out.get("deferred_to_decision") == "first"
    assert (_n(f"REJECT::{reason}"), _n("SETUP::VCP")) == (1, 0)
    pll.ingest_counterfactual(loss)                                 # its own decision resumes it
    assert lo.claim_state(lo.ledger_path_for(__import__("product.learning_policy_store",
                          fromlist=["x"]).policy_path()), "COUNTERFACTUAL", "cf:2026-09-01|TCS|" + reason) == lo.APPLIED


def test_qt04_repair_refuses_a_plan_made_before_a_new_observation(tmp_path):
    from product import forward_evidence as fe
    from product import learning_policy_repair as repair, learning_policy_store as lps
    from tests.test_learning_exactly_once import _n, _settle

    _review_freeze()
    _settle()
    plan = repair.plan(forward_path=fe.ledger_path(), cf_path=tmp_path / "empty.jsonl",
                       policy_file=lps.policy_path())
    fe.freeze_observation(dict(symbol="INFY", decision="BLOCK", reason_code="ENTRY_TOO_EXTENDED",
                               entry=100, stop=94, target=115),
                          cycle_id="second", as_of="2026-09-01", group="REJECTED",
                          provenance=fe.REAL_FORWARD_MARKET)
    _settle()
    assert _n() == 2
    with pytest.raises(repair.StalePlan):
        repair.apply_plan(plan, backup_dir=tmp_path / "backup")
    assert _n() == 2 and not (tmp_path / "backup").exists() or not any((tmp_path / "backup").iterdir())


def test_qt05_closed_book_retries_the_ladder_before_its_checkpoint(monkeypatch):
    from types import SimpleNamespace
    from product import learning_policy_store as lps, paper_learning_loop as pll
    from tests.test_learning_exactly_once import _n

    book = SimpleNamespace(closed=[dict(decision_id="d1", paper_intent_id="i1", context_key="ctx", symbol="TCS",
                                        setup_label="VCP", entry_date="2026-09-01", exit_date="2026-09-02",
                                        exit_reason="TARGET", realized_R=1.0)])
    record = lps.record_measured_outcome

    def fail_ladder(**kwargs):
        if kwargs["policy_id"] == "cond::ctx":
            raise OSError("ladder unavailable before claim")
        return record(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(lps, "record_measured_outcome", fail_ladder)
        with pytest.raises(Exception):
            pll.ingest_closed_book(book)
    pll.ingest_closed_book(book)
    assert _n("cond::ctx") == 1 and _n("SETUP::VCP") == 1
    pll.ingest_closed_book(book)
    assert _n("cond::ctx") == 1 and _n("SETUP::VCP") == 1
