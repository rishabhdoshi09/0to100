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
