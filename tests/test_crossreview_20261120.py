"""Regressions for the first independent GPT cross-review (2026-11-20, target 9636b030).

Every test here was a counterexample that FAILED on 9636b030 (the GPT finding
id is in the name) and passes after the fix. Money-path findings (C-9, C-11,
C-12, A-1) live in tests/test_money_paths.py::TestCrossReview20261120.
Network-free; every store is isolated.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from tests.evidence_fixtures import proven_provenance, rulebook_outcome


def _frame(rows, start="2026-02-03"):
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


# ── the rulebook: exact official sessions, no invented executions ────────────

def test_A7_a_missing_session_never_stretches_the_ten_session_window():
    from core.outcome_resolver import counterfactual_outcome

    days = pd.bdate_range("2026-02-03", periods=11)
    official = [d.date().isoformat() for d in days]
    bars = days.delete(1)                                     # the symbol has no bar on session 2
    frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.}, index=bars)
    frame.loc[bars[-1], "high"] = 111.                         # target first touched in session 11
    got = counterfactual_outcome("INFY", "2026-02-02", 100, 95, 110, frame=frame, sessions=official)
    assert got is None                                         # unverifiable: never a 10-session WIN
    # With the bar present and the touch inside the 10 sessions it IS a win.
    full = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.}, index=days)
    full.loc[days[5], "high"] = 111.
    assert counterfactual_outcome("INFY", "2026-02-02", 100, 95, 110, frame=full,
                                  sessions=official)["outcome"] == "WIN"


def test_B5_a_live_intraday_overlay_is_not_a_published_bar():
    from core.outcome_resolver import counterfactual_outcome

    overlay = pd.DataFrame({"open": [100.], "high": [111.], "low": [99.], "close": [105.]},
                           index=pd.to_datetime(["2026-11-19"]))
    overlay.attrs["quantterm_live_overlay_date"] = "2026-11-19"
    assert counterfactual_outcome("TCS", "2026-11-18", 100, 90, 110, frame=overlay) is None


def test_B8_a_bar_cannot_invent_an_execution_price():
    from core.evaluation_spec import simulate_trade

    gap = simulate_trade(100, 90, 120, [135], [125], [132], opens=[130])
    assert gap.outcome == "SCRATCH" and gap.r == 0.0 and gap.exit_price == gap.fill_price == 130
    missing = simulate_trade(100, 90, 120, [float("nan")], [79], [81], opens=[80])
    assert not missing.filled
    # A NaN bar after the fill neither exits nor marks the trade.
    held = simulate_trade(100, 90, 120, [101, float("nan"), 104], [99, 50, 101], [100.5, 1.0, 103], opens=[99, 1, 102])
    assert held.outcome == "FLAT" and held.exit_price == 103


def test_B4_a_market_wide_missing_archive_leaves_a_hole_in_covered_years():
    from core.session_window import official_session_index
    from research.intelligence.data import nse_calendar as CAL

    holidays = CAL.load_holidays()
    if 2026 not in CAL.covered_years(holidays):
        pytest.skip("bundled 2026 calendar manifest not verified in this checkout")
    days = [d for d in pd.bdate_range("2026-11-16", "2026-12-04").strftime("%Y-%m-%d")
            if d not in holidays]
    on_disk = [d for d in days if d != "2026-11-20"]            # one archive missing market-wide
    idx = official_session_index(on_disk)
    assert "2026-11-20" in idx                                  # the calendar session is still counted
    assert idx["2026-11-23"] - idx["2026-11-19"] == 2           # no compression across the gap
    future = official_session_index(on_disk + ["2027-01-04"])
    assert "2027-01-04" not in future                           # an uncovered future year is never verified


# ── one rulebook for every learning label ─────────────────────────────────────

def test_B7_historical_replay_learns_the_stop_even_if_price_recovers(monkeypatch):
    from core import outcome_resolver as OR
    from product import evidence_intelligence as EI, historical_replay as HR
    from tests.test_counterfactual_horizon import D, _bars

    frame = _bars([(99.5, 100.5, 99, 100), (99, 99.5, 93, 94.5)] + [(108, 109, 107, 108)] * 8)
    monkeypatch.setattr(OR, "_ohlcv", lambda _: frame)
    monkeypatch.setattr(HR, "ohlcv_as_of", lambda *a, **kw: frame)
    labels = []
    monkeypatch.setattr(EI, "record_resolved_prediction", lambda decision_id, r, **kw: labels.append((r, kw)))
    row, = HR.evaluate_outcomes([dict(symbol="TCS", as_of=D, decision="AVOID", canonical_decision_id="t",
                                      entry=100., stop=94., target=115.)])
    assert row["classification"] == "AVOIDED_LOSER"
    assert len(labels) == 1 and labels[0][0] < 0 and labels[0][1]["eval_spec"]


def test_A8_evolution_grades_equity_shadows_on_the_rulebook(tmp_path, monkeypatch):
    from core import outcome_resolver as OR
    from product.evolution import grading as G

    frame = pd.DataFrame([(100, 101, 99, 100), (90, 93, 89, 92)], columns=["open", "high", "low", "close"],
                         index=pd.to_datetime(["2026-09-02", "2026-09-03"]))
    monkeypatch.setattr(OR, "_ohlcv", lambda symbol: frame)
    row = dict(shadow_id="s", domain="EQUITY", decision="WAIT", symbol="INFY", as_of="2026-09-01",
               entry=100, stop=95, target=110)
    grade = G.grade_shadow_decision(row, path=tmp_path / "shadow.jsonl")
    expected = OR.counterfactual_outcome("INFY", row["as_of"], 100, 95, 110)
    assert grade["counterfactual_R"] == expected["net_r"] < -1.5       # the gap-down open, not -1R
    assert grade["eval_spec"] == expected["eval_spec"]


def test_own_decision_simulator_grades_on_the_rulebook(monkeypatch):
    from product import decision_simulator as DS
    from product.counterfactual_learning import AVOIDED_LOSER

    monkeypatch.setattr(DS, "_rulebook_outcome", lambda row: rulebook_outcome(-1.05, "LOSS"))
    assert DS._classify({"symbol": "X", "as_of": "2026-09-01", "decision": "REJECT"}) == AVOIDED_LOSER
    monkeypatch.setattr(DS, "_rulebook_outcome", lambda row: None)
    assert DS._classify({"symbol": "X", "as_of": "2026-09-01", "decision": "REJECT",
                         "forward_return_pct": 9.0}) == DS.INCONCLUSIVE      # a bare return is not judgeable


def test_own_integrity_reprobe_uses_the_resolver_the_settler_uses(tmp_path, monkeypatch):
    from product import forward_evidence as FE
    from research.autonomy import outcome_integrity as OI
    import core.outcome_resolver as OR

    path = tmp_path / "forward.jsonl"
    monkeypatch.setattr(FE, "DEFAULT_PATH", path)
    FE.freeze_observation({"symbol": "AAA", "decision_id": "rej"}, cycle_id="c", as_of="2026-09-01", path=path)
    FE.freeze_observation({"symbol": "BBB", "decision_id": "taken"}, cycle_id="c", as_of="2026-09-01",
                          entered=True, path=path)
    probed = []

    def boom(symbol, *a, **k):
        probed.append(symbol)
        raise RuntimeError("resolver broke")

    monkeypatch.setattr(OR, "counterfactual_outcome", boom)
    monkeypatch.setattr(FE, "load_ledger", lambda *a, **k: FE._read_jsonl(path))
    failures = OI._unreported_forward_failures()
    assert probed == ["AAA"]                                    # taken rows are not counterfactuals
    assert [f["symbol"] for f in failures] == ["AAA"]


def test_A9_a_signal_log_write_error_is_not_a_succeeded_outcome_job(monkeypatch):
    from research.autonomy import job_store as JS, jobs as J, outcome_integrity as OI

    base = lambda ctx: J.JobResult(JS.SUCCEEDED, "outcomes resolved",
                                   metadata={"signal_outcomes_error": "disk full"})
    monkeypatch.setattr(OI, "_BASE_HANDLER", base)
    monkeypatch.setattr(OI, "_unreported_forward_failures", lambda: [])
    out = OI.strict_outcome_resolution(NS())
    assert out.status == JS.RETRYABLE_FAILED and out.error_code == "SIGNAL_LOG_SETTLEMENT_FAILED"


def test_A10_the_scorecard_reads_the_recorded_policy_effect(tmp_path):
    from product.forward_evidence import freeze_observation
    from product.learning_scorecard import learning_influence

    row = freeze_observation({"symbol": "INFY", "decision": "BLOCK", "policy_effect": "BLOCK"},
                             cycle_id="one", as_of="2026-09-01", path=tmp_path / "forward.jsonl")
    assert row["policy_result"] == "BLOCK"
    policy = {"policy_id": "SETUP::VCP", "production_status": "ACTIVE"}
    assert learning_influence([policy], [row])["decisions_changed_by_learning"] == 1


def test_A6_a_simulated_rejection_never_feeds_a_forward_policy(tmp_path, monkeypatch):
    from product import forward_evidence as FE, forward_soak as FS
    from product.learning_policy_store import load_policies

    path = tmp_path / "forward.jsonl"
    monkeypatch.setattr(FE, "DEFAULT_PATH", path)
    FE.freeze_observation({"symbol": "INFY", "reason_code": "LOW_QUALITY_SETUP"}, cycle_id="sim",
                          as_of="2026-09-01", provenance=FE.BACKTEST, path=path)
    monkeypatch.setattr(FE, "load_ledger", lambda *a, **k: FE._read_jsonl(path))
    policy = tmp_path / "policy.json"
    FS.settle_pending_from_market(outcome_fn=lambda *a: rulebook_outcome(1.0, "WIN"), policy_path=policy)
    assert (load_policies(policy).get("policies") or []) == []


def test_A2_an_old_close_cannot_settle_a_reentry_in_the_same_symbol(tmp_path, monkeypatch):
    from product import forward_evidence as FE, forward_soak as FS
    from product import paper_learning_loop as PLL

    path = tmp_path / "forward.jsonl"
    monkeypatch.setattr(PLL, "_lookup_taken", lambda *a, **k: {})
    for did, day in [("a", "2026-09-01"), ("b", "2026-09-03")]:
        FE.freeze_observation({"symbol": "INFY", "decision_id": did, "paper_decision_id": f"pos-{did}"},
                              cycle_id=did, as_of=day, entered=True, path=path)
    close = dict(symbol="INFY", decision_id="pos-a", entry_date="2026-09-01", exit_date="2026-09-02",
                 realized_R=2.0, exit_reason="TARGET")
    FS.attach_closed_trades(NS(closed=[close]), path=path)
    FS.attach_closed_trades(NS(closed=[close]), path=path)              # every EOD re-iterates every close
    rows = {r["decision_id"]: r for r in FE.load_ledger(path)}
    assert rows["a"]["gross_R"] == 2.0 and rows["b"]["later_outcome"] is None
    # A legacy row (no paper id) joins on its entry session, and only when unambiguous.
    FE.freeze_observation({"symbol": "TCS", "decision_id": "l1"}, cycle_id="x", as_of="2026-09-05",
                          entered=True, path=path)
    FE.freeze_observation({"symbol": "TCS", "decision_id": "l2"}, cycle_id="y", as_of="2026-09-05",
                          entered=True, path=path)
    out = FS.attach_closed_trades(NS(closed=[dict(symbol="TCS", entry_date="2026-09-05", exit_date="2026-09-08",
                                                 realized_R=-1.0, exit_reason="STOP")]), path=path)
    assert out["attached"] == 0 and out["ambiguous_unresolved"] == 1


def test_A5_staging_keeps_each_freezes_own_outcome(tmp_path):
    from research import counterfactual_reconcile as RC

    base = dict(symbol="INFY", as_of="2026-09-01", reason_code="LOW_QUALITY_SETUP", classification="MISSED_WINNER",
                hypothetical_entry=100, hypothetical_target=115, outcome={"forward_return_pct": 9})
    rows = [dict(base, counterfactual_id="a", hypothetical_stop=90),
            dict(base, counterfactual_id="b", hypothetical_stop=98)]

    def resolve(symbol, day, entry, stop, target):
        return rulebook_outcome(1, "WIN") if stop == 90 else rulebook_outcome(-1, "LOSS")

    report = RC.reconcile([], rows, resolve)
    paths = RC.stage(report, [], rows, tmp_path)
    staged = RC._read_jsonl(paths["counterfactuals"])
    assert [r["classification"] for r in staged] == ["MISSED_WINNER", "AVOIDED_LOSER"]


def test_A4_a_legacy_counterfactual_label_does_not_train_the_challenger(monkeypatch):
    from core.evaluation_spec import spec_tag
    from product import challenger_learning as CL, trading_thesis as TT
    from research import feature_store

    def obs(meta):
        return {"meta": {"thesis_hash": "t"}, "outcome": 1.5, "outcome_meta": meta}

    legacy = obs({"evidence_class": "FORWARD_COUNTERFACTUAL", "not_pnl": True, "classification": "MISSED_WINNER"})
    rulebook = obs({"evidence_class": "FORWARD_COUNTERFACTUAL", "not_pnl": True, "eval_spec": spec_tag()})
    real = obs({"evidence_class": "PAPER_FORWARD"})
    monkeypatch.setattr(TT, "manifest", lambda: {"thesis_hash": "t"})
    monkeypatch.setattr(feature_store, "load_observations", lambda **kw: [legacy, rulebook, real])
    assert CL._rows() == [rulebook, real]


# ── BUY admission ─────────────────────────────────────────────────────────────

def _authority(snapshot_id="new", edges=None):
    return {"snapshot_id": snapshot_id, "backtest_provenance": proven_provenance(),
            "backtest_edges": edges or {"VCP": {"mean": .5, "lower": .2, "n": 90, "blocks": 30,
                                                "inference": "OVERLAP_ROBUST", "family": True}}}


def test_B3_a_later_report_does_not_retro_certify_an_earlier_scans_buy():
    from product import scan_store as SS

    def scan(evidence_id):
        return {"evidence_identity": {"id": evidence_id},
                "records": [{"symbol": "TCS", "signals": ["VCP"], "verdict": "BUY", "edge_evidence_id": evidence_id}]}

    assert SS.bind_evidence(scan("new"), authority=_authority("new"))["records"][0]["verdict"] == "BUY"
    old = SS.bind_evidence(scan("old"), authority=_authority("new"))
    assert old["records"][0]["verdict"] == "WATCH"
    assert any("not the current authority" in r for r in old["records"][0]["reasons"])


def test_C13_an_authority_outage_reads_evidence_unavailable_not_no_edge():
    from product import scan_store as SS
    from product.buy_admission import EVIDENCE_UNAVAILABLE, from_scan_payload

    out = SS.bind_evidence({"records": [{"symbol": "TCS", "verdict": "BUY"}], "evidence_identity": {"note": ""}},
                           authority=None)
    assert out["records"][0]["verdict"] == "WATCH"
    assert from_scan_payload(out, scan_current=True)["code"] == EVIDENCE_UNAVAILABLE


def test_C6_unverified_windows_or_stale_data_cannot_certify():
    from scan import unified_scanner as US

    assert US.evidence_provenance_note(proven_provenance()) == ""
    assert "official sessions" in US.evidence_provenance_note(proven_provenance(exact_horizon_check=False))
    assert "data is stale" in US.evidence_provenance_note(proven_provenance(data_last_session="2026-01-02"))


def test_C5_a_strong_signal_no_longer_carries_a_weak_one_past_the_gate():
    from scan import unified_scanner as US

    def ok(mean, lower):
        return {"mean": mean, "lower": lower, "n": 90, "blocks": 24, "inference": "OVERLAP_ROBUST", "family": True}

    edges = {"MOMENTUM": ok(0.60, 0.30), "VOL_SQUEEZE": ok(0.02, 0.01)}     # average +0.31, weakest +0.02
    assert US._evidence_verdict(80, ["MOMENTUM", "VOL_SQUEEZE"], signal_edges=edges, gate=True)[0] == "WATCH"
    edges["VOL_SQUEEZE"] = ok(0.20, 0.05)
    assert US._evidence_verdict(80, ["MOMENTUM", "VOL_SQUEEZE"], signal_edges=edges, gate=True)[0] == "BUY"


def test_C1_independent_blocks_are_disjoint_windows_not_occupied_bins():
    from research.overlap_inference import overlap_robust_test

    ordinals = [0] + [x for k in range(1, 10) for x in (20 * k - 1, 20 * k)] + [199]
    cells = {o: [1.0 + 0.01 * (i % 3), 1] for i, o in enumerate(ordinals)}
    out = overlap_robust_test(cells, 10)
    assert out["occupied_bins"] == 20 and out["independent_sessions"] == 11
    assert out["status"] != "OK"                                     # < 20 disjoint windows: no verdict


def test_C4_only_the_production_configuration_can_certify(tmp_path, monkeypatch):
    from research import alpha_ledger
    from scan import signal_backtest as SB

    assert SB.gate_program() == SB.gate_program(dict(SB.PRODUCTION_RUN_CONFIG))
    preview = SB.gate_program({**SB.PRODUCTION_RUN_CONFIG, "sample_step": 1})
    assert alpha_ledger.program_id(preview) != alpha_ledger.program_id(SB.gate_program())
    # The family is fixed before outcomes: untested catalogue signals count at p = 1.
    rows = {"A": {"inference": "OVERLAP_ROBUST", "expectancy_r_precise": 0.4, "se_r": 0.1, "df": 40,
                  "p_one_sided_raw": 0.0006}}
    small = SB.apply_family_correction(json.loads(json.dumps(rows)), {}, alpha=0.05 / 52)
    padded = SB.apply_family_correction(json.loads(json.dumps(rows)), {}, alpha=0.05 / 52, untested=1)
    assert small["survivors"] == 1 and padded["survivors"] == 0 and padded["family_size"] == 2


def test_C3_an_overridden_ledger_never_certifies_outside_tests(tmp_path, monkeypatch):
    from research import alpha_ledger

    att = proven_provenance()["alpha_accounting"]
    assert alpha_ledger.verify(att, program=None) == ""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert "overridden" in alpha_ledger.verify(dict(att, _bust=1), program=None)


def test_C3_a_lost_ledger_does_not_open_a_fresh_budget(tmp_path, monkeypatch):
    from scan import signal_backtest as SB

    out = tmp_path / "signal_backtest.json"
    out.write_text(json.dumps({"evidence_provenance": {"alpha_accounting": {"look_index": 7}}}))
    monkeypatch.setattr(SB, "_OUT", str(out))
    monkeypatch.setenv("QT_ALPHA_LEDGER", str(tmp_path / "gone.db"))
    assert SB._alpha_ledger_lost() is True
    out.write_text(json.dumps({}))
    assert SB._alpha_ledger_lost() is False


# ── point in time ─────────────────────────────────────────────────────────────

def _priced_breakout(scale):
    from tests.test_evidence_gate import _breakout_frame

    df = _breakout_frame()
    for c in ("close", "high", "low"):
        df[c] = df[c] * scale
    df["open"] = df["close"]
    df["volume"] = df["volume"] / scale
    return df


def test_B1_a_split_after_the_decision_cannot_remove_the_setup():
    from data.corporate_actions import adjust_frame, raw_price_multiplier
    from scan.unified_scanner import UnifiedScanner

    raw = _priced_breakout(30.0 / 212.0)                         # traded at ~₹30 on the decision day
    scanner = UnifiedScanner(calibration_snapshot={"snapshot_id": "neutral", "multipliers": {}},
                             evidence_gate=False)
    assert scanner._analyze("X", raw) is not None
    future = raw.index[-1] + pd.Timedelta(days=3)
    adjusted = adjust_frame(raw, [{"ex_date": future, "factor": 2.0, "type": "split"}])
    assert float(adjusted["close"].iloc[-1]) < 20 and raw_price_multiplier(adjusted) == 2.0
    assert scanner._analyze("X", adjusted) is not None           # judged on the ₹30 that traded
    assert raw_price_multiplier(adjust_frame(raw, [{"ex_date": raw.index[10], "factor": 2.0}])) == 1.0


def test_B2_replay_scores_the_past_with_neutral_not_later_calibration(monkeypatch):
    from product import historical_replay as HR
    from scan import calibration_snapshot as CS

    frame = _priced_breakout(1.0)
    day = str(frame.index[-1].date())

    def scored(weight):
        monkeypatch.setattr(CS, "get_or_create_snapshot", lambda: {
            "snapshot_id": f"november-{weight}", "multipliers": {k: weight for k in ("VOL_SQUEEZE", "POCKET_PIVOT",
                                                                                   "BREAKOUT_52W", "MOMENTUM")}})
        out = HR.scan_session(day, ["X"], ohlcv_fn=lambda _s: frame)
        return out["calibration_snapshot_id"], [r.get("score") for r in out.get("records") or []]

    a, b = scored(0.45), scored(1.25)
    assert a == b and a[0] == HR.NEUTRAL_REPLAY_CALIBRATION["snapshot_id"]


def test_B9_strict_coverage_ends_with_the_verified_window():
    from research import edge_diagnosis as ED

    days = pd.bdate_range("2021-08-30", "2022-10-03").strftime("%Y-%m-%d")
    coverage = {"windows": {"one": {"covered_from": "2021-08-28", "covered_to": "2022-08-31", "success": True}}}
    start, end = ED.coverage_span(coverage)
    tiers = ED.tier_map(dict(zip(days, range(len(days)))), start, end)
    assert end == "2022-08-31" and tiers["2022-10-03"] == "UNCOVERED" and tiers["2022-08-31"] == "STRICT"


def test_B10_a_parseable_but_wrong_membership_file_is_incomplete(tmp_path, monkeypatch):
    from data import nse_universe as U

    monkeypatch.setattr(U, "get_nse_universe", lambda: ["TCS"])
    for text in ('{"symbols":[]}', "[]", '[{"nope": 1}]'):
        path = tmp_path / "universe_history.json"
        path.write_text(text)
        assert U.point_in_time_universe("2026-09-18", path=path)["survivorship_complete"] is False


# ── second pass: GPT verified the fixes (01dec27c) and found these gaps ──────

def test_QT02_a_lost_ledger_stays_blocked_across_runs(tmp_path, monkeypatch):
    from tests.test_alpha_ledger import _stub_walk_forward

    sb = _stub_walk_forward(monkeypatch, tmp_path)
    ledger = tmp_path / "alpha.db"
    monkeypatch.setenv("QT_ALPHA_LEDGER", str(ledger))
    monkeypatch.setattr(sb, "MIN_SYMBOLS_TO_SAVE", 1)

    def accounting():
        return sb.run_backtest()["evidence_provenance"]["alpha_accounting"]

    assert accounting()["look_index"] == 1
    ledger.unlink()
    assert accounting()["alpha"] == 0
    assert accounting()["alpha"] == 0                          # the loss is latched, not forgotten
    assert not ledger.exists()
    (tmp_path / "signal_backtest.json").unlink()                # even with the report gone ...
    assert accounting()["alpha"] == 0                          # ... the ledger's sentinel remembers


def test_QT06_store_bars_without_the_session_list_never_settle(monkeypatch):
    from core import outcome_resolver as OR, session_window as SW

    days = pd.bdate_range("2026-02-03", periods=11).delete(1)
    frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.}, index=days)
    frame.loc[days[-1], "high"] = 111.
    monkeypatch.setattr(OR, "_ohlcv", lambda _: frame)
    monkeypatch.setattr(OR, "_store_loaded", lambda: True)      # production: the store is loaded
    monkeypatch.setattr(SW, "official_dates", lambda: [])
    assert OR.resolve_trade("TCS", "2026-02-02", 100, 95, 110) is None


def test_QT07_a_missing_or_incoherent_open_cannot_fill_below_the_range():
    from core.evaluation_spec import simulate_trade

    assert not simulate_trade(100, 90, 120, [111], [105], [108], opens=[float("nan")]).filled
    assert not simulate_trade(100, 90, 120, [111], [105], [108], opens=[95]).filled     # open below the low
    assert simulate_trade(100, 90, 120, [111], [105], [108], opens=[106]).fill_price == 106


def test_QT08_a_close_never_borrows_another_positions_context(monkeypatch):
    from product import _paper_learning_loop_core as core

    monkeypatch.setattr(core, "_read_jsonl", lambda _p: [
        {"symbol": "TCS", "paper_decision_id": "pos-b", "entry_date": "2026-09-01", "setup_label": "VCP"}])
    assert core._lookup_taken("TCS", decision_id="pos-a", entry_date="2026-09-01") == {}
    assert core._lookup_taken("TCS", decision_id="pos-b", entry_date="2026-09-01")["setup_label"] == "VCP"
    monkeypatch.setattr(core, "_read_jsonl", lambda _p: [
        {"symbol": "TCS", "entry_date": "2026-09-01", "setup_label": "A"},
        {"symbol": "TCS", "entry_date": "2026-09-01", "setup_label": "B"}])
    assert core._lookup_taken("TCS", entry_date="2026-09-01") == {}          # ambiguous: no guess


def test_QT08_a_legacy_close_cannot_settle_a_second_row_of_its_session(tmp_path, monkeypatch):
    from product import forward_evidence as FE, forward_soak as FS
    from product import paper_learning_loop as PLL

    path = tmp_path / "forward.jsonl"
    monkeypatch.setattr(PLL, "_lookup_taken", lambda *a, **k: {})
    for did in ("old", "new"):
        FE.freeze_observation({"symbol": "TCS", "decision_id": did}, cycle_id=did, as_of="2026-09-05",
                              entered=True, path=path)
    FE.attach_settlement("old", classification="STOP", gross_R=-1.0, path=path)   # attached by the old code
    out = FS.attach_closed_trades(NS(closed=[dict(symbol="TCS", entry_date="2026-09-05", exit_date="2026-09-08",
                                                 realized_R=-1.0, exit_reason="STOP")]), path=path)
    assert out["attached"] == 0
    assert {r["decision_id"]: r["later_outcome"] for r in FE.load_ledger(path)}["new"] is None


def test_QT09_promotion_ignores_legacy_equity_grades(tmp_path):
    from core.evaluation_spec import spec_tag
    from product.evolution.scorecard import paired_comparison

    ledger = tmp_path / "shadow.jsonl"
    base = {"domain": "EQUITY", "market_snapshot_id": "s", "outcome": {"settled": True}}
    rows = [dict(base, policy_id="champ", decision="WAIT", counterfactual_R=2),
            dict(base, policy_id="chal", decision="ENTER_NOW", counterfactual_R=2)]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert paired_comparison("champ", "chal", domain="EQUITY", path=ledger)["paired_snapshots"] == 0
    ledger.write_text("".join(json.dumps(dict(r, eval_spec=spec_tag())) + "\n" for r in rows))
    assert paired_comparison("champ", "chal", domain="EQUITY", path=ledger)["paired_snapshots"] == 1


def test_QT10_a_replayed_order_that_never_traded_has_no_return(monkeypatch):
    from product import historical_replay as HR
    from product.scorecards import reason_scorecards

    days = pd.bdate_range("2026-02-03", periods=10)
    frame = pd.DataFrame({"open": 94., "high": 95., "low": 93., "close": 94.}, index=days)
    monkeypatch.setattr(HR, "ohlcv_as_of", lambda *a, **kw: frame)
    rows = HR.evaluate_outcomes([{"symbol": "TCS", "as_of": "2026-02-02", "decision": "AVOID",
                                  "reason_code": "LOW_QUALITY_SETUP", "entry": 100, "stop": 90, "target": 110}])
    assert rows[0]["rulebook_outcome"] == "NO_FILL" and rows[0]["forward_return_pct"] is None
    card = (reason_scorecards(rows).get("rows") or [{}])[0]
    assert card.get("r_expectancy") is None


def test_QT11_an_unusable_frame_cannot_pad_a_thin_run(tmp_path, monkeypatch):
    from data import bhavcopy_store as BS
    from tests.test_alpha_ledger import _stub_walk_forward

    sb = _stub_walk_forward(monkeypatch, tmp_path)
    read = BS.get_ohlcv
    junk = pd.DataFrame(index=pd.bdate_range("2026-06-01", periods=100))
    monkeypatch.setattr(BS, "store_symbols", lambda: ["S0", "S1", "S2", "JUNK"])
    monkeypatch.setattr(BS, "get_ohlcv", lambda symbol: junk if symbol == "JUNK" else read(symbol))
    monkeypatch.setattr(sb, "MIN_SYMBOLS_TO_SAVE", 4)
    report = sb.run_backtest()
    assert report["evidence_provenance"]["symbols_evaluated"] == 3
    assert not (tmp_path / "signal_backtest.json").exists()


def test_QT13_a_null_listing_date_does_not_certify_membership(tmp_path, monkeypatch):
    from data import nse_universe as U

    table = tmp_path / "membership.json"
    monkeypatch.setattr(U, "get_nse_universe", lambda: ["TCS"])
    for rows in ('[{"symbol":"TCS","listed":null}]', '[{"symbol":"","listed":"2020-01-01"}]',
                 '[{"symbol":"TCS","listed":"2020-01-01","delisted":"2019-01-01"}]'):
        table.write_text(rows)
        assert U.point_in_time_universe("2026-09-01", table)["survivorship_complete"] is False


def test_QT14_a_default_neutral_replay_is_recognised_as_current(tmp_path):
    from product import historical_replay as HR

    dates = lambda: ["2026-09-01", "2026-09-02"]
    saved = HR.replay_identity(dates_fn=dates, calibration_snapshot=HR.NEUTRAL_REPLAY_CALIBRATION)
    HR.report_path(tmp_path).write_text(json.dumps(dict(saved, status="SUCCEEDED")))
    assert HR.replay_is_current(directory=tmp_path, dates_fn=dates)


def test_QT05_home_revokes_a_buy_when_the_authority_target_disappears(tmp_path, monkeypatch):
    from product import home_os as HO, scan_store as SS
    from scan import calibration_snapshot as CS

    target = tmp_path / "cal.json"
    target.write_text(json.dumps(_authority("cal")))
    (tmp_path / "current.json").write_text('{"snapshot_id":"cal"}')
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({"schema_version": 2, "provenance": {}, "evidence_identity": {"id": "cal"},
                                "records": [{"symbol": "TCS", "verdict": "BUY", "signals": ["VCP"],
                                             "edge_evidence_id": "cal"}]}))
    monkeypatch.setattr(CS, "DEFAULT_DIR", tmp_path)
    monkeypatch.setattr(SS, "DEFAULT_SCAN_PATH", scan)
    monkeypatch.setattr(SS, "scan_source_status", lambda _: (True, ""))
    monkeypatch.setattr(SS, "_AUTHORITY_CACHE", {"key": None, "snapshot": None})
    monkeypatch.setattr(HO, "_SCAN_EVIDENCE_CACHE", {"key": None, "value": None})
    load_scan = SS.load_scan
    monkeypatch.setattr(SS, "load_scan", lambda *a, **k: load_scan(scan))
    rows = [{"symbol": "TCS", "decision": "BUY", "entry": 100, "stop": 95, "target": 110}]
    assert HO.build_buy_only(rows, verified=True)["count"] == 1
    target.unlink()
    assert load_scan(scan)["records"][0]["verdict"] == "WATCH"
    assert HO.build_buy_only(rows, verified=True)["count"] == 0


def test_C8_the_desk_alert_never_sends_a_withdrawn_buy_as_a_pick(tmp_path, monkeypatch):
    """Matching scan generation, but load_scan() withdrew TCS: the alert gets it as research."""
    from product import recommendations_store as RS, scan_store as SS
    from research.autonomy.telegram_notifications import TelegramNotifier

    stamp = "2026-11-20T05:00:00+00:00"
    ws = {"scan_scanned_at": stamp, "categories": [{"cards": [
        {"symbol": "INFY", "reco_tier": "high_conviction"}, {"symbol": "TCS", "reco_tier": "high_conviction"}]}]}
    monkeypatch.setattr(RS, "load_recommendations", lambda *a, **k: ws)
    monkeypatch.setattr(SS, "load_scan", lambda *a, **k: {"scanned_at": stamp, "records": [
        {"symbol": "INFY", "verdict": "BUY"}, {"symbol": "TCS", "verdict": "WATCH"}]})
    seen = {}
    n = TelegramNotifier(tmp_path, engine_factory=lambda: None)
    monkeypatch.setattr(n, "notify_recommendations", lambda w: seen.update(w=w) or {"sent": False})
    monkeypatch.setattr(n, "notify_market_report_if_due", lambda: {})
    monkeypatch.setattr(n, "drain_fno_alerts", lambda: {}, raising=False)
    n.drain_desk_alerts()
    high, good = TelegramNotifier._reco_tiers(seen["w"])
    assert [c["symbol"] for c in high] == ["INFY"] and good == []


def test_C10_the_legacy_trade_command_is_dormant_in_the_canonical_stack(monkeypatch):
    """/trade reads the legacy auto_scan store, filled only by start_background_scan (legacy_app/ui)."""
    import execution.autopilot as AP
    import scan.auto_scan as AS
    from alerts import telegram_commands as TC

    calls = []
    monkeypatch.setattr(AP, "get_status", lambda: {"armed": True, "mode": "PAPER"})
    monkeypatch.setattr(AP, "consider", lambda **k: calls.append(k) or True)
    monkeypatch.setattr(AS, "_results", [])
    assert AS._thread_started is False                       # nothing canonical started the legacy worker
    msg = TC._trade_now()
    assert calls == [] and "Trade liya" not in msg            # nothing to trade: no legacy entry
    import inspect
    import terminal_product_api, operations.market_ops as MO, research.autonomy.supervisor as SUP
    for module in (terminal_product_api, MO, SUP):
        assert "start_background_scan" not in inspect.getsource(module)
