"""The learning ladder counts independent observations, not copies.

Owner's Mac 2026-11-11: 36 paper cycles re-froze the same 175 rejected
symbols as 3,744 forward-ledger rows. Each row was settled and ingested on its
own, so REJECT::LOW_QUALITY_SETUP reached n=2,989. A policy at n>=20 with a
large enough edge changes paper selection (SUPPORT / PENALIZE / BLOCK), so
copies of one day's names could have moved decisions with no shrinkage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from product import forward_evidence as FE
from product import forward_soak as FS
from product import learning_observations as LO
from product import learning_policy_store as LPS
from product.paper_learning_loop import ingest_closed_trade, ingest_counterfactual
from core.evaluation_spec import spec_tag
from tests.evidence_fixtures import rulebook_outcome


def _policy(path, policy_id):
    store = json.loads(path.read_text())
    return next(p for p in store["policies"] if p["policy_id"] == policy_id)


def _settled_reject(symbol="TCS", as_of="2026-09-01", reason="ENTRY_TOO_EXTENDED", cls="AVOIDED_LOSER"):
    return {
        "symbol": symbol,
        "as_of": as_of,
        "reason_code": reason,
        "classification": cls,
        "eval_spec": spec_tag(),
        "evidence": {"setup_label": "VCP", "regime": "RISK_ON"},
    }


def test_same_counterfactual_from_many_cycles_counts_once(tmp_path):
    path = tmp_path / "policies.json"
    for _cycle in range(36):
        ingest_counterfactual(_settled_reject(), path=path)
    assert _policy(path, "REJECT::ENTRY_TOO_EXTENDED")["sample_size"] == 1


def test_distinct_symbols_days_and_reasons_are_distinct_observations(tmp_path):
    path = tmp_path / "policies.json"
    ingest_counterfactual(_settled_reject("TCS", "2026-09-01"), path=path)
    ingest_counterfactual(_settled_reject("INFY", "2026-09-01"), path=path)
    ingest_counterfactual(_settled_reject("TCS", "2026-09-02"), path=path)
    ingest_counterfactual(_settled_reject("TCS", "2026-09-01", reason="DD_GATE_FAILED"), path=path)
    row = _policy(path, "REJECT::ENTRY_TOO_EXTENDED")
    assert row["sample_size"] == 3
    assert row["sessions_observed"] == 2
    assert _policy(path, "REJECT::DD_GATE_FAILED")["sample_size"] == 1


def test_duplicate_counterfactual_does_not_touch_calibration_record(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "product.evidence_intelligence.record_resolved_prediction",
        lambda *a, **k: calls.append(a),
    )
    path = tmp_path / "policies.json"
    row = {**_settled_reject(), "decision_id": "d-1", "counterfactual_R": -0.5}
    ingest_counterfactual(row, path=path)
    ingest_counterfactual({**row, "decision_id": "d-2"}, path=path)  # next cycle, same unit
    assert len(calls) == 1


def test_reingesting_one_closed_trade_counts_once(tmp_path):
    path = tmp_path / "policies.json"
    trade = {"symbol": "TCS", "setup_label": "VCP", "realized_R": -1.0, "exit_reason": "STOP",
             "entry_date": "2026-09-01", "exit_date": "2026-09-03"}
    for _ in range(5):
        ingest_closed_trade(dict(trade), path=path)
    assert _policy(path, "SETUP::VCP")["sample_size"] == 1


def test_large_n_from_few_sessions_cannot_influence_selection(tmp_path):
    """n=60 from 2 sessions is still one or two market paths: EXPERIMENTAL."""
    path = tmp_path / "policies.json"
    for i in range(60):
        ingest_counterfactual(
            _settled_reject(f"S{i:03d}", "2026-09-01" if i % 2 else "2026-09-02", cls="MISSED_WINNER"),
            path=path,
        )
    row = _policy(path, "REJECT::ENTRY_TOO_EXTENDED")
    assert row["sample_size"] == 60
    assert row["sessions_observed"] == 2
    assert row["production_status"] == LPS.EXPERIMENTAL
    assert row["confidence"] == "INSUFFICIENT_EVIDENCE"


def test_same_n_across_enough_sessions_is_promoted(tmp_path):
    path = tmp_path / "policies.json"
    for i in range(60):
        day = f"2026-{9 + i // 28:02d}-{i % 28 + 1:02d}"
        ingest_counterfactual(_settled_reject(f"S{i:03d}", day, cls="MISSED_WINNER"), path=path)
    row = _policy(path, "REJECT::ENTRY_TOO_EXTENDED")
    assert row["sessions_observed"] == 60
    assert row["production_status"] == LPS.ACTIVE


@pytest.mark.parametrize(
    "n,sessions,expected",
    [
        (100, None, LPS.EXPERIMENTAL),   # unknown coverage fails closed
        (100, 0, LPS.EXPERIMENTAL),
        (100, 19, LPS.EXPERIMENTAL),
        (100, 20, LPS.ELIGIBLE),
        (100, 39, LPS.ELIGIBLE),
        (100, 40, LPS.ACTIVE),
        (25, 40, LPS.ELIGIBLE),          # sample floor still binds
    ],
)
def test_session_floor_caps_status(n, sessions, expected):
    assert LPS._status_for(n, 0.5, source="paper_forward", sessions=sessions) == expected


def test_historical_sources_stay_capped_regardless_of_sessions():
    assert LPS._status_for(500, 1.0, source="backtest_reproduced", sessions=400) == LPS.EXPERIMENTAL


def test_a_crash_after_the_claim_is_replayed_exactly_once(tmp_path, monkeypatch):
    """Cross-review A-3: claim-then-apply used to LOSE the observation on a crash in between.

    The two-phase claim keeps it CLAIMED with its payload; the retry of the
    same unit resumes it and applies it once. Never zero, never twice.
    """
    path = tmp_path / "policies.json"
    real_save = LPS.save_policies
    boom = {"armed": True}

    def flaky_save(payload, p=None):
        if boom.pop("armed", False):
            raise RuntimeError("power cut")
        return real_save(payload, p)

    monkeypatch.setattr(LPS, "save_policies", flaky_save)
    with pytest.raises(RuntimeError):
        ingest_counterfactual(_settled_reject(), path=path)
    ingest_counterfactual(_settled_reject(), path=path)  # retry of the same unit
    assert _policy(path, "REJECT::ENTRY_TOO_EXTENDED")["sample_size"] == 1
    ingest_counterfactual(_settled_reject(), path=path)  # a third delivery changes nothing
    assert _policy(path, "REJECT::ENTRY_TOO_EXTENDED")["sample_size"] == 1
    ingest_counterfactual(_settled_reject("INFY"), path=path)
    assert _policy(path, "REJECT::ENTRY_TOO_EXTENDED")["sample_size"] == 2


def test_observation_ledger_lives_beside_its_policy_store(tmp_path):
    path = tmp_path / "policies.json"
    ingest_counterfactual(_settled_reject(), path=path)
    ledger = LO.ledger_path_for(path)
    assert ledger.exists()
    assert LO.count(ledger, "COUNTERFACTUAL") == 1
    assert LO.seen(ledger, "COUNTERFACTUAL", "cf:2026-09-01|TCS|ENTRY_TOO_EXTENDED")


def test_forward_soak_cycles_of_same_rejections_count_independent_units(tmp_path):
    """End to end: 12 cycles x 5 names settle 60 ledger rows but 5 observations."""
    policy_path = tmp_path / "policies.json"
    for cycle in range(12):
        FE.freeze_cycle({
            "as_of": "2026-09-01",
            "cycle_id": f"c{cycle}",
            "rules_hash": "hash",
            "rejections": [
                {"symbol": f"SYM{i}", "reason_code": "ENTRY_TOO_EXTENDED", "decision": "WAIT",
                 "entry": 100, "stop": 94, "target": 115, "setup_label": "VCP", "regime": "RISK_ON"}
                for i in range(5)
            ],
        })
    out = FS.settle_pending_from_market(outcome_fn=lambda _s, _d, _r: rulebook_outcome(-0.5, "LOSS", -3.0), policy_path=policy_path)
    assert out["updated"] == 60  # every ledger row keeps its own settlement (audit trail)
    assert _policy(policy_path, "REJECT::ENTRY_TOO_EXTENDED")["sample_size"] == 5


def test_learning_memory_counts_decisions_not_cycle_copies(tmp_path, monkeypatch):
    from product import autonomous_loop as LOOP

    cf = tmp_path / "cf.jsonl"
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(cf))
    monkeypatch.setattr(LOOP, "MEMORY_PATH", tmp_path / "memory.json")
    monkeypatch.setattr(LOOP, "LEARNING_PATH", tmp_path / "learning.jsonl")
    rows = []
    for _cycle in range(10):  # ten cycles re-freeze the same two decisions
        rows.append({"as_of": "2026-09-01", "symbol": "TCS", "reason_code": "DD_GATE_FAILED",
                     "classification": "MISSED_WINNER", "setup": "VCP", "eval_spec": spec_tag()})
        rows.append({"as_of": "2026-09-01", "symbol": "INFY", "reason_code": "DD_GATE_FAILED",
                     "classification": "CORRECT_REJECTION", "setup": "VCP", "eval_spec": spec_tag()})
    cf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    memory = LOOP.consume_learning_memory("2026-09-02")
    assert memory["ledger_rows"] == 20
    assert memory["observations"] == 2
    assert memory["classification_counts"] == {"MISSED_WINNER": 1, "CORRECT_REJECTION": 1}
    assert memory["reason_aggregations"]["DD_GATE_FAILED"]["decisions"] == 2


def test_repair_rebuilds_inflated_policy_from_independent_units_and_blocks_reingest(tmp_path):
    from product import learning_policy_repair as REPAIR

    policy_file = tmp_path / "product" / "policies.json"
    LPS.upsert_policy(  # what the duplicated ingest left behind: n=40 for 2 decisions
        policy_id="REJECT::DD_GATE_FAILED", dimension="reason_code", bucket="DD_GATE_FAILED",
        sample_size=40, expectancy_R=-0.40, source="counterfactual_not_pnl", path=policy_file,
        extra={"affects_selection": False},
    )
    LPS.upsert_policy(  # not counterfactual-owned: must be left alone
        policy_id="SETUP::VCP", dimension="setup", bucket="VCP", sample_size=7,
        expectancy_R=0.3, source="paper_forward_taken_gross_only", path=policy_file,
    )
    fwd = tmp_path / "fwd.jsonl"
    rows = []
    for cycle in range(20):
        for sym, cls in (("TCS", "MISSED_WINNER"), ("INFY", "CORRECT_REJECTION")):
            rows.append({"symbol": sym, "reason_code": "DD_GATE_FAILED", "market_timestamp": "2026-09-01",
                         "entered": False, "later_outcome": {"forward_return_pct": 1.0, "eval_spec": spec_tag()},
                         "counterfactual_classification": cls, "cycle_id": f"c{cycle}"})
    fwd.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = REPAIR.plan(forward_path=fwd, cf_path=tmp_path / "missing.jsonl", policy_file=policy_file)
    assert report["independent_units"] == 2
    (change,) = report["changes"]
    assert (change["stored_n"], change["independent_n"], change["expectancy_R"]) == (40, 2, 0.0)

    out = REPAIR.apply_plan(report, backup_dir=tmp_path / "backup")
    assert Path(out["backup"]).exists()
    rebuilt = _policy(policy_file, "REJECT::DD_GATE_FAILED")
    assert rebuilt["sample_size"] == 2
    assert rebuilt["affects_selection"] is False
    assert rebuilt["rebuilt_from_independent_units"]["previous_sample_size"] == 40
    assert _policy(policy_file, "SETUP::VCP")["sample_size"] == 7
    # The live ingest now treats those decisions as already counted.
    dup = ingest_counterfactual({"symbol": "TCS", "as_of": "2026-09-01", "reason_code": "DD_GATE_FAILED",
                                 "classification": "MISSED_WINNER", "eval_spec": spec_tag()}, path=policy_file)
    assert dup.get("duplicate_observation") is True
    assert _policy(policy_file, "REJECT::DD_GATE_FAILED")["sample_size"] == 2


def test_repair_never_rebuilds_from_legacy_horizon_labels(tmp_path):
    """The repair mirrors the live ingest: 5-session close-to-close labels are excluded,
    and a policy whose only evidence is legacy is reset to n = 0, not kept stale."""
    from product import learning_policy_repair as REPAIR

    policy_file = tmp_path / "product" / "policies.json"
    for reason in ("DD_GATE_FAILED", "LOW_QUALITY_SETUP"):
        LPS.upsert_policy(policy_id=f"REJECT::{reason}", dimension="reason_code", bucket=reason,
                          sample_size=30, expectancy_R=-0.40, source="counterfactual_not_pnl", path=policy_file,
                          extra={"affects_selection": False})
    rows = [  # LOW_QUALITY_SETUP has only legacy labels; DD_GATE_FAILED has one of each
        {"symbol": "A", "reason_code": "LOW_QUALITY_SETUP", "market_timestamp": "2026-09-01", "entered": False,
         "later_outcome": {"forward_return_pct": 9.0}, "counterfactual_classification": "MISSED_WINNER"},
        {"symbol": "B", "reason_code": "DD_GATE_FAILED", "market_timestamp": "2026-09-01", "entered": False,
         "later_outcome": {"forward_return_pct": 9.0}, "counterfactual_classification": "MISSED_WINNER"},
        {"symbol": "C", "reason_code": "DD_GATE_FAILED", "market_timestamp": "2026-09-01", "entered": False,
         "later_outcome": {"forward_return_pct": -6.0, "eval_spec": spec_tag()},
         "counterfactual_classification": "AVOIDED_LOSER"},
    ]
    fwd = tmp_path / "fwd.jsonl"
    fwd.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = REPAIR.plan(forward_path=fwd, cf_path=tmp_path / "missing.jsonl", policy_file=policy_file)
    assert report["independent_units"] == 1 and report["excluded_incompatible_eval_spec"] == {"forward_ledger": 2}
    by_id = {c["policy_id"]: c for c in report["changes"]}
    assert (by_id["REJECT::DD_GATE_FAILED"]["independent_n"], by_id["REJECT::DD_GATE_FAILED"]["expectancy_R"]) \
        == (1, 0.4)
    assert by_id["REJECT::LOW_QUALITY_SETUP"]["independent_n"] == 0
    REPAIR.apply_plan(report, backup_dir=tmp_path / "backup")
    reset = _policy(policy_file, "REJECT::LOW_QUALITY_SETUP")
    assert reset["sample_size"] == 0 and reset["production_status"] == "OBSERVING"


def test_scoreboard_counts_decisions_and_never_labels_unmeasured_trades_measured(monkeypatch):
    # The scoreboard counts REAL_FORWARD_MARKET rows only; pytest rows default to TEST_FIXTURE.
    monkeypatch.setenv("QT_EVIDENCE_PROVENANCE", FE.REAL_FORWARD_MARKET)
    for cycle in range(8):
        FE.freeze_cycle({
            "as_of": "2026-09-01",
            "cycle_id": f"c{cycle}",
            "rules_hash": "hash",
            "rejections": [
                {"symbol": f"SYM{i}", "reason_code": "ENTRY_TOO_EXTENDED", "decision": "WAIT",
                 "entry": 100, "stop": 94, "target": 115, "setup_label": "VCP", "regime": "RISK_ON"}
                for i in range(30)
            ],
        })
    FS.settle_pending_from_market(outcome_fn=lambda _s, _d, _r: rulebook_outcome(-0.5, "LOSS", -3.0))
    board = FS.scoreboard()
    assert board["real_forward_ledger_rows"] == 240
    assert board["real_forward_observations"] == 30
    assert board["rejected_candidates_settled"] == 30
    assert board["rejected_ledger_rows_settled"] == 240
    assert board["rejection_gate_evidence"]["ENTRY_TOO_EXTENDED"]["n"] == 30
    # 30 settled rejections and zero settled trades: the trade record is unmeasured.
    assert board["settled_trades"] == 0
    assert board["evidence_label"] == "INSUFFICIENT EVIDENCE"
    assert board["win_rate"] is None


def test_home_learning_impact_counts_decisions_not_cycle_copies(monkeypatch, tmp_path):
    """Home's LEARNING IMPACT showed 740 missed winners for ~185 decisions (2026-11-19)."""
    from product import autonomous_learning as AL

    cf = tmp_path / "cf.jsonl"
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(cf))
    rows = []
    for _cycle in range(12):
        rows.append({"as_of": "2026-09-01", "symbol": "TCS", "reason_code": "DD_GATE_FAILED", "classification": "MISSED_WINNER"})
        rows.append({"as_of": "2026-09-01", "symbol": "INFY", "reason_code": "DD_GATE_FAILED", "classification": "CORRECT_REJECTION"})
    cf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("QT_EVIDENCE_PROVENANCE", FE.REAL_FORWARD_MARKET)
    for cycle in range(5):
        FE.freeze_cycle({"as_of": "2026-09-01", "cycle_id": f"c{cycle}", "rules_hash": "h",
                         "rejections": [{"symbol": "TCS", "reason_code": "DD_GATE_FAILED", "decision": "WAIT",
                                         "entry": 100, "stop": 94, "target": 115}]})
    counts = AL._evidence_counts()
    assert counts["missed_winners"] == 1
    assert counts["correct_rejects"] == 1
    assert counts["forward_evidence_count"] == 1
