"""The reconciliation of legacy 5-session counterfactuals is a read-only dry run with staged copies."""

from __future__ import annotations

import json

import pytest

from core.evaluation_spec import spec_tag
from research import counterfactual_reconcile as RC
from tests.evidence_fixtures import rulebook_outcome

D = "2026-09-01"


def _ledger():
    base = {"market_timestamp": D, "reason_code": "LOW_QUALITY_SETUP", "entry": 100, "stop": 94, "target": 115}
    return [
        # legacy 5-session "missed winners": one was stopped out first, one never triggered
        {**base, "decision_id": "d1", "symbol": "STOPPED", "entered": False,
         "later_outcome": {"forward_return_pct": 9.0}, "counterfactual_classification": "MISSED_WINNER"},
        {**base, "decision_id": "d2", "symbol": "NOFILL", "entered": False,
         "later_outcome": {"forward_return_pct": 6.0}, "counterfactual_classification": "MISSED_WINNER"},
        {**base, "decision_id": "d3", "symbol": "WAITING", "entered": False,
         "later_outcome": {"forward_return_pct": -1.0}, "counterfactual_classification": "FLAT"},
        # already on the rulebook: untouched
        {**base, "decision_id": "d4", "symbol": "NEW", "entered": False,
         "later_outcome": {"forward_return_pct": 7.0, "eval_spec": spec_tag(), "counterfactual_R": 1.1},
         "counterfactual_classification": "MISSED_WINNER"},
        # a taken trade stamped with a counterfactual before its paper close: blocked
        {**base, "decision_id": "d5", "symbol": "TAKEN", "entered": True,
         "later_outcome": {"forward_return_pct": 4.0}, "counterfactual_classification": "FLAT"},
        # a taken trade with its real close: fine
        {**base, "decision_id": "d6", "symbol": "CLOSED", "entered": True, "gross_R": 1.4,
         "later_outcome": {"forward_return_pct": 8.0}, "counterfactual_classification": "TARGET"},
    ]


OUT = {"STOPPED": rulebook_outcome(-1.05, "LOSS", -6.0), "NOFILL": {**rulebook_outcome(0.0, "NO_FILL", 0.0),
       "net_r": 0.0}, "WAITING": None}


def _fn(sym, day, entry, stop, target):
    return OUT.get(sym)


def test_the_dry_run_names_every_legacy_flip_and_blocked_taken_trade():
    rep = RC.reconcile(_ledger(), [], _fn)
    assert rep["legacy_rejection_rows"] == 3 and rep["rejudged_pending"] == 1
    assert rep["label_transitions"] == {"MISSED_WINNER -> AVOIDED_LOSER": 1, "MISSED_WINNER -> NOT_TRIGGERED": 1,
                                        "FLAT -> PENDING": 1}
    assert [b["symbol"] for b in rep["blocked_taken_rows"]] == ["TAKEN"]
    assert rep["per_gate_before"]["LOW_QUALITY_SETUP"]["missed_winner_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert rep["per_gate_after"]["LOW_QUALITY_SETUP"]["missed_winner_rate"] == 0.0


def test_staging_writes_copies_and_preserves_every_legacy_value(tmp_path):
    rows = _ledger()
    original = json.dumps(rows, sort_keys=True)
    rep = RC.reconcile(rows, [], _fn)
    paths = RC.stage(rep, rows, [], tmp_path / "stage")
    assert json.dumps(rows, sort_keys=True) == original                      # inputs untouched
    staged = {r["symbol"]: r for r in map(json.loads, open(paths["forward_evidence"]).read().splitlines())}
    assert staged["STOPPED"]["counterfactual_classification"] == "AVOIDED_LOSER"
    assert staged["STOPPED"]["later_outcome"]["eval_spec"] == spec_tag()
    assert staged["STOPPED"]["legacy_counterfactual_classification"] == "MISSED_WINNER"
    assert staged["STOPPED"]["legacy_later_outcome"] == {"forward_return_pct": 9.0}
    assert staged["WAITING"] == rows[2]                                       # pending stays as it was
    assert staged["NEW"] == rows[3] and staged["CLOSED"] == rows[5]
    assert staged["TAKEN"]["later_outcome"] is None                          # the real close can attach now
    assert staged["TAKEN"]["legacy_later_outcome"] == {"forward_return_pct": 4.0}


def test_counterfactual_jsonl_rows_are_rejudged_on_their_hypothetical_levels(tmp_path):
    cf = [{"symbol": "STOPPED", "as_of": D, "reason_code": "LOW_QUALITY_SETUP", "classification": "MISSED_WINNER",
           "hypothetical_entry": 100, "hypothetical_stop": 94, "hypothetical_target": 115,
           "outcome": {"forward_return_pct": 9.0}}]
    rep = RC.reconcile([], cf, _fn)
    assert rep["counterfactual_jsonl_transitions"] == {"MISSED_WINNER -> AVOIDED_LOSER": 1}
    staged = json.loads(open(RC.stage(rep, [], cf, tmp_path / "s")["counterfactuals"]).read())
    assert staged["classification"] == "AVOIDED_LOSER" and staged["legacy_classification"] == "MISSED_WINNER"


def test_the_live_runtime_is_refused():
    with pytest.raises(SystemExit, match="SSD copy"):
        RC.main(["--ledger", "/Volumes/QuantTermStorage/x.jsonl", "--store-root", "/tmp/x"])
    with pytest.raises(SystemExit, match="SSD copy"):
        RC.stage({"_judged": [], "_cf_judged": [], "_split": {RC.BLOCKED_TAKEN: []}}, [], [],
                 "/Volumes/QuantTermStorage/stage")
