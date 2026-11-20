"""One rulebook for rejected and taken trades (the 5-vs-10-session learning inconsistency).

Until 2026-11-20 forward-ledger rejections were judged by a 5-session
close-to-close move that ignored the stop, while taken paper trades used the
10-session rulebook (core.evaluation_spec). The outcome job could also stamp
that counterfactual on a TAKEN row first, which blocked the trade's real
paper result from ever attaching. These tests pin the fix.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.evaluation_spec import spec_tag
from core.outcome_resolver import counterfactual_outcome, resolve_trade
from product import counterfactual_learning as CL
from product import forward_evidence as FE
from product import forward_soak as FS
from tests.evidence_fixtures import rulebook_outcome

D = "2026-09-01"


def _bars(rows, *, before_close=99.0, start="2026-08-03"):
    """A frame: 20 quiet sessions up to D (close `before_close`), then `rows` = (open, high, low, close)."""
    pre = pd.bdate_range(start, D)
    idx = list(pre) + list(pd.bdate_range(pd.Timestamp(D) + pd.Timedelta(days=1), periods=len(rows)))
    data = [(before_close, before_close + 0.5, before_close - 0.5, before_close)] * len(pre) + list(rows)
    return pd.DataFrame(data, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(idx))


def _cf(frame, entry=100.0, stop=94.0, target=115.0):
    return counterfactual_outcome("X", D, entry, stop, target, frame=frame)


# ── the rulebook judges a rejection the way a taken trade is judged ───────────

def test_stopped_out_then_recovering_is_a_loss_not_a_missed_winner():
    frame = _bars([(99.5, 100.5, 99.0, 100.0),     # fills the 100 buy-stop
                   (99.0, 99.5, 93.0, 94.5),       # stop 94 hit
                   (95.0, 101.0, 94.5, 100.0), (100, 104, 99.5, 103.0), (103, 108.5, 102.5, 108.0)]
                  + [(108, 109, 107, 108.0)] * 5)
    legacy = CL.classify_forward(entry=100, stop=94, target=115, forward_return_pct=(108.0 / 99.0 - 1) * 100)
    assert legacy == CL.MISSED_WINNER                                  # the old 5-session verdict
    out = _cf(frame)
    assert out["outcome"] == "LOSS" and out["net_r"] < 0 and out["eval_spec"] == spec_tag()
    assert CL.classify_rulebook(out) == CL.AVOIDED_LOSER


def test_a_buy_stop_that_never_trades_is_not_a_missed_winner():
    frame = _bars([(90 + i * 0.6, 90.4 + i * 0.6, 89.8 + i * 0.6, 90.2 + i * 0.6) for i in range(10)],
                  before_close=90.0)
    legacy = CL.classify_forward(entry=100, stop=94, target=115, forward_return_pct=(92.6 / 90.0 - 1) * 100)
    assert legacy != CL.NOT_TRIGGERED
    out = _cf(frame)
    assert out["outcome"] == "NO_FILL" and CL.classify_rulebook(out) == CL.NOT_TRIGGERED


def test_a_winner_that_arrives_after_session_five_is_counted():
    rows = [(99.5, 100.5, 99.0, 100.0)] + [(100, 100.8, 99.6, 100.3)] * 6 + [(101, 116.0, 100.5, 115.5)] \
        + [(115, 116, 114, 115)] * 2
    out = _cf(_bars(rows))
    legacy = CL.classify_forward(entry=100, stop=94, target=115, forward_return_pct=(100.3 / 99.0 - 1) * 100)
    assert legacy != CL.MISSED_WINNER
    assert out["outcome"] == "WIN" and CL.classify_rulebook(out) == CL.MISSED_WINNER


def test_rejected_and_taken_trades_get_the_same_r_and_the_same_grade():
    from core.costs import cost_in_r
    rows = [(99.5, 100.5, 99.0, 100.0)] + [(100 + i, 101.5 + i, 99.8 + i, 101 + i) for i in range(9)]
    frame = _bars(rows)
    taken = resolve_trade("X", D, 100.0, 94.0, 115.0, frame=frame)              # signal_log's resolver
    out = _cf(frame)
    assert out["gross_r"] == pytest.approx(taken.r)
    assert out["net_r"] == pytest.approx(taken.r - cost_in_r((taken.fill_price - 94.0) / taken.fill_price))
    assert CL.classify_rulebook(out) == CL.MISSED_WINNER
    assert CL.classify_rulebook(out, selected=True) == CL.WINNER_TAKEN               # one rule, both sides


def test_nothing_settles_before_the_horizon_is_published():
    rows = [(99.5, 100.5, 99.0, 100.0)] + [(100, 101, 99.5, 100.5)] * 5               # 6 of 10 sessions
    assert _cf(_bars(rows)) is None
    assert _cf(_bars(rows + [(100, 101, 99.5, 100.5)] * 4))["outcome"] == "FLAT"


@pytest.mark.parametrize("levels", [(None, 94, 115), (100, 101, 115), (100, 94, 99), ("x", 94, 115)])
def test_invalid_levels_are_unmeasured_never_guessed(levels):
    out = counterfactual_outcome("X", D, *levels, frame=_bars([(100, 101, 99, 100)] * 10))
    assert out["outcome"] == "NO_LEVELS" and out["net_r"] is None
    assert CL.classify_rulebook(out) == CL.UNMEASURED


# ── incompatible horizons never reach a learning score ────────────────────────

def _settled(cls, spec, symbol="TCS"):
    return {"symbol": symbol, "as_of": D, "reason_code": "ENTRY_TOO_EXTENDED", "classification": cls,
            "eval_spec": spec, "evidence": {"setup_label": "VCP", "regime": "RISK_ON"}}


@pytest.mark.parametrize("spec", [None, "", "sim:h5:be2:gap", "close:h5"])
def test_a_legacy_or_foreign_horizon_label_updates_no_policy(tmp_path, spec):
    from product.paper_learning_loop import ingest_counterfactual
    path = tmp_path / "policies.json"
    out = ingest_counterfactual(_settled(CL.MISSED_WINNER, spec), path=path)
    assert out["skipped"] == "INCOMPATIBLE_EVAL_SPEC"
    assert not path.exists() or "REJECT::" not in path.read_text()


def test_a_rulebook_label_updates_the_policy_once(tmp_path):
    from product.paper_learning_loop import ingest_counterfactual
    path = tmp_path / "policies.json"
    first = ingest_counterfactual(_settled(CL.AVOIDED_LOSER, spec_tag()), path=path)
    again = ingest_counterfactual(_settled(CL.AVOIDED_LOSER, spec_tag()), path=path)
    store = json.loads(path.read_text())
    pol = next(p for p in store["policies"] if p["policy_id"] == "REJECT::ENTRY_TOO_EXTENDED")
    assert "skipped" not in (first or {}) and again.get("duplicate_observation") is True
    assert pol["sample_size"] == 1


def test_an_unmeasured_rejection_updates_nothing(tmp_path):
    from product.paper_learning_loop import ingest_counterfactual
    out = ingest_counterfactual(_settled(CL.UNMEASURED, spec_tag()), path=tmp_path / "p.json")
    assert out["skipped"] == "UNMEASURED" and not (tmp_path / "p.json").exists()


# ── settlement: rejections only, tagged, idempotent; taken rows untouched ──────

def _freeze(n_rejected=3, taken=("TAKEN1",)):
    FE.freeze_cycle({
        "as_of": D, "cycle_id": "cyc-1", "rules_hash": "h",
        "taken": [{"symbol": s, "entry": 100, "stop": 94, "target": 115, "setup_label": "VCP"} for s in taken],
        "rejections": [{"symbol": f"R{i}", "reason_code": "ENTRY_TOO_EXTENDED", "decision": "WAIT",
                        "entry": 100, "stop": 94, "target": 115, "setup_label": "VCP"} for i in range(n_rejected)],
    })


def test_forward_soak_settles_rejections_on_the_rulebook_and_only_once(tmp_path):
    _freeze()
    fn = lambda s, d, r: rulebook_outcome(1.2, "WIN", 7.0) if s == "R0" else rulebook_outcome(-1.0, "LOSS", -6.0)
    first = FS.settle_pending_from_market(outcome_fn=fn, policy_path=tmp_path / "pol.json")
    again = FS.settle_pending_from_market(outcome_fn=fn, policy_path=tmp_path / "pol.json")
    assert first["updated"] == 3 and again["updated"] == 0
    rows = {r["symbol"]: r for r in FE.load_ledger()}
    assert rows["R0"]["counterfactual_classification"] == CL.MISSED_WINNER
    assert rows["R1"]["counterfactual_classification"] == CL.AVOIDED_LOSER
    assert all(rows[s]["later_outcome"]["eval_spec"] == spec_tag() for s in ("R0", "R1", "R2"))
    assert rows["R0"]["later_outcome"]["counterfactual_R"] == pytest.approx(1.2)
    assert rows["TAKEN1"]["later_outcome"] is None                     # taken trades never settle here


def test_the_outcome_job_never_stamps_a_counterfactual_on_a_taken_trade(tmp_path, monkeypatch):
    """Before the fix a taken trade open past session 5 was settled as a rejection and its
    real paper close could then never attach (a row settles once)."""
    from product import autonomous_loop as LOOP
    monkeypatch.setattr(LOOP, "MEMORY_PATH", tmp_path / "memory.json")
    monkeypatch.setattr(LOOP, "LEARNING_PATH", tmp_path / "learning.jsonl")
    monkeypatch.setattr("core.outcome_resolver.counterfactual_outcome",
                        lambda *a, **k: rulebook_outcome(2.0, "WIN", 12.0))
    _freeze(n_rejected=2)
    out = LOOP.settle_official_outcomes(D)
    rows = {r["symbol"]: r for r in FE.load_ledger()}
    assert rows["TAKEN1"]["later_outcome"] is None and out["n_settled"] == 2
    assert all(rows[s]["later_outcome"]["eval_spec"] == spec_tag() for s in ("R0", "R1"))
    FE.attach_settlement(rows["TAKEN1"]["decision_id"], classification="TARGET", gross_R=1.9)
    taken = next(r for r in FE.load_ledger() if r["symbol"] == "TAKEN1")
    assert taken["gross_R"] == 1.9                                     # the real close still attaches


def test_scorecards_pool_only_rulebook_rejections(tmp_path):
    from product.learning_scorecard import forward_evidence
    _freeze(n_rejected=4, taken=())
    ids = {r["symbol"]: r["decision_id"] for r in FE.load_ledger()}
    FE.attach_settlements([   # two legacy 5-session labels, two rulebook labels
        {"decision_id": ids["R0"], "classification": CL.MISSED_WINNER, "forward_return_pct": 9.0},
        {"decision_id": ids["R1"], "classification": CL.MISSED_WINNER, "forward_return_pct": 8.0},
        {"decision_id": ids["R2"], "classification": CL.AVOIDED_LOSER, "forward_return_pct": -6.0,
         "eval_spec": spec_tag(), "counterfactual_R": -1.0},
        {"decision_id": ids["R3"], "classification": CL.MISSED_WINNER, "forward_return_pct": 7.0,
         "eval_spec": spec_tag(), "counterfactual_R": 1.1},
    ])
    rows = [dict(r, provenance=FE.REAL_FORWARD_MARKET) for r in FE.load_ledger()]      # as on the host
    ev = forward_evidence(rows)
    assert ev["rejections_settled_independent"] == 2 and ev["rejections_legacy_excluded"] == 2
    gate = ev["rejection_gates"]["ENTRY_TOO_EXTENDED"]
    assert gate["independent_n"] == 2 and gate["missed_winner_rate"] == 0.5
    board = FS.rejection_gate_scorecard(FE.load_ledger())["ENTRY_TOO_EXTENDED"]
    assert board["n"] == 2 and board["mean_rejected_move_R"] == pytest.approx(0.05)


def test_learning_memory_counts_rulebook_labels_and_reports_legacy_aside(tmp_path, monkeypatch):
    from product import autonomous_loop as LOOP
    cf = tmp_path / "cf.jsonl"
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(cf))
    monkeypatch.setattr(LOOP, "MEMORY_PATH", tmp_path / "memory.json")
    monkeypatch.setattr(LOOP, "LEARNING_PATH", tmp_path / "learning.jsonl")
    rows = [{"as_of": D, "symbol": "A", "reason_code": "DD", "classification": CL.MISSED_WINNER},     # legacy
            {"as_of": D, "symbol": "B", "reason_code": "DD", "classification": CL.MISSED_WINNER,
             "eval_spec": spec_tag()},
            {"as_of": D, "symbol": "C", "reason_code": "DD", "classification": CL.GOOD_WAIT}]          # event label
    cf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    memory = LOOP.consume_learning_memory("2026-09-02")
    assert memory["observations"] == 2 and memory["legacy_eval_spec_excluded"] == 1
    assert memory["classification_counts"] == {CL.MISSED_WINNER: 1, CL.GOOD_WAIT: 1}


def test_the_settlement_horizon_is_the_rulebook_horizon():
    from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS
    import core.signal_outcome_tracker as SOT
    assert FS.SETTLE_HORIZON == OUTCOME_HORIZON_SESSIONS == 10
    assert SOT.eval_spec_tag() == spec_tag() == "sim:h10:be2:gap"
    assert CL.is_rulebook(spec_tag()) and not CL.is_rulebook("sim:h5:be2:gap") and not CL.is_rulebook(None)
