"""One complete revolution of the loop, proved rather than asserted.

The claim under test is the one the product rests on: that QuantTerm learns
from its own results. Architecture does not prove it. A statistics table does
not prove it either — a system can write numbers nothing reads. What proves it
is a single chain that can be walked end to end:

    decision -> intent -> order -> position -> outcome -> evidence -> a later
    decision ranked differently BECAUSE of that evidence

and a demonstration that the last arrow is real: same decisions, same scorer,
different rank, with the difference attributable to the outcome.

The safety half matters as much as the learning half. Two of these tests exist
to prove the loop does NOT close when it shouldn't: invented bars and replayed
history are allowed to prove plumbing and nothing else.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from product import conditional_evidence as CE
from product.decision import BUY, Decision, EvidenceItem, SUPPORTING, CONFLICTING, MISSING
from product.decision_chain import (
    BrokenChain,
    EvidenceUpdate,
    Outcome,
    PaperIntent,
    PaperOrder,
    PaperPositionRef,
    chain_ids,
    validate_chain,
)
from product.decision_ranking import decision_context_key, rank, ranking_explanation
from product.evidence_class import (
    HISTORICAL_REPLAY,
    PAPER_FORWARD,
    TEST_FIXTURE,
    is_market_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence_store(tmp_path, monkeypatch):
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))
    return path


def _decision(symbol: str, *, score: float, setup: str = "VCP",
              market: str = "HEALTHY", sector: str = "LEADING",
              confidence: float = 0.65) -> Decision:
    return Decision(
        symbol=symbol,
        state=BUY,
        setup=setup,
        score=score,
        calibrated_confidence=confidence,
        market_state=market,
        sector_state=sector,
        technical_evidence={"atr_pct": 2.5, "pct_from_pivot": 1.0},
        entry=100.0,
        stop=95.0,
        target=115.0,
        source_scan_id="scan-2026-09-12",
        evidence_class=PAPER_FORWARD,
        generated_at="2026-09-12T04:00:00+00:00",
        strategy_version="v1",
    )


def _settle(decision: Decision, *, realized_R: float, n: int,
            evidence_class: str = PAPER_FORWARD) -> list[EvidenceUpdate]:
    """Walk the full chain n times and fold each settled outcome into evidence."""
    key = decision_context_key(decision)
    updates: list[EvidenceUpdate] = []
    for i in range(n):
        intent = PaperIntent(
            decision_id=decision.decision_id, symbol=decision.symbol, qty=10,
            entry=decision.entry, stop=decision.stop, target=decision.target,
            created_at=f"2026-09-12T04:{i // 60:02d}:{i % 60:02d}+00:00",
            evidence_class=evidence_class,
        )
        order = PaperOrder(
            paper_intent_id=intent.intent_id, decision_id=decision.decision_id,
            symbol=decision.symbol, qty=10, fill_price=100.2,
            placed_at=intent.created_at,
        )
        position = PaperPositionRef(
            paper_order_id=order.order_id, paper_intent_id=intent.intent_id,
            decision_id=decision.decision_id, symbol=decision.symbol,
            opened_at=intent.created_at,
        )
        outcome = Outcome(
            position_id=position.position_id, paper_order_id=order.order_id,
            paper_intent_id=intent.intent_id, decision_id=decision.decision_id,
            symbol=decision.symbol, realized_R=realized_R, exit_reason="STOP",
            entry_session="2026-09-12", exit_session=f"2026-09-{13 + (i % 15):02d}",
            evidence_class=evidence_class,
            resolved_at=f"2026-09-30T10:{i // 60:02d}:{i % 60:02d}+00:00",
        )
        update = CE.record_outcome(
            outcome, context_key=key, evidence_class=evidence_class,
            calibrated_confidence=decision.calibrated_confidence,
        )
        validate_chain(
            decision_id=decision.decision_id, intent=intent, order=order,
            position=position, outcome=outcome, update=update,
        )
        updates.append(update)
    return updates


# ── the revolution ─────────────────────────────────────────────────────────
def test_a_settled_outcome_changes_a_later_ranking(evidence_store):
    """evidence X -> outcome -> evidence Y -> a later ranking reads Y."""
    loser = _decision("AAA", score=80.0)
    control = _decision("BBB", score=78.0, setup="CUP_HANDLE")

    # ── evidence state X ───────────────────────────────────────────────────
    key = decision_context_key(loser)
    before = CE.read(key)
    assert before["count"] == 0

    first = rank([loser, control])
    assert [r.symbol for r in first] == ["AAA", "BBB"]
    assert first[0].evidence_adjustment == 0.0
    assert first[0].evidence["reason"] == "INSUFFICIENT_EVIDENCE"

    # ── the loop runs: thirty settled losses in this exact context ─────────
    updates = _settle(loser, realized_R=-1.0, n=30)
    assert len(updates) == 30
    assert updates[0].before["count"] == 0
    assert updates[0].after["count"] == 1
    assert updates[-1].after["count"] == 30
    assert all(u.changed for u in updates)

    # ── evidence state Y ───────────────────────────────────────────────────
    after = CE.read(key)
    assert after["count"] == 30
    assert after["wins"] == 0
    assert after["expectancy_R"] == pytest.approx(-1.0)
    assert after["wilson_lower_bound"] == pytest.approx(0.0, abs=1e-9)
    assert after != before

    # ── a later ranking reads Y and changes deterministically ──────────────
    second = rank([loser, control])
    assert [r.symbol for r in second] == ["BBB", "AAA"], (
        "the measured loser must fall behind the control it out-scored"
    )
    demoted = next(r for r in second if r.symbol == "AAA")
    assert demoted.base_score == 80.0, "the scorer itself is untouched"
    assert demoted.evidence_adjustment < 0
    assert demoted.ranking_score < demoted.base_score
    assert demoted.evidence["reason"] == "MEASURED_NEGATIVE_EXPECTANCY"
    assert demoted.evidence["count"] == 30

    # and the change is explainable from the record alone
    assert "30 settled trades" in ranking_explanation(demoted)


def test_the_chain_links_evidence_back_to_the_decision(evidence_store):
    decision = _decision("AAA", score=80.0)
    intent = PaperIntent(decision_id=decision.decision_id, symbol="AAA", qty=10)
    order = PaperOrder(paper_intent_id=intent.intent_id,
                       decision_id=decision.decision_id, symbol="AAA", qty=10)
    position = PaperPositionRef(paper_order_id=order.order_id,
                                paper_intent_id=intent.intent_id,
                                decision_id=decision.decision_id, symbol="AAA")
    outcome = Outcome(position_id=position.position_id, paper_order_id=order.order_id,
                      paper_intent_id=intent.intent_id, decision_id=decision.decision_id,
                      symbol="AAA", realized_R=1.5, evidence_class=PAPER_FORWARD)
    update = CE.record_outcome(outcome, context_key=decision_context_key(decision),
                               evidence_class=PAPER_FORWARD)

    ids = chain_ids(decision_id=decision.decision_id, intent=intent, order=order,
                    position=position, outcome=outcome, update=update)
    assert all(ids.values()), f"a hop lost its id: {ids}"
    assert len(set(ids.values())) == len(ids), "ids must be distinguishable"
    validate_chain(decision_id=decision.decision_id, intent=intent, order=order,
                   position=position, outcome=outcome, update=update)


def test_a_broken_link_is_refused_not_counted():
    """A statistic whose provenance cannot be walked back is not a statistic."""
    with pytest.raises(BrokenChain):
        PaperIntent(decision_id="", symbol="AAA")

    intent = PaperIntent(decision_id="dec_one", symbol="AAA")
    stray = PaperOrder(paper_intent_id="int_someone_else",
                       decision_id="dec_one", symbol="AAA")
    with pytest.raises(BrokenChain):
        validate_chain(decision_id="dec_one", intent=intent, order=stray)


def test_evidence_survives_a_process_restart(evidence_store):
    """The loop must cross a restart: a new process reads the same belief."""
    decision = _decision("AAA", score=80.0)
    _settle(decision, realized_R=-1.0, n=30)
    key = decision_context_key(decision)

    probe = (
        "import json,os;"
        "from product.conditional_evidence import read, ranking_evidence;"
        f"cell = read({key!r});"
        f"print(json.dumps({{'count': cell['count'],"
        f" 'adjustment': ranking_evidence({key!r})['adjustment']}}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["count"] == 30
    assert payload["adjustment"] < 0


# ── the safety half ────────────────────────────────────────────────────────
def test_invented_bars_can_never_move_a_ranking(evidence_store):
    """A synthetic outcome proves code behaviour. It is not market evidence."""
    decision = _decision("QTTRUTHA", score=80.0)
    _settle(decision, realized_R=-3.0, n=50, evidence_class=TEST_FIXTURE)

    key = decision_context_key(decision)
    fixture_cell = CE.read(key, evidence_class=TEST_FIXTURE)
    assert fixture_cell["count"] == 50, "the rows were recorded"

    assert CE.read(key, evidence_class=PAPER_FORWARD)["count"] == 0, (
        "fixture rows must not land in the paper-forward cell"
    )
    ranked = rank([decision])
    assert ranked[0].evidence_adjustment == 0.0
    assert ranked[0].evidence["reason"] == "INSUFFICIENT_EVIDENCE"

    refused = CE.ranking_evidence(key, evidence_class=TEST_FIXTURE)
    assert refused["usable"] is False
    assert refused["reason"] == "NOT_MARKET_EVIDENCE"
    assert not is_market_evidence(TEST_FIXTURE)


def test_historical_replay_proves_plumbing_not_edge(evidence_store):
    """Replay may re-run the live path over the past. It may not size a trade."""
    decision = _decision("AAA", score=80.0)
    _settle(decision, realized_R=2.0, n=40, evidence_class=HISTORICAL_REPLAY)

    key = decision_context_key(decision)
    assert CE.read(key, evidence_class=HISTORICAL_REPLAY)["count"] == 40
    assert CE.read(key, evidence_class=PAPER_FORWARD)["count"] == 0
    assert rank([decision])[0].evidence_adjustment == 0.0
    assert CE.ranking_evidence(key, evidence_class=HISTORICAL_REPLAY)["usable"] is False


def test_thin_evidence_does_not_vote(evidence_store):
    """Twenty-nine settled trades is still no claim."""
    decision = _decision("AAA", score=80.0)
    _settle(decision, realized_R=-1.0, n=29)
    key = decision_context_key(decision)
    assert CE.read(key)["count"] == 29
    gated = CE.ranking_evidence(key)
    assert gated["usable"] is False
    assert gated["reason"] == "INSUFFICIENT_EVIDENCE"
    assert rank([decision])[0].evidence_adjustment == 0.0


def test_measured_winners_earn_no_promotion(evidence_store):
    """Evidence is demote-only: surviving your own history is not an edge."""
    decision = _decision("AAA", score=80.0)
    _settle(decision, realized_R=+1.5, n=40)
    ranked = rank([decision])[0]
    assert ranked.evidence["usable"] is True
    assert ranked.evidence["reason"] == "MEASURED_NOT_NEGATIVE"
    assert ranked.evidence_adjustment == 0.0
    assert ranked.ranking_score == ranked.base_score


def test_the_same_outcome_cannot_be_counted_twice(evidence_store):
    decision = _decision("AAA", score=80.0)
    key = decision_context_key(decision)
    intent = PaperIntent(decision_id=decision.decision_id, symbol="AAA", qty=10)
    order = PaperOrder(paper_intent_id=intent.intent_id,
                       decision_id=decision.decision_id, symbol="AAA", qty=10)
    position = PaperPositionRef(paper_order_id=order.order_id,
                                paper_intent_id=intent.intent_id,
                                decision_id=decision.decision_id, symbol="AAA")
    outcome = Outcome(position_id=position.position_id, paper_order_id=order.order_id,
                      paper_intent_id=intent.intent_id, decision_id=decision.decision_id,
                      symbol="AAA", realized_R=-1.0, evidence_class=PAPER_FORWARD)

    first = CE.record_outcome(outcome, context_key=key, evidence_class=PAPER_FORWARD)
    second = CE.record_outcome(outcome, context_key=key, evidence_class=PAPER_FORWARD)
    assert first.after["count"] == 1
    assert second.before == second.after, "a replayed outcome must not double-count"
    assert CE.read(key)["count"] == 1


def test_an_unresolved_trade_is_not_evidence(evidence_store):
    decision = _decision("AAA", score=80.0)
    intent = PaperIntent(decision_id=decision.decision_id, symbol="AAA", qty=10)
    order = PaperOrder(paper_intent_id=intent.intent_id,
                       decision_id=decision.decision_id, symbol="AAA", qty=10)
    position = PaperPositionRef(paper_order_id=order.order_id,
                                paper_intent_id=intent.intent_id,
                                decision_id=decision.decision_id, symbol="AAA")
    open_still = Outcome(position_id=position.position_id, paper_order_id=order.order_id,
                         paper_intent_id=intent.intent_id,
                         decision_id=decision.decision_id, symbol="AAA",
                         realized_R=None, evidence_class=PAPER_FORWARD)
    with pytest.raises(ValueError, match="has not settled"):
        CE.record_outcome(open_still, context_key=decision_context_key(decision),
                          evidence_class=PAPER_FORWARD)


def test_the_ranking_cell_is_the_cell_the_outcome_updates(evidence_store):
    """If these drift apart the loop stops learning without anything failing."""
    decision = _decision("AAA", score=80.0)
    key_at_ranking = decision_context_key(decision)
    updates = _settle(decision, realized_R=-1.0, n=1)
    assert updates[0].context_key == key_at_ranking


# ── the decision object itself ─────────────────────────────────────────────
def test_a_decision_without_an_exit_cannot_reach_the_paper_book():
    no_stop = Decision(symbol="AAA", state=BUY, entry=100.0, stop=None)
    assert no_stop.is_actionable
    assert not no_stop.may_open_paper_position


def test_non_actionable_states_never_open_positions():
    for state in ("WATCH", "WAIT", "AVOID", "NO_TRADE"):
        decision = Decision(symbol="AAA", state=state, entry=100.0, stop=95.0)
        assert not decision.is_actionable
        assert not decision.may_open_paper_position


def test_missing_evidence_stays_missing():
    decision = Decision(symbol="AAA", state=BUY).with_evidence(
        supporting=[EvidenceItem(id="volume", label="Volume expansion",
                                 direction=SUPPORTING)],
        conflicting=[EvidenceItem(id="rsi", label="RSI 74", direction=CONFLICTING)],
        missing=[EvidenceItem(id="funds", label="No fundamentals", direction=MISSING)],
    )
    counts = decision.evidence_counts()
    assert counts == {SUPPORTING: 1, CONFLICTING: 1, MISSING: 1}
    payload = decision.to_dict()
    assert payload["missing_evidence"][0]["id"] == "funds"
    assert Decision.from_dict(payload).to_dict() == payload


def test_enrichment_keeps_the_same_decision_id():
    decision = Decision(symbol="AAA", state=BUY, entry=100.0, stop=95.0)
    enriched = decision.with_evidence(
        supporting=[EvidenceItem(id="x", direction=SUPPORTING)]
    )
    assert enriched.decision_id == decision.decision_id
