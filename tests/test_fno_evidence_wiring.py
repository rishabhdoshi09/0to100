"""F&O real-forward paper evidence must move the SAME ranking engine the
equity desk uses -- proved the same way test_core_loop_revolution.py proves
it for equities: settle real outcomes, then show a later ranking reads them.

Before product/fno_evidence.py existed, FoPaperPosition.context_key was read
from the candidate dict at open time but nothing upstream ever populated it
(always ""), and settlement never called record_outcome. So no volume of
real, settled F&O paper history could ever move F&O ranking. These tests
prove that gap is closed, and that the same safety properties the equity
loop already guarantees (sample-gating, honest evidence-class labeling)
hold for F&O too.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import product.conditional_evidence as CE
from product.decision import BUY, Decision
from product.decision_ranking import decision_context_key, rank
from product.evidence_class import PAPER_FORWARD
from product.fno_evidence import fno_context_key, record_fno_settlement
from product.fo_paper_runtime import run_fo_paper_cycle
from product.fo_paper_store import FoPaperStore
from tests.test_fo_paper_runtime import _IntradayQuoteClient, _QuoteClient

IST = timezone(timedelta(hours=5, minutes=30))


def _fno_setup(*, direction="LONG", score=78.0, nifty_alignment=3.0, sector_strength=2.5):
    return {
        "direction": direction,
        "score": score,
        "atr_pct": 2.1,
        "breakout_distance_pct": 1.2,
        "components": {"nifty_alignment": nifty_alignment, "sector_strength": sector_strength},
    }


def _fno_decision(symbol: str, *, score: float, setup: dict) -> Decision:
    """An F&O candidate expressed the same way the ranking engine already
    understands equity candidates -- same Decision shape, FNO_-prefixed
    setup label so its evidence cell can never collide with an equity one."""
    key_setup = fno_context_key(setup)
    assert key_setup, "fixture setup must key to a valid context"
    return Decision(
        symbol=symbol,
        state=BUY,
        setup=f"FNO_{setup['direction']}",
        score=score,
        calibrated_confidence=setup["score"] / 100.0,
        market_state="RISK_ON" if setup["components"]["nifty_alignment"] >= 2.0 else "NEUTRAL",
        sector_state="LEADING" if setup["components"]["sector_strength"] >= 2.0 else "NEUTRAL",
        technical_evidence={
            "atr_pct": setup["atr_pct"],
            "pct_from_pivot": setup["breakout_distance_pct"],
        },
        entry=200.0,
        stop=190.0,
        target=230.0,
        source_scan_id="fno-scan-2026-09-30",
        evidence_class=PAPER_FORWARD,
        generated_at="2026-09-30T04:00:00+00:00",
        strategy_version="fno-v1",
    )


def _settled_trade_row(
    i: int, *, net_pnl: float, eligible: bool = True, trade_id_prefix="FNO"
) -> dict:
    return {
        "trade_id": f"{trade_id_prefix}-{i}",
        "underlying": "RELIANCE",
        "option_symbol": "RELIANCE26OCT1500CE",
        "entry_price": 200.0,
        "stop_price": 190.0,
        "exit_price": 200.0 + net_pnl / 10.0,
        "quantity": 10,
        "net_pnl": net_pnl,
        "exit_reason": "STOP" if net_pnl < 0 else "TARGET",
        "opened_at": "2026-09-30T04:00:00+00:00",
        "settled_at": f"2026-10-{1 + (i % 20):02d}T10:00:00+00:00",
        "mfe_pct": 0.5,
        "mae_pct": -1.0 if net_pnl < 0 else -0.2,
        "production_evidence_eligible": eligible,
    }


def test_fno_context_key_requires_a_direction():
    assert fno_context_key({}) == ""
    assert fno_context_key({"direction": "SIDEWAYS"}) == ""


def test_fno_context_key_never_collides_with_an_equity_setup_label():
    key = fno_context_key(_fno_setup())
    assert key.startswith("setup=FNO_LONG")
    assert "VCP" not in key


def test_ineligible_settlement_records_nothing(tmp_path, monkeypatch):
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _fno_setup()
    key = fno_context_key(setup)

    row = _settled_trade_row(0, net_pnl=-100.0, eligible=False)
    update = record_fno_settlement(row, context_key=key)
    assert update is None
    assert CE.read(key)["count"] == 0


def test_zero_risk_settlement_records_nothing(tmp_path, monkeypatch):
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _fno_setup()
    key = fno_context_key(setup)

    row = _settled_trade_row(0, net_pnl=-100.0)
    row["stop_price"] = row["entry_price"]  # zero risk per unit
    update = record_fno_settlement(row, context_key=key)
    assert update is None
    assert CE.read(key)["count"] == 0


def test_missing_context_key_records_nothing(tmp_path, monkeypatch):
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    row = _settled_trade_row(0, net_pnl=-100.0)
    assert record_fno_settlement(row, context_key="") is None


def test_tiny_fno_sample_does_not_move_ranking(tmp_path, monkeypatch):
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))

    setup = _fno_setup()
    key = fno_context_key(setup)
    loser = _fno_decision("RELIANCE", score=80.0, setup=setup)
    control = _fno_decision("TCS", score=78.0, setup=_fno_setup(sector_strength=0.0))

    for i in range(5):  # far below MIN_SAMPLE
        row = _settled_trade_row(i, net_pnl=-1000.0)
        update = record_fno_settlement(row, context_key=key)
        assert update is not None

    assert CE.read(key)["count"] == 5
    ranked = rank([loser, control], evidence_class=PAPER_FORWARD, path=path)
    assert [r.symbol for r in ranked] == ["RELIANCE", "TCS"], (
        "5 settled trades is not enough evidence to reorder F&O candidates"
    )
    demoted = next(r for r in ranked if r.symbol == "RELIANCE")
    assert demoted.evidence["reason"] == "INSUFFICIENT_EVIDENCE"
    assert demoted.evidence_adjustment == 0.0


def test_real_forward_fno_losses_demote_a_later_ranking(tmp_path, monkeypatch):
    """The exact shape of test_a_settled_outcome_changes_a_later_ranking,
    but for the F&O path: real settled paper losses -> evidence -> a LATER
    ranking call reorders two F&O candidates because of it."""
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))

    loser_setup = _fno_setup()
    key = fno_context_key(loser_setup)
    loser = _fno_decision("RELIANCE", score=80.0, setup=loser_setup)
    control = _fno_decision("TCS", score=78.0, setup=_fno_setup(sector_strength=0.0))

    assert decision_context_key(loser) == key, (
        "the cell a candidate is ranked against must be the exact cell its "
        "own settlement updates -- otherwise the loop silently stops learning"
    )

    before = rank([loser, control], evidence_class=PAPER_FORWARD, path=path)
    assert [r.symbol for r in before] == ["RELIANCE", "TCS"]

    for i in range(30):
        row = _settled_trade_row(i, net_pnl=-1000.0)  # a full stop-out every time
        update = record_fno_settlement(row, context_key=key, path=path)
        assert update is not None
        assert update.changed

    cell = CE.read(key, path=path)
    assert cell["count"] == 30
    assert cell["wins"] == 0
    assert cell["expectancy_R"] < 0

    after = rank([loser, control], evidence_class=PAPER_FORWARD, path=path)
    assert [r.symbol for r in after] == ["TCS", "RELIANCE"], (
        "30 real, fully-costed settled F&O losses in this exact context must "
        "demote the measured setup behind the unmeasured control"
    )
    demoted = next(r for r in after if r.symbol == "RELIANCE")
    assert demoted.base_score == 80.0, "the raw scorer is untouched"
    assert demoted.evidence_adjustment < 0
    assert demoted.evidence["reason"] == "MEASURED_NEGATIVE_EXPECTANCY"


def test_counterfactual_evidence_class_cannot_move_fno_ranking(tmp_path, monkeypatch):
    """Safety half: an F&O candidate that was only ever a hypothesis
    (COUNTERFACTUAL, not a real settled PAPER_FORWARD trade) must not be
    able to move ranking, no matter the sample size -- the same boundary
    test_historical_replay_proves_plumbing_not_edge proves for equities."""
    from product.decision_chain import Outcome
    from product.evidence_class import COUNTERFACTUAL

    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))

    setup = _fno_setup()
    key = fno_context_key(setup)
    loser = _fno_decision("RELIANCE", score=80.0, setup=setup)
    control = _fno_decision("TCS", score=78.0, setup=_fno_setup(sector_strength=0.0))

    for i in range(30):
        outcome = Outcome(
            position_id=f"CF-{i}", paper_order_id=f"CF-{i}", paper_intent_id=f"CF-{i}",
            decision_id=f"CF-{i}", symbol="RELIANCE", realized_R=-1.0,
            exit_reason="STOP", entry_session="2026-09-30",
            exit_session=f"2026-10-{1 + (i % 20):02d}", evidence_class=COUNTERFACTUAL,
            resolved_at=f"2026-10-{1 + (i % 20):02d}T10:00:00+00:00",
        )
        CE.record_outcome(outcome, context_key=key, evidence_class=COUNTERFACTUAL, path=path)

    ranked = rank([loser, control], evidence_class=PAPER_FORWARD, path=path)
    assert [r.symbol for r in ranked] == ["RELIANCE", "TCS"], (
        "30 counterfactual (never actually paper-traded) F&O outcomes must "
        "not be able to demote a ranking the way real settled evidence can"
    )


def _candidate_without_context_key(*, symbol="RELIANCE", score=82.0):
    return {
        "symbol": symbol,
        "direction": "LONG",
        "decision": "PAPER_OPTION_CANDIDATE",
        "setup": {
            "score": score,
            "direction": "LONG",
            "atr_pct": 2.0,
            "breakout_distance_pct": 1.5,
            "components": {"nifty_alignment": 3.0, "sector_strength": 2.5},
            "expected_move": {"holding_days": 2},
            "underlying_trade_plan": {"entry": 50.0},
        },
        "selected_contract": {
            "symbol": f"{symbol}26OCTCE",
            "option_type": "CE",
            "eligible": True,
            "lot_size": 25,
            "ask": 50.0,
            "score": 86.0,
            # Deliberately no "context_key" here to test the fallback that
            # derives ranking_context_key from `setup`. NOTE: a real
            # candidate from fo_snapshot_engine.evaluate_fo_snapshot always
            # DOES set this field, to a different, unrelated legacy FOCTX_V1
            # key -- see test_fno_full_lifecycle.py's
            # test_real_candidate_shape_keeps_the_two_context_keys_separate,
            # which exercises that real shape.
            "trade_plan": {"entry": 50.0, "stop": 40.0, "target": 70.0},
        },
    }


def _directional_without_context_key(token: int):
    return {
        "available": True,
        "status": "READY",
        "decision": "PAPER_CANDIDATES",
        "candidates": [
            {**_candidate_without_context_key(),
             "selected_contract": {
                 **_candidate_without_context_key()["selected_contract"],
                 "instrument_token": token,
             }},
        ],
        "paper_only": True,
        "live_execution_allowed": False,
    }


def test_production_settlement_path_records_real_evidence_end_to_end(tmp_path, monkeypatch):
    """Exercises the actual product entrypoint (run_fo_paper_cycle) end to
    end -- open, then settle -- with NO context_key supplied by the
    candidate. Proves both halves of the fix together: the fallback
    derivation at open time, and the record_fno_settlement call at
    settlement time.

    Checks ranking_context_key, not context_key. The latter is
    fo_evidence.py's separate, legacy FOCTX_V1 contract-level key that real
    candidates from fo_snapshot_engine.evaluate_fo_snapshot always populate
    (never empty in real production, unlike this synthetic fixture) for the
    unrelated "Forward evidence" UI panel; it happens to equal
    fno_context_key(setup) here too only because this fixture's fallback
    branch fires for both fields at once. See
    tests/test_fno_full_lifecycle.py::test_real_candidate_shape_keeps_the_two_context_keys_separate
    for the proof that a REAL candidate (built through fo_snapshot_engine,
    never a hand-built dict) sets context_key to a genuinely different,
    unrelated string and ranking_context_key is what settlement must key
    evidence on."""
    evidence_path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(evidence_path))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))

    fo_db = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)
    expected_key = fno_context_key(_candidate_without_context_key()["setup"])
    assert expected_key

    with FoPaperStore(fo_db) as store:
        opened = run_fo_paper_cycle(
            _directional_without_context_key(9001),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    client = _IntradayQuoteClient(
        last=30, high=32, low=28, bid=29,
        entry_minute_rows=[
            {"date": "2026-09-26T10:15:00+05:30",
             "open": 49.0, "high": 55.0, "low": 45.0, "close": 51.0},
        ],
        intraday_rows=[
            {"date": "2026-09-26T10:16:00+05:30",
             "open": 45.0, "high": 46.0, "low": 30.0, "close": 32.0},
        ],
    )
    with FoPaperStore(fo_db) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=10, minute=20, second=0),
            allow_new_entries=False,
            store=store,
            capital=200_000,
            cost_model=lambda entry, exit, qty: 25.0,
            cost_model_name="TEST_COSTS",
        )

    assert marked["settled_count"] == 1
    trade = marked["settled"][0]
    assert trade["ranking_context_key"] == expected_key, (
        "the position must have been opened with the SAME ranking key "
        "fno_context_key derives, or its settlement cannot update the cell "
        "it was ranked against"
    )
    assert trade["production_evidence_eligible"] is True

    cell = CE.read(expected_key, path=evidence_path)
    assert cell["count"] == 1, (
        "a fully-costed, fully-observed settled F&O trade from the real "
        "run_fo_paper_cycle entrypoint must land in the same evidence store "
        "product.decision_ranking.rank reads"
    )
