"""Deterministic acceptance tests for OPTION CONTRACT selection learning
(product/fno_contract_evidence.py) -- kept separate from underlying-setup
ranking (product/fno_evidence_fusion.py, tests/test_fno_evidence_fusion.py).

Each test isolates one policy rule, plus the two cross-cutting properties the
task explicitly calls out: (a) no historical/fabricated option-chain evidence
can ever move contract selection -- only genuine settled PAPER_FORWARD trades
can, and (b) underlying-call correctness and contract-shape correctness are
graded on entirely separate evidence.
"""
from __future__ import annotations

import product.conditional_evidence as CE
from product.decision_chain import Outcome
from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD
from product.fno_contract_evidence import (
    CLASSIFICATION_BAD_EXPIRY_CHOICE,
    CLASSIFICATION_DELTA_TOO_HIGH,
    CLASSIFICATION_DELTA_TOO_LOW,
    CLASSIFICATION_IV_CRUSH,
    CLASSIFICATION_POOR_LIQUIDITY,
    CLASSIFICATION_THETA_DAMAGE,
    CLASSIFICATION_UNDERLYING_RIGHT_CONTRACT_POOR,
    CLASSIFICATION_UNDERLYING_RIGHT_CONTRACT_RIGHT,
    CLASSIFICATION_UNDERLYING_WRONG_CONTRACT_OK_MECHANICS,
    CLASSIFICATION_UNDERLYING_WRONG_CONTRACT_WRONG,
    CLASSIFICATION_UNKNOWN,
    CLASSIFICATION_WIDE_SPREAD_COST,
    MIN_SAMPLE_DEMOTE,
    MIN_SAMPLE_PROMOTE,
    NEGATIVE_CAP,
    POSITIVE_CAP,
    classify_contract_outcome,
    contract_context_key,
    contract_context_modifiers,
    contract_ranking_adjustment,
    record_contract_settlement,
)


def _contract(**overrides):
    base = {
        "option_type": "CE",
        "moneyness": "ATM",
        "delta": 0.45,
        "dte": 5,
        "iv_percentile": 50.0,
        "spread_pct": 1.5,
        "score": 70.0,
    }
    base.update(overrides)
    return base


def _seed(*, context_key: str, evidence_class: str, n: int,
          realized_R: float | list[float], path, prefix: str) -> None:
    values = realized_R if isinstance(realized_R, list) else [realized_R] * n
    assert len(values) == n
    for i, r in enumerate(values):
        outcome = Outcome(
            position_id=f"{prefix}-{i}", paper_order_id=f"{prefix}-{i}",
            paper_intent_id=f"{prefix}-{i}", decision_id=f"{prefix}-{i}",
            symbol="TESTOPT", realized_R=r, exit_reason="TEST",
            entry_session="2026-01-01", exit_session=f"2026-01-{2 + (i % 25):02d}",
            evidence_class=evidence_class,
            resolved_at=f"2026-01-{2 + (i % 25):02d}T10:00:00+00:00",
        )
        CE.record_outcome(outcome, context_key=context_key, evidence_class=evidence_class, path=path)


# ── context key ──────────────────────────────────────────────────────────

def test_context_key_empty_without_option_type():
    assert contract_context_key({"delta": 0.5}) == ""
    assert contract_context_key(None) == ""


def test_context_key_stable_for_identical_shape_and_differs_by_dimension():
    a = contract_context_key(_contract())
    b = contract_context_key(_contract())
    assert a == b
    assert a != contract_context_key(_contract(option_type="PE"))
    assert a != contract_context_key(_contract(delta=0.05))
    assert a != contract_context_key(_contract(dte=25))


# ── ranking adjustment: sample floors, caps, promotion gates ────────────

def test_tiny_sample_cannot_affect_contract_selection(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE - 1,
          realized_R=-5.0, path=path, prefix="C")

    result = contract_ranking_adjustment(contract, path=path)
    assert result["adjustment"] == 0.0
    assert result["reason"] == "INSUFFICIENT_CONTRACT_SAMPLE"


def test_robust_negative_forward_evidence_demotes_contract(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=-3.0, path=path, prefix="C")

    result = contract_ranking_adjustment(contract, path=path)
    assert result["direction"] == "DEMOTE"
    assert result["adjustment"] < 0.0
    assert result["adjustment"] == NEGATIVE_CAP


def test_robust_positive_forward_evidence_promotes_contract(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_PROMOTE,
          realized_R=2.0, path=path, prefix="C")

    result = contract_ranking_adjustment(contract, path=path)
    assert result["direction"] == "PROMOTE"
    assert 0.0 < result["adjustment"] <= POSITIVE_CAP


def test_positive_expectancy_below_promotion_floor_does_not_promote(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=2.0, path=path, prefix="C")

    result = contract_ranking_adjustment(contract, path=path)
    assert result["usable"] is False
    assert result["reason"] == "INSUFFICIENT_SAMPLE_FOR_PROMOTION"
    assert result["adjustment"] == 0.0


def test_weak_win_rate_does_not_promote(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    values = [8.0] * 8 + [-0.3] * 32
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=len(values),
          realized_R=values, path=path, prefix="C")

    result = contract_ranking_adjustment(contract, path=path)
    assert result["usable"] is False
    assert result["reason"] == "WIN_RATE_LOWER_BOUND_TOO_WEAK_FOR_PROMOTION"


def test_unstable_evidence_does_not_promote(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    values = [-5.0] * 5 + [0.1] * 15 + [2.0] * 20
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=len(values),
          realized_R=values, path=path, prefix="C")

    result = contract_ranking_adjustment(contract, path=path)
    assert result["stability"]["checked"] is True
    assert result["stability"]["stable"] is False
    assert result["usable"] is False
    assert result["reason"] == "CONTRACT_UNSTABLE_ACROSS_PERIODS"


def test_caps_hold_under_extreme_inputs(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    neg_contract = _contract(delta=0.05)
    key = contract_context_key(neg_contract)
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=200, realized_R=-50.0, path=path, prefix="CN")
    neg = contract_ranking_adjustment(neg_contract, path=path)
    assert neg["adjustment"] == NEGATIVE_CAP

    pos_contract = _contract(delta=0.90)
    key2 = contract_context_key(pos_contract)
    _seed(context_key=key2, evidence_class=PAPER_FORWARD, n=200, realized_R=50.0, path=path, prefix="CP")
    pos = contract_ranking_adjustment(pos_contract, path=path)
    assert pos["adjustment"] == POSITIVE_CAP


def test_no_context_key_yields_zero_not_an_exception():
    result = contract_ranking_adjustment({"delta": 0.5})
    assert result["adjustment"] == 0.0
    assert result["context_key"] == ""
    assert result["reason"] == "NO_CONTRACT_CONTEXT_KEY"


# ── historical/fabricated option-chain evidence must never be used ─────

def test_historical_counterfactual_evidence_never_moves_contract_selection(tmp_path, monkeypatch):
    """No historical NSE option-chain source exists in this repository. Even a
    huge, robust COUNTERFACTUAL cell parked under the exact same contract
    context key must be completely invisible to contract_ranking_adjustment --
    it only ever reads PAPER_FORWARD.
    """
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    _seed(context_key=key, evidence_class=COUNTERFACTUAL, n=500, realized_R=10.0, path=path, prefix="H")

    result = contract_ranking_adjustment(contract, path=path)
    assert result["adjustment"] == 0.0
    assert result["reason"] == "INSUFFICIENT_CONTRACT_SAMPLE"
    assert result["count"] == 0, "the COUNTERFACTUAL cell must not be counted at all"


def test_genuine_forward_contract_evidence_changes_future_selection(tmp_path, monkeypatch):
    """Once genuine PAPER_FORWARD evidence exists for a contract shape, it
    measurably changes the learned score used for future selection.
    """
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    before = contract_ranking_adjustment(contract, path=path)
    assert before["adjustment"] == 0.0

    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=-3.0, path=path, prefix="C")
    after = contract_ranking_adjustment(contract, path=path)
    assert after["adjustment"] < before["adjustment"]


def test_record_contract_settlement_skips_ineligible_trade(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    row = {
        "trade_id": "T1", "entry_price": 10.0, "stop_price": 8.0, "exit_price": 12.0,
        "quantity": 50, "net_pnl": 100.0, "mfe_pct": 20.0, "mae_pct": -5.0,
        "opened_at": "2026-01-01", "settled_at": "2026-01-02",
        "production_evidence_eligible": False,
    }
    result = record_contract_settlement(row, context_key=key, path=path)
    assert result is None
    cell = CE.read(key, evidence_class=PAPER_FORWARD, path=path)
    assert cell["count"] == 0


def test_record_contract_settlement_records_eligible_trade(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    key = contract_context_key(contract)
    row = {
        "trade_id": "T1", "entry_price": 10.0, "stop_price": 8.0, "exit_price": 12.0,
        "quantity": 50, "net_pnl": 100.0, "mfe_pct": 20.0, "mae_pct": -5.0,
        "opened_at": "2026-01-01", "settled_at": "2026-01-02",
        "production_evidence_eligible": True, "option_symbol": "TESTOPT",
    }
    result = record_contract_settlement(row, context_key=key, path=path)
    assert result is not None
    cell = CE.read(key, evidence_class=PAPER_FORWARD, path=path)
    assert cell["count"] == 1


# ── underlying-vs-contract correctness are graded on separate evidence ──

def test_underlying_ranking_and_contract_ranking_read_disjoint_cells(tmp_path, monkeypatch):
    """The same settled trade's evidence must land in two DIFFERENT cells:
    one keyed by the underlying setup (fno_evidence_fusion), one keyed by the
    contract shape (this module). Recording strong evidence into the contract
    cell alone must not make the underlying-ranking cell usable, and vice
    versa -- proving the two are not secretly pooled.
    """
    from product.fno_evidence import fno_context_key
    from product.fno_evidence_fusion import fuse_fno_ranking_evidence

    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = {
        "score": 80.0, "direction": "LONG", "atr_pct": 2.0, "breakout_distance_pct": 1.5,
        "components": {"nifty_alignment": 3.0, "sector_strength": 2.5},
    }
    setup_key = fno_context_key(setup)
    contract = _contract()
    contract_key = contract_context_key(contract)
    assert setup_key != contract_key

    _seed(context_key=contract_key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_PROMOTE,
          realized_R=2.0, path=path, prefix="C")

    underlying = fuse_fno_ranking_evidence(setup, path=path)
    assert underlying["adjustment"] == 0.0, "contract-only evidence must not leak into underlying ranking"

    contract_result = contract_ranking_adjustment(contract, path=path)
    assert contract_result["adjustment"] > 0.0


# ── outcome classification: evidence-justified only, never invented ────

def _row(**overrides):
    base = {
        "entry_underlying_spot": 100.0, "exit_underlying_spot": 105.0,
        "option_type": "CE", "net_pnl": 500.0, "exit_reason": "TARGET",
        "quantity": 50, "opened_at": "2026-01-01", "settled_at": "2026-01-02",
    }
    base.update(overrides)
    return base


def test_classify_unknown_without_spot_evidence():
    result = classify_contract_outcome(_row(entry_underlying_spot=0.0, exit_underlying_spot=0.0))
    assert result["classification"] == CLASSIFICATION_UNKNOWN


def test_classify_iv_crush_from_exit_reason():
    result = classify_contract_outcome(_row(exit_reason="IV_CRUSH", net_pnl=-200.0))
    assert result["classification"] == CLASSIFICATION_IV_CRUSH


def test_classify_underlying_right_contract_right():
    result = classify_contract_outcome(_row())
    assert result["classification"] == CLASSIFICATION_UNDERLYING_RIGHT_CONTRACT_RIGHT
    assert result["underlying_direction"] == "RIGHT"


def test_classify_underlying_wrong_contract_wrong():
    result = classify_contract_outcome(_row(exit_underlying_spot=95.0, net_pnl=-300.0))
    assert result["classification"] == CLASSIFICATION_UNDERLYING_WRONG_CONTRACT_WRONG
    assert result["underlying_direction"] == "WRONG"


def test_classify_underlying_wrong_contract_ok_mechanics():
    result = classify_contract_outcome(_row(exit_underlying_spot=95.0, net_pnl=150.0))
    assert result["classification"] == CLASSIFICATION_UNDERLYING_WRONG_CONTRACT_OK_MECHANICS


def test_classify_wide_spread_cost():
    result = classify_contract_outcome(
        _row(net_pnl=-100.0, entry_spread_pct=6.0)
    )
    assert result["classification"] == CLASSIFICATION_WIDE_SPREAD_COST


def test_classify_bad_expiry_choice():
    result = classify_contract_outcome(
        _row(net_pnl=-100.0, entry_dte=1, exit_reason="MAX_HOLD")
    )
    assert result["classification"] == CLASSIFICATION_BAD_EXPIRY_CHOICE


def test_classify_poor_liquidity():
    result = classify_contract_outcome(
        _row(net_pnl=-100.0, entry_oi=100)
    )
    assert result["classification"] == CLASSIFICATION_POOR_LIQUIDITY


def test_classify_delta_too_low():
    result = classify_contract_outcome(
        _row(net_pnl=-100.0, entry_delta=0.1)
    )
    assert result["classification"] == CLASSIFICATION_DELTA_TOO_LOW


def test_classify_delta_too_high():
    result = classify_contract_outcome(
        _row(net_pnl=-100.0, entry_delta=0.92)
    )
    assert result["classification"] == CLASSIFICATION_DELTA_TOO_HIGH


def test_classify_theta_damage():
    result = classify_contract_outcome(
        _row(net_pnl=-100.0, exit_reason="MAX_HOLD",
             entry_theta_per_day=-3.0, opened_at="2026-01-01", settled_at="2026-01-05")
    )
    assert result["classification"] == CLASSIFICATION_THETA_DAMAGE


def test_classify_falls_back_to_contract_poor_when_no_specific_reason_evidenced():
    result = classify_contract_outcome(_row(net_pnl=-1.0))
    assert result["classification"] == CLASSIFICATION_UNDERLYING_RIGHT_CONTRACT_POOR


# ── hierarchical context modifiers: base + optional, sample-gated refinement ──

def test_absent_optional_dimensions_never_create_a_fabricated_modifier():
    contract = _contract()  # no oi/volume set
    modifiers = contract_context_modifiers(contract)
    assert set(modifiers.keys()) == {"oi", "volume"}, (
        "holding_horizon/regime/setup_type/daypart must never appear unless "
        "the caller actually supplies a real value"
    )
    assert modifiers["oi"].endswith("|oi=UNKNOWN")
    assert modifiers["volume"].endswith("|volume=UNKNOWN")


def test_supplied_dimensions_form_their_own_narrower_keys():
    contract = _contract()
    modifiers = contract_context_modifiers(
        contract, holding_horizon="multi_day", regime="risk_on", setup_type="long_buildup",
    )
    base = contract_context_key(contract)
    assert modifiers["holding_horizon"] == f"{base}|horizon=MULTI_DAY"
    assert modifiers["regime"] == f"{base}|regime=RISK_ON"
    assert modifiers["setup_type"] == f"{base}|setup=LONG_BUILDUP"


def test_no_key_yields_no_modifiers():
    assert contract_context_modifiers({"delta": 0.5}) == {}


def test_tiny_modifier_sample_cannot_override_a_usable_base(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    base_key = contract_context_key(contract)
    _seed(context_key=base_key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=-3.0, path=path, prefix="BASE")

    modifier_key = contract_context_modifiers(contract, holding_horizon="multi_day")["holding_horizon"]
    _seed(context_key=modifier_key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE - 1,
          realized_R=-3.0, path=path, prefix="MOD-TINY")

    result = contract_ranking_adjustment(contract, path=path, holding_horizon="multi_day")
    assert result["used_modifier"] is None
    assert result["context_key"] == base_key


def test_modifier_overrides_base_once_it_independently_qualifies_and_agrees(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    base_key = contract_context_key(contract)
    # Base: mature, demoted, capped at NEGATIVE_CAP.
    _seed(context_key=base_key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=-3.0, path=path, prefix="BASE")
    # A narrower, independently-supported modifier: also demoted (agrees),
    # milder magnitude, MORE sample -- more informative, so it should win.
    modifier_key = contract_context_modifiers(contract, holding_horizon="multi_day")["holding_horizon"]
    _seed(context_key=modifier_key, evidence_class=PAPER_FORWARD, n=50,
          realized_R=-1.0, path=path, prefix="MOD-AGREE")

    result = contract_ranking_adjustment(contract, path=path, holding_horizon="multi_day")
    assert result["used_modifier"] == "holding_horizon"
    assert result["context_key"] == modifier_key
    assert result["direction"] == "DEMOTE"
    assert result["adjustment"] > NEGATIVE_CAP, "the milder, more specific lesson must be used, not the base's cap"


def test_modifier_contradicting_a_usable_base_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    base_key = contract_context_key(contract)
    _seed(context_key=base_key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=-3.0, path=path, prefix="BASE")
    # A mature, stable, Wilson-clearing PROMOTE modifier -- but it disagrees
    # with the already-validated base, so it must be ignored entirely.
    modifier_key = contract_context_modifiers(contract, holding_horizon="multi_day")["holding_horizon"]
    _seed(context_key=modifier_key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_PROMOTE,
          realized_R=2.0, path=path, prefix="MOD-CONTRADICT")

    result = contract_ranking_adjustment(contract, path=path, holding_horizon="multi_day")
    assert result["used_modifier"] is None
    assert result["context_key"] == base_key
    assert result["direction"] == "DEMOTE"


def test_modifier_acts_alone_when_base_has_no_usable_evidence(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    modifier_key = contract_context_modifiers(contract, regime="risk_on")["regime"]
    _seed(context_key=modifier_key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_PROMOTE,
          realized_R=1.5, path=path, prefix="MOD-ALONE")

    result = contract_ranking_adjustment(contract, path=path, regime="risk_on")
    assert result["used_modifier"] == "regime"
    assert result["direction"] == "PROMOTE"
    assert result["context_key"] == modifier_key


def test_oi_and_volume_modifiers_activate_without_any_extra_caller_context(tmp_path, monkeypatch):
    """OI/volume are derived straight from the contract, so they must be able
    to act even when the caller supplies none of the other optional
    dimensions -- no extra plumbing required for the simplest, always-
    available refinement.
    """
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract(oi=50_000, volume=20_000)
    modifier_key = contract_context_modifiers(contract)["oi"]
    _seed(context_key=modifier_key, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_PROMOTE,
          realized_R=1.5, path=path, prefix="OI-ALONE")

    result = contract_ranking_adjustment(contract, path=path)
    assert result["used_modifier"] == "oi"
    assert result["direction"] == "PROMOTE"


def test_considered_list_reports_every_dimension_examined(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    contract = _contract()
    result = contract_ranking_adjustment(
        contract, path=path, holding_horizon="multi_day", regime="risk_on", setup_type="long_buildup",
    )
    considered_names = {row["modifier"] for row in result["considered"]}
    assert considered_names == {"base", "oi", "volume", "holding_horizon", "regime", "setup_type"}
