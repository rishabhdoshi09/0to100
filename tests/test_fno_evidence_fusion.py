"""Deterministic acceptance tests for the F&O evidence-fusion policy
(product/fno_evidence_fusion.py): the explicit rules for how historical
(COUNTERFACTUAL) and forward (PAPER_FORWARD) evidence combine into one
ranking adjustment.

Each test isolates ONE row of the policy table so a future change that
breaks any single rule fails exactly the test that names it, rather than a
single large end-to-end assertion.
"""
from __future__ import annotations

import product.conditional_evidence as CE
from product.decision_chain import Outcome
from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD
from product.fno_evidence import fno_context_key
from product.fno_evidence_fusion import (
    COMBINED_NEGATIVE_CAP,
    COMBINED_POSITIVE_CAP,
    FORWARD_MIN_SAMPLE_DEMOTE,
    FORWARD_MIN_SAMPLE_PROMOTE,
    FORWARD_NEGATIVE_CAP,
    FORWARD_POSITIVE_CAP,
    HISTORICAL_CAP,
    HISTORICAL_MIN_SAMPLE,
    fuse_fno_ranking_evidence,
)


def _setup(*, sector_strength: float = 2.5, nifty_alignment: float = 3.0, **overrides):
    # fno_context_key reads sector_strength/nifty_alignment from the NESTED
    # "components" dict, not top-level -- these must be passed through there,
    # or two setups meant to land in different buckets silently collide.
    base = {
        "score": 80.0,
        "direction": "LONG",
        "atr_pct": 2.0,
        "breakout_distance_pct": 1.5,
        "components": {"nifty_alignment": nifty_alignment, "sector_strength": sector_strength},
    }
    base.update(overrides)
    return base


def _seed(
    *, context_key: str, evidence_class: str, n: int, realized_R: float | list[float],
    path, prefix: str,
) -> None:
    values = realized_R if isinstance(realized_R, list) else [realized_R] * n
    assert len(values) == n
    for i, r in enumerate(values):
        outcome = Outcome(
            position_id=f"{prefix}-{i}", paper_order_id=f"{prefix}-{i}",
            paper_intent_id=f"{prefix}-{i}", decision_id=f"{prefix}-{i}",
            symbol="TESTCO", realized_R=r, exit_reason="TEST",
            entry_session="2026-01-01", exit_session=f"2026-01-{2 + (i % 25):02d}",
            evidence_class=evidence_class,
            resolved_at=f"2026-01-{2 + (i % 25):02d}T10:00:00+00:00",
        )
        CE.record_outcome(outcome, context_key=context_key, evidence_class=evidence_class, path=path)


def test_1_tiny_historical_sample_alone_cannot_affect_ranking(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    _seed(context_key=key, evidence_class=COUNTERFACTUAL, n=HISTORICAL_MIN_SAMPLE - 1,
          realized_R=-3.0, path=path, prefix="H")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["adjustment"] == 0.0
    assert result["historical_prior"] == 0.0
    assert result["historical"]["reason"] == "INSUFFICIENT_HISTORICAL_SAMPLE"
    assert result["status"] in {"NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"}


def test_2_large_stable_historical_evidence_creates_only_a_bounded_prior(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    # An extreme per-trade loss, repeated many times, would demote a REAL
    # forward cell to its full -30 cap. Historical must still be bounded far
    # below that even at this magnitude.
    _seed(context_key=key, evidence_class=COUNTERFACTUAL, n=200,
          realized_R=-10.0, path=path, prefix="H")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["status"] == "HISTORICAL_PRIOR_ONLY"
    assert result["forward_adjustment"] == 0.0
    assert result["historical_prior"] < 0.0
    assert abs(result["historical_prior"]) <= HISTORICAL_CAP
    assert result["adjustment"] == result["historical_prior"]


def test_3_forward_evidence_has_greater_weight_than_historical(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    # Same magnitude, same sign, same sample size -- forward's cap alone must
    # still be strictly larger than historical's cap.
    _seed(context_key=key, evidence_class=COUNTERFACTUAL, n=200, realized_R=-10.0, path=path, prefix="H")
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=200, realized_R=-10.0, path=path, prefix="F")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["status"] == "FORWARD_DOMINATES"
    assert abs(result["forward_adjustment"]) > HISTORICAL_CAP
    assert abs(result["forward_adjustment"]) >= abs(result["historical_prior"])
    assert result["forward_adjustment"] == FORWARD_NEGATIVE_CAP


def test_4_historical_positive_forward_negative_forward_dominates(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    _seed(context_key=key, evidence_class=COUNTERFACTUAL, n=200, realized_R=8.0, path=path, prefix="H")
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=FORWARD_MIN_SAMPLE_DEMOTE,
          realized_R=-2.0, path=path, prefix="F")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["status"] == "FORWARD_DOMINATES"
    assert result["forward_adjustment"] < 0.0
    assert result["historical_prior"] == 0.0, "opposite-signed historical evidence must be ignored entirely"
    assert result["historical_note"] == "DISAGREES_WITH_FORWARD_IGNORED"
    assert result["adjustment"] == result["forward_adjustment"]


def test_5_historical_negative_strong_forward_positive_can_recover_ranking(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    _seed(context_key=key, evidence_class=COUNTERFACTUAL, n=200, realized_R=-8.0, path=path, prefix="H")
    # All wins, well past the stricter promotion floor -- wilson lower bound
    # will be high, and the two chronological halves are identical (stable).
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=FORWARD_MIN_SAMPLE_PROMOTE,
          realized_R=2.0, path=path, prefix="F")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["status"] == "FORWARD_DOMINATES"
    assert result["forward_adjustment"] > 0.0, "strong validated forward evidence must be able to promote despite a negative historical prior"
    assert result["historical_prior"] == 0.0
    assert result["historical_note"] == "DISAGREES_WITH_FORWARD_IGNORED"


def test_6_strong_validated_positive_evidence_can_promote(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=FORWARD_MIN_SAMPLE_PROMOTE,
          realized_R=1.5, path=path, prefix="F")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["forward"]["direction"] == "PROMOTE"
    assert result["forward"]["reason"] == "FORWARD_POSITIVE_EXPECTANCY_VALIDATED"
    assert 0.0 < result["forward_adjustment"] <= FORWARD_POSITIVE_CAP
    assert result["adjustment"] > 0.0


def test_6b_positive_expectancy_below_promotion_sample_floor_does_not_promote(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    # Enough for demotion's floor (30) but short of promotion's stricter floor (40).
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=FORWARD_MIN_SAMPLE_DEMOTE,
          realized_R=1.5, path=path, prefix="F")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["forward"]["usable"] is False
    assert result["forward"]["reason"] == "INSUFFICIENT_SAMPLE_FOR_PROMOTION"
    assert result["adjustment"] == 0.0, "promotion must never fire below its own, stricter sample floor"


def test_6c_positive_expectancy_with_weak_win_rate_does_not_promote(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    # A few big wins and many small losses: enough sample and positive mean
    # expectancy, but few outright wins -> a weak Wilson lower bound on the
    # win rate. Promotion must refuse this even though "expectancy > 0".
    values = [8.0] * 8 + [-0.3] * 32
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=len(values),
          realized_R=values, path=path, prefix="F")

    result = fuse_fno_ranking_evidence(setup, path=path)
    cell = CE.read(key, evidence_class=PAPER_FORWARD, path=path)
    assert cell["expectancy_R"] > 0, "fixture sanity check: aggregate expectancy really is positive"
    assert result["forward"]["usable"] is False
    assert result["forward"]["reason"] == "WIN_RATE_LOWER_BOUND_TOO_WEAK_FOR_PROMOTION"
    assert result["adjustment"] == 0.0


def test_6d_unstable_positive_evidence_does_not_promote(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    # High win rate (35/40, clears the Wilson gate easily) and positive
    # overall expectancy, but the R-values' own chronological halves have
    # opposite-signed means: five large first-half losses outweigh many
    # small first-half wins, while the whole second half is solid wins. A
    # real regime break, not a durable edge -- the win-count-based Wilson
    # bound alone cannot see this, which is exactly why stability is a
    # separate, independent check.
    values = [-5.0] * 5 + [0.1] * 15 + [2.0] * 20
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=len(values),
          realized_R=values, path=path, prefix="F")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["forward"]["stability"]["checked"] is True
    assert result["forward"]["stability"]["stable"] is False
    assert result["forward"]["usable"] is False
    assert result["forward"]["reason"] == "FORWARD_UNSTABLE_ACROSS_PERIODS"
    assert result["adjustment"] == 0.0


def test_7_strong_validated_negative_evidence_can_demote(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    key = fno_context_key(setup)
    _seed(context_key=key, evidence_class=PAPER_FORWARD, n=FORWARD_MIN_SAMPLE_DEMOTE,
          realized_R=-2.0, path=path, prefix="F")

    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["forward"]["direction"] == "DEMOTE"
    assert result["forward_adjustment"] < 0.0
    assert result["adjustment"] == FORWARD_NEGATIVE_CAP


def test_8_adjustment_caps_hold_under_extreme_inputs(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))

    # Extreme negative agreement: forward AND historical both very negative.
    neg_setup = _setup(sector_strength=9.0)
    neg_key = fno_context_key(neg_setup)
    _seed(context_key=neg_key, evidence_class=COUNTERFACTUAL, n=200, realized_R=-50.0, path=path, prefix="HN")
    _seed(context_key=neg_key, evidence_class=PAPER_FORWARD, n=200, realized_R=-50.0, path=path, prefix="FN")
    neg = fuse_fno_ranking_evidence(neg_setup, path=path)
    assert neg["adjustment"] >= COMBINED_NEGATIVE_CAP, "combined adjustment must never exceed its floor"
    assert neg["forward_adjustment"] == FORWARD_NEGATIVE_CAP
    assert neg["historical_prior"] == -HISTORICAL_CAP

    # Extreme positive agreement: forward AND historical both very positive.
    pos_setup = _setup(sector_strength=-9.0)
    pos_key = fno_context_key(pos_setup)
    _seed(context_key=pos_key, evidence_class=COUNTERFACTUAL, n=200, realized_R=50.0, path=path, prefix="HP")
    _seed(context_key=pos_key, evidence_class=PAPER_FORWARD, n=200, realized_R=50.0, path=path, prefix="FP")
    pos = fuse_fno_ranking_evidence(pos_setup, path=path)
    assert pos["adjustment"] <= COMBINED_POSITIVE_CAP, "combined adjustment must never exceed its ceiling"
    assert pos["forward_adjustment"] == FORWARD_POSITIVE_CAP
    assert pos["historical_prior"] == HISTORICAL_CAP
    assert pos["historical_note"] == "AGREES_WITH_FORWARD_ADDED"


def test_no_evidence_anywhere_yields_the_untouched_baseline(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    setup = _setup()
    result = fuse_fno_ranking_evidence(setup, path=path)
    assert result["adjustment"] == 0.0
    assert result["historical_prior"] == 0.0
    assert result["forward_adjustment"] == 0.0
    assert result["status"] == "NO_EVIDENCE"


def test_no_context_key_yields_zero_not_an_exception():
    result = fuse_fno_ranking_evidence({"direction": "SIDEWAYS"})
    assert result["adjustment"] == 0.0
    assert result["context_key"] == ""
    assert result["status"] == "NO_CONTEXT_KEY"
