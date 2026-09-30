"""product.fno_learning_impact: the read-only F&O evidence/learning summary.

Proves the specific honesty properties the desk depends on:
  - historical (COUNTERFACTUAL) and forward (PAPER_FORWARD) evidence are
    reported in separate blocks and never pooled into one win rate
  - historical evidence can only report can_affect_ranking=True once it is
    large AND stable enough to pass product.fno_evidence_fusion's own prior
    gate (see tests/test_fno_evidence_fusion.py for that policy itself) --
    this module never claims more or less than what that gate would do
  - a ranking claim is only made when a real candidate actually carries a
    nonzero, usable ranking_adjustment -- never inferred or assumed
"""
from __future__ import annotations

import json

import product.conditional_evidence as CE
from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD
from product.fno_learning_impact import build_fno_learning_impact


def _seed_cell(path, *, evidence_class: str, context_key: str, count: int) -> None:
    store = CE.empty_store()
    key = CE.cell_key(evidence_class, context_key)
    r_values = [0.1] * count
    wins = sum(1 for r in r_values if r > 0)
    store["cells"][key] = {
        "context_key": context_key,
        "evidence_class": evidence_class,
        "count": count,
        "r_values": r_values,
        # A real cell always carries these (product.conditional_evidence
        # ._recompute); write them explicitly since this test seeds the
        # store file directly rather than going through record_outcome.
        "wins": wins,
        "expectancy_R": sum(r_values) / count if count else None,
        "wilson_lower_bound": CE.wilson_lower_bound(wins, count) if count else None,
    }
    path.write_text(json.dumps(store), encoding="utf-8")


def test_historical_and_forward_are_never_pooled(tmp_path, monkeypatch):
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(evidence_path))
    _seed_cell(
        evidence_path, evidence_class=COUNTERFACTUAL,
        context_key="setup=FNO_LONG|regime=NEUTRAL|sector=NEUTRAL|vol=UNKNOWN|ext=UNKNOWN|conf=UNKNOWN",
        count=50,
    )
    monkeypatch.setattr(
        "product.fno_historical_loop.status",
        lambda: {
            "available": True, "last_run_at": "2026-01-01T00:00:00Z", "cursor_date": "2026-01-01",
            "coverage_complete": False, "total_sessions_processed": 12,
            "total_candidates_evaluated": 20, "total_settled": 20,
            "classification_counts": {"CORRECT_REJECTION": 5, "MISSED_WINNER": 2},
        },
    )

    class _EmptyStore:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def status(self):
            return {"open_positions": 0}

        def load_trades(self, limit=5000):
            return []

    monkeypatch.setattr("product.fo_paper_store.FoPaperStore", lambda *a, **k: _EmptyStore())

    result = build_fno_learning_impact(directional=None)

    assert result["historical"]["evidence_class"] == "HISTORICAL_COUNTERFACTUAL"
    assert result["forward"]["evidence_class"] == "PAPER_FORWARD"
    assert result["historical"]["evidence_cells"] == 1
    assert result["forward"]["evidence_cells"] == 0
    # count=50 at expectancy=0.1R is large enough and stable enough to clear
    # fno_evidence_fusion's own historical-prior gate (see
    # test_fno_evidence_fusion.py), so this module must report it honestly --
    # a fixed can_affect_ranking=False would be a claim this module cannot back.
    assert result["historical"]["can_affect_ranking"] is True
    assert result["historical"]["cells_large_enough_for_a_prior"] == 1
    assert result["historical"]["richest_priors"][0]["count"] == 50
    assert result["historical"]["correct_rejects"] == 5
    assert result["historical"]["missed_winners"] == 2
    # No combined/pooled win-rate key exists anywhere in the payload.
    flat = json.dumps(result)
    assert "combined_win_rate" not in flat
    assert "pooled" not in flat.lower()


def test_forward_evidence_maturity_gates_can_affect_ranking(tmp_path, monkeypatch):
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(evidence_path))
    context_key = "setup=FNO_SHORT|regime=NEUTRAL|sector=NEUTRAL|vol=UNKNOWN|ext=UNKNOWN|conf=UNKNOWN"
    _seed_cell(evidence_path, evidence_class=PAPER_FORWARD, context_key=context_key, count=5)
    monkeypatch.setattr(
        "product.fno_historical_loop.status",
        lambda: {"available": False, "classification_counts": {}},
    )

    class _EmptyStore:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def status(self):
            return {"open_positions": 0}

        def load_trades(self, limit=5000):
            return []

    monkeypatch.setattr("product.fo_paper_store.FoPaperStore", lambda *a, **k: _EmptyStore())

    result = build_fno_learning_impact(directional=None)
    # count=5 < MIN_SAMPLE=30: a real cell exists but has not matured.
    assert result["forward"]["evidence_cells"] == 1
    assert result["forward"]["matured_cells"] == 0
    assert result["forward"]["can_affect_ranking"] is False


def test_ranking_impact_reflects_real_candidate_adjustment_only():
    directional = {
        "candidates": [
            {
                "symbol": "DEMOTED",
                "direction": "LONG",
                "base_score": 80.0,
                "ranking_score": 65.0,
                "ranking_adjustment": -15.0,
                "ranking_evidence": {
                    "usable": True, "reason": "MEASURED_NEGATIVE_EXPECTANCY",
                    "count": 40, "expectancy_R": -0.3,
                },
            },
            {
                "symbol": "UNCHANGED",
                "direction": "SHORT",
                "base_score": 70.0,
                "ranking_score": 70.0,
                "ranking_adjustment": 0.0,
                "ranking_evidence": {"usable": False, "reason": "INSUFFICIENT_EVIDENCE"},
            },
        ],
    }
    result = build_fno_learning_impact(directional=directional)
    impact = result["ranking_impact"]
    assert impact["status"] == "RANKING_CHANGED_BY_VALIDATED_EVIDENCE"
    assert impact["influenced_count"] == 1
    assert impact["influenced"][0]["symbol"] == "DEMOTED"


def test_ranking_impact_makes_no_claim_when_nothing_moved():
    directional = {
        "candidates": [
            {
                "symbol": "FLAT",
                "base_score": 70.0,
                "ranking_score": 70.0,
                "ranking_adjustment": 0.0,
                "ranking_evidence": {"usable": False, "reason": "INSUFFICIENT_EVIDENCE"},
            },
        ],
    }
    result = build_fno_learning_impact(directional=directional)
    impact = result["ranking_impact"]
    assert impact["status"] == "NO_RANKING_CHANGE_YET"
    assert impact["influenced_count"] == 0


def test_never_promotes_or_touches_live_money():
    result = build_fno_learning_impact(directional=None)
    assert result["live_locked"] is True
    # Historical evidence CAN move ranking now (as a small, bounded prior --
    # see product.fno_evidence_fusion), but it can never touch live money,
    # and it is never pooled with forward evidence into one statistic.
    assert result["policy"]["historical_alone_can_move_ranking"] is True
    assert result["policy"]["historical_prior_is_small_and_bounded"] is True
    assert result["policy"]["live_money_affected"] is False
    assert result["policy"]["historical_and_forward_kept_in_separate_cells"] is True
