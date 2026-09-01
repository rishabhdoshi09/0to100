"""Phase 10 — ML only as a Challenger."""

from __future__ import annotations

from product.ml_challenger import MlChallenger, leakage_check
from product.strategy_catalog import ensemble_identity


def _row(i, *, leak=False, pnl=0.4):
    as_of = f"2026-01-{i:02d}"
    features = {"rs": 50 + i, "extension": 0.1 * i, "dd": 0.5}
    if leak:
        features["future_return"] = 9.9
    return {
        "as_of": as_of,
        "decision_as_of": as_of,
        "feature_ts": "2026-06-01" if leak else as_of,
        "features": features,
        "label": i % 2 == 0,
        "pnl": pnl if i % 2 == 0 else -0.1,
    }


def test_ml_starts_as_challenger_and_cannot_execute(tmp_path):
    ml = MlChallenger(path=tmp_path / "ch.json")
    row = ml.register(hypothesis="rank finalists", features=["rs", "extension", "dd"])
    assert row["can_execute"] is False
    assert ml.can_execute is False
    assert ml.role == "CHALLENGER"
    assert ensemble_identity()["rules_hash"]  # champion untouched


def test_leakage_is_rejected():
    leaks = leakage_check([_row(1, leak=True)])
    assert leaks
    ml = MlChallenger()
    out = ml.fit([_row(i, leak=True) for i in range(1, 12)], [_row(20, leak=True)])
    assert out["fitted"] is False


def test_no_ai_score_and_no_auto_promote(tmp_path):
    ml = MlChallenger(path=tmp_path / "ch.json")
    ml.register(hypothesis="h", features=["rs"])
    train = [_row(i, pnl=0.5) for i in range(1, 16)]
    val = [_row(i, pnl=0.4) for i in range(16, 22)]
    oos = [_row(i, pnl=0.3) for i in range(22, 28)]
    fitted = ml.fit(train, val)
    assert fitted["fitted"] is True
    assert fitted["can_execute"] is False
    scored = ml.score_oos(oos, champion_pnls=[0.25] * len(oos))
    assert scored["ai_score_written"] is False
    assert scored["recommend_promote"] is False
    assert scored["can_execute"] is False
    assert scored["feature_contributions"] is not None
    assert scored["same_execution_reality"] is True
    wf = ml.walk_forward([{"train": train, "val": val, "oos": oos}])
    assert wf["can_execute"] is False
    # Champion hash unchanged
    before = ensemble_identity()["rules_hash"]
    ml.score_oos(oos, champion_pnls=[0.2] * 10)
    assert ensemble_identity()["rules_hash"] == before
