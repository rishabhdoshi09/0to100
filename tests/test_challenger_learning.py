from __future__ import annotations

from datetime import datetime, timedelta, timezone

from product import challenger_learning as CL
from research import feature_store as FS


def _seed(
    *,
    n: int,
    db,
    model_version: str = "",
    forward_compare: bool = False,
):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        strong = (i % 10) >= 4
        outcome = 1.0 if strong else -1.0
        rs = 90.0 if strong else 35.0
        champion_p = 0.50
        challenger_p = (0.90 if strong else 0.10) if forward_compare else None
        meta = {
            "setup": "VCP_BREAKOUT",
            "market_state": "TRENDING_BULL",
            "sector_state": "STRONG",
            "predicted_p": champion_p,
            "challenger_predicted_p": challenger_p,
            "challenger_model_version": model_version if forward_compare else "",
        }
        oid = f"decision::seed-{model_version or 'train'}-{i}"
        frozen = FS.snapshot(
            oid,
            f"S{i:03d}",
            "DECISION",
            {
                "rsi": 62.0,
                "atr_pct": 2.3,
                "rs_percentile": rs,
                "quality_score": rs,
                "volume_ratio": 1.7 if strong else 0.9,
                "breadth_pct_above_50dma": 65.0,
                "sector_strength": 1.4,
                "regime": "TRENDING_BULL",
                "index_trend": "UP",
                "correlation_regime": "NORMAL",
            },
            ts=(base + timedelta(days=i)).isoformat(),
            meta=meta,
        )
        assert frozen["status"] in {"frozen", "exists"}
        lane = "PAPER_FORWARD" if (forward_compare or i < max(20, n // 3)) else "FORWARD_COUNTERFACTUAL"
        settled = FS.set_outcome(
            oid,
            outcome,
            outcome_meta={
                "evidence_class": lane,
                "not_pnl": lane != "PAPER_FORWARD",
            },
        )
        assert settled["status"] in {"settled", "exists"}


def test_challenger_is_shadow_until_forward_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "features.db")
    path = tmp_path / "challenger.json"
    _seed(n=90, db=tmp_path)

    model = CL.train(path=path)
    assert model["trained_n"] == 90
    assert model["real_forward_n"] >= 20
    assert model["status"] in {CL.SHADOW_CANDIDATE, CL.OBSERVING}
    assert model["status"] != CL.PAPER_ACTIVE
    assert model["affects_selection"] is False
    assert model["live_locked"] is True

    card = {
        "symbol": "ABC",
        "setup_label": "VCP_BREAKOUT",
        "rsi": 63,
        "atr_pct": 2.2,
        "rs_percentile": 92,
        "quality_score": 90,
        "volume_ratio": 1.8,
        "breadth_pct_above_50dma": 65,
        "sector_strength": 1.4,
        "regime": "TRENDING_BULL",
        "index_trend": "UP",
        "correlation_regime": "NORMAL",
        "entry": 100,
        "stop": 95,
        "target": 110,
    }
    adj = CL.paper_selection_adjustment(card, path=path)
    assert adj["adjustment"] == 0.0
    assert adj["affects_selection"] is False


def test_exact_version_forward_evidence_can_promote_paper_only(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "features.db")
    path = tmp_path / "challenger.json"
    _seed(n=110, db=tmp_path)
    model = CL.train(path=path)
    assert model["status"] == CL.SHADOW_CANDIDATE
    version = model["model_version"]

    # Forward observations freeze both champion and exact challenger predictions.
    _seed(n=40, db=tmp_path, model_version=version, forward_compare=True)

    promoted_store = CL.maybe_promote(path=path)
    current = promoted_store["current"]
    assert current["status"] == CL.PAPER_ACTIVE
    assert current["affects_selection"] is True
    assert current["live_locked"] is True
    assert current["forward_validation"]["n"] >= 30
    assert current["forward_validation"]["improvement"] > 0
    assert current["forward_validation"]["improvement_lower_95"] > 0

    card = {
        "symbol": "XYZ",
        "setup_label": "VCP_BREAKOUT",
        "rsi": 63,
        "atr_pct": 2.2,
        "rs_percentile": 92,
        "quality_score": 90,
        "volume_ratio": 1.8,
        "breadth_pct_above_50dma": 65,
        "sector_strength": 1.4,
        "regime": "TRENDING_BULL",
        "index_trend": "UP",
        "correlation_regime": "NORMAL",
        "entry": 100,
        "stop": 95,
        "target": 110,
    }
    adj = CL.paper_selection_adjustment(card, path=path)
    assert adj["affects_selection"] is True
    assert -5.0 <= adj["adjustment"] <= 3.0
