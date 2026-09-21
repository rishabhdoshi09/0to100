from __future__ import annotations

from research.autonomy import evidence_acquisition as EA
from research.autonomy import hypotheses as HYP
from research.autonomy import research_loop as RL


def test_forward_gap_cannot_be_satisfied_by_historical_replay(tmp_path, monkeypatch):
    target = tmp_path / "request.json"
    monkeypatch.setattr(EA, "DEFAULT_REQUEST_PATH", target)
    gap = HYP.EvidenceGap(
        kind="insufficient_sample",
        strategy_id="MOM",
        diagnosis="Only 5 resolved forward trades.",
        economic_impact=0.4,
        confidence=1.0,
        data_available=True,
        data_mining_risk=0.8,
        evidence_origin="FORWARD_PAPER",
        current_samples=5,
        target_samples=30,
    )

    out = RL.execute_pipeline(
        object(),
        gap=gap,
        parent=object(),
        session_date="2026-09-21",
    )

    req = out["evidence_request"]
    assert out["decision"] == "EVIDENCE_ACQUISITION"
    assert req["current_samples"] == 5
    assert req["target_samples"] == 30
    assert req["sample_deficit"] == 25
    assert req["allowed_lanes"] == ("FORWARD_PAPER",)
    assert EA.open_request_for_lane("HISTORICAL_REPLAY", path=target) == {}
    assert EA.open_request_for_lane("FORWARD_PAPER", path=target)["request_id"] == req["request_id"]


def test_research_validation_request_names_missing_metrics(tmp_path):
    gap = HYP.EvidenceGap(
        kind="poor_calibration",
        strategy_id="MOM",
        diagnosis="Validation evidence is incomplete.",
        economic_impact=0.6,
        confidence=0.8,
        data_available=True,
        data_mining_risk=0.4,
        evidence_origin="RESEARCH_VALIDATION",
        current_samples=40,
        target_samples=40,
    )
    context = {
        "evidence_origin": "RESEARCH_VALIDATION",
        "n_trades": 40,
        "raw": {
            "n_trades": 40,
            "benchmark_available": True,
            "walk_forward_ok": True,
        },
    }

    req = EA.request_from_gap(
        gap,
        session_date="2026-09-21",
        context=context,
        path=tmp_path / "request.json",
    )

    assert req["allowed_lanes"] == ("HISTORICAL_REPLAY",)
    assert req["sample_deficit"] == 0
    assert set(req["missing_metrics"]) == {
        "deflated_sharpe",
        "fdr_significant",
        "reality_check_p",
    }
    assert "COMPUTE_DEFLATED_SHARPE" in req["acquisition_tasks"]
    assert "RUN_REALITY_CHECK" in req["acquisition_tasks"]
    assert "RUN_FDR_CONTROL" in req["acquisition_tasks"]


def test_evidence_request_id_is_deterministic_for_same_gap():
    kwargs = dict(
        session_date="2026-09-21",
        strategy_id="MOM",
        gap_kind="insufficient_sample",
        diagnosis="Need more independent validation trades.",
        evidence_origin="RESEARCH_VALIDATION",
        current_samples=12,
        target_samples=30,
        missing_metrics=("walk_forward_ok",),
        priority=0.5,
        thesis_hash="thesis-a",
    )
    a = EA.build_request(**kwargs)
    b = EA.build_request(**{**kwargs, "session_date": "2026-09-22"})
    assert a.request_id == b.request_id
    assert a.sample_deficit == 18
    assert "HISTORICAL_REPLAY" in a.allowed_lanes
