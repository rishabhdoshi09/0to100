from __future__ import annotations

from product.decision_adapter import decision_from_card
from product.paper_autopilot import AutopilotDecision, _canonical_decision
from product.evidence_class import PAPER_FORWARD
from research import feature_store as FS
from product import challenger_learning as CL


def test_ui_and_paper_paths_share_exact_canonical_decision_id(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "features.db")
    monkeypatch.setattr(CL, "DEFAULT_PATH", tmp_path / "challenger.json")

    scan_ts = "2026-09-18T09:45:00+05:30"
    card = {
        "symbol": "RELIANCE",
        "reco_tier": "good_setup",
        "setup_label": "VCP_BREAKOUT",
        "entry_state": "enter_now",
        "entry": 100.0,
        "stop": 95.0,
        "target": 110.0,
        "score": 82.0,
        "market_state": "RISK_ON",
        "sector_state": "STRONG",
        "rsi": 62.0,
        "atr_pct": 2.2,
        "rs_percentile": 91.0,
        "volume_ratio": 1.6,
    }

    expected = decision_from_card(
        card,
        source_scan_id=scan_ts,
        market_state="RISK_ON",
        sector_state="STRONG",
        evidence_class=PAPER_FORWARD,
        generated_at=scan_ts,
    )
    paper = AutopilotDecision(
        symbol="RELIANCE",
        decision="ENTER_NOW",
        reason_code="ELIGIBLE",
        card=dict(card),
    )
    got_id, context = _canonical_decision(
        paper,
        as_of="2026-09-18",
        snapshot_id=scan_ts,
    )
    assert got_id == expected.decision_id
    assert context
