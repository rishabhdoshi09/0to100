"""Tests for the ensemble veto."""
from unittest.mock import MagicMock

from sq_ai.backend.ensemble import (
    EnsembleVeto,
    build_decision_prompt,
    parse_decision,
)
from sq_ai.portfolio.tracker import PortfolioTracker


def _tracker(tmp_path):
    return PortfolioTracker(db_path=str(tmp_path / "t.db"))


def test_parse_decision_uppercases_action():
    d = parse_decision('{"action":"buy","size_pct":5,"confidence":0.7}')
    assert d["action"] == "BUY"
    assert d["size_pct"] == 5.0
    assert d["confidence"] == 0.7


def test_parse_decision_rejects_invalid():
    assert parse_decision(None) is None
    assert parse_decision("nope") is None
    assert parse_decision('{"foo":1}') is None


def test_build_decision_prompt_contains_symbol_and_news():
    p = build_decision_prompt(
        "RELIANCE.NS",
        {"close": 2500, "rsi": 55, "atr": 30, "regime": 2, "ml_proba_up": 0.6},
        [{"source": "ET", "title": "RELIANCE Q3 beats estimates"}],
        {"cash": 1e6, "equity": 1e6, "exposure_pct": 0, "daily_pnl_pct": 0},
    )
    assert "RELIANCE.NS" in p
    assert "Q3 beats" in p
    assert "regime=2" in p


def test_no_veto_when_size_below_threshold(tmp_path):
    veto = EnsembleVeto(tracker=_tracker(tmp_path), threshold_pct=10.0,
                        claude=MagicMock(), deepseek=MagicMock())
    out = veto.maybe_veto("X.NS",
                          {"action": "BUY", "size_pct": 5, "confidence": 0.7},
                          prompt="p")
    assert out["below_threshold"] is True
    assert out["final_action"] == "BUY"
    assert out["vetoed"] is False


def test_veto_executes_when_both_buy(tmp_path):
    deepseek = MagicMock()
    deepseek.generate.return_value = '{"action":"BUY","size_pct":12,"confidence":0.8}'
    veto = EnsembleVeto(tracker=_tracker(tmp_path), threshold_pct=10.0,
                        claude=MagicMock(), deepseek=deepseek)
    out = veto.maybe_veto("X.NS",
                          {"action": "BUY", "size_pct": 12, "confidence": 0.7},
                          prompt="p")
    assert out["below_threshold"] is False
    assert out["final_action"] == "BUY"
    assert out["vetoed"] is False
    assert out["deepseek_action"] == "BUY"


def test_veto_blocks_when_deepseek_disagrees(tmp_path):
    tracker = _tracker(tmp_path)
    deepseek = MagicMock()
    deepseek.generate.return_value = '{"action":"HOLD","size_pct":0,"confidence":0.9}'
    veto = EnsembleVeto(tracker=tracker, threshold_pct=10.0,
                        claude=MagicMock(), deepseek=deepseek)
    out = veto.maybe_veto("X.NS",
                          {"action": "BUY", "size_pct": 12, "confidence": 0.8},
                          prompt="p")
    assert out["vetoed"] is True
    assert out["final_action"] == "HOLD"
    rows = tracker.latest_disagreements()
    assert len(rows) == 1
    assert rows[0]["claude_action"] == "BUY"
    assert rows[0]["deepseek_action"] == "HOLD"
    assert rows[0]["final_action"] == "HOLD"
    assert rows[0]["prompt_hash"]


def test_veto_blocks_when_deepseek_unparseable(tmp_path):
    deepseek = MagicMock()
    deepseek.generate.return_value = "garbage"
    veto = EnsembleVeto(tracker=_tracker(tmp_path), threshold_pct=10.0,
                        claude=MagicMock(), deepseek=deepseek)
    out = veto.maybe_veto("X.NS",
                          {"action": "BUY", "size_pct": 15, "confidence": 0.9},
                          prompt="p")
    assert out["vetoed"] is True
    assert out["final_action"] == "HOLD"
    assert out["deepseek_action"] == "UNKNOWN"
