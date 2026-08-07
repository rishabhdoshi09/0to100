"""Learning bridge: full-universe backtest edge → scan ranking / demotion."""
from __future__ import annotations

from product.pre_trade import CAUTION, GO, NO_GO, build_pre_trade
from product.scan_store import build_scan_payload
from scan.measured_edge import apply_measured_edge


def _actionable_report() -> dict:
    return {
        "generated_at": "2026-08-01T00:00:00+00:00",
        "symbols": 800,
        "universe": {"run": 800, "available": 800, "truncated": False},
        "signals": {
            "MOMENTUM": {"trades": 120, "expectancy_r": 0.18},
            "PRE_BREAKOUT": {"trades": 80, "expectancy_r": -0.12},
            "VOLUME_SPIKE": {"trades": 60, "expectancy_r": 0.02},
        },
    }


def _patch_actionable(monkeypatch, report_fn=_actionable_report):
    import scan.measured_edge as me
    import scan.signal_backtest as sb

    monkeypatch.setattr(sb, "load_report", report_fn)
    monkeypatch.setattr(sb, "report_is_actionable", lambda _r=None: True)
    monkeypatch.setattr(sb, "universe_evidence_note", lambda _r=None: "800 stocks (full-universe backtest)")
    # measured_edge imported these by name — patch module bindings too
    monkeypatch.setattr(me, "load_report", report_fn)
    monkeypatch.setattr(me, "report_is_actionable", lambda _r=None: True)
    monkeypatch.setattr(me, "universe_evidence_note", lambda _r=None: "800 stocks (full-universe backtest)")


def test_apply_measured_edge_tags_and_demotes(monkeypatch):
    _patch_actionable(monkeypatch)

    rows = [
        {"symbol": "WIN", "signals": ["MOMENTUM"], "score": 70, "verdict": "BUY", "reasons": []},
        {
            "symbol": "LOSE",
            "signals": ["PRE_BREAKOUT"],
            "score": 90,
            "verdict": "BUY",
            "reasons": ["breakout"],
        },
        {"symbol": "THIN", "signals": ["VOLUME_SPIKE"], "score": 80, "verdict": "WATCH", "reasons": []},
    ]
    tagged = apply_measured_edge(rows)
    assert tagged == 3
    by_sym = {r["symbol"]: r for r in rows}
    assert by_sym["WIN"]["edge_r"] == 0.18
    assert by_sym["LOSE"]["verdict"] == "WATCH"
    assert by_sym["LOSE"]["edge_r"] == -0.12
    assert rows[0]["symbol"] == "WIN"  # ranked above demoted loser


def test_apply_measured_edge_skips_thin_report(monkeypatch):
    import scan.measured_edge as me
    import scan.signal_backtest as sb

    thin = lambda: {"signals": {}, "universe": {"run": 10}}
    monkeypatch.setattr(sb, "load_report", thin)
    monkeypatch.setattr(sb, "report_is_actionable", lambda _r=None: False)
    monkeypatch.setattr(me, "load_report", thin)
    monkeypatch.setattr(me, "report_is_actionable", lambda _r=None: False)
    rows = [{"symbol": "X", "signals": ["MOMENTUM"], "score": 99, "verdict": "BUY"}]
    assert apply_measured_edge(rows) == 0
    assert rows[0].get("edge_r") is None


def test_scan_store_persists_edge(monkeypatch):
    _patch_actionable(monkeypatch)

    class Sig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    payload = build_scan_payload(
        {"WIN": "Win Co", "LOSE": "Lose Co"},
        [
            Sig(symbol="WIN", signals=["MOMENTUM"], score=70, verdict="BUY",
                price=100, momentum_5d=2, rsi=55, volume_ratio=1.2,
                entry=100, stop=95, target=110, chase_risk=False, reasons=["momo"]),
            Sig(symbol="LOSE", signals=["PRE_BREAKOUT"], score=90, verdict="BUY",
                price=50, momentum_5d=5, rsi=70, volume_ratio=2.0,
                entry=50, stop=48, target=60, chase_risk=False, reasons=["break"]),
        ],
    )
    assert payload["summary"]["with_measured_edge"] == 2
    assert payload["records"][0]["symbol"] == "WIN"
    assert payload["records"][0]["edge_r"] == 0.18
    assert payload["records"][1]["verdict"] == "WATCH"


def test_pre_trade_blocks_negative_edge(monkeypatch):
    _patch_actionable(monkeypatch)
    body = build_pre_trade(
        symbol="LOSE",
        plan={
            "available": True,
            "tradeable": True,
            "heat_verdict": "OK",
            "market_health": "healthy",
            "market_risk_factor": 1.0,
            "correlation_status": "independent",
            "summary": "ok",
        },
        scan_record={"symbol": "LOSE", "signals": ["PRE_BREAKOUT"], "edge_r": -0.15, "verdict": "WATCH"},
        readiness={"lanes": [], "retail_research_checklist": {"gaps": []}},
    )
    assert body["verdict"] == NO_GO
    assert body["measured_edge_r"] == -0.15
    assert any("loser" in b.lower() or "negative" in b.lower() or "-0.15" in b for b in body["blockers"])


def test_pre_trade_go_without_scan_stays_go(monkeypatch):
    import scan.signal_backtest as sb

    monkeypatch.setattr(sb, "load_report", lambda: None)
    monkeypatch.setattr(sb, "report_is_actionable", lambda _r=None: False)

    body = build_pre_trade(
        symbol="ACME",
        plan={
            "available": True,
            "tradeable": True,
            "heat_verdict": "OK",
            "market_health": "healthy",
            "market_risk_factor": 1.0,
            "correlation_status": "independent",
            "summary": "ok",
        },
        readiness={"lanes": [], "retail_research_checklist": {"gaps": []}},
    )
    assert body["verdict"] == GO
    assert body["learning"]["signal_backtest_actionable"] is False


def test_pre_trade_warns_stale_scan_when_report_actionable(monkeypatch):
    _patch_actionable(monkeypatch)

    body = build_pre_trade(
        symbol="ACME",
        plan={
            "available": True,
            "tradeable": True,
            "heat_verdict": "OK",
            "market_health": "healthy",
            "market_risk_factor": 1.0,
            "correlation_status": "independent",
            "summary": "ok",
        },
        scan_record={"symbol": "ACME", "signals": ["MOMENTUM"], "verdict": "BUY"},
        readiness={"lanes": [], "retail_research_checklist": {"gaps": []}},
    )
    assert body["verdict"] == CAUTION
    assert any("edge_r" in w.lower() or "rescan" in w.lower() for w in body["warnings"])
