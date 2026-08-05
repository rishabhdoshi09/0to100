"""Desk composition reuses existing stores without inventing data."""
from __future__ import annotations

from pathlib import Path

import product.desk_composition as DC


def test_stock_desk_tape_composes_soft_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_LOW_POWER", "1")
    monkeypatch.setattr(DC, "ROOT", tmp_path)
    monkeypatch.setattr(DC, "PULSE_PATH", tmp_path / "missing_pulse.json")

    monkeypatch.setattr(
        DC,
        "_evidence_completeness",
        lambda symbol: {
            "available": True,
            "score_pct": 40,
            "fresh": 2,
            "attached": 2,
            "total": 7,
            "missing": ["Management commentary"],
            "note": "40%",
        },
    )
    monkeypatch.setattr(
        DC,
        "_buy_context",
        lambda symbol: {"in_buy_book": False, "in_holdings": False, "health": None},
    )
    monkeypatch.setattr(
        DC,
        "_flow_context",
        lambda symbol: {
            "available": True,
            "bias": "FII_SELLING",
            "note": "test",
            "latest_fii_net_cr": -1200,
            "latest_dii_net_cr": 800,
            "as_of": "2026-08-01",
            "bulk_deals": [],
        },
    )

    tape = DC.build_stock_desk_tape(
        "TCS",
        workspace={
            "state": "TECHNICAL_ONLY",
            "confidence_pct": 55,
            "scanner": {"status": "MOMENTUM", "score": 82, "signals": ["MOMENTUM"]},
            "technical": {"volume_ratio": 1.4, "atr_pct": 2.1, "from_high_pct": -4.0},
            "news": [{"title": "TCS wins deal", "published_at": "2026-08-01", "impact_score": 70}],
            "growth_outlook": {"thesis": {"label": "INCOMPLETE", "text": "Need guidance evidence"}},
            "peers": {"peer_rank": 3, "total_peers": 12, "average_pe": 28},
        },
    )
    assert tape["symbol"] == "TCS"
    assert tape["places_orders"] is False
    assert tape["info_score_pct"] >= 40
    assert any("FII/DII" in b or "FII" in b for b in tape["bullets"])
    assert tape["evidence"]["score_pct"] == 40
    assert tape["scan"]["relative_strength_proxy"] == 82


def test_market_command_attaches_to_home(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_LOW_POWER", "1")
    monkeypatch.setattr(DC, "ROOT", tmp_path)
    monkeypatch.setattr(DC, "PULSE_PATH", tmp_path / "pulse.json")
    (tmp_path / "pulse.json").write_text(
        '{"generated_at":"2026-08-01T00:00:00+00:00","takeaways":["Breadth soft","IT leading"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        DC,
        "_brain_posture_light",
        lambda: {
            "available": True,
            "mode": "low_power_light",
            "posture": "DEFENSIVE",
            "posture_reason": "test",
            "verdict_line": "Careful day",
            "regime": "CHOPPY",
            "flows_bias": "MIXED",
            "options_bias": "",
            "breadth": "Narrow",
        },
    )
    monkeypatch.setattr(
        DC,
        "_flow_context",
        lambda symbol: {
            "available": True,
            "bias": "MIXED",
            "note": "ok",
            "latest_fii_net_cr": -10,
            "latest_dii_net_cr": 20,
            "as_of": "2026-08-01",
            "bulk_deals": [],
        },
    )
    import product.buy_health as BH

    monkeypatch.setattr(
        BH,
        "evaluate_book",
        lambda: {"summary": {"critical": 1, "warn": 2}, "items": []},
    )

    home = DC.build_market_command(
        home={"market_health": "Cautious", "counts": {"breakouts": 3, "momentum": 5, "long_term_picks": 2}}
    )
    assert home["command"]["available"] is True
    assert home["command"]["posture"] == "DEFENSIVE"
    assert "Breadth soft" in home["command"]["takeaways"]
    assert home["command"]["active_buy_critical"] == 1


def test_watchlist_briefing_badges(tmp_path, monkeypatch):
    monkeypatch.setattr(DC, "ROOT", tmp_path)
    monkeypatch.setattr(DC, "PULSE_PATH", tmp_path / "pulse.json")
    (tmp_path / "pulse.json").write_text(
        '{"relative_strength":[{"symbol":"INFY"}],"breakouts_today":[{"symbol":"TCS"}]}',
        encoding="utf-8",
    )
    class FakeStore:
        def recent(self, **kwargs):
            return [{"title": "x"}, {"title": "y"}]

    monkeypatch.setattr("news.curator_store.NewsCuratorStore", lambda *a, **k: FakeStore())
    monkeypatch.setattr("product.buy_book.load_book", lambda: {"items": []})
    monkeypatch.setattr("product.sniper_board.load_board", lambda: {"hits": [{"symbol": "TCS"}]})

    briefing = DC.build_watchlist_briefing(
        [
            {"id": 1, "symbol": "TCS", "added_date": "2026-08-01", "snapshot": {"setup_label": "Breakout"}},
            {"id": 2, "symbol": "INFY", "added_date": "2026-08-01", "snapshot": {"status": "WATCH"}},
        ]
    )
    assert briefing["count"] == 2
    tcs = next(r for r in briefing["items"] if r["symbol"] == "TCS")
    assert tcs["briefing"]["sniper_hit"] is True
    assert tcs["briefing"]["in_pulse_movers"] is True
    assert any(b.startswith("SETUP:") for b in tcs["briefing"]["badges"])


def test_build_stock_workspace_includes_desk_tape(monkeypatch):
    from product import stock_workspace as SW

    monkeypatch.setattr(
        SW,
        "_default_inputs",
        lambda symbol: {
            "scan": {"records": [], "scanned_at": ""},
            "long_term": {"records": [], "scanned_at": ""},
            "raw": {"available": False, "data": {}},
            "frame": None,
            "news": [],
            "fno": {"underlyings": [], "generated_at": ""},
        },
    )
    monkeypatch.setattr(
        "product.desk_composition.build_stock_desk_tape",
        lambda symbol, workspace=None: {
            "symbol": symbol,
            "info_score_pct": 12,
            "bullets": ["Desk state DATA INCOMPLETE · data confidence 0%"],
            "places_orders": False,
        },
    )
    ws = SW.build_stock_workspace("DEMO")
    assert "desk_tape" in ws
    assert ws["desk_tape"]["info_score_pct"] == 12
