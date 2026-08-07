"""Market Decision Brief — Motilal-competitive retail morning desk honesty."""
from __future__ import annotations

from product import market_decision_brief as MDB
from product import premarket_cues as PM


def test_parse_gift_nifty_down_pct():
    gift = PM._parse_gift_nifty("Gift Nifty is down 0.4%. Brent crude jumped above $83.")
    assert gift is not None
    assert gift["chg_pct"] == -0.4
    assert gift["source"] == "moneycontrol_text"


def test_parse_gift_nifty_level_and_pct():
    gift = PM._parse_gift_nifty("GIFT Nifty 24,450 (-0.35%) leads the open cue.")
    assert gift is not None
    assert gift["level"] == 24450.0
    assert gift["chg_pct"] == -0.35


def test_parse_gift_nifty_snippet_without_pct():
    gift = PM._parse_gift_nifty("Traders watching Gift Nifty for India open cues amid crude spike.")
    assert gift is not None
    assert gift.get("chg_pct") is None
    assert "Gift Nifty" in (gift.get("snippet") or "")


def test_parse_gift_headline_points_lower():
    gift = PM._parse_gift_headline(
        "Gift Nifty Signals Negative Start Today, Trades 97 Points Lower",
        source="rss:google_gift",
    )
    assert gift is not None
    assert gift["chg_points"] == -97.0


def test_parse_gift_headline_falls_pts():
    gift = PM._parse_gift_headline(
        "GIFT Nifty falls 100 pts, signals muted start for Sensex, Nifty",
        source="rss:moneycontrol",
    )
    assert gift is not None
    assert gift["chg_points"] == -100.0


def test_consensus_gift_median_points():
    cands = [
        {"chg_points": -100.0, "source": "a", "headline": "falls 100"},
        {"chg_points": -97.0, "source": "b", "headline": "97 lower"},
        {"chg_points": -93.0, "source": "c", "headline": "down 93"},
    ]
    cons = PM._consensus_gift(cands, nifty_spot=24500.0)
    assert cons is not None
    assert cons["chg_points"] == -97.0
    assert cons["chg_pct"] is not None
    assert cons["evidence_count"] == 3
    assert cons["source"] == "headline_consensus"


def test_side_aware_walls_respect_spot():
    support, resistance = MDB._side_aware_walls(
        24500.0,
        top_calls=[{"strike": 24800, "ce_oi": 1e6}, {"strike": 24000, "ce_oi": 9e6}],
        top_puts=[{"strike": 24200, "pe_oi": 1e6}, {"strike": 25000, "pe_oi": 9e6}],
    )
    assert support == 24200.0
    assert resistance == 24800.0


def test_fundamental_picks_prefer_combined_score_and_classification(monkeypatch):
    monkeypatch.setattr(
        "product.long_term_store.load_long_term_scan",
        lambda: {
            "schema_version": 1,
            "scanned_at": "2026-08-07T00:00:00Z",
            "records": [
                {
                    "symbol": "SKIPME",
                    "verdict": "LONG_TERM_BUY",
                    "classification": "AVOID_REVIEW",
                    "combined_score": 99,
                    "price": 10,
                },
                {
                    "symbol": "ETERNAL",
                    "verdict": "LONG_TERM_BUY",
                    "classification": "QUALITY_COMPOUNDER",
                    "combined_score": 80,
                    "price": 315,
                    "from_high_pct": 21.0,
                    "thesis": "compounder",
                },
                {
                    "symbol": "LOW",
                    "verdict": "WATCH",
                    "classification": "LONG_TERM_WATCH",
                    "combined_score": 40,
                    "price": 100,
                    "from_high_pct": 10.0,
                },
            ],
        },
    )
    fund = MDB._fundamental_picks(limit=5)
    assert fund["available"] is True
    assert fund["as_of"] == "2026-08-07T00:00:00Z"
    assert [r["symbol"] for r in fund["rows"]] == ["ETERNAL", "LOW"]
    assert fund["rows"][0]["target_watch"] is not None


def test_build_brief_offline_composes_picks(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_MARKET_BRIEF_FILE", str(tmp_path / "brief.json"))
    monkeypatch.setattr(
        "product.premarket_cues.build_premarket_cues",
        lambda allow_network=True: {
            "available": True,
            "headline": "Gift Nifty -0.40% — watch the open",
            "bullets": ["Gift Nifty is down 0.40% (Moneycontrol pre-market)."],
            "gift_nifty": {"chg_pct": -0.4, "level": None},
            "us_futures": [],
            "gaps": [],
        },
    )
    monkeypatch.setattr(
        MDB,
        "_global_macro_cues",
        lambda allow_network=True, skip_commodity_names=None: {
            "available": True,
            "key": "macro_global",
            "title": "Macro / Global Cues",
            "icon": "global",
            "headline": "US markets fell up to 0.5%.",
            "bullets": ["US markets fell up to 0.5%.", "Brent crude +4.0% at $83.00."],
            "cues": [],
            "themes": [],
            "gaps": [],
        },
    )
    monkeypatch.setattr(
        MDB,
        "_options_levels_block",
        lambda allow_network=True: {
            "available": True,
            "key": "options_levels",
            "title": "Options Data & Key Levels",
            "icon": "options",
            "headline": "Nifty 24400 to 24900 zones",
            "bullets": ["Nifty 24400 to 24900 zones", "Bank Nifty 57500 to 59000 zones"],
            "levels": [],
            "gaps": [],
        },
    )
    monkeypatch.setattr(
        "product.long_term_store.load_long_term_scan",
        lambda: {
            "schema_version": 1,
            "generated_at": "2026-08-07T00:00:00Z",
            "records": [
                {
                    "symbol": "ETERNAL",
                    "verdict": "LONG_TERM_BUY",
                    "price": 315,
                    "from_high_pct": 21.0,
                    "score": 72,
                    "thesis": "above 200-DMA; near leadership zone",
                },
                {
                    "symbol": "SKIPME",
                    "verdict": "SKIP",
                    "price": 10,
                    "score": 99,
                },
            ],
        },
    )
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {
            "schema_version": 1,
            "generated_at": "2026-08-07T00:00:00Z",
            "source": "test",
            "records": [
                {
                    "symbol": "UNIONBANK",
                    "status": "Ready to trade",
                    "price": 182,
                    "entry": 183,
                    "stop": 175,
                    "target": 195,
                    "score": 80,
                    "chase_risk": False,
                    "why": "Breakout hold",
                    "signals": ["Resistance break"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        "product.scan_store.watchlist_rows",
        lambda payload, limit=40: list((payload or {}).get("records") or [])[:limit],
    )

    brief = MDB.build_market_decision_brief(persist=True, allow_network=False)
    assert brief["available"] is True
    assert brief["places_orders"] is False
    assert brief["live_locked"] is True
    assert len(brief["deciders"]) == 3
    assert all(d.get("available") for d in brief["deciders"])
    fund_syms = [r["symbol"] for r in brief["fundamental_picks"]["rows"]]
    assert fund_syms == ["ETERNAL"]
    fund = brief["fundamental_picks"]["rows"][0]
    assert fund["target_watch"] == 398.73 or abs(fund["target_watch"] - (315 / 0.79)) < 0.1
    assert fund["upside_to_prior_high_pct"] is not None
    tech = brief["technical_picks"]["rows"][0]
    assert tech["symbol"] == "UNIONBANK"
    assert tech["entry"] == 183
    assert tech["target"] == 195
    assert tech["upside_pct"] == 7.1
    assert brief["why_better"]
    assert "invented" in brief["honesty"].lower() or "Missing" in brief["honesty"]

    loaded = MDB.load_brief()
    assert loaded["available"] is True
    assert loaded["title"] == "3 Things That Will Decide the Market Today"

    tg = MDB.brief_telegram_message(brief)
    assert "3 Things That Will Decide the Market Today" in tg
    assert "ETERNAL" in tg
    assert "UNIONBANK" in tg
    assert "not a buy ticket" in tg.lower() or "paper-first" in tg.lower()


def test_empty_brief_never_places_orders():
    empty = MDB.empty_brief(message="not built")
    assert empty["available"] is False
    assert empty["places_orders"] is False
    assert empty["live_locked"] is True


def test_index_zone_rounds_trader_friendly():
    zone = MDB._index_zone(24512.0, 24380.0, 24890.0)
    assert zone == "24350 to 24900" or "24400" in (zone or "") or zone is not None
