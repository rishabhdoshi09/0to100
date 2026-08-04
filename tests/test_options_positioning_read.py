"""Options positioning read is research context — never a buy ticket."""
from __future__ import annotations

from options.positioning_read import SUPPORTIVE, HOSTILE, INCOMPLETE, build_positioning_read


def _chain(**overrides):
    base = {
        "available": True,
        "symbol": "NIFTY",
        "expiry": "04-Aug-2026",
        "pcr": 1.35,
        "max_pain": 24600,
        "spot": 24400,
        "atm_iv": 14.0,
        "iv_rank": 40.0,
        "top_call_oi": [{"strike": 24800, "ce_oi": 500000}],
        "top_put_oi": [{"strike": 24200, "pe_oi": 600000}],
        "chain": [
            {"strike": 24300, "ce_coi": -1000, "pe_coi": 5000, "ce_oi": 1, "pe_oi": 1},
            {"strike": 24400, "ce_coi": -500, "pe_coi": 4000, "ce_oi": 1, "pe_oi": 1},
            {"strike": 24500, "ce_coi": 200, "pe_coi": 3000, "ce_oi": 1, "pe_oi": 1},
        ],
        "bias": "BULLISH",
    }
    base.update(overrides)
    return base


def test_supportive_read_is_not_a_buy_label():
    read = build_positioning_read(_chain())
    assert read["available"] is True
    assert read["stance"] == SUPPORTIVE
    assert read["signal_desk"] is False
    assert read["places_orders"] is False
    assert "buy" not in str(read["headline"]).lower() or "research" in str(read["headline"]).lower()
    assert "tomorrow_watch" in read["consider_for"]
    assert "Not a buy/sell signal" in read["honesty"]


def test_hostile_when_calls_dominate_and_spot_above_pain():
    read = build_positioning_read(
        _chain(
            pcr=0.55,
            spot=25000,
            max_pain=24500,
            top_call_oi=[{"strike": 25100, "ce_oi": 900000}],
            top_put_oi=[{"strike": 24000, "pe_oi": 100000}],
            chain=[{"strike": 25000, "ce_coi": 8000, "pe_coi": -2000}],
        )
    )
    assert read["stance"] == HOSTILE
    assert "avoid_chase" in read["consider_for"]


def test_incomplete_without_chain():
    read = build_positioning_read({"available": False, "message": "down"})
    assert read["stance"] == INCOMPLETE
    assert read["available"] is False


def test_cash_scan_overlay_can_align():
    read = build_positioning_read(
        _chain(symbol="RELIANCE"),
        scan_row={
            "symbol": "RELIANCE",
            "verdict": "BUY",
            "score": 78,
            "edge_r": 0.12,
            "chase_risk": False,
        },
    )
    assert read["cash_scan_joined"] is True
    assert read["stance"] in {SUPPORTIVE, "CAUTION", "NEUTRAL"}
    assert any("Cash scan" in r or "Measured edge" in r for r in read["reasons"])


def test_chase_risk_blocks_watch_tag():
    read = build_positioning_read(
        _chain(pcr=1.4),
        scan_row={"verdict": "BUY", "score": 80, "edge_r": 0.1, "chase_risk": True},
    )
    assert "avoid_chase" in read["consider_for"] or any("chase" in r.lower() for r in read["risks"])


def test_options_workspace_attaches_positioning_read(monkeypatch):
    from product import market_api as MA
    from options.positioning_read import attach_positioning_read

    fake = _chain()
    monkeypatch.setattr(
        "options.chain_fetch.chain_workspace_cached",
        lambda symbol, spot=None, force=False: dict(fake),
    )
    monkeypatch.setattr("options.eod_store.history", lambda symbol, days=14: [])
    monkeypatch.setattr(MA, "_scan_row_for", lambda symbol: None)
    payload = MA.options_workspace("NIFTY")
    assert "positioning_read" in payload
    assert payload["positioning_read"]["stance"] == SUPPORTIVE
    assert payload["positioning_read"]["signal_desk"] is False
    # attach helper keeps honesty at top-level note
    attached = attach_positioning_read(fake)
    assert attached["bias"] == "BULLISH"
