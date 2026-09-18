"""A trade level the scanner never produced must read as missing, not zero.

The scan record previously coerced price/entry/stop/target through
``float(x or 0.0)``. The desk renders any non-null number, so an absent stop
became "Stop Rs0.00" — a fabricated trade plan presented as a real one.
"""
from __future__ import annotations

from product.scan_store import build_scan_payload, derive_trade_plan


class Bare:
    """A signal the scanner graded but produced no trade levels for."""
    symbol = "BARE"
    score = 55.0
    status = "Watch"


class Planned:
    symbol = "PLAN"
    score = 82.0
    status = "Ready to trade"
    price = 100.0
    entry = 101.0
    stop = 96.0
    target = 121.0


def _row(payload, symbol):
    return next(r for r in payload["records"] if r["symbol"] == symbol)


def test_absent_levels_are_null_not_zero():
    payload = build_scan_payload({"BARE": "Bare Co"}, [Bare()], freshness={})
    row = _row(payload, "BARE")
    for field in ("price", "entry", "stop", "target"):
        assert row[field] is None, f"{field} fabricated as {row[field]!r}"
    assert row["plan_complete"] is False
    assert set(row["plan_missing"]) == {"entry", "stop", "target"}


def test_real_levels_survive_and_derive_correct_geometry():
    payload = build_scan_payload({"PLAN": "Planned Co"}, [Planned()], freshness={})
    row = _row(payload, "PLAN")
    assert row["entry"] == 101.0 and row["stop"] == 96.0 and row["target"] == 121.0
    assert row["risk_per_share"] == 5.0
    assert row["reward_per_share"] == 20.0
    assert row["reward_risk"] == 4.0
    assert row["upside_pct"] == 19.8      # 20/101
    assert row["downside_pct"] == 4.95    # 5/101
    assert row["plan_complete"] is True
    assert row["plan_missing"] == []


def test_partial_plan_names_the_missing_input():
    plan = derive_trade_plan({"entry": 100.0, "stop": 95.0, "target": None, "price": 100.0})
    assert plan["downside_pct"] == 5.0
    assert plan["upside_pct"] is None
    assert plan["reward_risk"] is None
    assert plan["plan_missing"] == ["target"]
    assert plan["plan_complete"] is False


def test_inverted_levels_do_not_produce_negative_geometry():
    """A stop above entry is incoherent; report nothing rather than a negative R."""
    plan = derive_trade_plan({"entry": 100.0, "stop": 110.0, "target": 90.0, "price": 100.0})
    assert plan["risk_per_share"] is None
    assert plan["reward_per_share"] is None
    assert plan["reward_risk"] is None


def test_price_alone_still_anchors_a_partial_plan():
    plan = derive_trade_plan({"entry": None, "stop": 90.0, "target": 120.0, "price": 100.0})
    assert plan["plan_reference_price"] == 100.0
    assert plan["downside_pct"] == 10.0
    assert plan["upside_pct"] == 20.0
    assert plan["reward_risk"] == 2.0
