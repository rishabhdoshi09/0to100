"""Breakout sniper re-armed via autonomy (product scan records)."""
from __future__ import annotations

from product.scan_store import build_scan_payload
from scan.breakout_sniper import build_watch_map
from research.autonomy.sniper_bridge import ensure_breakout_sniper, records_from_payload


class _Sig:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def categories(self):
        return {"PreBreakout"} if "PRE_BREAKOUT" in self.signals else set()


def test_scan_store_preserves_sniper_fields():
    sig = _Sig(
        symbol="KAYNES", price=100, momentum_5d=2, rsi=55, volume_ratio=1.5,
        signals=["PRE_BREAKOUT"], reasons=["near pivot"], score=70,
        entry=101, stop=95, target=110, verdict="WATCH", chase_risk=False,
        pivot_distance_pct=1.2, avg_vol20=500000,
        breakout_grade="A", breakout_conviction=72.0,
    )
    payload = build_scan_payload({"KAYNES": "Kaynes"}, [sig])
    row = payload["records"][0]
    assert "PreBreakout" in row["categories"]
    assert row["pivot_distance_pct"] == 1.2
    assert row["avg_vol20"] == 500000
    assert row["breakout_grade"] == "A"
    assert row["breakout_conviction"] == 72.0


def test_build_watch_map_accepts_product_records(monkeypatch):
    monkeypatch.setattr(
        "scan.breakout_sniper.InstrumentManager",
        type("IM", (), {"tokens_for": staticmethod(lambda syms: {s: i + 1 for i, s in enumerate(syms)})}),
        raising=False,
    )
    # Patch where used
    import scan.breakout_sniper as BS

    class FakeIM:
        def tokens_for(self, syms):
            return {s: 1000 + i for i, s in enumerate(syms)}

    monkeypatch.setattr(BS, "InstrumentManager", FakeIM, raising=False)
    # InstrumentManager is imported inside function — patch data.instruments
    import data.instruments as INS
    monkeypatch.setattr(INS, "InstrumentManager", FakeIM)

    rows = [{
        "symbol": "KAYNES",
        "signals": ["PRE_BREAKOUT"],
        "status": "Watch for breakout",
        "categories": ["PreBreakout"],
        "pivot_distance_pct": 1.0,
        "entry": 100, "stop": 90, "target": 120,
        "avg_vol20": 1_000_000, "rsi": 50, "chase_risk": False,
    }]
    watch = build_watch_map(rows)
    assert len(watch) == 1
    tok = next(iter(watch))
    assert watch[tok]["symbol"] == "KAYNES"
    assert watch[tok]["trigger"] == 100


def test_ensure_sniper_kite_unavailable(monkeypatch):
    monkeypatch.setattr(
        "research.autonomy.sniper_bridge.SCH",
        type("S", (), {"market_is_open": staticmethod(lambda *a, **k: True)}),
        raising=False,
    )
    # Force import path inside ensure
    import research.autonomy.sniper_bridge as SB

    def _fake_open(*a, **k):
        return True

    monkeypatch.setattr(
        "research.autonomy.schedules.market_is_open", _fake_open
    )
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {"records": [{
            "symbol": "X", "signals": ["PRE_BREAKOUT"], "status": "Watch for breakout",
            "categories": ["PreBreakout"], "pivot_distance_pct": 1.0,
            "entry": 10, "stop": 9, "target": 12, "avg_vol20": 1000, "rsi": 40,
            "chase_risk": False,
        }]},
    )
    monkeypatch.setattr("scan.breakout_sniper.start_sniper", lambda: False)
    out = ensure_breakout_sniper()
    assert out["ok"] is False
    assert out["error"] == "kite_unavailable"


def test_records_from_payload():
    assert records_from_payload(None) == []
    assert records_from_payload({"records": [{"symbol": "A"}]}) == [{"symbol": "A"}]
