"""scan/conviction.py: a bare news-headline substring match (no sentiment,
no entity-linking -- see _fetch_buzz_map) must never by itself be what
crosses a marginal technical setup from WATCH into BUY/STRONG BUY. This is
CLAUDE.md's "evidence over vibes" invariant, applied to the live 15-min
scan's conviction layer (scan/auto_scan.py calls build_conviction on every
cycle).
"""
from __future__ import annotations

import re

import scan.conviction as conv


def _row(symbol: str, *, score: float, signals=None, reasons=None,
         volume_ratio: float = 1.0, chase_risk: bool = False, verdict: str = "WATCH") -> dict:
    return {
        "symbol": symbol, "score": score, "verdict": verdict,
        "signals": signals or [], "reasons": reasons or [],
        "volume_ratio": volume_ratio, "chase_risk": chase_risk,
    }


def _no_earnings(symbol: str) -> dict:
    return {"growth_pct": None, "days_to_results": None}


def test_buzz_alone_cannot_flip_a_marginal_setup_to_buy(monkeypatch):
    monkeypatch.setattr(conv, "_fetch_buzz_map", lambda symbols: {"XYZ": "XYZ wins big contract"})
    monkeypatch.setattr(conv, "_earnings_info", _no_earnings)
    # One real technical check (n_pos=1 without buzz) + a borderline score
    # that only clears 55 once the +8 buzz bonus is added.
    row = _row("XYZ", score=48.0, signals=["Breakout"], reasons=["Cleared resistance"])

    out = conv.build_conviction([row])

    assert out[0]["buzzing"] is True, "buzz should still be recorded/shown"
    assert any("Buzzing" in c for c in out[0]["checks"]), "buzz should still appear in the checklist"
    assert out[0]["conviction"] == 56.0, "the displayed number still reflects the buzz bonus, honestly"
    assert out[0]["verdict"] == "WATCH", (
        "conviction only clears 55 BECAUSE of the buzz bonus (48 base -> 56) -- "
        "a bare headline match must never be the deciding factor for BUY"
    )


def test_real_technical_evidence_alone_still_reaches_buy(monkeypatch):
    monkeypatch.setattr(conv, "_fetch_buzz_map", lambda symbols: {})
    monkeypatch.setattr(conv, "_earnings_info", _no_earnings)
    # Two independent, real technical checks and a score that already clears
    # 55 without any buzz/earnings bonus at all.
    row = _row("ABC", score=60.0, signals=["Breakout"], reasons=["Cleared resistance"], verdict="BUY",
               volume_ratio=2.0)

    out = conv.build_conviction([row])

    assert out[0]["verdict"] == "BUY"
    assert out[0].get("buzzing") is not True


def test_buzz_still_boosts_display_score_for_an_already_qualifying_setup(monkeypatch):
    monkeypatch.setattr(conv, "_fetch_buzz_map", lambda symbols: {"ABC": "ABC beats estimates"})
    monkeypatch.setattr(conv, "_earnings_info", _no_earnings)
    row = _row("ABC", score=60.0, signals=["Breakout"], reasons=["Cleared resistance"], verdict="BUY",
               volume_ratio=2.0)

    out = conv.build_conviction([row])

    # Real evidence alone (score 60 + volume check) already clears BUY --
    # buzz here is not the deciding factor, so it's free to still show and
    # to nudge the displayed number/sort order.
    assert out[0]["verdict"] == "BUY"
    assert out[0]["buzzing"] is True
    assert out[0]["conviction"] == 60.0 + 4.0 + 8.0


def test_earnings_growth_is_real_evidence_and_still_counts_toward_the_gate(monkeypatch):
    monkeypatch.setattr(conv, "_fetch_buzz_map", lambda symbols: {})
    monkeypatch.setattr(conv, "_earnings_info", lambda symbol: {"growth_pct": 20.0, "days_to_results": None})
    # One technical check + earnings growth (a real, backtestable-in-principle
    # fact, not a vibe) should be able to reach BUY on their own.
    row = _row("DEF", score=50.0, signals=["Breakout"], reasons=["Cleared resistance"], verdict="BUY")

    out = conv.build_conviction([row])

    assert out[0]["verdict"] == "BUY"


def test_chase_risk_stays_watch_even_with_buzz(monkeypatch):
    monkeypatch.setattr(conv, "_fetch_buzz_map", lambda symbols: {"XYZ": "XYZ rallies"})
    monkeypatch.setattr(conv, "_earnings_info", _no_earnings)
    row = _row("XYZ", score=80.0, signals=["Breakout", "Momentum"],
               reasons=["don't chase", "Cleared resistance", "RSI strong"],
               volume_ratio=2.0, chase_risk=True)

    out = conv.build_conviction([row])

    assert out[0]["verdict"] == "WATCH"


def test_conviction_can_never_promote_a_row_the_evidence_gate_held_at_watch(monkeypatch):
    """2026-11-20 review: build_conviction was a legacy BUY path that bypassed the evidence gate."""
    monkeypatch.setattr(conv, "_fetch_buzz_map", lambda symbols: {})
    monkeypatch.setattr(conv, "_earnings_info", lambda symbol: {"growth_pct": 40.0, "days_to_results": 6})
    row = _row("GATED", score=95.0, signals=["Breakout", "Momentum"], reasons=["Cleared resistance"],
               volume_ratio=3.0, verdict="WATCH")
    out = conv.build_conviction([row])
    assert out[0]["verdict"] == "WATCH" and out[0]["conviction"] > 0


def test_no_canonical_module_reaches_the_legacy_conviction_path():
    """Only the archived auto_scan/legacy_app/ui paths may call build_conviction."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    allowed = {"scan/conviction.py", "scan/auto_scan.py", "legacy_app.py"}
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("tests/", "ui/", ".claude/", "venv/", "node_modules/")) or rel in allowed:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"\bbuild_conviction\s*\(", text) or "scan.conviction" in text \
                or "from scan import conviction" in text:
            offenders.append(rel)
    assert offenders == [], offenders


def test_the_canonical_conviction_shortlist_inherits_the_gate_instead_of_bypassing_it():
    """product.conviction classifies; HIGH_CONVICTION needs a gate-certified 'Ready to trade' row."""
    from product.conviction import build_conviction_shortlist
    rows = [{"symbol": "GATED", "verdict": "WATCH", "status": "Watch", "score": 99, "volume_ratio": 3.0,
             "rsi": 60, "edge_gated": True},
            {"symbol": "CERT", "verdict": "BUY", "status": "Ready to trade", "score": 99, "volume_ratio": 3.0,
             "rsi": 60}]
    from types import SimpleNamespace
    view = SimpleNamespace(health="Healthy", leaders=(), laggards=())
    out = {r["symbol"]: r for r in build_conviction_shortlist({"records": rows}, view, sector_lookup=lambda s: "")}
    assert out["GATED"]["classification"] != "HIGH_CONVICTION"
    assert out["CERT"]["classification"] == "HIGH_CONVICTION"
