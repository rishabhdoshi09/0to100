"""
Tests for the risk-first Trade Plan projection (deterministic, network-free).

Proves the "R lens" composes the authoritative sizing + book-risk + correlation functions honestly:
exact shares/₹-risk/R, invalidation, before/after open-risk, correlation as new-vs-adding-to a bet,
regime throttle — and degrades honestly (invalid stop / tiny capital never fabricate a size).
"""
from __future__ import annotations

from product import trade_plan as TP


def _fake_sizer(entry, stop, capital, risk_pct):
    per = entry - stop
    if per <= 0 or capital <= 0:
        return {"qty": 0, "invested": 0.0, "max_loss": 0.0, "capped": False}
    qty = int((capital * risk_pct) / per)
    if qty < 1:
        return {"qty": 0, "invested": 0.0, "max_loss": 0.0, "capped": False}
    return {"qty": qty, "invested": qty * entry, "max_loss": qty * per,
            "risk_pct_used": risk_pct, "capped": False}


def _reporter(before_pct, after_pct, verdict="OK", warnings=()):
    def report(extra):
        return {"open_risk_pct": after_pct if extra else before_pct,
                "verdict": verdict if extra else "OK", "warnings": list(warnings) if extra else []}
    return report


# ── core sizing / R / % of capital ───────────────────────────────────────────────
def test_exact_shares_rupee_risk_and_r():
    # entry 100, stop 95 → ₹5/share risk; 1% of ₹1,00,000 = ₹1,000 → 200 shares, ₹1,000 risk
    p = TP.build_trade_plan("ACME", 100, 95, 115, capital=100_000, risk_pct=0.01, sizer=_fake_sizer)
    assert p.tradeable and p.qty == 200
    assert p.rupee_risk == 1000 and p.risk_pct_of_capital == 1.0
    assert p.invested == 20000 and p.pct_of_capital == 20.0
    assert p.reward_risk == 3.0                      # (115-100)/(100-95)
    assert p.invalidation_pct == 5.0                 # stop is 5% below entry


def test_missing_target_gives_no_reward_risk():
    p = TP.build_trade_plan("ACME", 100, 95, None, capital=100_000, sizer=_fake_sizer)
    assert p.tradeable and p.reward_risk is None and "reward:risk unknown" in p.summary.lower()


# ── honest untradeable states (never fabricate a size) ───────────────────────────
def test_invalid_stop_is_not_tradeable():
    p = TP.build_trade_plan("ACME", 100, 105, 120, capital=100_000, sizer=_fake_sizer)
    assert p.tradeable is False and p.qty == 0 and "stop must be below entry" in p.reason


def test_capital_too_small_is_not_tradeable():
    # 0.01% of ₹100 with ₹5 risk/share → 0 shares
    p = TP.build_trade_plan("ACME", 100, 95, 120, capital=100, risk_pct=0.0001, sizer=_fake_sizer)
    assert p.tradeable is False and p.qty == 0 and "too small" in p.reason


# ── book heat before vs after (authoritative reporter, composed) ─────────────────
def test_open_risk_before_after_and_danger_verdict():
    rep = _reporter(3.2, 5.4, verdict="DANGER", warnings=["open risk above 5%"])
    p = TP.build_trade_plan("ACME", 100, 95, 120, capital=100_000, sizer=_fake_sizer,
                            portfolio_report=rep)
    assert p.open_risk_pct_before == 3.2 and p.open_risk_pct_after == 5.4
    assert p.heat_verdict == "DANGER" and "close something first" in p.summary
    assert p.heat_warnings == ("open risk above 5%",)


# ── correlation: new bet vs adding to an existing one ────────────────────────────
def test_correlation_adds_to_existing_bet():
    p = TP.build_trade_plan("HDFCBANK", 100, 95, 120, capital=100_000, sizer=_fake_sizer,
                            correlated_with=["ICICIBANK", "AXISBANK"])
    assert p.correlation_status == TP.ADDS_TO_BET
    assert p.correlated_with == ("ICICIBANK", "AXISBANK")
    assert "not a new bet" in p.summary.lower()


def test_correlation_new_independent_bet():
    p = TP.build_trade_plan("SUNPHARMA", 100, 95, 120, capital=100_000, sizer=_fake_sizer,
                            correlated_with=[])
    assert p.correlation_status == TP.NEW_BET and "new, independent bet" in p.summary.lower()


def test_correlation_unknown_without_history():
    p = TP.build_trade_plan("ACME", 100, 95, 120, capital=100_000, sizer=_fake_sizer,
                            correlated_with=None)
    assert p.correlation_status == TP.UNKNOWN and p.correlated_with == ()


# ── regime throttle scales the risk budget and says so ───────────────────────────
def test_regime_throttle_halves_risk_and_explains():
    full = TP.build_trade_plan("ACME", 100, 95, 120, capital=100_000, risk_pct=0.01,
                               regime_factor=1.0, sizer=_fake_sizer)
    weak = TP.build_trade_plan("ACME", 100, 95, 120, capital=100_000, risk_pct=0.01,
                               regime_factor=0.5, sizer=_fake_sizer)
    assert weak.suggested_risk_pct == 0.005 and weak.qty == full.qty // 2
    assert "throttled" in weak.summary.lower()


# ── composes the REAL sizer (single source of truth, no re-derivation) ───────────
def test_uses_canonical_sizer_by_default():
    import inspect
    src = inspect.getsource(TP)
    assert "from risk.position_sizer import size_position" in src   # composes, never re-derives
    assert "open_risk" not in src.replace("open_risk_pct", "")      # no re-derived open-risk math
    # default path calls the canonical size_position
    p = TP.build_trade_plan("ACME", 250, 240, 300, capital=100_000, risk_pct=0.01)
    assert p.qty >= 1 and p.rupee_risk > 0
