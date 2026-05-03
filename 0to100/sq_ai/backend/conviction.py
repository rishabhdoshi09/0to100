"""Conviction scoring engine — 5-pillar BUY / SELL / HOLD verdict.

Pillars & weights
-----------------
  Technical    30 %  — trend alignment, momentum, volume, 52 W position
  Fundamental  25 %  — quality, profitability, balance sheet health
  Smart Money  20 %  — FII / DII / promoter ownership signals
  Momentum     15 %  — multi-period returns, volume trend
  Valuation    10 %  — P/E, P/B positioning

Final score 0-100:
  ≥ 68 → STRONG BUY | 55-67 → BUY | 45-54 → HOLD |
  35-44 → SELL      | < 35  → STRONG SELL
"""
from __future__ import annotations

import math
from typing import Any

from sq_ai.backend.cache import cached
from sq_ai.backend.data_fetcher import fetch_yf_history
from sq_ai.backend.financials import get_ratios
from sq_ai.backend.shareholding import get_shareholding
from sq_ai.backend.screener_engine import _technical_features
from sq_ai.signals.composite_signal import adx


def _tag(tags: list[str], label: str) -> None:
    tags.append(label)


# ─────────────────────────────────────────────────────────────────────────────
# Pillar scorers — each returns (score 0-100, signal_tags list[str])
# ─────────────────────────────────────────────────────────────────────────────

def _technical_score(feat: dict[str, Any], adx_val: float) -> tuple[float, list[str]]:
    s = 0.0
    tags: list[str] = []
    price = feat.get("price", 0)

    if price and feat.get("sma_20") and price > feat["sma_20"]:
        s += 12
        _tag(tags, "above_sma20")
    if price and feat.get("sma_50") and price > feat["sma_50"]:
        s += 12
        _tag(tags, "above_sma50")
    sma200 = feat.get("sma_200")
    if price and sma200 and not math.isnan(sma200) and price > sma200:
        s += 14
        _tag(tags, "above_sma200")

    rsi_v = feat.get("rsi", 50)
    if 45 <= rsi_v <= 65:
        s += 14
        _tag(tags, "rsi_healthy")
    elif 35 <= rsi_v < 45:
        s += 6
    elif 65 < rsi_v <= 75:
        s += 8
        _tag(tags, "rsi_momentum")
    elif rsi_v > 75:
        s -= 8
        _tag(tags, "rsi_overbought")
    elif rsi_v < 30:
        _tag(tags, "rsi_oversold")

    if adx_val >= 40:
        s += 20
        _tag(tags, "strong_trend")
    elif adx_val >= 25:
        s += 14
        _tag(tags, "trending")
    elif adx_val >= 15:
        s += 5

    if feat.get("macd_state") == "bullish":
        s += 10
        _tag(tags, "macd_bullish")

    vol_ratio = feat.get("vol_ratio", 1.0)
    if vol_ratio >= 2.0:
        s += 14
        _tag(tags, "volume_surge_2x")
    elif vol_ratio >= 1.5:
        s += 9
        _tag(tags, "volume_above_avg")
    elif vol_ratio >= 1.2:
        s += 4

    from_52h = feat.get("from_52w_high_pct", -1.0)
    if from_52h >= -0.05:
        s += 14
        _tag(tags, "near_52w_high")
    elif from_52h >= -0.15:
        s += 9
        _tag(tags, "within_15pct_high")
    elif from_52h >= -0.30:
        s += 4

    return min(max(s, 0), 100), tags


def _fundamental_score(ratios: dict[str, Any]) -> tuple[float, list[str]]:
    if not ratios:
        return 50.0, ["no_fundamental_data"]
    s = 0.0
    tags: list[str] = []

    roe = ratios.get("roe") or 0
    if isinstance(roe, float) and roe < 1:
        roe *= 100
    if roe >= 25:
        s += 28
        _tag(tags, "roe_excellent")
    elif roe >= 18:
        s += 22
        _tag(tags, "roe_strong")
    elif roe >= 12:
        s += 14
        _tag(tags, "roe_decent")
    elif roe >= 5:
        s += 6
    elif roe < 0:
        _tag(tags, "roe_weak")

    de = ratios.get("debt_to_equity") or 0
    if de == 0:
        s += 22
        _tag(tags, "debt_free")
    elif de < 0.3:
        s += 18
        _tag(tags, "low_debt")
    elif de < 0.7:
        s += 12
        _tag(tags, "manageable_debt")
    elif de < 1.5:
        s += 5
    else:
        _tag(tags, "high_debt")

    pe = ratios.get("pe") or 0
    if 0 < pe < 12:
        s += 18
        _tag(tags, "cheap_pe")
    elif 12 <= pe < 20:
        s += 14
        _tag(tags, "fair_pe")
    elif 20 <= pe < 30:
        s += 8
    elif 30 <= pe < 45:
        s += 3
    elif pe >= 45:
        _tag(tags, "expensive")

    mc = ratios.get("market_cap") or 0
    if mc >= 50_000_000_000:
        s += 10
        _tag(tags, "large_cap")
    elif mc >= 10_000_000_000:
        s += 6
        _tag(tags, "mid_cap")
    elif mc > 0:
        s += 2
        _tag(tags, "small_cap")

    dy = ratios.get("dividend_yield") or 0
    if dy >= 0.02:
        s += 6
        _tag(tags, "dividend_payer")
    elif dy >= 0.01:
        s += 3

    if (ratios.get("eps") or 0) > 0:
        s += 6

    return min(max(s, 0), 100), tags


def _smart_money_score(sh: dict[str, Any]) -> tuple[float, list[str]]:
    if not sh:
        return 50.0, ["no_ownership_data"]
    cur = sh.get("current") or {}
    tags: list[str] = []
    s = 0.0

    fii = cur.get("fii") or 0
    if fii >= 25:
        s += 35
        _tag(tags, "high_fii_holding")
    elif fii >= 15:
        s += 26
        _tag(tags, "solid_fii_holding")
    elif fii >= 8:
        s += 16
        _tag(tags, "moderate_fii")
    elif fii >= 3:
        s += 8
    else:
        _tag(tags, "low_fii")

    dii = cur.get("dii") or 0
    if dii >= 15:
        s += 25
        _tag(tags, "high_dii_holding")
    elif dii >= 8:
        s += 18
        _tag(tags, "solid_dii_holding")
    elif dii >= 4:
        s += 10
    elif dii >= 1:
        s += 4

    promoter = cur.get("promoter") or 0
    if promoter >= 65:
        s += 25
        _tag(tags, "high_promoter")
    elif promoter >= 50:
        s += 20
        _tag(tags, "majority_promoter")
    elif promoter >= 35:
        s += 12
    elif promoter >= 20:
        s += 6
    else:
        _tag(tags, "low_promoter")

    hist = sh.get("history") or []
    if len(hist) >= 2:
        fii_old = hist[-2].get("fii") or 0
        fii_now = hist[-1].get("fii") or 0
        if fii_now > fii_old + 0.5:
            s += 15
            _tag(tags, "fii_accumulating")
        elif fii_now < fii_old - 0.5:
            s -= 10
            _tag(tags, "fii_selling")

    return min(max(s, 0), 100), tags


def _momentum_score(feat: dict[str, Any]) -> tuple[float, list[str]]:
    tags: list[str] = []
    s = 0.0

    r1w = feat.get("ret_1w", 0)
    if r1w >= 0.05:
        s += 20
        _tag(tags, "1w_strong")
    elif r1w >= 0.02:
        s += 12
        _tag(tags, "1w_positive")
    elif r1w >= 0:
        s += 5
    else:
        s -= 5
        _tag(tags, "1w_negative")

    r1m = feat.get("ret_1m", 0)
    if r1m >= 0.12:
        s += 28
        _tag(tags, "1m_strong")
    elif r1m >= 0.06:
        s += 20
        _tag(tags, "1m_positive")
    elif r1m >= 0.02:
        s += 10
    elif r1m < -0.08:
        s -= 10
        _tag(tags, "1m_negative")

    r3m = feat.get("ret_3m", 0)
    if r3m >= 0.25:
        s += 30
        _tag(tags, "3m_strong")
    elif r3m >= 0.12:
        s += 22
        _tag(tags, "3m_positive")
    elif r3m >= 0.04:
        s += 10
    elif r3m < -0.15:
        s -= 12
        _tag(tags, "3m_negative")

    vol_ratio = feat.get("vol_ratio", 1.0)
    if vol_ratio >= 1.5:
        s += 12
        _tag(tags, "vol_confirmation")
    elif vol_ratio >= 1.2:
        s += 6

    if feat.get("from_52w_high_pct", -1.0) >= -0.08:
        s += 10
        _tag(tags, "52w_high_breakout_zone")

    return min(max(s, 0), 100), tags


def _valuation_score(ratios: dict[str, Any]) -> tuple[float, list[str]]:
    if not ratios:
        return 50.0, []
    tags: list[str] = []
    s = 0.0

    pe = ratios.get("pe") or 0
    if 0 < pe < 10:
        s += 45
        _tag(tags, "deep_value")
    elif 10 <= pe < 18:
        s += 36
        _tag(tags, "value")
    elif 18 <= pe < 28:
        s += 24
        _tag(tags, "fair_value")
    elif 28 <= pe < 40:
        s += 12
    elif pe >= 40:
        s += 4
        _tag(tags, "growth_premium")

    pb = ratios.get("pb") or 0
    if 0 < pb < 1.5:
        s += 30
        _tag(tags, "below_book")
    elif 1.5 <= pb < 3:
        s += 22
        _tag(tags, "reasonable_pb")
    elif 3 <= pb < 6:
        s += 12
    elif pb >= 6:
        s += 4

    return min(max(s, 0), 100), tags


# ─────────────────────────────────────────────────────────────────────────────
# Claude narrative (optional — graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

@cached("conviction_narrative", ttl_seconds=21600)
def _claude_narrative(
    symbol: str, verdict: str, conviction: int,
    tech_tags: str, funda_tags: str, smart_tags: str, mom_tags: str,
) -> list[str]:
    try:
        from sq_ai.backend.llm_clients import ClaudeClient
        client = ClaudeClient()
        if not client.available:
            return []
        prompt = (
            f"Stock: {symbol} | Verdict: {verdict} | Score: {conviction}/100\n"
            f"Technical signals: {tech_tags}\n"
            f"Fundamental signals: {funda_tags}\n"
            f"Smart money signals: {smart_tags}\n"
            f"Momentum signals: {mom_tags}\n\n"
            "Write exactly 3 short bullet points (each ≤ 15 words) explaining "
            "the key reasons for this verdict. Start each with '• '. No other text."
        )
        raw = client.generate(prompt, max_tokens=120, temperature=0.1)
        if not raw:
            return []
        return [line.strip("• ").strip()
                for line in raw.strip().splitlines()
                if line.strip().startswith("•")][:3]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _verdict(score: float) -> str:
    if score >= 68:
        return "STRONG BUY"
    if score >= 55:
        return "BUY"
    if score >= 45:
        return "HOLD"
    if score >= 35:
        return "SELL"
    return "STRONG SELL"


@cached("conviction", ttl_seconds=1800)
def get_conviction(symbol: str) -> dict[str, Any]:
    """Compute the 5-pillar conviction score for *symbol*."""
    df = fetch_yf_history(symbol, period="1y", interval="1d")
    if df is None or len(df) < 60:
        return {"symbol": symbol, "error": "insufficient price history"}

    feat = _technical_features(df)

    try:
        adx_val = float(adx(df, 14).iloc[-1])
    except Exception:
        adx_val = 0.0

    ratios = get_ratios(symbol) or {}
    sh = get_shareholding(symbol) or {}

    tech_sc, tech_tags = _technical_score(feat, adx_val)
    funda_sc, funda_tags = _fundamental_score(ratios)
    smart_sc, smart_tags = _smart_money_score(sh)
    mom_sc, mom_tags = _momentum_score(feat)
    val_sc, val_tags = _valuation_score(ratios)

    final = round(
        0.30 * tech_sc
        + 0.25 * funda_sc
        + 0.20 * smart_sc
        + 0.15 * mom_sc
        + 0.10 * val_sc, 1
    )
    verdict = _verdict(final)

    price = feat["price"]
    atr_v = feat["atr"]
    stop = round(price - 2 * atr_v, 2)
    target = round(price + 3 * atr_v, 2)
    rr = round((target - price) / (price - stop), 2) if price - stop > 0 else 0.0

    bullets = _claude_narrative(
        symbol, verdict, int(final),
        ",".join(tech_tags[:4]), ",".join(funda_tags[:4]),
        ",".join(smart_tags[:3]), ",".join(mom_tags[:3]),
    )

    return {
        "symbol": symbol,
        "verdict": verdict,
        "conviction": final,
        "breakdown": {
            "technical":   {"score": round(tech_sc),  "weight": 30, "signals": tech_tags},
            "fundamental": {"score": round(funda_sc), "weight": 25, "signals": funda_tags},
            "smart_money": {"score": round(smart_sc), "weight": 20, "signals": smart_tags},
            "momentum":    {"score": round(mom_sc),   "weight": 15, "signals": mom_tags},
            "valuation":   {"score": round(val_sc),   "weight": 10, "signals": val_tags},
        },
        "price": price,
        "stop": stop,
        "target": target,
        "risk_reward": rr,
        "adx": round(adx_val, 1),
        "from_52w_high_pct": round(feat.get("from_52w_high_pct", 0) * 100, 1),
        "vol_ratio": round(feat.get("vol_ratio", 1), 2),
        "claude_bullets": bullets,
        "name": ratios.get("name"),
        "sector": ratios.get("sector"),
        "market_cap": ratios.get("market_cap"),
    }


__all__ = ["get_conviction"]
