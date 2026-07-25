"""
🌍 Macro Pulse — the news radar the technical stack was blind to.

A breakout scanner reads price. It does NOT read the reason a market gaps
down 2% at open: tariffs, a crude spike, a rate decision, a geopolitical
shock. Lately the tape has been macro-DRIVEN — so a purely technical read
gets blindsided. This module reads the news stream the system already pulls
(news.fetcher — RSS + Marketaux) and extracts the market-MOVING themes, with
direction, so the Brain can lean careful when the tape is news-driven.

HONEST framing (evidence-over-vibes): this is a keyword CONTEXT RADAR, not a
predictive signal. It answers "kya tape aaj news ke haath mein hai, aur kis
taraf?" — a weather report, not a buy/sell call. It NEVER places or blocks a
trade on its own; it feeds the Brain's posture as demote-only caution (same
role as breadth NARROW). Corroboration-gated: one headline is noise — a theme
counts only when several fresh articles agree.

Pure + testable: detect_macro_themes / macro_pulse take a list of article
dicts ({headline, summary}) and return a structured read. No I/O here.
"""
from __future__ import annotations

import os as _os

# Direction cues — which way a driver is moving. Kept deliberately small and
# high-signal; the theme table below decides what that motion MEANS for Indian
# equities (crude UP hurts, crude DOWN helps, etc.).
# NB: deliberately excludes equity-move words ("rally", "gains") — those refer
# to STOCKS and pollute commodity/currency themes (e.g. "OMCs rally" in a
# crude-FALLING story). Motion here is about the DRIVER, not the market.
_MOTION_UP = ("surge", "surges", "surging", "spike", "spikes", "jump", "jumps",
              "soar", "soars", "rise", "rises", "rising", "rose", "climb",
              "climbs", "jumped", "record high", "hits high", "all-time high",
              "higher", "hike", "hikes", "raises")
_MOTION_DOWN = ("fall", "falls", "falling", "fell", "drop", "drops", "plunge",
                "plunges", "slump", "slumps", "tumble", "tumbles", "ease",
                "eases", "easing", "cool", "cools", "cooling", "cut", "cuts",
                "record low", "hits low", "declines", "lower", "slips", "slide")

# ── Theme table — the macro drivers that actually move NSE ────────────────────
# up_is: what an UPWARD move of this driver means for equities (bearish|bullish).
# base:  polarity when no clear motion word is present (the usual newsworthy
#        direction). sectors: the obvious first-order impact (context tags).
_THEMES = {
    "TARIFF": {
        "kw": ("tariff", "trade war", "import duty", "sanction", "export ban",
               "trade deal", "customs duty"),
        "up_is": "bearish", "base": "bearish", "de_escalation": ("deal", "truce",
        "agreement", "resolved", "rollback"),
        "label": "Tariffs / trade war",
        "sectors_hit": ["IT", "Pharma", "Auto", "Metals"],
    },
    "CRUDE": {
        "kw": ("crude", "brent", "oil price", "opec", "wti", "fuel price"),
        "up_is": "bearish", "base": "bearish",
        "label": "Crude / oil",
        "sectors_hit": ["OMC (BPCL/IOC/HPCL)", "Paints", "Aviation", "Tyres"],
        "sectors_help": ["Upstream (ONGC/OIL)"],
    },
    "GEOPOLITICS": {
        "kw": ("war", "attack", "airstrike", "conflict", "geopolit",
               "middle east", "israel", "iran", "russia", "ukraine", "missile",
               "tensions"),
        "up_is": "bearish", "base": "bearish",
        "label": "Geopolitics / war",
        "sectors_hit": ["broad market (risk-off)"],
        "sectors_help": ["Defence", "Gold"],
    },
    "RATES": {
        "kw": ("rate hike", "rate cut", "repo rate", "fomc", "fed ", "rbi ",
               "powell", "interest rate", "monetary policy", "basis points",
               "bps"),
        "up_is": "bearish", "base": "neutral",   # hike=bearish, cut=bullish
        "label": "Rates / central bank",
        "sectors_hit": ["Banks", "NBFC", "Realty", "Auto"],
    },
    "INFLATION": {
        "kw": ("inflation", "cpi ", "wpi ", "retail inflation", "price rise"),
        "up_is": "bearish", "base": "bearish",
        "label": "Inflation",
        "sectors_hit": ["FMCG", "Consumer", "Rate-sensitives"],
    },
    "RUPEE": {
        "kw": ("rupee", "usd/inr", "dollar index", "currency"),
        "up_is": "bullish", "base": "bearish",   # rupee UP = strong = mild+;
        "label": "Rupee / currency",             # base bearish (usually news = fall)
        "sectors_hit": ["Importers", "Oil", "Aviation"],
        "sectors_help": ["IT", "Pharma (exporters)"],
    },
    "FLOWS": {
        "kw": ("fii sell", "fpi outflow", "foreign investors sold",
               "fii outflow", "fpi sell", "foreign selling", "fii buy",
               "fpi inflow"),
        "up_is": "bullish", "base": "bearish",
        "label": "FII / FPI flows",
        "sectors_hit": ["broad market"],
    },
    "MARKET": {
        "kw": ("sensex", "nifty", "market crash", "bloodbath", "selloff",
               "sell-off", "meltdown", "market rally", "d-street"),
        "up_is": "bullish", "base": "neutral",
        "label": "Index move",
        "sectors_hit": ["broad market"],
    },
}

_MIN_ARTICLES = int(_os.getenv("QT_MACRO_MIN_ARTICLES", "2") or 2)


def _motion(text: str) -> str:
    """'up' | 'down' | '' — which way the driver is described as moving."""
    up = any(w in text for w in _MOTION_UP)
    down = any(w in text for w in _MOTION_DOWN)
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return ""                       # ambiguous / both → let the theme's base decide


def _polarity(theme: dict, text: str) -> str:
    """bearish | bullish | neutral — what this theme in this article means for
    equities. Motion + the theme's up_is decide; de-escalation words soften a
    risk-off theme; base is the fallback when there's no clear motion."""
    # de-escalation of a risk-off theme (a trade DEAL, a truce) → not bearish
    for w in theme.get("de_escalation", ()):
        if w in text:
            return "bullish"
    m = _motion(text)
    if m == "up":
        return theme["up_is"]
    if m == "down":
        return "bullish" if theme["up_is"] == "bearish" else "bearish"
    return theme["base"]


def detect_macro_themes(articles: list[dict]) -> list[dict]:
    """Per-theme detection across the news stream. Corroboration-gated: a theme
    is returned only when ≥ _MIN_ARTICLES fresh articles mention it (one
    headline is noise, not a market driver). Each theme carries its net
    direction, the corroborating count, a sample headline, and the sectors it
    first-order hits/helps — pure context, no trade call."""
    hits: dict[str, dict] = {}
    for a in articles or []:
        text = f"{a.get('headline', '')} {a.get('summary', '')}".lower()
        if not text.strip():
            continue
        for name, theme in _THEMES.items():
            if not any(k in text for k in theme["kw"]):
                continue
            pol = _polarity(theme, text)
            h = hits.setdefault(name, {"name": name, "label": theme["label"],
                                       "count": 0, "bearish": 0, "bullish": 0,
                                       "sample": a.get("headline", ""),
                                       "sectors_hit": theme.get("sectors_hit", []),
                                       "sectors_help": theme.get("sectors_help", [])})
            h["count"] += 1
            if pol == "bearish":
                h["bearish"] += 1
            elif pol == "bullish":
                h["bullish"] += 1
    out = []
    for h in hits.values():
        if h["count"] < _MIN_ARTICLES:
            continue                        # not corroborated → drop as noise
        h["direction"] = ("bearish" if h["bearish"] > h["bullish"]
                          else "bullish" if h["bullish"] > h["bearish"]
                          else "mixed")
        out.append(h)
    # strongest (most-corroborated) first
    return sorted(out, key=lambda x: x["count"], reverse=True)


def macro_pulse(articles: list[dict]) -> dict:
    """One market-mood read from the news stream.

    Returns {mood, heat, risk_off, themes, note}:
      mood    — RISK_OFF | CAUTIOUS | NEUTRAL | RISK_ON
      heat    — 0-100, how much the tape is news-DRIVEN right now
      risk_off— True when the Brain should lean careful (demote-only caution)
      themes  — the corroborated drivers (from detect_macro_themes)
      note    — plain-English one-liner for the Pulse / Brain hero."""
    themes = detect_macro_themes(articles)
    if not themes:
        return {"mood": "NEUTRAL", "heat": 0, "risk_off": False,
                "themes": [], "note": "News-driven macro risk abhi shaant."}
    bear = sum(t["count"] for t in themes if t["direction"] == "bearish")
    bull = sum(t["count"] for t in themes if t["direction"] == "bullish")
    total = sum(t["count"] for t in themes)
    heat = min(100, total * 12)             # ~8 corroborating articles → pegged
    net = bear - bull
    if net >= 3 and heat >= 40:
        mood = "RISK_OFF"
    elif net >= 1:
        mood = "CAUTIOUS"
    elif net <= -3 and heat >= 40:
        mood = "RISK_ON"
    else:
        mood = "NEUTRAL"
    risk_off = mood in ("RISK_OFF", "CAUTIOUS")
    lead = themes[0]
    dir_word = {"bearish": "🔴 dabaav", "bullish": "🟢 support",
                "mixed": "⚪ mixed"}[lead["direction"]]
    note = (f"{lead['label']} {dir_word} ({lead['count']} khabrein)"
            + (f" +{len(themes) - 1} aur theme" if len(themes) > 1 else ""))
    return {"mood": mood, "heat": heat, "risk_off": risk_off,
            "themes": themes, "note": note}
