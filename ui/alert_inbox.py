"""Alert inbox — ML-scored stream of technical breakouts and news triggers."""
from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import NamedTuple

import streamlit as st

try:
    import feedparser  # type: ignore
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://feeds.feedburner.com/ndtvprofit-latest",
]

KEYWORDS_BUY  = {"breakout", "rally", "surge", "upgrade", "beat", "strong", "bullish", "high"}
KEYWORDS_SELL = {"crash", "fall", "downgrade", "miss", "weak", "bearish", "low", "cut", "below"}


class Alert(NamedTuple):
    timestamp: str
    category:  str       # TECHNICAL | NEWS | VOLUME | PATTERN
    symbol:    str
    message:   str
    score:     float     # 0-100 relevance


def _relevance(title: str, watchlist: set[str]) -> float:
    title_l = title.lower()
    words   = set(title_l.split())
    sym_hit = 20.0 if any(s.lower() in title_l for s in watchlist) else 0.0
    buy_hit = len(words & KEYWORDS_BUY)  * 10.0
    sel_hit = len(words & KEYWORDS_SELL) * 10.0
    return min(100.0, sym_hit + buy_hit + sel_hit)


@st.cache_data(ttl=180)
def _fetch_news_alerts(watchlist_tuple: tuple) -> list[Alert]:
    if not HAS_FEEDPARSER:
        return []
    wl    = set(watchlist_tuple)
    items: list[Alert] = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                score = _relevance(title, wl)
                if score < 5:
                    continue
                ts = datetime.now().strftime("%H:%M:%S")
                # Detect mentioned symbol
                sym = next((s for s in wl if s.lower() in title.lower()), "MARKET")
                items.append(Alert(ts, "NEWS", sym, title[:120], score))
        except Exception:
            continue
    items.sort(key=lambda a: -a.score)
    return items[:20]


def _session_alerts() -> list[Alert]:
    """Return manually injected technical alerts stored in session state."""
    return st.session_state.get("devbloom_alerts", [])


def push_alert(category: str, symbol: str, message: str, score: float = 50.0):
    """Push a programmatic alert (called by anomaly scanner, screener, etc.)"""
    if "devbloom_alerts" not in st.session_state:
        st.session_state["devbloom_alerts"] = []
    st.session_state["devbloom_alerts"].insert(
        0,
        Alert(datetime.now().strftime("%H:%M:%S"), category, symbol, message, score),
    )
    # Keep only last 50
    st.session_state["devbloom_alerts"] = st.session_state["devbloom_alerts"][:50]


def render_alert_inbox(watchlist: list[str] | None = None):
    wl = tuple(watchlist or ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"])

    news_alerts = _fetch_news_alerts(wl)
    tech_alerts = _session_alerts()
    all_alerts  = sorted(tech_alerts + news_alerts, key=lambda a: -a.score)

    badge_colors = {
        "TECHNICAL": ("rgba(0,212,255,.15)", "#00d4ff"),
        "NEWS":      ("rgba(255,184,0,.12)", "#ffb800"),
        "VOLUME":    ("rgba(0,255,136,.12)", "#00ff88"),
        "PATTERN":   ("rgba(160,80,255,.12)", "#a050ff"),
    }

    if not all_alerts:
        st.markdown(
            "<div style='color:#8892a4;font-size:.85rem;text-align:center;padding:1.5rem'>No alerts yet — they appear here as signals trigger.</div>",
            unsafe_allow_html=True,
        )
        return

    filter_cat = st.selectbox(
        "Filter", ["ALL", "TECHNICAL", "NEWS", "VOLUME", "PATTERN"],
        key="alert_filter", label_visibility="collapsed",
    )

    for a in all_alerts:
        if filter_cat != "ALL" and a.category != filter_cat:
            continue
        bg, fg = badge_colors.get(a.category, ("rgba(255,255,255,.06)", "#e8eaf0"))
        score_bar = int(a.score)
        st.markdown(
            f"<div style='background:{bg};border:1px solid {fg}33;border-radius:10px;"
            f"padding:.6rem .9rem;margin-bottom:.4rem;font-size:.82rem'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"  <span style='color:{fg};font-weight:600;font-size:.7rem'>{a.category} · {a.symbol}</span>"
            f"  <span style='color:#8892a4;font-size:.68rem'>{a.timestamp} · {a.score:.0f}/100</span>"
            f"</div>"
            f"<div style='color:#c8cfe0;margin-top:.25rem'>{a.message}</div>"
            f"<div style='margin-top:.35rem;height:3px;border-radius:2px;"
            f"background:linear-gradient(to right,{fg} {score_bar}%,rgba(255,255,255,0.06) {score_bar}%)'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
