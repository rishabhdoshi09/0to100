"""
Marketaux News Feed — Streamlit panel.
Rendered as a sub-tab inside the 🤖 Agents top-level tab.
"""
from __future__ import annotations

import streamlit as st

_SENTIMENT_COLOR = {
    "positive": "#00d4a0",
    "negative": "#ff4b4b",
    "neutral":  "#8892a4",
}
_SENTIMENT_EMOJI = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral":  "⚪",
}


@st.cache_data(ttl=900, show_spinner=False)   # 15-minute cache
def _cached_fetch(symbols_key: str, limit: int, days_back: int) -> list:
    from news.marketaux_news import MarketauxNews
    client = MarketauxNews()
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()]
    return client.fetch_for_symbols(symbols, limit=limit, days_back=days_back)


def render_news_feed() -> None:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.4rem;margin:0'>📰 Marketaux News Feed</h2>"
        "<p style='color:#8892a4;font-size:.8rem;margin:.2rem 0 1rem'>"
        "AI-tagged, sentiment-scored stock news · 15-min cache · 1 API call per refresh</p>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        raw_tickers = st.text_input(
            "Tickers (comma-separated)",
            value="RELIANCE,INFY,TCS,HDFCBANK",
            key="mx_tickers",
            label_visibility="collapsed",
            placeholder="RELIANCE,INFY,TCS …",
        )
    with col2:
        limit = st.selectbox("Articles", [5, 10, 20, 50], index=1, key="mx_limit")
    with col3:
        days_back = st.selectbox("Days", [1, 3, 7, 14], index=1, key="mx_days")
    with col4:
        sentiment_filter = st.selectbox(
            "Sentiment",
            ["All", "positive", "negative", "neutral"],
            key="mx_sentiment",
        )

    fetch_btn = st.button("🔄 Fetch News", key="mx_fetch", type="primary")
    if not fetch_btn and "mx_articles" not in st.session_state:
        st.info(
            "Enter ticker symbols above and click **Fetch News**.\n\n"
            "Results are cached for 15 minutes to conserve API quota (100 req/day free tier)."
        )
        return

    if fetch_btn:
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        if not tickers:
            st.error("Enter at least one ticker.")
            return
        symbols_key = ",".join(tickers)
        try:
            with st.spinner(f"Fetching news for {symbols_key} from Marketaux…"):
                articles = _cached_fetch(symbols_key, limit, days_back)
            st.session_state["mx_articles"] = articles
            st.session_state["mx_tickers_used"] = tickers
        except Exception as exc:
            st.error(f"Marketaux API error: {exc}")
            if "MARKETAUX_API_KEY" in str(exc) or "api_token" in str(exc).lower():
                st.code("Add MARKETAUX_API_KEY=your_key to .env", language="bash")
            return

    articles = st.session_state.get("mx_articles", [])
    tickers_used = st.session_state.get("mx_tickers_used", [])

    if not articles:
        st.warning("No articles returned. Try a wider date range or different tickers.")
        return

    # ── Filter by sentiment ───────────────────────────────────────────────────
    sf = sentiment_filter if sentiment_filter != "All" else None
    displayed = [
        a for a in articles
        if sf is None or (a.get("sentiment_label") or "neutral").lower() == sf
    ]

    # ── Stats strip ───────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Articles", len(displayed))
    pos = sum(1 for a in displayed if (a.get("sentiment_label") or "") == "positive")
    neg = sum(1 for a in displayed if (a.get("sentiment_label") or "") == "negative")
    s2.metric("🟢 Positive", pos)
    s3.metric("🔴 Negative", neg)
    s4.metric("⚪ Neutral", len(displayed) - pos - neg)

    st.divider()

    # ── Per-ticker sections ───────────────────────────────────────────────────
    grouped: dict[str, list] = {}
    for a in displayed:
        for sym in (a.get("tickers") or tickers_used):
            sym = sym.upper()
            if sym in [t.upper() for t in tickers_used]:
                grouped.setdefault(sym, []).append(a)
                break
        else:
            grouped.setdefault("OTHER", []).append(a)

    if not grouped:
        for a in displayed:
            grouped.setdefault("GENERAL", []).append(a)

    for sym, sym_articles in grouped.items():
        with st.expander(f"📌 {sym}  ({len(sym_articles)} articles)", expanded=True):
            for a in sym_articles:
                label = (a.get("sentiment_label") or "neutral").lower()
                color = _SENTIMENT_COLOR.get(label, "#8892a4")
                emoji = _SENTIMENT_EMOJI.get(label, "⚪")
                score = a.get("sentiment_score")
                score_str = f" ({score:+.2f})" if score is not None else ""
                pub = (a.get("published_at") or "")[:10]

                st.markdown(
                    f"<div style='background:rgba(255,255,255,.03);"
                    f"border-left:3px solid {color};"
                    f"border-radius:0 6px 6px 0;padding:.6rem .9rem;margin:.3rem 0'>"
                    f"<span style='color:{color};font-size:.7rem;font-weight:700'>"
                    f"{emoji} {label.upper()}{score_str}</span> "
                    f"<span style='color:#8892a4;font-size:.65rem'>{pub} · {a.get('source','')}</span><br>"
                    f"<a href='{a.get('url','')}' target='_blank' "
                    f"style='color:#e8eaf0;font-size:.88rem;font-weight:600;"
                    f"text-decoration:none'>{a.get('title','')}</a><br>"
                    f"<span style='color:#8892a4;font-size:.8rem;line-height:1.5'>"
                    f"{(a.get('description') or '')[:220]}…</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
