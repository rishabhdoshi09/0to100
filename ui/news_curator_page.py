"""Retail-first high-coverage market-news curator page."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from news.curator_service import get_news_curator_service

_IST = ZoneInfo("Asia/Kolkata")


def _published_label(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(_IST).strftime("%d %b %Y · %I:%M %p IST")
    except Exception:
        return value


def _direction_label(value: str) -> str:
    return {
        "likely_positive": "Potentially positive",
        "likely_negative": "Potentially negative",
        "mixed": "Mixed",
        "unclear": "Direction unclear",
    }.get(value, value.replace("_", " ").title())


def _render_article(article) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f"### {article.headline}")
            badges = [
                article.category.replace("_", " ").title(),
                article.event_type.replace("_", " ").title(),
                _direction_label(article.direction),
                f"Impact {article.impact_score}/100",
            ]
            if article.official:
                badges.insert(0, "Official source")
            if article.fno_symbols:
                badges.append("F&O linked")
            st.caption(" · ".join(badges))
        with right:
            if article.url:
                st.link_button("Open source", article.url, width="stretch")
        st.write(article.summary or "No publisher summary was supplied.")
        st.info(f"**Why it matters:** {article.why_it_matters}")
        details = [article.source, _published_label(article.published_at)]
        if article.corroboration_count > 1:
            details.append(f"Seen across {article.corroboration_count} sources")
        st.caption(" · ".join(details))
        tags = []
        if article.mentioned_symbols:
            tags.append("Stocks: " + ", ".join(article.mentioned_symbols[:12]))
        if article.fno_symbols:
            tags.append("F&O: " + ", ".join(article.fno_symbols[:12]))
        if article.sectors:
            tags.append("Sectors: " + ", ".join(article.sectors))
        if tags:
            st.caption(" | ".join(tags))


def _render_list(rows, empty_message: str, limit: int) -> None:
    if not rows:
        st.info(empty_message)
        return
    for article in rows[:limit]:
        _render_article(article)
    if len(rows) > limit:
        st.caption(f"Showing {limit} of {len(rows)} matching stories. Increase 'Stories to show' above.")


def render_news_curator_page() -> None:
    service = get_news_curator_service()
    curator = service.curator

    st.title("Market News")
    st.caption(
        "Economy, regulation, companies, sectors and F&O-impacting news in one place. "
        "QuantTerm removes repeated copies, maps stories to NSE stocks, and explains why they may matter."
    )
    st.warning(
        "News is context—not a trade signal. A headline never bypasses price, evidence, liquidity or safety checks."
    )

    status_col, refresh_col = st.columns([5, 1])
    with status_col:
        status = service.status()
        if status["last_error"]:
            st.error(f"Latest refresh problem: {status['last_error']}")
        elif status["last_refresh_at"]:
            st.caption(f"Automatic curator running · Last refresh: {_published_label(status['last_refresh_at'])}")
        else:
            st.caption("Automatic curator started. First refresh is in progress.")
    with refresh_col:
        if st.button("Refresh now", type="primary", width="stretch"):
            from research.autonomy.controls import request_control, REFRESH_NEWS_NOW
            request_control(REFRESH_NEWS_NOW, reason="owner requested news refresh")
            st.success("News refresh queued for the autonomy supervisor.")

    stats = curator.stats(hours=24)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stories · 24h", stats["total"])
    c2.metric("Important now", stats["important"])
    c3.metric("F&O linked", stats["fno_linked"])
    c4.metric("Economy / policy", stats["macro"])
    c5.metric("Sources contributing", stats["sources"])

    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        hours_label = f1.selectbox(
            "Time window",
            ["6 hours", "24 hours", "3 days", "7 days", "30 days"],
            index=1,
        )
        hours = {"6 hours": 6, "24 hours": 24, "3 days": 72, "7 days": 168, "30 days": 720}[hours_label]
        min_impact = f2.slider("Minimum impact", 0, 90, 0, 5)
        fno_only = f3.checkbox("F&O-linked only")
        stories_to_show = f4.selectbox("Stories to show", [50, 100, 250, 500, 1000], index=2)
        q1, q2, q3 = st.columns(3)
        search = q1.text_input("Search headline, summary or source")
        symbol = q2.text_input("NSE symbol", placeholder="RELIANCE")
        category = q3.selectbox(
            "Category",
            ["All", "Company", "Market", "Economy", "Regulation", "Derivatives", "Global"],
        )

    filters = {
        "hours": hours,
        "limit": 2000,
        "min_impact": min_impact,
        "fno_only": fno_only,
        "symbol": symbol.strip().upper() or None,
        "search": search.strip() or None,
        "category": None if category == "All" else category.lower(),
    }
    rows = curator.latest(**filters)

    tabs = st.tabs([
        "Important Now",
        "Stocks & F&O",
        "Economy & Policy",
        "Sectors",
        "All News",
        "Source Health",
    ])
    with tabs[0]:
        important = [row for row in rows if row.impact_score >= 70]
        _render_list(important, "No story currently crosses the important-news threshold.", stories_to_show)
    with tabs[1]:
        stocks = [row for row in rows if row.mentioned_symbols or row.fno_symbols]
        _render_list(stocks, "No stock-linked stories match these filters.", stories_to_show)
    with tabs[2]:
        macro = [row for row in rows if row.category in {"economy", "regulation", "global"}]
        _render_list(macro, "No economy or policy stories match these filters.", stories_to_show)
    with tabs[3]:
        sector_rows = [row for row in rows if row.sectors]
        _render_list(sector_rows, "No sector-tagged stories match these filters.", stories_to_show)
    with tabs[4]:
        _render_list(rows, "No stories match these filters.", stories_to_show)
    with tabs[5]:
        health = curator.source_health()
        if not health:
            st.info("Source health will appear after the first refresh.")
        else:
            frame = pd.DataFrame([
                {
                    "Source": item.source_name,
                    "Status": item.status,
                    "Articles": item.article_count,
                    "Latency ms": item.latency_ms,
                    "Last checked": _published_label(item.fetched_at),
                    "Problem": item.error,
                }
                for item in health
            ])
            st.dataframe(frame, hide_index=True, width="stretch")
            working = sum(1 for item in health if item.status == "OK")
            st.caption(f"Working sources: {working} of {len(health)}. A failed source never hides the rest of the feed.")
