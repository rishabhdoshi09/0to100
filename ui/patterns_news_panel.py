import streamlit as st

try:
    from features.numta_wrapper import detect_chart_patterns
except ImportError:
    def detect_chart_patterns(df):
        return {}

try:
    from news.marketaux_news import MarketauxNews
    _NEWS_AVAILABLE = True
except Exception:
    _NEWS_AVAILABLE = False


def render_patterns_news(symbol: str, df):
    st.subheader(f"Patterns & News — {symbol}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Chart Patterns**")
        patterns = detect_chart_patterns(df) if df is not None and len(df) > 0 else {}
        if not patterns:
            st.info("No patterns detected (numta not installed or insufficient data).")
        else:
            for name, val in patterns.items():
                if val != 0:
                    direction = "Bullish" if val > 0 else "Bearish"
                    color = "green" if val > 0 else "red"
                    st.markdown(f":{color}[**{name.replace('_', ' ').title()}** — {direction}]")

    with col2:
        st.markdown("**Recent News**")
        if not _NEWS_AVAILABLE:
            st.info("News module unavailable. Set MARKETAUX_API_KEY in .env.")
        else:
            try:
                news = MarketauxNews().fetch(symbol, limit=5)
                if not news:
                    st.info("No recent news found.")
                for item in news:
                    sentiment = item.get("sentiment", "neutral")
                    color = "green" if sentiment == "positive" else ("red" if sentiment == "negative" else "gray")
                    st.markdown(f":{color}[{item.get('title', '')}]")
                    st.caption(item.get("published_at", ""))
            except Exception as e:
                st.warning(f"News fetch failed: {e}")
