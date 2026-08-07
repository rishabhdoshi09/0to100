"""
Earnings Call Analyst page.
Rendered as a sub-tab inside the 🤖 Agents top-level tab.
"""

from __future__ import annotations

import streamlit as st


def render_earnings_page() -> None:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.4rem;margin:0'>🎙️ Earnings Call Analyst</h2>"
        "<p style='color:#8892a4;font-size:.8rem;margin:.2rem 0 1rem'>"
        "Paste any YouTube earnings call URL · DeepSeek R1 extracts structured insights</p>",
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("earnings_form"):
        fc1, fc2 = st.columns([3, 2])
        with fc1:
            url = st.text_input(
                "YouTube Earnings Call URL",
                placeholder="https://www.youtube.com/watch?v=...",
                key="earnings_url",
            )
        with fc2:
            company = st.text_input(
                "Company Name",
                placeholder="e.g. RELIANCE, INFY, TCS",
                key="earnings_company",
            ).strip().upper()

        preview_only = st.checkbox(
            "Preview transcript only (no LLM call — quick sanity check)",
            key="earnings_preview_only",
        )
        submitted = st.form_submit_button("🎙️ Analyse Earnings Call", type="primary")

    if not submitted:
        st.info(
            "**How it works:**\n"
            "1. Paste the YouTube URL of any NSE company's earnings/investor call\n"
            "2. Enter the company name\n"
            "3. Click Analyse — DeepSeek R1 reads the transcript and extracts:\n"
            "   revenue beat/miss, EPS, guidance change, key quotes, red flags, outlook\n\n"
            "**Note:** The video must have captions/subtitles enabled (most investor calls do)."
        )
        return

    if not url.strip():
        st.error("Please enter a YouTube URL.")
        return
    if not company:
        st.error("Please enter the company name.")
        return

    try:
        from agents.earnings_agent import EarningsAgent
        agent = EarningsAgent()

        if preview_only:
            with st.spinner("Fetching transcript preview…"):
                preview = agent.get_transcript_preview(url.strip(), chars=3000)
            st.subheader("Transcript Preview (first 3000 chars)")
            st.text_area("Transcript", preview, height=300, label_visibility="collapsed")
            return

        with st.spinner(
            f"Fetching transcript and analysing {company} with DeepSeek R1… "
            "(R1 uses chain-of-thought — may take 30–60s)"
        ):
            result = agent.analyze(url.strip(), company)

        _render_result(result)

        # ── Save to memory ────────────────────────────────────────────────────
        try:
            from ai.mem0_store import get_memory
            mem = get_memory()
            summary = (
                f"Earnings analysis {company}: "
                f"Revenue beat={result.get('revenue_beat')}, "
                f"EPS beat={result.get('eps_beat')}, "
                f"Guidance={result.get('guidance_change')}, "
                f"Sentiment={result.get('overall_sentiment')}"
            )
            mem.add(summary, category="insight", metadata={"type": "earnings", "company": company})
            st.caption("✅ Analysis saved to Memory Vault.")
        except Exception:
            pass

    except ImportError as exc:
        st.error(f"Missing dependency: {exc}")
        st.code("pip install youtube-transcript-api", language="bash")
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)


def _render_result(result: dict) -> None:
    company = result.get("company", "")
    sentiment = str(result.get("overall_sentiment", "NEUTRAL")).upper()

    # ── Overall verdict banner ────────────────────────────────────────────────
    _colors = {"BULLISH": "#00d4a0", "BEARISH": "#ff4b4b", "NEUTRAL": "#f0a500"}
    color = _colors.get(sentiment, "#8892a4")
    st.markdown(
        f"<div style='background:{color}22;border:1.5px solid {color};"
        f"border-radius:10px;padding:1rem 1.5rem;margin:.5rem 0'>"
        f"<span style='color:{color};font-size:2rem;font-weight:700'>{sentiment}</span>"
        f" <span style='color:#8892a4;font-size:.9rem;margin-left:1rem'>"
        f"Transcript: {result.get('transcript_chars', 0):,} chars · "
        f"Management tone: {result.get('management_tone', 'N/A')}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Key metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    _bool_str = {True: "✅ Beat", False: "❌ Miss", None: "N/A"}
    m1.metric("Revenue", _bool_str.get(result.get("revenue_beat"), "N/A"))
    m2.metric("EPS", _bool_str.get(result.get("eps_beat"), "N/A"))
    m3.metric("Guidance", result.get("guidance_change", "N/A"))
    m4.metric("Analyst Q&A", result.get("analyst_qa_sentiment", "N/A"))

    # ── Revenue / EPS detail ──────────────────────────────────────────────────
    with st.expander("📊 Revenue & EPS Detail"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Revenue**")
            st.write(f"Actual: {result.get('revenue_actual', 'N/A')}")
            st.write(f"Estimate: {result.get('revenue_estimate', 'N/A')}")
        with d2:
            st.markdown("**EPS**")
            st.write(f"Actual: {result.get('eps_actual', 'N/A')}")
            st.write(f"Estimate: {result.get('eps_estimate', 'N/A')}")
        st.markdown(f"**Guidance:** {result.get('guidance_detail', 'N/A')}")

    # ── Key quotes ────────────────────────────────────────────────────────────
    quotes = result.get("key_quotes", [])
    if quotes:
        st.markdown("#### Key Quotes")
        for i, q in enumerate(quotes, 1):
            st.markdown(
                f"<div style='background:rgba(255,255,255,.04);border-left:3px solid #00d4ff;"
                f"border-radius:0 6px 6px 0;padding:.5rem .8rem;margin:.3rem 0;"
                f"color:#e8eaf0;font-size:.88rem;font-style:italic'>\"{ q }\"</div>",
                unsafe_allow_html=True,
            )

    # ── Red flags and catalysts ───────────────────────────────────────────────
    fl1, fl2 = st.columns(2)
    with fl1:
        flags = result.get("red_flags", [])
        if flags:
            st.markdown("#### 🚩 Red Flags")
            for f in flags:
                st.markdown(f"- {f}")
    with fl2:
        cats = result.get("catalysts", [])
        if cats:
            st.markdown("#### 🚀 Catalysts")
            for c in cats:
                st.markdown(f"- {c}")

    # ── Outlook ───────────────────────────────────────────────────────────────
    outlook = result.get("outlook", "")
    if outlook:
        st.markdown("#### Outlook")
        st.info(outlook)

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("Raw JSON"):
        st.json({k: v for k, v in result.items() if k not in ("transcript_chars",)})
