"""
SimpleQuant AI — Streamlit frontend for the Quant Red Flag Analyst.

A clutter-free, layman-friendly web view of the same forensic analysis
that powers `python main.py <SYMBOL>` on the CLI. No Kite/DeepSeek
credentials needed — only stock fundamentals (yfinance).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from forensics import QuantRedFlagAnalyst
from forensics.models import Severity, Verdict
from forensics.simple_report import (
    VERDICT_PLAIN, BUCKET_PLAIN, SEV_PLAIN, _plain_flag, _health_rows,
)
from screener.momentum_screener import scan
from logger import configure_logging, quiet_console

configure_logging()
quiet_console()

st.set_page_config(page_title="SimpleQuant — Stock Check", page_icon="📊", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 760px; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_check, tab_screen = st.tabs(["🔍 Check a stock", "🚀 Find winning trades"])

with tab_screen:
    st.subheader("Momentum & breakout screener")
    st.caption("Scans by market cap and ranks stocks on trend, volume, and breakout strength. "
               "Pulls live daily candles from Kite — run `python main.py login` each morning "
               "if KITE_ACCESS_TOKEN has expired.")

    buckets = st.multiselect("Market cap", ["Large", "Mid", "Small"],
                              default=["Large", "Mid", "Small"])
    c1, c2 = st.columns(2)
    min_score = c1.slider("Minimum momentum score", 0, 100, 60)
    breakout_only = c2.checkbox("Breakouts only", value=False)
    run_screen = st.button("Scan now", type="primary", use_container_width=True, key="scan_btn")

    if run_screen:
        if not buckets:
            st.warning("Pick at least one market-cap bucket.")
        else:
            with st.spinner("Scanning the universe…"):
                try:
                    hits = scan(buckets=buckets, min_score=min_score, breakout_only=breakout_only)
                except Exception as exc:
                    st.error(f"Scan failed — {exc}")
                    hits = []
            if not hits:
                st.info("No stocks matched. Try lowering the minimum score.")
            else:
                st.success(f"{len(hits)} stock(s) matched, ranked by momentum.")
                for h in hits[:30]:
                    icon = "🟢" if h.breakout else "🔵"
                    with st.container(border=True):
                        cols = st.columns([3, 1])
                        cols[0].markdown(f"{icon} **{h.symbol}** — {h.name}")
                        cols[0].caption(f"{h.bucket} cap · {h.sector} · {h.verdict}")
                        cols[1].metric("Score", f"{h.momentum_score:.0f}")
                        st.caption(
                            f"1M: {h.return_1m_pct:+.1f}%  ·  3M: "
                            f"{h.return_3m_pct:+.1f}%  ·  RSI: {h.rsi_14:.0f}  ·  "
                            f"{abs(h.pct_from_52w_high):.1f}% off 52w high  ·  "
                            f"volume {h.volume_ratio:.1f}x avg"
                            if h.return_3m_pct is not None and h.volume_ratio is not None
                            else "Not enough data for full stats"
                        )
    st.caption("Price/volume scan only — not a fundamentals check. "
               "Pair a hit with the 🔍 Check a stock tab before trading it.")

with tab_check:
    st.title("📊 Is this stock any good?")
    st.caption("Type a stock symbol. We check the accounts, the risk, and the red flags — in plain English.")

    symbol = st.text_input("Stock symbol", placeholder="e.g. RELIANCE, TCS, AAPL").strip().upper()
    go = st.button("Check it", type="primary", use_container_width=True)

    if go and not symbol:
        st.warning("Type a symbol first.")

    if go and symbol:
        with st.spinner(f"Checking {symbol}…"):
            try:
                report = QuantRedFlagAnalyst().analyze(symbol)
            except Exception as exc:
                st.error(f"Couldn't analyze {symbol} — {exc}")
                st.stop()

        comp = report.composite
        headline, meaning, _ = VERDICT_PLAIN[comp.verdict]
        color = {
            Verdict.STRONG_CANDIDATE: "green", Verdict.WATCHLIST: "green",
            Verdict.NEUTRAL: "orange", Verdict.CAUTION: "orange",
            Verdict.HIGH_RISK: "red", Verdict.AVOID: "red",
        }[comp.verdict]

        st.divider()
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(report.company)
            st.markdown(f":{color}[**{headline}**]")
            st.caption(meaning)
        with c2:
            st.metric("Quality score", f"{comp.score:.0f}/100" if comp.score is not None else "N/A")

        st.markdown("#### Health check")
        for label, score in _health_rows(report):
            col1, col2 = st.columns([2, 3])
            col1.write(label)
            if score is None:
                col2.caption("not enough data")
            else:
                col2.progress(min(max(score / 100, 0.0), 1.0), text=f"{score:.0f}/100")

        st.markdown("#### What worries us")
        serious = [f for f in report.flags
                   if f.severity in (Severity.CRITICAL, Severity.HIGH)][:5]
        if serious:
            for f in serious:
                word, _ = SEV_PLAIN[f.severity]
                icon = "🔴" if f.severity == Severity.CRITICAL else "🟠"
                st.markdown(f"{icon} **{word}** — {_plain_flag(f.title)}")
            more = len([f for f in report.flags
                        if f.severity in (Severity.CRITICAL, Severity.HIGH)]) - len(serious)
            if more > 0:
                st.caption(f"…and {more} more")
        else:
            st.success("No serious warning signs found.")

        if comp.strengths:
            _negative = ("inflated", "lower than", "risk", "weak", "erratic",
                         "aggressive", "not backed", "deteriorat")
            plain = [s.split(" — ", 1)[-1] for s in comp.strengths]
            good = [p for p in plain if not any(n in p.lower() for n in _negative)][:3]
            if good:
                st.markdown("#### What looks good")
                for p in good:
                    st.markdown(f"✅ {p}")

        st.markdown("#### Bottom line")
        if report.sizing:
            bucket_text = BUCKET_PLAIN.get(report.sizing.bucket, "")
            st.info(bucket_text)
            if report.sizing.bucket not in ("Avoid", "Tracking"):
                st.caption(f"Suggested size: about {report.sizing.suggested_weight:.0f}% of a portfolio")

        st.caption(f"Full details on the CLI: `python main.py {report.symbol} --pro`  ·  "
                   "Automated analysis of public data — not investment advice.")
