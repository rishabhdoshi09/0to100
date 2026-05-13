"""
Signal Tracker Page — "Does the scanner actually work?"

Renders a Streamlit page that shows the feedback loop:
  - Logs BUY/WATCH signals when the scanner runs
  - Auto-checks price 3, 5, and 10 days later
  - Displays win rate, avg return, best/worst trades
  - Per-symbol signal history lookup
"""
from __future__ import annotations

import streamlit as st


def render_signal_tracker() -> None:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:4px'>"
        "📊 Signal Tracker — Does the scanner actually work?</h2>"
        "<p style='color:#4a5568;font-size:.76rem;margin-bottom:1rem'>"
        "Every BUY/WATCH signal is logged. Prices are auto-checked 3, 5, and 10 "
        "trading days later. WIN = 5d return &gt; 2% · LOSS = 5d return &lt; -2%.</p>",
        unsafe_allow_html=True,
    )

    try:
        from screener.signal_tracker import SignalTracker
        tracker = SignalTracker()
    except Exception as e:
        st.error(f"Could not initialise Signal Tracker: {e}")
        return

    # ── Refresh pending rows on every page load ────────────────────────────
    with st.spinner("Refreshing pending outcomes…"):
        try:
            n_updated = tracker.update_outcomes()
            if n_updated:
                st.caption(f"✅ Updated {n_updated} pending signal(s).")
        except Exception as upd_err:
            st.caption(f"⚠️ Outcome refresh error: {upd_err}")

    stats = tracker.get_stats()

    # ── Top metrics row ────────────────────────────────────────────────────
    m0, m1, m2, m3 = st.columns(4)

    with m0:
        st.metric("Total Signals", stats["total_signals"])

    with m1:
        wr = stats["win_rate_pct"]
        wr_color = "normal" if wr >= 50 else "inverse"
        st.metric("Win Rate", f"{wr:.1f}%", delta=None)

    with m2:
        avg_r = stats["avg_return_5d"]
        st.metric("Avg 5d Return", f"{avg_r:+.2f}%", delta=None)

    with m3:
        best = stats["best_trade"]
        if best:
            st.metric(
                "Best Trade",
                best["symbol"],
                delta=f"{best['return_5d']:+.1f}%",
            )
        else:
            st.metric("Best Trade", "—")

    st.divider()

    # ── Win/Loss/Neutral bar chart ─────────────────────────────────────────
    recent = tracker.get_recent(limit=500)
    if recent:
        import pandas as pd
        import plotly.graph_objects as go

        df_all = pd.DataFrame(recent)

        # Counts by signal type × outcome
        outcome_order = ["WIN", "NEUTRAL", "LOSS", "PENDING"]
        outcome_colors = {
            "WIN":     "#00d4a0",
            "NEUTRAL": "#8892a4",
            "LOSS":    "#ff4b4b",
            "PENDING": "#f59e0b",
        }

        fig = go.Figure()
        for sig_type in ("BUY", "WATCH"):
            subset = df_all[df_all["signal"] == sig_type] if "signal" in df_all.columns else pd.DataFrame()
            for outcome in outcome_order:
                count = int((subset["outcome"] == outcome).sum()) if not subset.empty else 0
                fig.add_trace(go.Bar(
                    name=f"{sig_type} — {outcome}",
                    x=[sig_type],
                    y=[count],
                    marker_color=outcome_colors[outcome],
                    text=[count] if count > 0 else [""],
                    textposition="inside",
                    legendgroup=outcome,
                    showlegend=(sig_type == "BUY"),  # show legend label once
                ))

        fig.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1e0", size=12),
            title=dict(
                text="Signal Outcomes by Type",
                font=dict(color="#00d4ff", size=14),
                x=0.0,
            ),
            legend=dict(
                orientation="h",
                y=-0.25,
                font=dict(size=11),
            ),
            height=280,
            margin=dict(l=0, r=0, t=40, b=60),
            xaxis=dict(
                tickfont=dict(size=13, color="#c9d1e0"),
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(size=11, color="#8892a4"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No signals logged yet. Run the scanner to start building the feedback loop.", icon="📡")

    st.divider()

    # ── Recent signals table ───────────────────────────────────────────────
    st.markdown(
        "<span style='color:#00d4ff;font-size:.78rem;font-weight:700;"
        "letter-spacing:1.5px'>RECENT SIGNALS</span>",
        unsafe_allow_html=True,
    )

    recent50 = tracker.get_recent(limit=50)
    if recent50:
        import pandas as pd

        df = pd.DataFrame(recent50)

        # Select and rename display columns
        display_cols = {
            "symbol":          "Symbol",
            "signal":          "Signal",
            "price_at_signal": "Price @ Signal",
            "logged_at":       "Date",
            "score":           "Score",
            "rsi":             "RSI",
            "return_3d":       "Ret 3d %",
            "return_5d":       "Ret 5d %",
            "return_10d":      "Ret 10d %",
            "outcome":         "Outcome",
        }
        df_disp = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)

        # Format numeric columns
        for col in ["Price @ Signal"]:
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].apply(lambda x: f"₹{x:,.0f}" if x else "—")
        for col in ["Ret 3d %", "Ret 5d %", "Ret 10d %"]:
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].apply(
                    lambda x: f"{x:+.2f}%" if x is not None and str(x) != "nan" else "—"
                )
        for col in ["Score", "RSI"]:
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].apply(
                    lambda x: f"{x:.1f}" if x is not None and str(x) != "nan" else "—"
                )

        # Color-code Outcome column using styler
        def _color_outcome(val: str) -> str:
            if val == "WIN":
                return "color: #00d4a0; font-weight: 700"
            if val == "LOSS":
                return "color: #ff4b4b; font-weight: 700"
            if val == "PENDING":
                return "color: #f59e0b"
            return "color: #8892a4"  # NEUTRAL

        styled = df_disp.style.applymap(_color_outcome, subset=["Outcome"])

        st.dataframe(styled, hide_index=True, use_container_width=True, height=420)
    else:
        st.caption("No signals yet.")

    st.divider()

    # ── Symbol history lookup ──────────────────────────────────────────────
    st.markdown(
        "<span style='color:#00d4ff;font-size:.78rem;font-weight:700;"
        "letter-spacing:1.5px'>SYMBOL HISTORY LOOKUP</span>",
        unsafe_allow_html=True,
    )

    lookup_sym = st.text_input(
        "Enter symbol (e.g. RELIANCE)",
        key="tracker_sym_lookup",
        placeholder="RELIANCE",
        label_visibility="collapsed",
    )

    if lookup_sym:
        sym_history = tracker.get_symbol_history(lookup_sym.strip().upper())
        if sym_history:
            import pandas as pd

            df_sym = pd.DataFrame(sym_history)
            display_cols = {
                "signal":          "Signal",
                "price_at_signal": "Price @ Signal",
                "logged_at":       "Date",
                "return_3d":       "Ret 3d %",
                "return_5d":       "Ret 5d %",
                "return_10d":      "Ret 10d %",
                "outcome":         "Outcome",
            }
            df_sym_disp = df_sym[[c for c in display_cols if c in df_sym.columns]].rename(columns=display_cols)

            for col in ["Price @ Signal"]:
                if col in df_sym_disp.columns:
                    df_sym_disp[col] = df_sym_disp[col].apply(lambda x: f"₹{x:,.0f}" if x else "—")
            for col in ["Ret 3d %", "Ret 5d %", "Ret 10d %"]:
                if col in df_sym_disp.columns:
                    df_sym_disp[col] = df_sym_disp[col].apply(
                        lambda x: f"{x:+.2f}%" if x is not None and str(x) != "nan" else "—"
                    )

            def _color_outcome(val: str) -> str:
                if val == "WIN":
                    return "color: #00d4a0; font-weight: 700"
                if val == "LOSS":
                    return "color: #ff4b4b; font-weight: 700"
                if val == "PENDING":
                    return "color: #f59e0b"
                return "color: #8892a4"

            styled_sym = df_sym_disp.style.applymap(_color_outcome, subset=["Outcome"])
            st.markdown(
                f"<span style='color:#e8eaf0;font-size:.82rem;font-weight:700'>"
                f"{lookup_sym.upper()}</span>"
                f"<span style='color:#4a5568;font-size:.75rem;margin-left:.5rem'>"
                f"{len(sym_history)} signal(s) found</span>",
                unsafe_allow_html=True,
            )
            st.dataframe(styled_sym, hide_index=True, use_container_width=True)
        else:
            st.info(f"No signals logged for **{lookup_sym.upper()}** yet.")
