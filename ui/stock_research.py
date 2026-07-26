"""
🔎 Stock Research view — the full research report for any searched symbol.

Renders research.stock_report.research_stock() into one scannable page: quote,
the scanner's verdict + trade plan, the plain-English "why", similar history,
strategy health, technicals, fundamentals, and market context. Cached per symbol
so the (network-touching) aggregation runs once. Fully fail-open — a missing
section is simply omitted.
"""
from __future__ import annotations


def _dots(conf: str) -> str:
    return {"HIGH": "●●●○", "MEDIUM": "●●○○", "LOW": "●○○○"}.get(
        (conf or "").upper(), "○○○○")


def render_stock_research(symbol: str, is_us: bool = False) -> None:
    import streamlit as st

    @st.cache_data(ttl=180, show_spinner="🔎 Researching…")
    def _load(sym: str, us: bool, cap: float) -> dict:
        from research.stock_report import research_stock
        return research_stock(sym, is_us=us, capital=cap)

    cap = float(st.session_state.get("user_capital") or 100_000.0)
    try:
        rep = _load(symbol, is_us, cap)
    except Exception as exc:                                # pragma: no cover
        st.error(f"Research unavailable: {exc}")
        return

    cur = rep.get("cur", "₹")
    q = rep.get("quote", {})
    setup = rep.get("setup", {})

    # ── Header: price + verdict ───────────────────────────────────────────────
    h1, h2 = st.columns([3, 2])
    with h1:
        px = q.get("price")
        chg = q.get("change_pct")
        sub = ""
        if chg is not None:
            col = "#00d4a0" if chg >= 0 else "#ff4b4b"
            sub = f"<span style='color:{col};font-weight:700'>{'▲' if chg>=0 else '▼'} {chg:+.2f}%</span>"
        rng = ""
        if q.get("week52_low") and q.get("week52_high"):
            rng = (f"<div style='font-size:.72rem;color:#6b7280'>52W: "
                   f"{cur}{q['week52_low']:,.0f} – {cur}{q['week52_high']:,.0f}</div>")
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace'>"
            f"<span style='font-size:1.3rem;font-weight:800'>{rep['symbol']}</span>"
            f"<span style='font-size:1.1rem;margin-left:10px'>"
            f"{cur}{px:,.2f}</span> &nbsp;{sub}{rng}</div>"
            if px else f"<b>{rep['symbol']}</b> — <span style='color:#8892a4'>price data unavailable</span>",
            unsafe_allow_html=True)
    with h2:
        v = setup.get("verdict", "")
        vcol = {"STRONG BUY": "#22d3ee", "BUY": "#00d4a0"}.get(v, "#8892a4")
        st.markdown(
            f"<div style='text-align:right'><span style='background:{vcol}18;"
            f"border:1px solid {vcol}66;border-radius:8px;padding:4px 14px;"
            f"font-weight:800;color:{vcol}'>{v or 'NO SETUP'}</span>"
            + (f"<div style='font-size:.72rem;color:#8892a4;margin-top:4px'>"
               f"score {setup.get('score',0):.0f}"
               + (f" · conviction {setup.get('conviction',0):.0f}" if setup.get('conviction') else "")
               + "</div>" if setup.get("score") else "")
            + "</div>", unsafe_allow_html=True)

    # ── The trade plan (if a setup exists) ────────────────────────────────────
    if v in ("STRONG BUY", "BUY"):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Entry", f"{cur}{setup['entry']:,.1f}")
        p2.metric("Stop", f"{cur}{setup['stop']:,.1f}")
        p3.metric("Target", f"{cur}{setup['target']:,.1f}")
        p4.metric("Reward:Risk", f"{setup.get('rr',0):.1f}×")
        sz = rep.get("sizing") or {}
        if sz.get("qty"):
            st.caption(f"📏 1% rule: **{sz['qty']} shares** "
                       f"(₹{sz.get('invested',0):,.0f} lagenge · max loss "
                       f"₹{sz.get('max_loss',0):,.0f})")
        if setup.get("signals"):
            st.caption("Setups: " + " · ".join(setup["signals"]))

    st.divider()

    # ── Why (the plain-English case) ──────────────────────────────────────────
    why = rep.get("why") or {}
    if v in ("STRONG BUY", "BUY"):
        wb = why.get("why_buy") or {}
        if wb.get("summary"):
            st.markdown(f"**✓ Why buy** — {wb['summary']}")
    else:
        wn = why.get("why_not") or {}
        if wn.get("found"):
            st.markdown(f"**✗ Why not right now** — {wn.get('summary','')}")

    ev = why.get("evidence") or {}
    if ev.get("belief"):
        st.markdown(f"**📚 Evidence** — {ev['summary']}")
    tr = why.get("trust") or {}
    if tr.get("summary", "").startswith("Trust basis"):
        st.markdown(f"**🛡️ Trust** — {tr['summary']}")

    sh = why.get("similar_history") or {}
    if sh.get("found"):
        st.markdown(f"**🕒 Similar history** — {sh['summary']}")
        if sh.get("environment"):
            st.caption("Environment: " + " · ".join(sh["environment"]))

    # ── Strategy health for this stock's setups ───────────────────────────────
    hlth = rep.get("strategy_health") or []
    if hlth:
        for s in hlth:
            icon = {"DECAYING": "📉", "RECOVERING": "📈",
                    "STRENGTHENING": "📈"}.get(s.get("status"), "📊")
            st.markdown(f"{icon} **Strategy health** — {s.get('insight','')}")

    # ── Technicals · Fundamentals · Context (folded) ──────────────────────────
    c1, c2, c3 = st.columns(3)
    tech = rep.get("technicals") or {}
    with c1:
        with st.expander("📐 Technicals"):
            rows = [(k, v) for k, v in tech.items()
                    if k not in ("error",) and not isinstance(v, (dict, list))]
            if rows:
                for k, val in rows[:12]:
                    st.caption(f"**{k.replace('_',' ').title()}**: {val}")
            else:
                st.caption("Technicals unavailable.")
    fund = rep.get("fundamentals") or {}
    with c2:
        with st.expander("📊 Fundamentals"):
            rows = [(k, v) for k, v in fund.items()
                    if k not in ("error",) and not isinstance(v, (dict, list))]
            if rows:
                for k, val in rows[:12]:
                    st.caption(f"**{k.replace('_',' ').title()}**: {val}")
            else:
                st.caption("Fundamentals unavailable.")
    ctx = rep.get("context") or {}
    with c3:
        with st.expander("🌤 Market context"):
            if ctx.get("sector"):
                st.caption(f"**Sector**: {ctx['sector']}")
            if ctx.get("market_mood"):
                st.caption(f"**Market mood**: {ctx['market_mood']}")
            if ctx.get("market_health"):
                st.caption(f"**Market health**: {ctx['market_health']}"
                           + (f" ({ctx['pct_above_50dma']:.0f}% above 50-DMA)"
                              if ctx.get("pct_above_50dma") is not None else ""))
            if not any(ctx.values()):
                st.caption("Context unavailable.")

    if setup.get("verdict") == "NO SETUP" and not (q.get("price")):
        st.info("Is symbol ka saaf data nahi mila — spelling check karo "
                "(jaise RELIANCE, TCS, INFY).")
