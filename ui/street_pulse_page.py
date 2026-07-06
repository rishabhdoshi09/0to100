"""
📰 Daily Pulse page — the system's own "Daily Street Pulse" newsletter.

Rendered live from system data (bhavcopy + scanner + Google Finance +
news). Sections mirror a professional daily note: cover takeaways,
market snapshot, movers, buzzing stock, gaining/losing momentum,
breakouts today & tomorrow, global cues, headlines.
"""
from __future__ import annotations

import streamlit as st

_CARD = ("background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
         "padding:14px 18px;margin-bottom:10px")


@st.cache_data(ttl=600, show_spinner=False)
def _pulse() -> dict:
    from reports.street_pulse import build_pulse
    return build_pulse()


@st.cache_data(ttl=1800, show_spinner=False)
def _global_cues() -> list[dict]:
    """S&P, Nasdaq, Gold, Crude, Nikkei — few tickers, cached 30 min."""
    out = []
    try:
        import yfinance as yf
        for name, tk in (("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"),
                         ("Gold", "GC=F"), ("Crude", "CL=F"),
                         ("Nikkei", "^N225"), ("Hang Seng", "^HSI")):
            try:
                fi = yf.Ticker(tk).fast_info
                last = float(getattr(fi, "last_price", 0) or 0)
                prev = float(getattr(fi, "previous_close", 0) or 0)
                if last and prev:
                    out.append({"name": name, "price": last,
                                "chg_pct": (last - prev) / prev * 100})
            except Exception:
                continue
    except Exception:
        pass
    return out


def _chg_span(v: float) -> str:
    color = "#00d4a0" if v >= 0 else "#ff4b4b"
    arrow = "▲" if v >= 0 else "▼"
    return f"<span style='color:{color};font-weight:600'>{arrow} {v:+.2f}%</span>"


def _section_title(txt: str) -> None:
    st.markdown(
        f"<div style='font-size:.68rem;color:#8892a4;text-transform:uppercase;"
        f"letter-spacing:.12em;margin:14px 0 8px 0'>{txt}</div>",
        unsafe_allow_html=True)


def _stock_line(r: dict) -> str:
    reason = (r.get("reasons") or [""])[0]
    return (f"<div style='margin-bottom:8px'>"
            f"<span style='color:#e2e8f0;font-weight:700;"
            f"font-family:JetBrains Mono,monospace'>{r['symbol']}</span> "
            f"<span style='color:#e2e8f0'>₹{r['price']:,.1f}</span> "
            f"{_chg_span(r.get('change_pct', 0))}"
            f"<div style='font-size:.76rem;color:#94a3b8;margin-top:2px'>{reason}</div>"
            f"</div>")


def render_street_pulse() -> None:
    with st.spinner("Aaj ka pulse ban raha hai…"):
        p = _pulse()

    # ── Cover ─────────────────────────────────────────────────────────────────
    take_html = "".join(f"<div style='font-size:.9rem;color:#c9d1d9;"
                        f"margin:4px 0'>• {t}</div>" for t in p["takeaways"])
    st.markdown(
        f"<div style='{_CARD};border-left:3px solid #00d4ff'>"
        f"<div style='font-size:1.25rem;font-weight:800;color:#e6edf3'>"
        f"📰 Daily Street Pulse</div>"
        f"<div style='font-size:.8rem;color:#8892a4;margin-bottom:8px'>{p['date']} · "
        f"{p.get('scanned', 0)} stocks scanned</div>"
        f"{take_html}</div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # ── Buzzing stock ─────────────────────────────────────────────────────
        _section_title("🔥 Buzzing Stock of the Day")
        if p.get("buzzing"):
            b = p["buzzing"]
            st.markdown(
                f"<div style='{_CARD};border-left:3px solid #f59e0b'>"
                f"<span style='font-size:1.05rem;font-weight:800;color:#e6edf3;"
                f"font-family:JetBrains Mono,monospace'>{b['symbol']}</span> "
                f"<span style='color:#e2e8f0'>₹{b['price']:,.1f}</span> "
                f"{_chg_span(b.get('change_pct', 0))}"
                f"<div style='font-size:.82rem;color:#c9d1d9;margin-top:6px'>{b['note']}</div>"
                f"</div>", unsafe_allow_html=True)
        else:
            st.caption("Aaj koi bada buzz nahi.")

        # ── Gaining strength ──────────────────────────────────────────────────
        _section_title("💪 Stock Gaining Strength")
        if p.get("strength"):
            s = p["strength"]
            dist = s.get("pivot_distance_pct", 0)
            st.markdown(
                f"<div style='{_CARD};border-left:3px solid #00d4a0'>"
                f"<span style='font-size:1.05rem;font-weight:800;color:#e6edf3;"
                f"font-family:JetBrains Mono,monospace'>{s['symbol']}</span> "
                f"<span style='color:#e2e8f0'>₹{s['price']:,.1f}</span>"
                f"<div style='font-size:.82rem;color:#c9d1d9;margin-top:6px'>"
                f"{(s.get('reasons') or [''])[0]}</div>"
                f"<div style='font-size:.78rem;color:#94a3b8;margin-top:4px'>"
                f"Pivot ₹{s['entry']:,.0f} se {dist:.1f}% neeche · "
                f"Stop ₹{s['stop']:,.0f} · Target ₹{s['target']:,.0f}</div>"
                f"</div>", unsafe_allow_html=True)
        else:
            st.caption("Koi strong accumulation candidate nahi mila.")

        # ── Losing momentum ───────────────────────────────────────────────────
        _section_title("⚠️ Stock Losing Momentum")
        if p.get("weak"):
            w = p["weak"]
            st.markdown(
                f"<div style='{_CARD};border-left:3px solid #ff4b4b'>"
                f"<span style='font-size:1.05rem;font-weight:800;color:#e6edf3;"
                f"font-family:JetBrains Mono,monospace'>{w['symbol']}</span> "
                f"<span style='color:#e2e8f0'>₹{w['price']:,.1f}</span> "
                f"{_chg_span(w.get('chg_5d', 0))}<span style='font-size:.7rem;"
                f"color:#8892a4'> (5 din)</span>"
                f"<div style='font-size:.82rem;color:#c9d1d9;margin-top:6px'>{w['note']}</div>"
                f"</div>", unsafe_allow_html=True)
        else:
            st.caption("Koi bada breakdown nahi aaj.")

    with c2:
        # ── Market snapshot ───────────────────────────────────────────────────
        _section_title("📊 Market Snapshot")
        snap = p.get("snapshot", {})
        idx_html = "".join(
            f"<div style='margin:4px 0'><span style='color:#e2e8f0;font-weight:600'>"
            f"{i['name']}</span> <span style='color:#e2e8f0'>{i['price']:,.0f}</span> "
            f"{_chg_span(i['chg_pct'])}</div>"
            for i in snap.get("indices", []))
        comm = snap.get("commentary", "")
        st.markdown(
            f"<div style='{_CARD}'>{idx_html or '<span style=color:#8892a4>Index data unavailable</span>'}"
            + (f"<div style='font-size:.8rem;color:#94a3b8;margin-top:6px'>{comm}</div>" if comm else "")
            + "</div>", unsafe_allow_html=True)

        # ── Movers ────────────────────────────────────────────────────────────
        _section_title("📈 Top Gainers / 📉 Losers")
        g_html = "".join(
            f"<div style='font-size:.82rem;margin:2px 0'>"
            f"<span style='color:#e2e8f0;font-family:JetBrains Mono,monospace'>{r['symbol']}</span> "
            f"{_chg_span(r['chg_pct'])}</div>" for r in p.get("gainers", [])[:5])
        l_html = "".join(
            f"<div style='font-size:.82rem;margin:2px 0'>"
            f"<span style='color:#e2e8f0;font-family:JetBrains Mono,monospace'>{r['symbol']}</span> "
            f"{_chg_span(r['chg_pct'])}</div>" for r in p.get("losers", [])[:5])
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"<div style='{_CARD}'>{g_html or 'N/A'}</div>",
                        unsafe_allow_html=True)
        with mc2:
            st.markdown(f"<div style='{_CARD}'>{l_html or 'N/A'}</div>",
                        unsafe_allow_html=True)

        # ── Global cues ───────────────────────────────────────────────────────
        _section_title("🌍 Global Cues")
        cues = _global_cues()
        cue_html = "".join(
            f"<div style='font-size:.82rem;margin:2px 0'>"
            f"<span style='color:#e2e8f0'>{c['name']}</span> "
            f"{_chg_span(c['chg_pct'])}</div>" for c in cues)
        st.markdown(f"<div style='{_CARD}'>{cue_html or 'Unavailable right now'}</div>",
                    unsafe_allow_html=True)

    # ── Breakouts today / tomorrow ────────────────────────────────────────────
    bc1, bc2 = st.columns(2)
    with bc1:
        _section_title("💥 Breakout Stocks — Today")
        today = p.get("breakouts_today", [])
        if today:
            st.markdown(f"<div style='{_CARD}'>"
                        + "".join(_stock_line(r) for r in today)
                        + "</div>", unsafe_allow_html=True)
        else:
            st.caption("Aaj koi confirmed breakout nahi.")
    with bc2:
        _section_title("⏳ Breakout Stocks — Tomorrow?")
        tom = p.get("breakouts_tomorrow", [])
        if tom:
            st.markdown(f"<div style='{_CARD}'>"
                        + "".join(_stock_line(r) for r in tom)
                        + "</div>", unsafe_allow_html=True)
        else:
            st.caption("Pivot ke kareeb koi stock nahi abhi.")

    # ── Headlines ─────────────────────────────────────────────────────────────
    heads = p.get("headlines", [])
    if heads:
        _section_title("🗞 Top Updates")
        st.markdown(f"<div style='{_CARD}'>"
                    + "".join(f"<div style='font-size:.82rem;color:#c9d1d9;"
                              f"margin:4px 0'>• {h}</div>" for h in heads)
                    + "</div>", unsafe_allow_html=True)

    # ── Actions ───────────────────────────────────────────────────────────────
    a1, a2, _ = st.columns([1.2, 1.2, 2.6])
    with a1:
        if st.button("📨 Telegram pe bhejo", key="pulse_tg", use_container_width=True):
            try:
                from alerts.telegram_alerts import AlertEngine
                from reports.street_pulse import pulse_to_telegram
                if AlertEngine().send(pulse_to_telegram(p)):
                    st.success("Bhej diya ✓")
                else:
                    st.warning("Telegram configured nahi hai (.env dekho)")
            except Exception as exc:
                st.warning(f"Nahi bheja ja saka: {exc}")
    with a2:
        if st.button("⟳ Refresh pulse", key="pulse_refresh", use_container_width=True):
            _pulse.clear()
            st.rerun()
