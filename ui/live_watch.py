"""
⚡ Live Watch — Kite WebSocket se tick-by-tick prices, ek board pe.

Stream mode (Kite logged in): MODE_QUOTE ticks, ~sub-second fresh,
board 2s pe repaint. Fallback (no login): REST quote chain, honestly
labelled — stale kabhi live jaisa nahi dikhega.
"""
from __future__ import annotations

import streamlit as st

_TILE = ("background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
         "padding:12px 14px")


def _default_symbols() -> list[str]:
    """Positions + high-conviction picks — jo already matter karte hain."""
    syms: list[str] = []
    try:
        from risk.position_manager import review_positions
        syms += [p["symbol"] for p in review_positions()]
    except Exception:
        pass
    try:
        from scan.auto_scan import get_results
        results, _, _, _ = get_results()
        syms += [r["symbol"] for r in results if r.get("high_conviction")][:4]
        if len(syms) < 4:
            syms += [r["symbol"] for r in results[:4]]
    except Exception:
        pass
    seen: set = set()
    return [s for s in syms if not (s in seen or seen.add(s))][:12]


def render_live_watch() -> None:
    st.markdown("### ⚡ Live Watch")

    default = ", ".join(st.session_state.get("lw_syms",
                                             _default_symbols()))
    raw = st.text_input(
        "Symbols (comma-separated)", value=default, key="lw_input",
        help="Default: tumhari positions + high-conviction picks")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()][:20]
    st.session_state["lw_syms"] = symbols
    if not symbols:
        st.caption("Symbols daalo — positions/watchlist se khud bhar jayega")
        return

    # Stream first; REST fallback with an honest badge
    streaming = False
    try:
        from data.live_ticker import watch, get_ticks, status
        streaming = watch(symbols)
        ticks = get_ticks(symbols) if streaming else {}
    except Exception:
        ticks = {}

    if streaming:
        stt = status()
        _age = stt.get("last_tick_age_s")
        _live_now = _age is not None and _age < 10
        badge = ("<span style='color:#00d4a0'>● KITE STREAM</span>"
                 if _live_now else
                 "<span style='color:#f59e0b'>● stream connected — ticks "
                 "ka wait (market band?)</span>")
    else:
        badge = ("<span style='color:#f59e0b'>◌ REST poll (Kite login "
                 "nahi — stream ke liye login karo)</span>")
        try:
            from data.live_quotes import get_live_quotes
            q = get_live_quotes(symbols)
            ticks = {s: {**v, "age_s": None, "high": 0, "low": 0,
                         "volume": 0} for s, v in q.items()}
        except Exception:
            ticks = {}

    st.markdown(f"<div style='font-size:.72rem;font-family:JetBrains Mono,"
                f"monospace;margin-bottom:8px'>{badge}</div>",
                unsafe_allow_html=True)

    cols = st.columns(3)
    for i, sym in enumerate(symbols):
        d = ticks.get(sym)
        with cols[i % 3]:
            if not d or not d.get("price"):
                st.markdown(
                    f"<div style='{_TILE};opacity:.55'>"
                    f"<b style='font-family:JetBrains Mono,monospace'>{sym}"
                    f"</b><br><span style='font-size:.72rem;color:#8892a4'>"
                    f"data nahi (symbol check karo)</span></div>",
                    unsafe_allow_html=True)
                continue
            chg = float(d.get("chg_pct") or 0)
            c = "#00d4a0" if chg >= 0 else "#ff4b4b"
            age = d.get("age_s")
            if age is None:
                age_txt = "REST · ~8s cache"
                age_col = "#8892a4"
            elif age < 5:
                age_txt = f"tick {age:.0f}s ago"
                age_col = "#00d4a0"
            else:
                age_txt = f"tick {age:.0f}s ago"
                age_col = "#f59e0b"
            hl = ""
            if d.get("high"):
                hl = (f"<br><span style='font-size:.68rem;color:#64748b'>"
                      f"day ₹{d['low']:,.1f}–₹{d['high']:,.1f}"
                      + (f" · vol {d['volume']:,}" if d.get("volume") else "")
                      + "</span>")
            st.markdown(
                f"<div style='{_TILE};margin-bottom:8px'>"
                f"<span style='font-weight:800;font-family:JetBrains Mono,"
                f"monospace;color:#e2e8f0'>{sym}</span> "
                f"<span style='float:right;font-size:.62rem;color:{age_col}'>"
                f"{age_txt}</span><br>"
                f"<span style='font-size:1.35rem;font-weight:800;"
                f"color:#e8eaf0'>₹{d['price']:,.2f}</span> "
                f"<span style='color:{c};font-weight:700'>{chg:+.2f}%</span>"
                f"{hl}</div>",
                unsafe_allow_html=True)

    # Auto-repaint: stream 2s, REST 10s
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=2000 if streaming else 10000, key="lw_tick")
    except Exception:
        if st.button("⟳ Refresh", key="lw_refresh"):
            st.rerun()
