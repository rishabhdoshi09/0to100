"""
Smart Scanner — the whole market, always current, zero wait.

A background auto-scan (scan/auto_scan.py) covers the ENTIRE NSE and
refreshes every 15 min during market hours. This page just reads the
store — results appear instantly. Every BUY is logged to the outcome
tracker, so the accuracy shown here is measured, not promised.
"""
from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

_CATEGORY_TABS = ["🔥 All Signals", "⏳ Breakout Soon", "🚀 Momentum",
                  "💥 Breakouts", "📐 Chart Patterns"]
_CATEGORY_MAP = {
    "⏳ Breakout Soon": "PreBreakout",
    "🚀 Momentum": "Momentum",
    "💥 Breakouts": "Breakout",
    "📐 Chart Patterns": "Pattern",
}

_VERDICT_STYLE = {
    "STRONG BUY": ("🔥 Strong Buy", "#22d3ee"),
    "BUY":        ("⚡ Buy Signal", "#00d4a0"),
    "WATCH":      ("👁 Watch",      "#f59e0b"),
}
_BUY_VERDICTS = ("STRONG BUY", "BUY")

_CAT_COLOR = {"Momentum": "#38bdf8", "Breakout": "#a78bfa", "Pattern": "#f472b6",
              "PreBreakout": "#facc15"}


@st.cache_data(ttl=180, show_spinner=False)
def _live_quotes(symbols_key: str) -> dict:
    """Fresh Google Finance quotes for the stocks on screen. Cached 3 min."""
    try:
        from data.google_finance import get_quotes
        return get_quotes(symbols_key.split(",")[:80])
    except Exception:
        return {}


def _apply_live_prices(results: list[dict]) -> None:
    """Overlay current prices at render time. Marks each card live/EOD —
    a stale price must LOOK stale, never pretend to be current."""
    syms = [r["symbol"] for r in results[:80]]
    if not syms:
        return
    live = _live_quotes(",".join(syms))
    for r in results[:80]:
        q = live.get(r["symbol"])
        if not (q and q.get("price")):
            r["live"] = False
            continue
        r["live"] = True
        r["price"] = q["price"]
        r["change_pct"] = q["chg_pct"]
        entry = float(r.get("entry") or 0)
        if entry and q["price"] < entry * 0.97 and r.get("verdict") in ("STRONG BUY", "BUY"):
            slip = (entry - q["price"]) / entry * 100
            r["verdict"] = "WATCH"
            r.setdefault("reasons", []).insert(
                0, f"⚠ Live ₹{q['price']:,.0f} — entry ₹{entry:,.0f} se "
                   f"{slip:.0f}% neeche, setup pullback mein hai")
            if r.get("checks"):
                r["checks"].insert(
                    0, f"⚠ Live ₹{q['price']:,.0f} entry se {slip:.0f}% neeche — "
                       f"chase mat karo")


@st.cache_data(ttl=600, show_spinner=False)
def _accuracy_stat() -> str:
    """Measured accuracy from the outcome tracker — '' if not enough data yet."""
    try:
        from core.signal_outcome_tracker import get_accuracy_report
        rep = get_accuracy_report()
        closed = (rep.get("wins", 0) or 0) + (rep.get("losses", 0) or 0)
        if closed >= 5:
            return (f"🎯 Verified accuracy: <b>{rep.get('overall_accuracy', 0):.0f}%</b> "
                    f"on {closed} tracked signals")
    except Exception:
        pass
    return ""


@st.cache_data(ttl=300, show_spinner=False)
def _health() -> list[tuple[str, bool, str]]:
    """[(name, ok, detail)] — data-source health so failures can't hide."""
    out = []
    # NSE data — show the actual latest bar date (live intraday vs EOD)
    try:
        from datetime import date as _date
        from data.bhavcopy_store import is_ready, get_ohlcv
        ok = is_ready()
        detail = "not built"
        if ok:
            _df = get_ohlcv("RELIANCE")
            if _df is not None and len(_df):
                _last = _df.index[-1].date()
                detail = (f"aaj ka live bar ✓" if _last == _date.today()
                          else f"EOD {_last.strftime('%d %b')}")
        out.append(("NSE data", ok, detail))
    except Exception:
        out.append(("NSE data", False, "error"))
    # Google Finance live quotes — test the STOCK path (what cards actually use)
    try:
        from data.google_finance import get_quote
        q = get_quote("RELIANCE", timeout=6)
        ok = bool(q and q.get("price"))
        out.append(("Live quotes", ok,
                    f"Google Finance ₹{q['price']:,.0f}" if ok else "parse/network fail"))
    except Exception:
        out.append(("Live quotes", False, "unreachable"))
    # Kite
    try:
        from config import settings
        out.append(("Kite", bool(settings.kite_access_token), "token set" if settings.kite_access_token else "not logged in"))
    except Exception:
        out.append(("Kite", False, "—"))
    # Telegram
    try:
        from alerts.telegram_alerts import AlertEngine
        ok = AlertEngine().is_configured()
        out.append(("Telegram", ok, "alerts on" if ok else "not set"))
    except Exception:
        out.append(("Telegram", False, "—"))
    return out


def _render_health() -> None:
    items = _health()
    html = " &nbsp;&nbsp; ".join(
        f"<span style='color:{'#00d4a0' if ok else '#ff4b4b'}'>●</span> "
        f"<span style='color:#94a3b8'>{name}</span> "
        f"<span style='color:#4a5568;font-size:.68rem'>({detail})</span>"
        for name, ok, detail in items)
    st.markdown(
        f"<div style='font-size:.74rem;padding:4px 0 8px 0'>{html}</div>",
        unsafe_allow_html=True)


def _freshness(ts: float) -> str:
    if not ts:
        return ""
    mins = int((time.time() - ts) / 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    return datetime.fromtimestamp(ts).strftime("%I:%M %p")


@st.cache_data(ttl=900, show_spinner=False)
def _sparkline_svg(symbol: str, entry: float = 0.0) -> str:
    """Inline 30-day price sparkline SVG from the bhav store. '' on miss."""
    try:
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(symbol)
        if df is None or len(df) < 10:
            return ""
        closes = df["close"].values[-30:].astype(float)
        lo, hi = closes.min(), closes.max()
        if hi <= lo:
            return ""
        W, H, PAD = 150, 38, 3
        xs = [PAD + i * (W - 2 * PAD) / (len(closes) - 1) for i in range(len(closes))]
        ys = [H - PAD - (c - lo) / (hi - lo) * (H - 2 * PAD) for c in closes]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        up = closes[-1] >= closes[0]
        color = "#00d4a0" if up else "#ff4b4b"
        # entry level line (if inside the visible range)
        entry_line = ""
        if entry and lo <= entry <= hi:
            ey = H - PAD - (entry - lo) / (hi - lo) * (H - 2 * PAD)
            entry_line = (f"<line x1='{PAD}' y1='{ey:.1f}' x2='{W-PAD}' y2='{ey:.1f}' "
                          f"stroke='#f59e0b' stroke-width='1' stroke-dasharray='3,3'/>")
        return (f"<svg width='{W}' height='{H}' style='vertical-align:middle'>"
                f"<polyline points='{pts}' fill='none' stroke='{color}' "
                f"stroke-width='1.6' stroke-linejoin='round'/>"
                f"{entry_line}</svg>")
    except Exception:
        return ""


# ── Card renderer ─────────────────────────────────────────────────────────────

def _render_card(s: dict, key_prefix: str = "") -> None:
    label, vcolor = _VERDICT_STYLE.get(s["verdict"], _VERDICT_STYLE["WATCH"])
    chg = s["change_pct"]
    chg_color = "#00d4a0" if chg >= 0 else "#ff4b4b"
    chg_arrow = "▲" if chg >= 0 else "▼"
    # Live vs EOD transparency — stale price must never look current
    live_tag = ("" if s.get("live")
                else "<span style='background:#1e293b;border-radius:4px;padding:1px 6px;"
                     "font-size:.62rem;color:#8892a4;margin-left:6px'>EOD close</span>")

    chips = "".join(
        f"<span style='background:#1e293b;border-radius:5px;padding:2px 8px;"
        f"font-size:.68rem;color:#94a3b8;margin-right:6px'>{sig}</span>"
        for sig in s["signals"]
    )

    # Conviction checklist (from JARVIS layer) or plain scanner reasons
    checks = s.get("checks") or []
    if checks:
        reason = "<br>".join(
            f"<span style='color:{'#00d4a0' if c.startswith('✓') else '#f59e0b' if c.startswith('⚠') else '#94a3b8'}'>"
            f"{c}</span>"
            for c in checks[:5]
        )
    else:
        reason = " · ".join(s["reasons"][:2])

    # Position size — the 1% discipline line
    size_html = ""
    try:
        from risk.position_sizer import size_position
        cap = st.session_state.get("user_capital")
        ps = size_position(float(s["entry"]), float(s["stop"]), capital=cap)
        if ps["qty"] >= 1:
            size_html = (
                f"<div style='margin-top:4px;font-size:.75rem;color:#7dd3fc'>"
                f"📏 Tumhare liye: <b>{ps['qty']} shares</b> (₹{ps['invested']:,.0f} lagenge) "
                f"· max loss ₹{ps['max_loss']:,.0f} ({ps['risk_pct_used']:.1f}% of capital)"
                + (" · position cap laga" if ps["capped"] else "")
                + "</div>")
    except Exception:
        pass

    plan = (
        f"<div style='margin-top:7px;font-size:.78rem;color:#94a3b8'>"
        f"Entry <span style='color:#f59e0b;font-weight:700'>₹{s['entry']:,.0f}</span>"
        f" &nbsp;·&nbsp; Stop <span style='color:#ff4b4b;font-weight:700'>₹{s['stop']:,.0f}</span>"
        f" &nbsp;·&nbsp; Target <span style='color:#00d4a0;font-weight:700'>₹{s['target']:,.0f}</span>"
        + (f" &nbsp;·&nbsp; <span style='color:#e2e8f0'>Reward {s['rr']:.1f}×</span>" if s["rr"] > 0 else "")
        + f"</div>{size_html}"
    )

    spark = _sparkline_svg(s["symbol"], float(s.get("entry") or 0))
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
            f"padding:12px 16px;margin-bottom:6px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"  <div>"
            f"    <span style='color:#e2e8f0;font-weight:700;font-size:1rem;"
            f"      font-family:JetBrains Mono,monospace'>{s['symbol']}</span>"
            f"    <span style='color:#e2e8f0;font-size:.95rem;margin-left:12px'>₹{s['price']:,.1f}</span>"
            f"    <span style='color:{chg_color};font-size:.82rem;margin-left:6px;font-weight:600'>"
            f"      {chg_arrow}{abs(chg):.1f}%</span>{live_tag}"
            f"  </div>"
            f"  <div style='display:flex;align-items:center;gap:12px'>{spark}"
            f"  <span style='background:{vcolor}18;border:1px solid {vcolor}55;border-radius:6px;"
            f"    padding:3px 10px;font-size:.7rem;font-weight:700;color:{vcolor}'>{label}</span>"
            f"  </div>"
            f"</div>"
            f"<div style='margin-top:7px'>{chips}</div>"
            f"<div style='margin-top:6px;font-size:.78rem;color:#c9d1d9'>{reason}</div>"
            f"{plan}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("Analyse →", key=f"scan_{key_prefix}_{s['symbol']}", use_container_width=True):
            st.session_state["sidebar_nav"] = "Terminal"
            st.session_state["terminal_symbol"] = s["symbol"]
            st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render_scanner(universe: list[str]) -> None:
    from scan.auto_scan import (start_background_scan, get_results,
                                run_manual_scan, set_auto_enabled, is_auto_enabled)
    start_background_scan()   # idempotent — first visitor kicks it off

    try:
        from data.nse_universe import get_nifty500_universe
        nifty500 = get_nifty500_universe()
    except Exception:
        nifty500 = universe[:500]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Smart Scanner")
    st.caption("Momentum · Breakouts · Chart patterns — button dabao, "
               "poore market se ache setups nikal ke aayenge")
    _render_health()

    # ── Your controls ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2.2, 1.4, 1.4])
    with c1:
        scope_options = {
            "NIFTY 500 — fast (~1 min)": nifty500,
            f"Poora NSE — {len(universe)} stocks (~3-4 min)": universe,
        }
        scope_label = st.selectbox("Kahan scan karein", list(scope_options.keys()),
                                   index=0, key="scanner_scope")
        scan_syms = scope_options[scope_label]
    with c2:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("🔍 Scan Market", key="scanner_run",
                                 type="primary", use_container_width=True)
    with c3:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        auto_on = st.toggle("Auto-refresh (15 min)", value=is_auto_enabled(),
                            key="scanner_auto",
                            help="On: system market hours mein khud scan karta rahega. "
                                 "Off: sirf tumhare button dabane par scan hoga.")
        set_auto_enabled(auto_on)

    # ── Signal accuracy (walk-forward backtest on real NSE data) ─────────────
    with st.expander("📊 Signal accuracy — kaunsa signal kitna sahi (backtest)"):
        try:
            from scan.signal_backtest import (load_report, run_in_background,
                                              get_state)
            from scan.unified_scanner import SIGNAL_META
            _bt_state = get_state()
            _rep = load_report()
            bc1, bc2 = st.columns([1.4, 2.6])
            with bc1:
                if st.button("▶ Backtest chalao", key="bt_run",
                             disabled=_bt_state["running"],
                             use_container_width=True):
                    run_in_background()
                    st.rerun()
                if _bt_state["running"]:
                    st.caption(f"⏳ Chal raha hai… {_bt_state['progress']}/"
                               f"{_bt_state['total']} stocks")
            with bc2:
                st.caption("Har signal ko pichhle ~100 sessions pe walk-forward "
                           "test karta hai (no lookahead): entry ke baad target "
                           "(2R) pehle laga ya stop. Isse pata chalta hai kis "
                           "signal pe kitna bharosa karna chahiye.")
            if _rep:
                rows = []
                for key, s in sorted(_rep.get("signals", {}).items(),
                                     key=lambda kv: -kv[1].get("win_rate", 0)):
                    if s.get("closed", 0) < 10:
                        continue
                    label = SIGNAL_META.get(key, (key,))[0]
                    rows.append({
                        "Signal": label,
                        "Win rate": f"{s['win_rate']:.0f}%",
                        "Expectancy": f"{s['expectancy_r']:+.2f}R",
                        "Trades": s["closed"],
                    })
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                    st.caption(f"Last run: {_rep.get('generated_at')} · "
                               f"{_rep.get('symbols')} stocks · "
                               f"{_rep.get('horizon_days')}-day horizon · "
                               f"Expectancy +0.5R se upar = solid edge")
            else:
                st.caption("Abhi tak backtest nahi chala — ▶ dabao (~2-3 min).")
        except Exception as _bt_exc:
            st.caption(f"Backtest unavailable: {_bt_exc}")

    # ── Capital for position sizing ───────────────────────────────────────────
    with st.expander("💰 Position sizing — apna capital set karo"):
        try:
            from config import settings as _cfg
            _default_cap = float(st.session_state.get("user_capital")
                                 or _cfg.trading_capital)
        except Exception:
            _default_cap = 100_000.0
        _cap = st.number_input(
            "Trading capital (₹)", min_value=10_000.0, max_value=100_000_000.0,
            value=_default_cap, step=10_000.0, key="capital_input",
            help="Har card pe exact shares dikhengi — 1% risk rule ke hisaab se. "
                 "Yeh discipline hi account ko zinda rakhti hai.")
        st.session_state["user_capital"] = _cap
        st.caption(f"Har trade pe max risk: ₹{_cap * 0.01:,.0f} (1%) · "
                   f"ek stock mein max: ₹{_cap * 0.10:,.0f} (10%)")

    # ── Manual scan (user-controlled, with live progress) ────────────────────
    if scan_clicked:
        pbar = st.progress(0.0, text=f"Scanning {len(scan_syms)} stocks…")

        def _on_progress(done: int, total: int) -> None:
            pct = done / total if total else 0.0
            pbar.progress(min(pct, 0.95),
                          text=f"Downloading market data… {done}/{total} stocks")

        run_manual_scan(universe=scan_syms, progress=_on_progress)
        pbar.progress(1.0, text="Done — setups ready ✓")
        st.rerun()

    results, universe_size, last_ts, status = get_results()
    _apply_live_prices(results)   # Google Finance live overlay at render time
    if last_ts:
        fresh_note = (f"Last scan: **{_freshness(last_ts)}** · {universe_size} stocks "
                      f"covered · prices live via Google Finance (3-min refresh)")
        if status == "scanning":
            fresh_note += " · ⟳ refreshing in background…"
        st.caption(fresh_note)

    # ── No results yet ────────────────────────────────────────────────────────
    if not results:
        if status in ("scanning", "idle"):
            st.info("⏳ Pehla scan background mein chal raha hai — ya **Scan Market** "
                    "dabao aur progress dekho. Baaki app use kar sakte ho, results "
                    "yahin milenge.")
            if st.button("⟳ Check again", key="scanner_poll"):
                st.rerun()
        else:
            st.warning("Scan market data nahi laa paya. Internet check karke "
                       "**Scan Market** dobara dabao.")
        return

    # ── Summary strip ─────────────────────────────────────────────────────────
    n_strong = sum(1 for r in results if r["verdict"] == "STRONG BUY")
    n_buy = sum(1 for r in results if r["verdict"] in _BUY_VERDICTS)
    n_mom = sum(1 for r in results if "Momentum" in r["categories"])
    n_brk = sum(1 for r in results if "Breakout" in r["categories"])
    n_pat = sum(1 for r in results if "Pattern" in r["categories"])
    n_pre = sum(1 for r in results if "PreBreakout" in r["categories"])
    acc = _accuracy_stat()
    st.markdown(
        f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:8px;"
        f"padding:9px 14px;margin:4px 0 12px 0;font-size:.84rem;color:#c9d1d9'>"
        f"<b>{universe_size}</b> stocks scanned → <b>{len(results)}</b> signals &nbsp;·&nbsp; "
        + (f"<span style='color:#22d3ee'>🔥 {n_strong} strong buy</span> &nbsp;·&nbsp; " if n_strong else "")
        + f"<span style='color:#00d4a0'>⚡ {n_buy} buy</span> &nbsp;·&nbsp; "
        + (f"<span style='color:{_CAT_COLOR['PreBreakout']}'>⏳ {n_pre} breakout soon</span> &nbsp;·&nbsp; " if n_pre else "")
        + f"<span style='color:{_CAT_COLOR['Momentum']}'>🚀 {n_mom} momentum</span> &nbsp;·&nbsp; "
        f"<span style='color:{_CAT_COLOR['Breakout']}'>💥 {n_brk} breakouts</span> &nbsp;·&nbsp; "
        f"<span style='color:{_CAT_COLOR['Pattern']}'>📐 {n_pat} patterns</span>"
        + (f" &nbsp;·&nbsp; {acc}" if acc else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Hot sectors strip (sector heat working = visible here) ───────────────
    _sec_counts: dict = {}
    for r in results:
        if r.get("sector"):
            _sec_counts[r["sector"]] = _sec_counts.get(r["sector"], 0) + 1
    if _sec_counts:
        _hot_html = " ".join(
            f"<span style='background:#f59e0b18;border:1px solid #f59e0b44;"
            f"border-radius:6px;padding:2px 10px;font-size:.72rem;color:#f59e0b;"
            f"margin-right:6px'>🔥 {sec} · {cnt} stocks</span>"
            for sec, cnt in sorted(_sec_counts.items(), key=lambda kv: -kv[1])[:4])
        st.markdown(
            f"<div style='margin:-4px 0 10px 0'>"
            f"<span style='font-size:.7rem;color:#8892a4;margin-right:8px'>"
            f"HOT SECTORS</span>{_hot_html}</div>",
            unsafe_allow_html=True)

    # ── Calibration badge (backtest weights active = visible here) ───────────
    try:
        from scan.unified_scanner import _load_calibration
        _cal = _load_calibration()
        if _cal:
            _boosted = sum(1 for v in _cal.values() if v > 1)
            _cut = sum(1 for v in _cal.values() if v < 1)
            st.caption(f"🎯 Scores backtest-calibrated: {_boosted} signal(s) boosted, "
                       f"{_cut} discounted — evidence se, andaaze se nahi")
    except Exception:
        pass

    # ── Telegram push hint (only when not configured) ─────────────────────────
    try:
        from alerts.telegram_alerts import AlertEngine
        if not AlertEngine().is_configured():
            st.caption("💡 Naye setups **khud aap tak** pahunch sakte hain — `.env` mein "
                       "`TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID` daalo, har scan ke baad "
                       "fresh Buy setups Telegram pe milenge.")
    except Exception:
        pass

    # ── Search within results ─────────────────────────────────────────────────
    q = st.text_input("Filter", placeholder="🔍 Filter by symbol…",
                      key="scanner_filter", label_visibility="collapsed")
    if q:
        qq = q.strip().upper()
        results = [r for r in results if qq in r["symbol"]]

    # ── Category tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs(_CATEGORY_TABS)
    for tab, tab_name in zip(tabs, _CATEGORY_TABS):
        with tab:
            cat = _CATEGORY_MAP.get(tab_name)
            subset = results if cat is None else [
                r for r in results if cat in r["categories"]]
            if cat == "PreBreakout":
                # closest to the pivot first — those may break out any moment
                subset = sorted(subset,
                                key=lambda r: r.get("pivot_distance_pct") or 99)
            if not subset:
                st.caption("Nothing in this category right now.")
                continue

            buy_only = st.toggle("⚡ Buy signals only", value=False,
                                 key=f"buyonly_{tab_name}")
            if buy_only:
                subset = [r for r in subset if r["verdict"] in _BUY_VERDICTS]

            buys = [r for r in subset if r["verdict"] in _BUY_VERDICTS]
            watch = [r for r in subset if r["verdict"] == "WATCH"]

            kp = tab_name.split(" ")[-1].lower()
            for r in buys[:20]:
                _render_card(r, key_prefix=kp)
            if watch and not buy_only:
                st.markdown("###### 👁 Worth watching")
                for r in watch[:20]:
                    _render_card(r, key_prefix=kp)
