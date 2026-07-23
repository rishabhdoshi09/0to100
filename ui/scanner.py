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
              "PreBreakout": "#facc15", "Pullback": "#34d399"}


@st.cache_data(ttl=60, show_spinner=False)
def _live_quotes(symbols_key: str) -> dict:
    """Live quotes — Kite → NSE → Google (unified source). Cached 1 min;
    STRICT: no cap — every displayed card gets a render-time price
    (Kite bulk quote is 500 symbols/call)."""
    try:
        from data.live_quotes import get_live_quotes
        return get_live_quotes(symbols_key.split(","))
    except Exception:
        return {}


def _mkt_open() -> bool:
    try:
        from scan.auto_scan import _is_market_hours
        return _is_market_hours()
    except Exception:
        return False


def _apply_live_prices(results: list[dict]) -> None:
    """Overlay current prices at render time — EVERY card, not a top slice.
    Marks each card live/EOD — a stale price must LOOK stale, never
    pretend to be current. Off-hours nothing is 'live' by definition."""
    syms = [r["symbol"] for r in results]
    if not syms:
        return
    live = _live_quotes(",".join(syms))
    is_open = _mkt_open()
    for r in results:
        q = live.get(r["symbol"])
        if not (q and q.get("price")):
            r["live"] = False
            continue
        r["live"] = is_open      # market band → koi card 'live' nahi
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
def _sector_perf() -> list[dict]:
    """Sector performance from the bhav store. Cached 10 min."""
    try:
        from scan.sector_heat import sector_performance
        return sector_performance()
    except Exception:
        return []


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
    # Live quotes — unified source (Kite → NSE → Google), shows which won
    try:
        from data.live_quotes import source_health
        ok, detail = source_health("RELIANCE")
        out.append(("Live quotes", ok, detail))
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


def _data_freshness() -> dict:
    """What data are cards actually built on — for the loud banner."""
    from datetime import date as _date
    info = {"scan_date": None, "is_today": False, "detail": ""}
    try:
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv("RELIANCE")
        if df is not None and len(df):
            last = df.index[-1].date()
            info["scan_date"] = last
            info["is_today"] = (last == _date.today())
    except Exception:
        pass
    return info


def _render_freshness_banner(results: list[dict]) -> None:
    """
    Loud, unmissable banner separating the two data paths that confused
    the user: displayed PRICE (live via Kite/NSE/Google, unified chain) vs SCAN DATA
    (NSE bhavcopy + today's intraday overlay).
    """
    from datetime import date as _date
    fi = _data_freshness()
    live_cards = sum(1 for r in results[:80] if r.get("live"))
    total_cards = min(80, len(results))

    # Scan-data half
    if fi["is_today"]:
        scan_txt = "🟢 Scan data: <b>aaj ka (live intraday)</b>"
        scan_col = "#00d4a0"
    elif fi["scan_date"]:
        days_old = (_date.today() - fi["scan_date"]).days
        scan_txt = (f"🟡 Scan data: <b>{fi['scan_date'].strftime('%d %b')} close</b> "
                    f"({days_old} din purana — NSE intraday abhi nahi mila)")
        scan_col = "#f59e0b"
    else:
        scan_txt = "🔴 Scan data: <b>load nahi hua</b>"
        scan_col = "#ff4b4b"

    # Displayed-price half
    if total_cards and live_cards >= total_cards * 0.6:
        price_txt = f"🟢 Card prices: <b>LIVE</b> (Kite/NSE/Google, {live_cards}/{total_cards})"
        price_col = "#00d4a0"
    elif live_cards:
        price_txt = f"🟡 Card prices: <b>{live_cards}/{total_cards} live</b>, baaki EOD"
        price_col = "#f59e0b"
    else:
        price_txt = "🔴 Card prices: <b>EOD close</b> (live quotes nahi mile)"
        price_col = "#ff4b4b"

    st.markdown(
        f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
        f"padding:10px 16px;margin-bottom:10px;display:flex;gap:24px;flex-wrap:wrap;"
        f"font-size:.82rem'>"
        f"<span style='color:{scan_col}'>{scan_txt}</span>"
        f"<span style='color:{price_col}'>{price_txt}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if not fi["is_today"]:
        st.caption("💡 NSE intraday snapshot sirf market hours (9:15–4 PM, "
                   "weekday) mein milta hai. Us waqt scan chalao toh signals aaj "
                   "ke move pe banenge. Card ke prices phir bhi Kite/NSE/Google "
                   "(unified chain) se live hote hain.")


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


# ── Trade Ticket (scan → execution, one click) ────────────────────────────────

def _trade_ticket_body(s: dict) -> None:
    """Prefilled order ticket: qty from 1% rule, stop/target from the setup."""
    from execution.trade_executor import place_trade, kite_ready
    from risk.position_sizer import size_position

    live_mode = kite_ready()
    sym = s["symbol"]
    px = float(s.get("price") or 0)
    d_entry = float(s.get("entry") or px)
    d_stop = float(s.get("stop") or round(d_entry * 0.95, 1))
    d_target = float(s.get("target") or round(d_entry * 1.10, 1))

    if live_mode:
        st.markdown("<div style='background:#00d4a018;border:1px solid #00d4a055;"
                    "border-radius:8px;padding:6px 12px;font-size:.8rem;color:#00d4a0'>"
                    "🟢 <b>LIVE</b> — order seedha Zerodha jayega, GTT stop+target ke saath"
                    "</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:#f59e0b18;border:1px solid #f59e0b55;"
                    "border-radius:8px;padding:6px 12px;font-size:.8rem;color:#f59e0b'>"
                    "📝 <b>PAPER</b> — Kite logged-in nahi; trade sirf record hoga, "
                    "paisa nahi lagega</div>", unsafe_allow_html=True)

    st.markdown(f"#### {sym} &nbsp; <span style='font-size:1rem;color:#94a3b8'>"
                f"live ₹{px:,.1f}</span>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        entry_type = st.radio("Order type", ["LIMIT", "MARKET"], horizontal=True,
                              key=f"tt_type_{sym}")
        entry_price = st.number_input("Entry price (₹)", min_value=1.0,
                                      value=float(d_entry), step=0.5,
                                      key=f"tt_entry_{sym}",
                                      disabled=(entry_type == "MARKET"))
        product = st.radio("Product", ["CNC (delivery)", "MIS (intraday)"],
                           horizontal=True, key=f"tt_prod_{sym}")
    with c2:
        cap = st.session_state.get("user_capital")
        ps = size_position(d_entry, d_stop, capital=cap)
        qty = st.number_input("Quantity (1% rule suggest karta hai: "
                              f"{ps['qty']})", min_value=1,
                              value=max(1, int(ps["qty"] or 1)), step=1,
                              key=f"tt_qty_{sym}")
        stop = st.number_input("Stop loss (₹)", min_value=0.1,
                               value=float(d_stop), step=0.5, key=f"tt_stop_{sym}")
        target = st.number_input("Target (₹)", min_value=0.1,
                                 value=float(d_target), step=0.5,
                                 key=f"tt_tgt_{sym}")

    eff_entry = px if entry_type == "MARKET" else entry_price
    deploy = qty * eff_entry
    max_loss = qty * max(0.0, eff_entry - stop)
    reward = qty * max(0.0, target - eff_entry)
    rr = (target - eff_entry) / (eff_entry - stop) if eff_entry > stop else 0

    st.markdown(
        f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:8px;"
        f"padding:10px 14px;font-size:.85rem;color:#c9d1d9;margin:8px 0'>"
        f"💵 Deploy: <b>₹{deploy:,.0f}</b> &nbsp;·&nbsp; "
        f"<span style='color:#ff4b4b'>Max loss: <b>₹{max_loss:,.0f}</b></span> &nbsp;·&nbsp; "
        f"<span style='color:#00d4a0'>Reward: <b>₹{reward:,.0f}</b></span> &nbsp;·&nbsp; "
        f"R:R <b>{rr:.1f}×</b>"
        f"</div>", unsafe_allow_html=True)

    # Portfolio impact — is trade ke BAAD account kaisa dikhega
    try:
        from risk.portfolio_risk import check_new_trade
        pr = check_new_trade(sym, int(qty), float(eff_entry), float(stop))
        _v_col = {"OK": "#00d4a0", "CAUTION": "#f59e0b", "DANGER": "#ff4b4b"}[pr["verdict"]]
        st.markdown(
            f"<div style='font-size:.78rem;color:{_v_col};margin:2px 0'>"
            f"Portfolio after this trade: {pr['n_positions']} positions · "
            f"total open risk {pr['open_risk_pct']:.1f}% · verdict "
            f"<b>{pr['verdict']}</b></div>", unsafe_allow_html=True)
        for w in pr["warnings"]:
            st.markdown(f"<div style='font-size:.76rem;color:#f59e0b'>{w}</div>",
                        unsafe_allow_html=True)
    except Exception:
        pr = {"verdict": "OK"}

    label = ("🚀 Place LIVE order (Zerodha)" if live_mode
             else "📝 Record paper trade")
    if pr["verdict"] == "DANGER":
        st.error("🔴 Portfolio DANGER zone mein hai — yeh trade lene se pehle "
                 "warnings padho. Button phir bhi neeche hai, decision tumhara.")
    if st.button(label, key=f"tt_go_{sym}", type="primary",
                 width="stretch"):
        res = place_trade(symbol=sym, qty=int(qty), entry_type=entry_type,
                          entry_price=float(eff_entry), stop=float(stop),
                          target=float(target),
                          product="CNC" if product.startswith("CNC") else "MIS")
        if res["ok"]:
            st.success(res["message"])
            if res.get("entry_order_id"):
                st.caption(f"Order ID: `{res['entry_order_id']}`"
                           + (f" · GTT ID: `{res['gtt_id']}`" if res.get("gtt_id") else ""))
        else:
            st.error(res["message"])


def _open_trade_ticket(s: dict) -> None:
    if hasattr(st, "dialog"):
        @st.dialog(f"💰 Trade — {s['symbol']}", width="large")
        def _dlg():
            _trade_ticket_body(s)
        _dlg()
    else:
        st.session_state["trade_ticket_row"] = s
        st.rerun()


# ── 🎯 High Conviction strip — evidence tier, priority display ────────────────

def _render_high_conviction(results: list[dict]) -> None:
    """Golden strip of the setups where EVERYTHING measured lines up:
    buy verdict + score ≥75 + backtested edge ≥ +0.10R (tagged by the
    scan pipeline). These deserve the first look — and the autopilot
    fills its daily slots from this same ranking."""
    require_live = _mkt_open()
    hc = sorted([r for r in results if r.get("high_conviction")
                 and (r.get("live") or not require_live)],
                key=lambda r: float(r.get("conviction_rank", 0)),
                reverse=True)[:5]
    if not hc:
        return
    st.markdown(
        "<div style='font-size:.68rem;color:#fbbf24;text-transform:uppercase;"
        "letter-spacing:.15em;margin:6px 0 4px'>🎯 High Conviction — "
        "score + measured edge dono strong, pehle yeh dekho</div>",
        unsafe_allow_html=True)
    for r in hc:
        edge = float(r.get("edge_r") or 0)
        sigs = " · ".join((r.get("signals") or [])[:3])
        live_tag = ("<span style='color:#00d4a0;font-size:.62rem'>● live</span>"
                    if r.get("live") else
                    "<span style='color:#64748b;font-size:.62rem'>EOD</span>")
        st.markdown(
            f"<div style='background:#1a1509;border:1px solid #fbbf2455;"
            f"border-left:4px solid #fbbf24;border-radius:8px;"
            f"padding:8px 14px;margin-bottom:6px;font-size:.82rem;"
            f"color:#e2e8f0'>"
            f"<b style='font-family:JetBrains Mono,monospace'>{r['symbol']}</b> "
            f"₹{float(r.get('price') or 0):,.1f} {live_tag} &nbsp;·&nbsp; "
            f"<span style='color:#fbbf24'>{r.get('verdict')}</span> "
            f"score {float(r.get('score') or 0):.0f} · "
            f"edge <b style='color:#00d4a0'>{edge:+.2f}R</b> (measured)"
            f"<br><span style='font-size:.72rem;color:#94a3b8'>"
            f"{sigs} &nbsp;·&nbsp; entry ₹{float(r.get('entry') or 0):,.1f} / "
            f"stop ₹{float(r.get('stop') or 0):,.1f} / "
            f"target ₹{float(r.get('target') or 0):,.1f}</span></div>",
            unsafe_allow_html=True)


# ── Best Trade hero card ──────────────────────────────────────────────────────

def _pick_best_trade(results: list[dict]) -> dict | None:
    """The single most actionable setup: buy verdict + non-negative measured
    edge + live price not broken below entry. Results are already sorted
    verdict-first + edge-weighted, so the first qualifier IS the best.

    STRICT: market hours mein bina live-verified price ke koi card
    'THE trade' nahi ban sakta — unverified hero = stale hero."""
    require_live = _mkt_open()
    for r in results[:25]:
        if r.get("verdict") not in ("STRONG BUY", "BUY"):
            continue
        if (r.get("edge_r") is not None) and r["edge_r"] < 0:
            continue
        if require_live and not r.get("live"):
            continue
        return r
    return None


def _render_best_trade(results: list[dict]) -> None:
    best = _pick_best_trade(results)
    if not best:
        return
    from risk.position_sizer import size_position

    sym = best["symbol"]
    chg = best.get("change_pct", 0)
    chg_col = "#00d4a0" if chg >= 0 else "#ff4b4b"
    arrow = "▲" if chg >= 0 else "▼"
    label, vcol = _VERDICT_STYLE.get(best["verdict"], _VERDICT_STYLE["WATCH"])

    checks = (best.get("checks") or [f"✓ {r}" for r in best.get("reasons", [])])[:3]
    checks_html = "".join(
        f"<div style='font-size:.8rem;color:"
        f"{'#00d4a0' if c.startswith('✓') else '#f59e0b' if c.startswith('⚠') else '#94a3b8'};"
        f"margin:2px 0'>{c}</div>" for c in checks)

    edge = best.get("edge_r")
    edge_html = (f"<span style='color:#00d4a0;font-size:.75rem;font-weight:700'>"
                 f"🎯 Measured edge {edge:+.2f}R/trade</span>"
                 if edge is not None and edge >= 0.05 else "")

    cap = st.session_state.get("user_capital")
    ps = size_position(float(best.get("entry") or 0), float(best.get("stop") or 0),
                       capital=cap)
    size_line = (f"📏 <b>{ps['qty']} shares</b> = ₹{ps['invested']:,.0f} · "
                 f"max loss ₹{ps['max_loss']:,.0f}" if ps["qty"] else "")

    spark = _sparkline_svg(sym, float(best.get("entry") or 0))

    hc1, hc2 = st.columns([4.2, 1])
    with hc1:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#0d1f2d,#0d1421);"
            f"border:1px solid {vcol}66;border-radius:12px;padding:16px 20px;"
            f"margin-bottom:4px'>"
            f"<div style='font-size:.68rem;color:{vcol};text-transform:uppercase;"
            f"letter-spacing:.15em;margin-bottom:6px'>⭐ Aaj ka Best Trade</div>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"  <div>"
            f"    <span style='color:#e6edf3;font-weight:800;font-size:1.35rem;"
            f"      font-family:JetBrains Mono,monospace'>{sym}</span>"
            f"    <span style='color:#e2e8f0;font-size:1.05rem;margin-left:14px'>"
            f"      ₹{best['price']:,.1f}</span>"
            f"    <span style='color:{chg_col};font-size:.9rem;margin-left:8px;"
            f"      font-weight:700'>{arrow}{abs(chg):.1f}%</span>"
            f"  </div>"
            f"  <div style='display:flex;align-items:center;gap:14px'>{spark}"
            f"    <span style='background:{vcol}22;border:1px solid {vcol};"
            f"      border-radius:8px;padding:5px 14px;font-size:.8rem;"
            f"      font-weight:800;color:{vcol}'>{label}</span>"
            f"  </div>"
            f"</div>"
            f"<div style='margin-top:8px'>{checks_html}</div>"
            f"<div style='margin-top:8px;font-size:.85rem;color:#c9d1d9'>"
            f"Entry <b style='color:#f59e0b'>₹{best['entry']:,.0f}</b> · "
            f"Stop <b style='color:#ff4b4b'>₹{best['stop']:,.0f}</b> · "
            f"Target <b style='color:#00d4a0'>₹{best['target']:,.0f}</b> · "
            f"Reward <b>{best.get('rr', 0):.1f}×</b>"
            + (f" &nbsp;&nbsp;{edge_html}" if edge_html else "")
            + f"</div>"
            + (f"<div style='margin-top:4px;font-size:.8rem;color:#7dd3fc'>{size_line}</div>"
               if size_line else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    with hc2:
        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        if st.button("💰 Trade Now", key="hero_trade", type="primary",
                     width="stretch"):
            _open_trade_ticket(best)
        if st.button("Analyse →", key="hero_analyse", width="stretch"):
            st.session_state["sidebar_nav"] = "Terminal"
            st.session_state["terminal_symbol"] = sym
            st.rerun()


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
    # Measured edge badge — backtest verdict in one glance, no table needed
    edge = s.get("edge_r")
    if edge is not None:
        if edge >= 0.25:
            e_col, e_txt = "#00d4a0", f"🎯 Proven edge {edge:+.2f}R/trade"
        elif edge >= 0.05:
            e_col, e_txt = "#7dd3fc", f"🎯 Mild edge {edge:+.2f}R/trade"
        elif edge >= -0.02:
            e_col, e_txt = "#8892a4", f"🎯 No edge yet ({edge:+.2f}R)"
        else:
            e_col, e_txt = "#ff4b4b", f"🎯 Negative edge {edge:+.2f}R — skip"
        chips += (f"<span style='background:{e_col}18;border:1px solid {e_col}44;"
                  f"border-radius:5px;padding:2px 8px;font-size:.68rem;"
                  f"color:{e_col};font-weight:600'>{e_txt}</span>")

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
        if st.button("💰 Trade", key=f"trade_{key_prefix}_{s['symbol']}",
                     type="primary", width="stretch"):
            _open_trade_ticket(s)
        if st.button("Analyse →", key=f"scan_{key_prefix}_{s['symbol']}", width="stretch"):
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
            f"Poora NSE — {len(universe)} stocks (warm ~1 min · pehli baar ~3-4 min)": universe,
        }
        scope_label = st.selectbox("Kahan scan karein", list(scope_options.keys()),
                                   index=0, key="scanner_scope")
        scan_syms = scope_options[scope_label]
    with c2:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("🔍 Scan Market", key="scanner_run",
                                 type="primary", width="stretch")
    with c3:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        auto_on = st.toggle("Auto-refresh (15 min)", value=is_auto_enabled(),
                            key="scanner_auto",
                            help="On: system market hours mein khud scan karta rahega. "
                                 "Off: sirf tumhare button dabane par scan hoga.")
        set_auto_enabled(auto_on)

    # ── Signal accuracy (walk-forward backtest on real NSE data) ─────────────
    try:
        from scan.signal_backtest import get_state as _btgs
        _bt_running_now = _btgs()["running"]
    except Exception:
        _bt_running_now = False
    with st.expander("📊 Signal accuracy — kaunsa signal kitna sahi (backtest)",
                     expanded=_bt_running_now or st.session_state.get("bt_open", False)):
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
                             width="stretch"):
                    st.session_state["bt_open"] = True
                    run_in_background()
                    st.rerun()
                if _bt_state["running"]:
                    _pct = (_bt_state["progress"] / _bt_state["total"]
                            if _bt_state["total"] else 0.0)
                    st.progress(_pct, text=f"⏳ {_bt_state['progress']}/"
                                           f"{_bt_state['total']} stocks…")
                    # Live progress — page khud refresh hota rahega
                    try:
                        from streamlit_autorefresh import st_autorefresh
                        st_autorefresh(interval=2500, limit=300, key="bt_poll")
                    except Exception:
                        if st.button("⟳ Progress dekho", key="bt_poll_btn"):
                            st.rerun()
                elif st.session_state.pop("bt_open", False):
                    st.success("✅ Backtest complete — table update ho gayi")
            with bc2:
                st.caption("**Yeh data ab khud kaam karta hai — tumhe kuch nahi "
                           "karna:** har card pe 🎯 edge badge, negative-edge "
                           "combos auto-demote (Buy → Watch), proven-edge setups "
                           "sabse upar, aur backtest roz raat khud refresh hota "
                           "hai. Neeche ki table sirf transparency ke liye hai.")
            if _rep:
                # ── 🧭 Trading playbook — backtest as ACTION ──────────────
                try:
                    from scan.signal_backtest import trading_playbook
                    _pb = trading_playbook()
                except Exception:
                    _pb = None
                if _pb:
                    _reg = _pb["regime"]
                    _reg_col = {"BULL": "#00d4a0", "BEAR": "#ff4b4b",
                                "CHOP": "#f59e0b"}.get(_reg, "#8892a4")
                    _best_txt = " · ".join(
                        f"<b>{SIGNAL_META.get(b['signal'], (b['signal'],))[0]}</b> "
                        f"{b['expectancy_r']:+.2f}R"
                        + (" (is regime mein)" if b["basis"] == "regime" else "")
                        for b in _pb["best"]) or "koi positive-edge signal nahi"
                    _avoid_txt = ", ".join(
                        SIGNAL_META.get(k, (k,))[0] for k in _pb["avoid"])
                    st.markdown(
                        f"<div style='background:#0d1421;border:1px solid "
                        f"{_reg_col}55;border-left:4px solid {_reg_col};"
                        f"border-radius:8px;padding:10px 14px;margin-bottom:8px;"
                        f"font-size:.8rem;color:#c9d1d9'>"
                        f"🧭 <b style='color:{_reg_col}'>Aaj ka tape: {_reg}"
                        f"</b><br>Isme sabse zyada earn: {_best_txt}"
                        + (f"<br>❌ Avoid (proven losers): {_avoid_txt}"
                           if _avoid_txt else "")
                        + "</div>",
                        unsafe_allow_html=True)
                    _tsw = _pb.get("target_sweep") or {}
                    if _tsw:
                        _rec = _pb.get("recommended_target_pct")
                        _bits = []
                        for _lbl in sorted(_tsw):
                            _t = _tsw[_lbl]
                            _hot = (_rec is not None
                                    and _lbl == f"+{_rec:.0f}%")
                            _bits.append(
                                (f"<b style='color:#fbbf24'>{_lbl}: "
                                 if _hot else f"{_lbl}: ")
                                + f"{_t['expectancy_r']:+.2f}R · "
                                  f"{_t['hit_rate']:.0f}% hit"
                                + ("</b> ⭐" if _hot else ""))
                        st.markdown(
                            "<div style='font-size:.74rem;color:#94a3b8;"
                            "margin-bottom:8px'>🎯 Target geometry (same "
                            "stops, measured): " + " &nbsp;|&nbsp; ".join(_bits)
                            + " — ⭐ wala Autopilot ke target % ke liye "
                              "evidence-backed choice hai</div>",
                            unsafe_allow_html=True)

                _vchip = {"PROVEN": "🟢 PROVEN", "POSITIVE": "🟢 positive",
                          "NEUTRAL": "🟡 neutral", "LOSER": "🔴 LOSER",
                          "THIN": "⚪ thin"}
                rows = []
                for key, s in sorted(_rep.get("signals", {}).items(),
                                     key=lambda kv: -kv[1].get("expectancy_r", 0)):
                    if s.get("closed", 0) < 10:
                        continue
                    label = SIGNAL_META.get(key, (key,))[0]
                    _ci = s.get("wr_ci_pp")
                    _br = s.get("by_regime") or {}

                    def _rexp(reg):
                        d = _br.get(reg)
                        return (f"{d['expectancy_r']:+.2f}R"
                                if d and d.get("trades", 0) >= 20 else "—")
                    rows.append({
                        "Signal": label,
                        "Verdict": _vchip.get(s.get("verdict", ""), ""),
                        "Win rate": f"{s['win_rate']:.0f}%"
                                    + (f" ±{_ci:.0f}" if _ci else ""),
                        "Expectancy": f"{s['expectancy_r']:+.2f}R",
                        "Bull": _rexp("BULL"),
                        "Chop": _rexp("CHOP"),
                        "Bear": _rexp("BEAR"),
                        "Trades": s["closed"],
                    })
                if rows:
                    st.dataframe(rows, width="stretch", hide_index=True)
                    st.caption(f"Last run: {_rep.get('generated_at')} · "
                               f"{_rep.get('symbols')} stocks · "
                               f"{_rep.get('horizon_days')}-day horizon · "
                               f"± = 95% confidence · regime columns 20+ "
                               f"trades pe hi number dikhate hain")
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
    _apply_live_prices(results)   # Kite/NSE/Google live overlay at render time

    # ── Loud data-freshness banner (the two paths, explicit) ──────────────────
    if results:
        fb1, fb2 = st.columns([5, 1])
        with fb2:
            if st.button("🔄 Live data", key="refresh_live",
                         width="stretch",
                         help="Live prices (Kite/NSE/Google) + NSE intraday snapshot dubara laao"):
                _live_quotes.clear()
                _health.clear()
                try:
                    from data.nse_live import apply_live_to_store
                    apply_live_to_store()
                except Exception:
                    pass
                st.rerun()
        with fb1:
            _render_freshness_banner(results)
    if last_ts:
        fresh_note = f"Last scan: **{_freshness(last_ts)}** · {universe_size} stocks covered"
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

    # ── ⭐ Best Trade — sabse pehle, sabse bada, ek click pe trade ────────────
    _render_best_trade(results)

    # ── 🎯 High Conviction — evidence tier, priority mein sabse upar ─────────
    _render_high_conviction(results)

    # ── Fallback trade panel (older Streamlit without st.dialog) ──────────────
    if not hasattr(st, "dialog") and st.session_state.get("trade_ticket_row"):
        with st.container(border=True):
            tc1, tc2 = st.columns([6, 1])
            with tc2:
                if st.button("✕ Band karo", key="tt_close"):
                    st.session_state.pop("trade_ticket_row", None)
                    st.rerun()
            _trade_ticket_body(st.session_state["trade_ticket_row"])

    # ── 📍 My book — shared panel (same truth as My Portfolio page) ──────────
    try:
        from ui.positions_panel import render_open_positions, render_trade_journal
        from risk.position_manager import review_positions as _rp
        _n_pos = len(_rp())
        if _n_pos:
            with st.expander(f"📍 Meri positions ({_n_pos} open)", expanded=True):
                render_open_positions()
        render_trade_journal()
    except Exception:
        pass

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

    # ── 📊 Sector Pulse — kaunsa sector bhaag raha hai + kitne setups ─────────
    # Annotate every result with its sector (cheap lookup) for counting/filter
    try:
        from scan.sector_heat import sector_of as _sec_of
        for r in results:
            if not r.get("sector"):
                r["sector"] = _sec_of(r["symbol"]) or ""
    except Exception:
        pass
    _sec_counts: dict = {}
    for r in results:
        if r.get("sector"):
            _sec_counts[r["sector"]] = _sec_counts.get(r["sector"], 0) + 1

    _perf = _sector_perf()
    if _perf:
        top5 = _perf[:5]
        worst = _perf[-1] if len(_perf) > 5 else None
        chips = ""
        for p in top5:
            col = "#00d4a0" if p["chg_1d"] >= 0 else "#ff4b4b"
            arrow = "▲" if p["chg_1d"] >= 0 else "▼"
            cnt = _sec_counts.get(p["sector"], 0)
            chips += (f"<span style='background:{col}14;border:1px solid {col}44;"
                      f"border-radius:6px;padding:3px 10px;font-size:.72rem;"
                      f"color:{col};margin-right:6px'>"
                      f"{p['sector']} {arrow}{abs(p['chg_1d']):.1f}%"
                      f"<span style='color:#8892a4'> · 5d {p['chg_5d']:+.1f}%"
                      + (f" · {cnt} setups" if cnt else "")
                      + "</span></span>")
        if worst and worst["chg_1d"] < 0:
            chips += (f"<span style='background:#ff4b4b14;border:1px solid #ff4b4b44;"
                      f"border-radius:6px;padding:3px 10px;font-size:.72rem;"
                      f"color:#ff4b4b'>Sabse weak: {worst['sector']} "
                      f"▼{abs(worst['chg_1d']):.1f}%</span>")
        st.markdown(
            f"<div style='margin:-4px 0 10px 0'>"
            f"<span style='font-size:.7rem;color:#8892a4;margin-right:8px'>"
            f"📊 SECTOR PULSE</span>{chips}</div>",
            unsafe_allow_html=True)
    elif _sec_counts:
        # Fallback: signal counts only (store not warm yet)
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

    # ── Search + sector filter ────────────────────────────────────────────────
    fc1, fc2 = st.columns([2.5, 1.5])
    with fc1:
        q = st.text_input("Filter", placeholder="🔍 Filter by symbol…",
                          key="scanner_filter", label_visibility="collapsed")
    with fc2:
        _sectors_avail = sorted({r["sector"] for r in results if r.get("sector")})
        sec_pick = st.selectbox("Sector", ["Saare sectors"] + _sectors_avail,
                                key="scanner_sector_filter",
                                label_visibility="collapsed")
    if q:
        qq = q.strip().upper()
        results = [r for r in results if qq in r["symbol"]]
    if sec_pick and sec_pick != "Saare sectors":
        results = [r for r in results if r.get("sector") == sec_pick]
        st.caption(f"Sirf **{sec_pick}** ke {len(results)} setups dikh rahe hain")

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
