"""My Holdings — your real Zerodha portfolio with tax timer."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import streamlit as st

try:
    from ui.context import go_to, nav_button
except Exception:
    def go_to(page, symbol=None, jarvis_query=None):
        st.session_state["_nav_pending"] = page
        if symbol:
            st.session_state["selected_symbol"] = symbol
        if jarvis_query:
            st.session_state["jarvis_prefill"] = jarvis_query
        st.rerun()

    def nav_button(label, page, symbol=None, jarvis_query=None, key="", **kwargs):
        if st.button(label, key=key, **kwargs):
            go_to(page, symbol=symbol, jarvis_query=jarvis_query)

# ── Sector map for common NSE symbols ─────────────────────────────────────────
_SECTOR_MAP: dict[str, str] = {
    # Financials
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials",
    "AXISBANK": "Financials", "KOTAKBANK": "Financials", "BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials", "INDUSINDBK": "Financials", "BANDHANBNK": "Financials",
    "FEDERALBNK": "Financials", "IDFCFIRSTB": "Financials", "PNB": "Financials",
    "CANBK": "Financials", "BANKBARODA": "Financials", "MUTHOOTFIN": "Financials",
    "CHOLAFIN": "Financials", "M&MFIN": "Financials", "LICHSGFIN": "Financials",
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "MPHASIS": "IT", "COFORGE": "IT", "PERSISTENT": "IT",
    "OFSS": "IT",
    # Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "IOC": "Energy", "BPCL": "Energy",
    "HINDPETRO": "Energy", "GAIL": "Energy", "PETRONET": "Energy",
    "ADANIGREEN": "Energy", "TATAPOWER": "Energy", "POWERGRID": "Energy",
    "NTPC": "Energy", "ADANITRANS": "Energy",
    # Consumer
    "HINDUNILVR": "Consumer", "ITC": "Consumer", "NESTLEIND": "Consumer",
    "BRITANNIA": "Consumer", "DABUR": "Consumer", "MARICO": "Consumer",
    "GODREJCP": "Consumer", "COLPAL": "Consumer", "EMAMILTD": "Consumer",
    "TATACONSUM": "Consumer", "VBL": "Consumer",
    # Auto
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "ASHOKLEY": "Auto",
    "TVSMOTORS": "Auto", "BALKRISIND": "Auto", "BOSCHLTD": "Auto",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "BIOCON": "Pharma", "AUROPHARMA": "Pharma",
    "TORNTPHARM": "Pharma", "ALKEM": "Pharma", "LALPATHLAB": "Pharma",
    "METROPOLIS": "Pharma",
    # Metals
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "COALINDIA": "Metals", "NMDC": "Metals",
    "JINDALSTEL": "Metals", "SAIL": "Metals", "NATIONALUM": "Metals",
    # Infrastructure
    "LT": "Infrastructure", "ULTRACEMCO": "Infrastructure", "GRASIM": "Infrastructure",
    "SHREECEM": "Infrastructure", "ACC": "Infrastructure", "AMBUJACEMENT": "Infrastructure",
    "ADANIPORTS": "Infrastructure", "ADANIENT": "Infrastructure", "GMRINFRA": "Infrastructure",
    # Telecom
    "BHARTIARTL": "Telecom", "IDEA": "Telecom",
    # Retail / Others
    "TITAN": "Consumer", "DMART": "Retail", "NYKAA": "Retail",
    "ZOMATO": "Internet", "PAYTM": "Internet", "POLICYBZR": "Internet",
}


# ── Demo data shown when Kite is not connected ────────────────────────────────
_DEMO_HOLDINGS: list[dict] = [
    {
        "tradingsymbol": "INFY",
        "quantity": 10,
        "average_price": 1420.0,
        "last_price": 1586.0,
        "pnl": 1660.0,
        "t1_quantity": 0,
        "day_change": 12.5,
        "day_change_percentage": 0.79,
    },
    {
        "tradingsymbol": "HDFCBANK",
        "quantity": 5,
        "average_price": 1650.0,
        "last_price": 1724.0,
        "pnl": 370.0,
        "t1_quantity": 0,
        "day_change": -8.2,
        "day_change_percentage": -0.47,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_days_held(holding: dict[str, Any]) -> int:
    """
    Heuristic: t1_quantity > 0 means bought today (T+1 settlement).
    Otherwise we don't have a real buy date from Kite holdings API,
    so we estimate 180 days as a conservative middle-ground.
    Users should track actual buy dates in their journal.
    """
    if holding.get("t1_quantity", 0) > 0:
        return 1
    return 180


def _tax_badge(days_held: int, unrealised_pnl: float) -> str:
    if days_held < 330:
        color = "#f59e0b"
        label = f"Short-term &nbsp;(20% tax)"
        return (
            f"<span style='background:rgba(245,158,11,.12);color:{color};"
            f"border:1px solid {color}44;border-radius:5px;"
            f"padding:2px 8px;font-size:.68rem;font-weight:600'>{label}</span>"
        )
    elif days_held <= 364:
        days_left = 365 - days_held
        saving = unrealised_pnl * 0.075 if unrealised_pnl > 0 else 0
        color = "#facc15"
        label = (
            f"⏰ {days_left} day{'s' if days_left != 1 else ''} to long-term"
            + (f" — save ₹{saving:,.0f}" if saving > 0 else "")
        )
        return (
            f"<span style='background:rgba(250,204,21,.10);color:{color};"
            f"border:1px solid {color}44;border-radius:5px;"
            f"padding:2px 8px;font-size:.68rem;font-weight:600'>{label}</span>"
        )
    else:
        color = "#00d4a0"
        label = "Long-term &nbsp;(12.5% tax)"
        return (
            f"<span style='background:rgba(0,212,160,.10);color:{color};"
            f"border:1px solid {color}44;border-radius:5px;"
            f"padding:2px 8px;font-size:.68rem;font-weight:600'>{label}</span>"
        )


def _sector_chart(holdings: list[dict]) -> None:
    """Simple horizontal bar chart showing sector weights by current value."""
    sector_value: dict[str, float] = {}
    total = 0.0
    for h in holdings:
        sym = h["tradingsymbol"]
        val = h["quantity"] * h["last_price"]
        sector = _SECTOR_MAP.get(sym, "Other")
        sector_value[sector] = sector_value.get(sector, 0.0) + val
        total += val

    if total == 0:
        return

    sorted_sectors = sorted(sector_value.items(), key=lambda x: x[1], reverse=True)

    st.markdown(
        "<div style='font-size:.72rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.07em;margin:.8rem 0 .4rem'>Sector Concentration</div>",
        unsafe_allow_html=True,
    )
    for sector, val in sorted_sectors:
        pct = val / total * 100
        bar_color = "#00d4ff" if pct < 40 else "#f59e0b"
        st.markdown(
            f"<div style='margin:.25rem 0'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:.74rem;color:#c8cfe0;margin-bottom:2px'>"
            f"<span>{sector}</span><span>{pct:.1f}%</span></div>"
            f"<div style='background:rgba(255,255,255,.07);border-radius:4px;height:6px'>"
            f"<div style='width:{pct:.1f}%;background:{bar_color};"
            f"border-radius:4px;height:6px'></div></div></div>",
            unsafe_allow_html=True,
        )


# ── Main render ───────────────────────────────────────────────────────────────

def render_real_holdings() -> None:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:4px'>"
        "MY HOLDINGS</h2>"
        "<p style='color:#4a5568;font-size:.76rem;margin-bottom:1.2rem'>"
        "Your real Zerodha portfolio — with a tax-saving timer.</p>",
        unsafe_allow_html=True,
    )

    # ── Try loading real Kite holdings ────────────────────────────────────────
    holdings: list[dict] = []
    kite_connected = False
    kite_error = ""

    try:
        from data.kite_client import KiteClient
        kc = KiteClient()
        holdings = kc.kite.holdings()
        kite_connected = True
    except Exception as exc:
        kite_error = str(exc)

    is_demo = False
    if not kite_connected or not holdings:
        # Show friendly prompt
        st.markdown(
            "<div style='background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.3);"
            "border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.2rem'>"
            "<span style='color:#f59e0b;font-weight:700;font-size:.85rem'>"
            "Connect your Zerodha account to see real holdings.</span><br>"
            "<span style='color:#8892a4;font-size:.78rem'>"
            "Add <code>KITE_ACCESS_TOKEN</code> to your <code>.env</code> file, "
            "then run <code>python main.py login</code> each morning.</span></div>",
            unsafe_allow_html=True,
        )
        if not kite_connected:
            st.caption(f"Demo mode — showing sample data. (Kite error: {kite_error})")
        else:
            st.caption("Demo mode — no holdings found in your Zerodha account.")
        holdings = _DEMO_HOLDINGS
        is_demo = True

    if not holdings:
        st.info("No holdings found.")
        return

    # ── Portfolio summary header ──────────────────────────────────────────────
    total_invested   = sum(h["quantity"] * h["average_price"] for h in holdings)
    total_current    = sum(h["quantity"] * h["last_price"] for h in holdings)
    total_pnl        = total_current - total_invested
    total_pnl_pct    = (total_pnl / total_invested * 100) if total_invested else 0.0
    pnl_color        = "#00d4a0" if total_pnl >= 0 else "#ff4466"
    pnl_sign         = "+" if total_pnl >= 0 else ""

    # Count holdings approaching LTCG
    approaching_ltcg = sum(
        1 for h in holdings
        if 330 <= _estimate_days_held(h) <= 364
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Invested", f"₹{total_invested:,.0f}")
    s2.metric("Current Value", f"₹{total_current:,.0f}")
    s3.metric(
        "Total P&L",
        f"{pnl_sign}₹{abs(total_pnl):,.0f}",
        delta=f"{pnl_sign}{total_pnl_pct:.1f}%",
        delta_color="normal",
    )
    s4.metric(
        "Near Long-Term Tax",
        str(approaching_ltcg),
        help="Holdings within 35 days of qualifying for the lower 12.5% long-term tax rate.",
    )

    if approaching_ltcg > 0:
        st.markdown(
            f"<div style='background:rgba(250,204,21,.08);border:1px solid rgba(250,204,21,.25);"
            f"border-radius:8px;padding:.6rem 1rem;margin:.5rem 0;font-size:.78rem;color:#facc15'>"
            f"⏰ You have {approaching_ltcg} holding(s) close to qualifying for the lower "
            f"long-term capital gains tax (12.5%). Hold a little longer to save money!</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Top portfolio review button ───────────────────────────────────────────
    nav_button(
        "🤖 Review my portfolio with JARVIS",
        "🤖  JARVIS",
        jarvis_query="Review my current portfolio holdings and give me a risk assessment",
        key="holdings_jarvis_review",
    )

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    # ── Holding cards ─────────────────────────────────────────────────────────
    for h in holdings:
        sym        = h["tradingsymbol"]
        qty        = h["quantity"]
        avg_px     = h["average_price"]
        last_px    = h["last_price"]
        pnl        = h.get("pnl", qty * (last_px - avg_px))
        day_chg    = h.get("day_change", 0.0)
        day_chg_p  = h.get("day_change_percentage", 0.0)

        invested   = qty * avg_px
        cur_val    = qty * last_px
        pnl_pct    = (pnl / invested * 100) if invested else 0.0

        pnl_color  = "#00d4a0" if pnl >= 0 else "#ff4466"
        pnl_sign   = "+" if pnl >= 0 else ""
        day_color  = "#00d4a0" if day_chg >= 0 else "#ff4466"
        day_sign   = "+" if day_chg >= 0 else ""

        days_held  = _estimate_days_held(h)
        badge_html = _tax_badge(days_held, pnl)

        sector     = _SECTOR_MAP.get(sym, "Other")

        card = (
            f"<div style='background:rgba(255,255,255,.03);"
            f"border:1px solid rgba(255,255,255,.08);"
            f"border-radius:10px;padding:14px 18px;margin-bottom:10px'>"
            # Header
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
            f"<div>"
            f"<span style='color:#e8eaf0;font-size:1.05rem;font-weight:700;"
            f"font-family:JetBrains Mono,monospace'>{sym}</span>"
            f"<span style='color:#4a5568;font-size:.72rem;margin-left:.6rem'>{sector}</span>"
            f"</div>"
            f"<div>{badge_html}</div>"
            f"</div>"
            # Qty × avg price row
            f"<div style='font-size:.75rem;color:#6b7280;margin-bottom:6px'>"
            f"{qty} shares × ₹{avg_px:,.2f} = "
            f"<span style='color:#8892a4'>₹{invested:,.0f} invested</span>"
            f"</div>"
            # Current value + P&L
            f"<div style='display:flex;gap:1.5rem;align-items:baseline;flex-wrap:wrap'>"
            f"<div>"
            f"<span style='color:#8892a4;font-size:.7rem'>Current value&nbsp;</span>"
            f"<span style='color:#e8eaf0;font-size:1rem;font-weight:700;"
            f"font-family:JetBrains Mono,monospace'>₹{cur_val:,.0f}</span>"
            f"</div>"
            f"<div>"
            f"<span style='color:#8892a4;font-size:.7rem'>P&amp;L&nbsp;</span>"
            f"<span style='color:{pnl_color};font-size:.92rem;font-weight:700'>"
            f"{pnl_sign}₹{abs(pnl):,.0f} ({pnl_sign}{abs(pnl_pct):.1f}%)</span>"
            f"</div>"
            f"<div>"
            f"<span style='color:#8892a4;font-size:.7rem'>Today&nbsp;</span>"
            f"<span style='color:{day_color};font-size:.82rem'>"
            f"{day_sign}₹{abs(day_chg):,.2f} ({day_sign}{abs(day_chg_p):.2f}%)</span>"
            f"</div>"
            f"</div>"
            f"</div>"
        )
        st.markdown(card, unsafe_allow_html=True)

        # Per-holding action buttons
        hact1, hact2 = st.columns(2)
        with hact1:
            nav_button(
                "⚡ Trade",
                "⚡  Terminal",
                symbol=sym,
                key=f"holding_trade_{sym}",
                use_container_width=True,
            )
        with hact2:
            nav_button(
                "🤖 Ask JARVIS",
                "🤖  JARVIS",
                jarvis_query=f"I hold {sym}. Current P&L is {pnl_pct:.1f}%. Should I hold, add, or exit?",
                key=f"holding_jarvis_{sym}",
                use_container_width=True,
            )

    # ── Sector concentration (3+ holdings) ───────────────────────────────────
    if len(holdings) >= 3:
        _sector_chart(holdings)

    st.divider()

    # ── Open positions (intraday / F&O) ───────────────────────────────────────
    positions: list[dict] = []
    if kite_connected:
        try:
            from data.kite_client import KiteClient
            kc2 = KiteClient()
            net_pos = kc2.kite.positions().get("net", [])
            positions = [p for p in net_pos if p.get("quantity", 0) != 0]
        except Exception:
            pass

    if positions:
        st.markdown(
            "<div style='font-size:.72rem;color:#8892a4;text-transform:uppercase;"
            "letter-spacing:.07em;margin-bottom:.6rem'>Open Positions (Intraday / F&O)</div>",
            unsafe_allow_html=True,
        )
        for pos in positions:
            sym_p  = pos.get("tradingsymbol", "")
            qty_p  = pos.get("quantity", 0)
            pnl_p  = pos.get("pnl", 0.0)
            pc     = "#00d4a0" if pnl_p >= 0 else "#ff4466"
            ps     = "+" if pnl_p >= 0 else ""
            st.markdown(
                f"<div style='background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);"
                f"border-radius:8px;padding:.55rem 1rem;margin-bottom:.4rem;font-size:.82rem;"
                f"display:flex;justify-content:space-between'>"
                f"<span style='color:#e8eaf0;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>{sym_p}</span>"
                f"<span style='color:#8892a4'>Qty: {qty_p}</span>"
                f"<span style='color:{pc}'>{ps}₹{abs(pnl_p):,.2f} P&amp;L</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if is_demo:
        st.caption(
            "This is demo data. Connect your Zerodha account to see your real portfolio."
        )
    else:
        st.caption("Holdings synced from Zerodha. Tax estimates are approximate — consult a tax adviser.")
