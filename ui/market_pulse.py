"""
Market Pulse Strip — horizontal ticker bar for NIFTY, BANKNIFTY, VIX, A/D, Market Status.
Call render_market_pulse() at the top of any Streamlit page.
"""
from __future__ import annotations

import streamlit as st
from datetime import datetime
import pytz


def _is_market_open() -> bool:
    """Return True if NSE is currently open (9:15–15:30 IST, Mon–Fri)."""
    try:
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception:
        return False


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_pulse_data() -> dict:
    """Fetch NIFTY, BANKNIFTY, VIX via yfinance. Cached 3 min."""
    import yfinance as yf

    result: dict = {}

    def _safe_fetch(ticker_sym: str) -> dict:
        try:
            info = yf.Ticker(ticker_sym).fast_info
            last  = float(getattr(info, "last_price", 0) or 0)
            prev  = float(getattr(info, "previous_close", 0) or 0)
            chg   = ((last - prev) / prev * 100) if prev else 0.0
            return {"price": last, "prev": prev, "chg": chg}
        except Exception:
            return {}

    result["nifty"]    = _safe_fetch("^NSEI")
    result["banknifty"] = _safe_fetch("^NSEBANK")
    result["vix"]      = _safe_fetch("^INDIAVIX")
    return result


def _advance_decline_snippet() -> str:
    """Try to grab today's advance/decline count from market_breadth data."""
    try:
        from ui.market_breadth import _compute_breadth
        # _compute_breadth returns a DataFrame; grab the latest row
        bd = _compute_breadth()
        if bd is not None and not bd.empty:
            latest = bd.iloc[-1]
            adv = int(latest.get("advances", 0))
            dec = int(latest.get("declines", 0))
            if adv or dec:
                color = "#00d4a0" if adv > dec else "#ff4b4b"
                return (
                    f"<span style='color:{color}'>"
                    f"A/D {adv}↑/{dec}↓"
                    f"</span>"
                )
    except Exception:
        pass
    return ""


def render_market_pulse() -> None:
    """Render a single-line dark strip with NIFTY / BANKNIFTY / VIX / market status."""
    try:
        data   = _fetch_pulse_data()
        is_open = _is_market_open()

        def _arrow(chg: float) -> str:
            return "▲" if chg >= 0 else "▼"

        def _col(chg: float) -> str:
            return "#00d4a0" if chg >= 0 else "#ff4b4b"

        # NIFTY
        nifty  = data.get("nifty", {})
        n_px   = nifty.get("price", 0)
        n_chg  = nifty.get("chg", 0)

        # BANKNIFTY
        bank   = data.get("banknifty", {})
        b_px   = bank.get("price", 0)
        b_chg  = bank.get("chg", 0)

        # VIX
        vix    = data.get("vix", {})
        v_px   = vix.get("price", 0)
        v_chg  = vix.get("chg", 0)
        v_dir  = "↑" if v_chg >= 0 else "↓"

        # Market status badge
        status_html = (
            "<span style='color:#00d4a0;font-weight:700'>🟢 MARKET OPEN</span>"
            if is_open else
            "<span style='color:#ff4b4b'>🔴 MARKET CLOSED</span>"
        )

        # Advance/Decline (optional)
        ad_html = _advance_decline_snippet()
        ad_sep  = f"  <span style='color:#2d3748'>|</span>  {ad_html}" if ad_html else ""

        # Compose strip
        parts = []
        if n_px:
            parts.append(
                f"<span style='color:{_col(n_chg)}'>{_arrow(n_chg)}</span> "
                f"<span style='color:#e8eaf0;font-weight:600'>NIFTY</span> "
                f"<span style='color:#e8eaf0'>{n_px:,.0f}</span> "
                f"<span style='color:{_col(n_chg)}'>{n_chg:+.2f}%</span>"
            )
        if b_px:
            parts.append(
                f"<span style='color:{_col(b_chg)}'>{_arrow(b_chg)}</span> "
                f"<span style='color:#e8eaf0;font-weight:600'>BANKNIFTY</span> "
                f"<span style='color:#e8eaf0'>{b_px:,.0f}</span> "
                f"<span style='color:{_col(b_chg)}'>{b_chg:+.2f}%</span>"
            )
        if v_px:
            v_col = "#ff4b4b" if v_chg >= 0 else "#00d4a0"  # rising VIX = bad
            parts.append(
                f"<span style='color:#e8eaf0;font-weight:600'>VIX</span> "
                f"<span style='color:{v_col}'>{v_px:.1f} {v_dir}</span>"
            )

        sep = "  <span style='color:#2d3748'>|</span>  "
        body = sep.join(parts)
        if ad_html:
            body += sep + ad_html

        st.markdown(
            f"""<div style='
                background:#0d1421;
                border:1px solid #1e293b;
                border-radius:6px;
                padding:6px 14px;
                font-family:"JetBrains Mono",monospace;
                font-size:0.75rem;
                margin-bottom:8px;
                display:flex;
                align-items:center;
                gap:0;
                flex-wrap:wrap;
                line-height:1.6;
            '>{body}{sep}{status_html}</div>""",
            unsafe_allow_html=True,
        )

    except Exception:
        st.markdown(
            "<div style='background:#0d1421;border:1px solid #1e293b;border-radius:6px;"
            "padding:6px 14px;font-size:0.75rem;color:#4a5568;font-family:\"JetBrains Mono\",monospace'>"
            "Market data unavailable</div>",
            unsafe_allow_html=True,
        )
