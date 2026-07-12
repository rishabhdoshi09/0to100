"""
Market Pulse Strip — minimal single-line bar showing NIFTY, BANKNIFTY, VIX, market status.
Moneycontrol-style: clean, no clutter.
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
    """NIFTY, BANKNIFTY, VIX — Kite first (real-time), Google/yfinance backup. Cached 3 min."""
    result: dict = {}

    # Primary: unified index quotes (Kite → Google Finance)
    try:
        from data.live_quotes import get_index_quotes
        q = get_index_quotes(["NIFTY", "BANKNIFTY", "VIX"])
        for key, name in (("nifty", "NIFTY"), ("banknifty", "BANKNIFTY"), ("vix", "VIX")):
            if q.get(name, {}).get("price"):
                result[key] = {"price": q[name]["price"], "chg": q[name]["chg_pct"]}
    except Exception:
        pass

    # Backup: yfinance for anything still missing
    try:
        import yfinance as yf

        def _safe_fetch(ticker_sym: str) -> dict:
            try:
                info = yf.Ticker(ticker_sym).fast_info
                last  = float(getattr(info, "last_price", 0) or 0)
                prev  = float(getattr(info, "previous_close", 0) or 0)
                chg   = ((last - prev) / prev * 100) if prev else 0.0
                return {"price": last, "chg": chg}
            except Exception:
                return {}

        if "nifty" not in result:
            result["nifty"] = _safe_fetch("^NSEI")
        if "banknifty" not in result:
            result["banknifty"] = _safe_fetch("^NSEBANK")
        if "vix" not in result:
            result["vix"] = _safe_fetch("^INDIAVIX")
    except Exception:
        pass
    return result


@st.cache_data(ttl=900, show_spinner=False)
def _cached_regime() -> str:
    try:
        from scan.signal_backtest import current_regime_simple
        return current_regime_simple()
    except Exception:
        return "UNKNOWN"


def render_market_pulse() -> None:
    """
    Render a minimal single-line dark bar:
      ▲ NIFTY 23,719 +0.30%   ▲ BANKNIFTY 54,055 +1.07%   VIX 17.9   🔴 CLOSED
    Nothing else.
    """
    try:
        data    = _fetch_pulse_data()
        is_open = _is_market_open()

        def _arrow(chg: float) -> str:
            return "▲" if chg >= 0 else "▼"

        def _chg_col(chg: float) -> str:
            return "#00d4a0" if chg >= 0 else "#ff4b4b"

        nifty = data.get("nifty", {})
        bank  = data.get("banknifty", {})
        vix   = data.get("vix", {})

        n_px  = nifty.get("price", 0)
        n_chg = nifty.get("chg", 0)
        b_px  = bank.get("price", 0)
        b_chg = bank.get("chg", 0)
        v_px  = vix.get("price", 0)

        sep = "  <span style='color:#2d3748'>|</span>  "

        parts = []
        if n_px:
            parts.append(
                f"<span style='color:{_chg_col(n_chg)}'>{_arrow(n_chg)}</span> "
                f"<span style='color:#e8eaf0;font-weight:600'>NIFTY</span> "
                f"<span style='color:#e8eaf0'>{n_px:,.0f}</span> "
                f"<span style='color:{_chg_col(n_chg)}'>{n_chg:+.2f}%</span>"
            )
        if b_px:
            parts.append(
                f"<span style='color:{_chg_col(b_chg)}'>{_arrow(b_chg)}</span> "
                f"<span style='color:#e8eaf0;font-weight:600'>BANKNIFTY</span> "
                f"<span style='color:#e8eaf0'>{b_px:,.0f}</span> "
                f"<span style='color:{_chg_col(b_chg)}'>{b_chg:+.2f}%</span>"
            )
        if v_px:
            parts.append(
                f"<span style='color:#e8eaf0;font-weight:600'>VIX</span> "
                f"<span style='color:#e8eaf0'>{v_px:.1f}</span>"
            )

        status_html = (
            "<span style='color:#00d4a0;font-weight:700'>🟢 OPEN</span>"
            if is_open else
            "<span style='color:#ff4b4b;font-weight:700'>🔴 CLOSED</span>"
        )
        parts.append(status_html)

        # 🧭 Tape regime — same rule as the backtest buckets (cached 15 min)
        try:
            reg = _cached_regime()
            if reg and reg != "UNKNOWN":
                _rc = {"BULL": "#00d4a0", "BEAR": "#ff4b4b",
                       "CHOP": "#f59e0b"}.get(reg, "#8892a4")
                parts.append(f"<span style='color:{_rc};font-weight:700'>"
                             f"🧭 {reg}</span>")
        except Exception:
            pass
        # 🤖 Autopilot chip — armed state visible on EVERY page
        try:
            from execution.autopilot import get_status as _ap_status
            _s = _ap_status()
            if _s["armed"]:
                _ac = "#00d4a0" if _s["mode"] == "PAPER" else "#ff4b4b"
                parts.append(f"<span style='color:{_ac};font-weight:700'>"
                             f"🤖 {_s['mode']}</span>")
        except Exception:
            pass

        body = sep.join(parts)

        st.markdown(
            f"<div style='"
            f"background:#0d1421;"
            f"border:1px solid #1e293b;"
            f"border-radius:6px;"
            f"padding:6px 14px;"
            f"font-family:\"JetBrains Mono\",monospace;"
            f"font-size:0.75rem;"
            f"margin-bottom:8px;"
            f"line-height:1.6;"
            f"'>{body}</div>",
            unsafe_allow_html=True,
        )

    except Exception:
        st.markdown(
            "<div style='background:#0d1421;border:1px solid #1e293b;border-radius:6px;"
            "padding:6px 14px;font-size:0.75rem;color:#4a5568;"
            "font-family:\"JetBrains Mono\",monospace'>"
            "Market data unavailable</div>",
            unsafe_allow_html=True,
        )
