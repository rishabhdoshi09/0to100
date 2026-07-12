"""
ui/command_center.py — Moneycontrol-style Command Center

Clean, familiar layout:
  Section A — Market Status Card (regime + Nifty + sector summary + action tip)
  Section B — Today's Setups    (plain stock cards like MC Top Gainers)
  Section C — JARVIS Market Brief (collapsible expander, default collapsed)

No badge rows, no score boxes, no regime pulse column, no metric grids.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark card style (consistent with overall theme)
# ---------------------------------------------------------------------------
_CARD = (
    "background:#0d1421;border:1px solid #1e293b;border-radius:12px;"
    "padding:1.1rem 1.25rem;margin-bottom:.85rem"
)

# ---------------------------------------------------------------------------
# Regime helpers
# ---------------------------------------------------------------------------

def _load_regime() -> dict:
    """Return regime dict; never raises."""
    try:
        from core.regime_engine import compute_regime
        r = compute_regime()
        return r.to_dict() if hasattr(r, "to_dict") else {}
    except Exception as exc:
        logger.debug("regime unavailable: %s", exc)
        return {}


def _regime_summary(regime: dict) -> tuple[str, str, str]:
    """
    Returns (headline_label, action_tip, color).
    e.g. ("🟢 Market: Bullish — Good day to trade", "Favour breakout entries.", "#00d4a0")
    """
    name = regime.get("market_regime", "CHOPPY")
    _MAP = {
        "TRENDING_BULL": (
            "🟢 Market: Bullish — Good day to trade",
            "Favour breakout and momentum entries. Size up to 1.2× normal.",
            "#00d4a0",
        ),
        "EXPANSION": (
            "🟢 Market: Expanding — Maximum aggression mode",
            "All breakout plays valid. Trail stops — let winners run.",
            "#00d4a0",
        ),
        "CHOPPY": (
            "🟡 Market: Choppy — Trade carefully today",
            "Take smaller positions than usual. Prefer mean-reversion over breakouts.",
            "#f59e0b",
        ),
        "COMPRESSION": (
            "⚪ Market: Sideways — Wait for breakouts",
            "Set alerts for volume expansion. No new entries until price moves.",
            "#8892a4",
        ),
        "DISTRIBUTION": (
            "🟠 Market: Distribution — Tighten stops",
            "Institutions are selling. Reduce longs and tighten stops.",
            "#fb923c",
        ),
        "TRENDING_BEAR": (
            "🔴 Market: Bearish — Stay defensive",
            "Cash or short bias. No new long breakout trades.",
            "#ff4b4b",
        ),
    }
    label, tip, color = _MAP.get(
        name,
        (
            "🟡 Market: Choppy — Trade carefully today",
            "Take smaller positions than usual.",
            "#f59e0b",
        ),
    )
    return label, tip, color


def _sector_line(regime: dict) -> str:
    """One plain sentence about leading/lagging sectors."""
    leading = regime.get("leading_sectors", []) or []
    lagging = regime.get("lagging_sectors", []) or []
    parts = []
    if leading:
        parts.append(
            f"{', '.join(leading[:3])} {'is' if len(leading) == 1 else 'are'} leading"
        )
    if lagging:
        parts.append(
            f"{', '.join(lagging[:3])} {'is' if len(lagging) == 1 else 'are'} weak"
        )
    return ".  ".join(parts) + "." if parts else ""


@st.cache_data(ttl=180, show_spinner=False)
def _nifty_change() -> tuple[float, float]:
    """Return (nifty_price, nifty_chg_pct). Falls back to zeros."""
    try:
        import yfinance as yf
        fi   = yf.Ticker("^NSEI").fast_info
        px   = float(getattr(fi, "last_price", 0) or 0)
        prev = float(getattr(fi, "previous_close", 0) or 0)
        chg  = ((px - prev) / prev * 100) if prev else 0.0
        return px, chg
    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Section A — Market Status Card
# ---------------------------------------------------------------------------

def _render_market_status_card(regime: dict) -> None:
    """
    One clean card:
      🟡 Market: Choppy — Trade carefully today        Nifty ▲0.27%
      IT and Realty are leading. Metals are weak.
      Take smaller positions than usual.
    """
    label, tip, color = _regime_summary(regime)
    sector_line        = _sector_line(regime)

    try:
        nifty_px, nifty_chg = _nifty_change()
        arrow   = "▲" if nifty_chg >= 0 else "▼"
        n_color = "#00d4a0" if nifty_chg >= 0 else "#ff4b4b"
        nifty_html = (
            f"<span style='font-size:.85rem;font-weight:700;color:{n_color}'>"
            f"Nifty {arrow}{abs(nifty_chg):.2f}%</span>"
        ) if nifty_px else ""
    except Exception:
        nifty_html = ""

    sector_html = (
        f"<div style='font-size:.8rem;color:#8892a4;margin-top:.35rem'>{sector_line}</div>"
        if sector_line else ""
    )
    tip_html = (
        f"<div style='font-size:.8rem;color:#c9d1d9;margin-top:.25rem'>{tip}</div>"
    )

    st.markdown(
        f"<div style='{_CARD}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='font-size:1rem;font-weight:700;color:{color}'>{label}</span>"
        f"{nifty_html}"
        f"</div>"
        f"{sector_html}"
        f"{tip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section B — Today's Setups
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _run_command_scan(universe_key: str) -> list[dict]:
    """Run ScanPipeline and return serialisable dicts."""
    try:
        from scan.pipeline import ScanPipeline
        from core.regime_engine import compute_regime
        regime  = compute_regime()
        results = ScanPipeline(max_workers=12, min_quality_score=40).run(
            universe_key.split(","),
            top_n=5,
            regime_state=regime,
            skip_liquidity_filter=True,
        )
        out = []
        for r in results:
            try:
                d = vars(r)
            except TypeError:
                d = r if isinstance(r, dict) else {}
            clean: dict = {}
            for k, v in d.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    clean[k] = v
                elif isinstance(v, list):
                    clean[k] = [str(x) for x in v]
                else:
                    clean[k] = str(v)
            out.append(clean)
        return out
    except Exception as exc:
        logger.warning("command_scan failed: %s", exc)
        return []


def _signal_badge(archetype: str, tier: str) -> tuple[str, str]:
    """Return (label, color) for the signal badge."""
    arch = archetype.upper()
    t    = tier.upper()
    if t in ("ELITE_A_PLUS", "A") or "BUY" in arch or "BREAKOUT" in arch or "MOMENTUM" in arch:
        return "⚡ Buy Setup", "#00d4a0"
    if t == "AVOID" or "SHORT" in arch or "BEAR" in arch:
        return "✗ Avoid", "#ff4b4b"
    return "👁 Watch", "#f59e0b"


def _render_setup_card(setup: dict) -> None:
    """
    Plain Moneycontrol-style stock card:
      WIPRO                          ⚡ Buy Setup
      ₹203.40  ▲ +1.2%
      Entry ₹211 · Stop ₹186 · Target ₹261 · 2.5×
      [View Analysis →]
    """
    symbol    = setup.get("symbol", "?")
    archetype = setup.get("archetype", "").replace("_", " ")
    tier      = setup.get("quality_tier", "WATCHLIST")
    price     = float(setup.get("price") or 0)
    pivot     = float(setup.get("pivot_level") or 0)
    stop      = float(setup.get("stop_level") or 0)

    # % change from prev_close if available
    prev_close = float(setup.get("prev_close") or 0)
    if prev_close and price:
        chg     = (price - prev_close) / prev_close * 100
        chg_str = f"{'▲' if chg >= 0 else '▼'} {chg:+.1f}%"
        chg_col = "#00d4a0" if chg >= 0 else "#ff4b4b"
    else:
        chg_str = ""
        chg_col = "#8892a4"

    # Target = pivot + 2 × risk; R:R ratio
    risk   = (pivot - stop) if pivot > stop else 0
    target = pivot + risk * 2 if risk > 0 else pivot
    rr     = (target - pivot) / risk if risk > 0 else 0

    badge_label, badge_col = _signal_badge(archetype, tier)

    price_html = (
        f"<span style='font-size:.9rem;color:#e8eaf0;font-weight:600'>₹{price:,.2f}</span>"
        + (f"  <span style='font-size:.82rem;color:{chg_col}'>{chg_str}</span>" if chg_str else "")
    )
    levels_html = (
        f"Entry ₹{pivot:,.0f} &nbsp;·&nbsp; Stop ₹{stop:,.0f} &nbsp;·&nbsp; "
        f"Target ₹{target:,.0f} &nbsp;·&nbsp; {rr:.1f}×"
        if pivot and stop else archetype
    )

    st.markdown(
        f"<div style='{_CARD}'>"
        # Row 1: symbol + badge
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:.3rem'>"
        f"<span style='font-size:1rem;font-weight:700;color:#e6edf3;"
        f"font-family:\"JetBrains Mono\",monospace'>{symbol}</span>"
        f"<span style='color:{badge_col};font-size:.8rem;font-weight:700'>{badge_label}</span>"
        f"</div>"
        # Row 2: price + change
        f"<div style='margin-bottom:.3rem'>{price_html}</div>"
        # Row 3: levels
        f"<div style='font-size:.75rem;color:#8892a4;font-family:\"JetBrains Mono\",monospace'>"
        f"{levels_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        f"View Analysis →  {symbol}",
        key=f"cc_view_{symbol}_{setup.get('archetype', '')}",
        width="stretch",
    ):
        st.session_state["sidebar_nav"]     = "Terminal"
        st.session_state["terminal_symbol"] = symbol
        st.session_state["_nav_pending"]    = "Terminal"
        st.rerun()


def _render_wiring_strip() -> None:
    """One compact line connecting the subsystems: autopilot state,
    today's best-earning signal for the current tape, avoid list."""
    bits = []
    try:
        from execution.autopilot import get_status
        s = get_status()
        if s["armed"]:
            _c = "#00d4a0" if s["mode"] == "PAPER" else "#ff4b4b"
            bits.append(f"<span style='color:{_c}'>🤖 Autopilot ARMED "
                        f"({s['mode']}) · {s['trades_today_count']}/"
                        f"{s['max_trades_per_day']} today · pool "
                        f"₹{s['pool']:,.0f}</span>")
        elif s["allocation"] > 0:
            bits.append("<span style='color:#8892a4'>🤖 Autopilot off</span>")
    except Exception:
        pass
    try:
        from scan.signal_backtest import trading_playbook
        from scan.unified_scanner import SIGNAL_META
        pb = trading_playbook()
        if pb and pb.get("best"):
            b = pb["best"][0]
            lbl = SIGNAL_META.get(b["signal"], (b["signal"],))[0]
            bits.append(f"🧭 {pb['regime']} tape · best: <b>{lbl}</b> "
                        f"{b['expectancy_r']:+.2f}R")
            if pb.get("avoid"):
                bits.append("❌ avoid: " + ", ".join(
                    SIGNAL_META.get(k, (k,))[0] for k in pb["avoid"][:2]))
    except Exception:
        pass
    if bits:
        st.markdown(
            "<div style='background:#0d1421;border:1px solid #1e293b;"
            "border-radius:8px;padding:7px 14px;margin-bottom:10px;"
            "font-size:.74rem;color:#c9d1d9;font-family:JetBrains Mono,"
            "monospace'>" + " &nbsp;·&nbsp; ".join(bits) + "</div>",
            unsafe_allow_html=True)


def _render_setups_section(universe: list[str]) -> None:
    """Section B — top picks from the whole-market auto-scan (instant, no wait)."""
    hdr_col, btn_col = st.columns([5, 1])
    with hdr_col:
        st.markdown(
            "<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
            "letter-spacing:.1em;margin-bottom:.5rem'>Today's Setups · whole market</div>",
            unsafe_allow_html=True,
        )
    with btn_col:
        if st.button("⟳", key="cc_rescan", help="Re-scan whole market"):
            try:
                from scan.auto_scan import force_rescan
                force_rescan()
            except Exception:
                pass
            st.rerun()

    # Read from the background auto-scan store — zero wait
    try:
        from scan.auto_scan import start_background_scan, get_results
        start_background_scan()
        results, _, _, status = get_results()
    except Exception:
        results, status = [], "error"

    # 🎯 High-conviction tier pehle — same priority order as scanner/autopilot
    _hc = [r for r in results if r.get("high_conviction")]
    _buys = [r for r in results
             if r.get("verdict") in ("STRONG BUY", "BUY")
             and not r.get("high_conviction")]
    picks = (_hc + _buys)[:5]
    if not picks:
        picks = results[:5]

    # Render-time live prices (same treatment the scanner cards get)
    try:
        @st.cache_data(ttl=120, show_spinner=False)
        def _cc_live(_syms_key: str) -> dict:
            from data.live_quotes import get_live_quotes
            return get_live_quotes(_syms_key.split(","))
        _live = _cc_live(",".join(p["symbol"] for p in picks))
        for p in picks:
            q = _live.get(p["symbol"])
            if q and q.get("price"):
                p["price"] = q["price"]
    except Exception:
        pass

    if not picks:
        msg = ("⏳ Whole-market scan is running — top picks will appear here "
               "in a few minutes." if status in ("scanning", "idle")
               else "No strong setups today. Wait for clearer market conditions.")
        st.markdown(
            f"<div style='{_CARD};text-align:center;padding:1.5rem'>"
            f"<span style='color:#8892a4;font-size:.88rem'>{msg}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for s in picks:
        # Adapt unified-scanner dict to the setup card's expected fields
        _arch = " + ".join(s.get("signals", [])[:2])
        if s.get("high_conviction"):
            _arch = "🎯 " + _arch      # evidence tier — score+edge dono strong
        _render_setup_card({
            "symbol": s["symbol"],
            "archetype": _arch,
            "quality_tier": "A" if s.get("verdict") == "BUY" else "WATCHLIST",
            "price": s.get("price"),
            "pivot_level": s.get("entry"),
            "stop_level": s.get("stop"),
        })


# ---------------------------------------------------------------------------
# Section C — Quick AI Brief (collapsible)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _gen_brief(regime_json: str) -> str:
    """Generate 5-bullet morning brief via DeepSeek. Empty string on failure."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        prompt = (
            "You are a senior institutional trader reviewing the market at open.\n"
            f"Market data: {regime_json}\n\n"
            "Write a concise 5-bullet morning brief for an Indian equity trader:\n"
            "• Regime & Score: [1 line]\n"
            "• Key Levels: [Nifty vs SMA50/SMA200, what it means]\n"
            "• Sector Rotation: [who's leading, who's lagging]\n"
            "• Playbooks for Today: [top 2 with brief why]\n"
            "• Risk to Watch: [1 specific risk]\n\n"
            "Be direct, institutional tone, no fluff. Each bullet max 25 words."
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.5,
            timeout=15,
        )
        if resp and resp.choices:
            return resp.choices[0].message.content or ""
        return ""
    except Exception as exc:
        err = str(exc)
        if "402" in err or "Insufficient Balance" in err:
            err = "DeepSeek balance khatam — recharge karo (brief tab tak skip)"
        elif "<" in err:
            err = "DeepSeek API unavailable (gateway timeout)"
        elif len(err) > 120:
            err = err[:120] + "…"
        logger.warning("brief gen failed: %s", err)
        return ""


def _fallback_brief(regime: dict) -> str:
    """Build a brief from regime data without AI."""
    name    = regime.get("market_regime", "UNKNOWN")
    score   = regime.get("regime_score", 0) or 0
    nifty   = regime.get("nifty_price", 0) or 0
    vix     = regime.get("vix", 0) or 0
    leading = ", ".join((regime.get("leading_sectors") or [])[:3]) or "none identified"
    lagging = ", ".join((regime.get("lagging_sectors") or [])[:3]) or "none identified"
    pbs     = ", ".join((regime.get("recommended_playbooks") or [])[:2]) or "WAIT"
    sma50   = regime.get("sma50", 0) or 0
    sma200  = regime.get("sma200", 0) or 0
    vs50    = "above" if nifty > sma50 > 0 else "below"
    vs200   = "above" if nifty > sma200 > 0 else "below"

    return "\n\n".join([
        f"• **Regime & Score:** {name} — {score:.0f}/100.",
        f"• **Key Levels:** Nifty {nifty:,.0f} is {vs50} SMA50 and {vs200} SMA200.",
        f"• **Sector Rotation:** Leading: {leading}. Lagging: {lagging}.",
        f"• **Playbooks for Today:** {pbs}.",
        f"• **Risk to Watch:** VIX at {vix:.1f}. "
        + (
            "Elevated — widen stops." if vix > 20 else
            "Normal range." if vix >= 14 else
            "Low vol — watch for breakout."
        ),
    ])


def _render_brief_expander(regime: dict) -> None:
    """Section C — collapsible JARVIS Market Brief (default collapsed)."""
    with st.expander("🤖 JARVIS Market Brief", expanded=False):
        _, col_btn = st.columns([4, 1])
        with col_btn:
            if st.button("↺ Refresh", key="cc_refresh_brief"):
                _gen_brief.clear()
                st.rerun()

        regime_compact = {
            k: v for k, v in regime.items()
            if k in (
                "market_regime", "regime_score", "volatility_regime",
                "breadth_label", "risk_mode", "nifty_price",
                "sma50", "sma200", "vix",
                "leading_sectors", "lagging_sectors",
                "recommended_playbooks",
            )
        }
        regime_json = json.dumps(regime_compact, sort_keys=True, default=str)
        now_hour    = datetime.now().hour
        market_open = now_hour >= 9

        with st.spinner("Fetching brief…"):
            brief = _gen_brief(regime_json) if market_open else ""

        if not brief:
            brief = _fallback_brief(regime)

        st.markdown(brief)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_command_center(universe: list[str]) -> None:
    """
    Render the Moneycontrol-style Command Center.
    Three clean sections: Market Status → Setups → JARVIS Brief.
    """
    # Minimal page header
    st.markdown(
        "<div style='padding:.4rem 0 1rem'>"
        "<span style='font-size:1.3rem;font-weight:800;color:#e6edf3;"
        "font-family:Inter,system-ui,sans-serif'>Today's Market</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load regime once (shared across all sections)
    with st.spinner("Loading market data…"):
        regime = _load_regime()

    if not regime:
        regime = {}

    # ── Section A: Market Status Card ────────────────────────────────────────
    _render_market_status_card(regime)

    # ── Wiring strip: autopilot + aaj ka playbook, ek nazar mein ─────────────
    _render_wiring_strip()

    # ── Section B: Today's Setups ─────────────────────────────────────────────
    _render_setups_section(universe)

    # ── Section C: JARVIS Brief (collapsed by default) ────────────────────────
    _render_brief_expander(regime)
