"""
ui/command_center.py — Smart Command Center

The first screen the user sees. Answers ONE question: "What should I do today?"

Layout (3-column):
  Left  (38%) — Today's Intelligence Brief (DeepSeek-generated, 1hr cache)
  Center(38%) — Top Setups Right Now (ScanPipeline, 5min cache)
  Right (24%) — Regime Pulse + Implications
  Bottom row  — Portfolio Pulse + Today's Stats
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_TIER_COLORS: dict[str, str] = {
    "ELITE_A_PLUS": "#f5c518",   # gold
    "A":            "#00d4a0",   # green
    "B":            "#00d4ff",   # blue
    "WATCHLIST":    "#8892a4",   # grey
    "AVOID":        "#ff4b4b",
}

_TIER_LABELS: dict[str, str] = {
    "ELITE_A_PLUS": "ELITE",
    "A":            "A",
    "B":            "B",
    "WATCHLIST":    "WATCH",
    "AVOID":        "AVOID",
}

_REGIME_COLORS: dict[str, str] = {
    "TRENDING_BULL": "#00d4a0",
    "EXPANSION":     "#00d4a0",
    "CHOPPY":        "#f59e0b",
    "COMPRESSION":   "#8892a4",
    "DISTRIBUTION":  "#fb923c",
    "TRENDING_BEAR": "#ff4b4b",
}

_CARD_STYLE = (
    "background:#0d1117;border:1px solid #21262d;border-radius:12px;"
    "padding:1rem;margin-bottom:.75rem"
)

# ---------------------------------------------------------------------------
# Regime helpers
# ---------------------------------------------------------------------------

def _load_regime() -> dict:
    """Return regime as a plain dict; never raises."""
    try:
        from core.regime_engine import compute_regime
        r = compute_regime()
        return r.to_dict() if hasattr(r, "to_dict") else {}
    except Exception as exc:
        logger.debug("regime unavailable: %s", exc)
        return {}


def _regime_color(regime_name: str) -> str:
    return _REGIME_COLORS.get(regime_name, "#8892a4")


def _get_implications(regime_dict: dict) -> list[str]:
    """Try RegimeMonitor first; fall back to static table."""
    try:
        from core.regime_monitor import RegimeMonitor
        monitor = RegimeMonitor()
        regime_name = regime_dict.get("market_regime", "CHOPPY")
        vix = float(regime_dict.get("vix") or 0.0)
        return monitor._implications_for(regime_name, vix)[:4]  # type: ignore[attr-defined]
    except Exception:
        pass
    # Static fallback
    regime_name = regime_dict.get("market_regime", "CHOPPY")
    _STATIC: dict[str, list[str]] = {
        "TRENDING_BULL": [
            "Favour VCP_BREAKOUT and EARLY_LEADER setups",
            "Increase position sizing to 1.15–1.25x normal",
            "Trail stops aggressively — let winners run",
        ],
        "EXPANSION": [
            "Maximum aggression — all breakout plays valid",
            "MOMENTUM_EXPANSION has highest probability",
        ],
        "CHOPPY": [
            "Reduce position size to 0.7x — mean reversion only",
            "Avoid breakout entries; they whipsaw in this regime",
        ],
        "COMPRESSION": [
            "Wait mode — small size, no new breakouts yet",
            "Set alerts for volume expansion",
        ],
        "DISTRIBUTION": [
            "Reduce longs — institutions are selling",
            "Tighten stops on all open positions",
        ],
        "TRENDING_BEAR": [
            "Cash or short bias — strict stop discipline",
            "No new long breakout trades",
        ],
    }
    return _STATIC.get(regime_name, [f"{regime_name}: monitor closely"])[:4]


def _get_recent_regime_events(limit: int = 3) -> list[dict]:
    """Pull last N regime shift events from SQLite."""
    try:
        db_path = Path("logs/regime_history.db")
        if not db_path.exists():
            return []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT timestamp, from_regime, to_regime, regime_score, vix "
            "FROM regime_shifts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _time_in_current_regime(regime_dict: dict) -> str:
    """How long has the market been in the current regime (from DB)?"""
    try:
        db_path = Path("logs/regime_history.db")
        if not db_path.exists():
            return "unknown"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT timestamp FROM regime_shifts ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if not row:
            return "first reading"
        shifted_at = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - shifted_at
        days = delta.days
        hours = delta.seconds // 3600
        return f"{days}d {hours}h"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Scan caching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _run_command_scan(universe_key: str) -> list[dict]:
    """Run ScanPipeline and return serialised results (dicts)."""
    from scan.pipeline import ScanPipeline
    from core.regime_engine import compute_regime
    try:
        regime = compute_regime()
        results = ScanPipeline(
            max_workers=12,
            min_quality_score=40,
        ).run(
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
            # Ensure JSON-serialisable
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


# ---------------------------------------------------------------------------
# DeepSeek morning brief
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _gen_morning_brief(regime_json: str) -> str:
    """
    Generate a 5-bullet morning brief via DeepSeek.
    regime_json is a JSON string used as the cache key.
    Returns a markdown-formatted string.
    """
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
            "• Sector Rotation: [who's leading, who's lagging, rotation theme]\n"
            "• Playbooks for Today: [top 2 with brief why]\n"
            "• Risk to Watch: [1 specific risk]\n\n"
            "Be direct, institutional tone, no fluff. Each bullet max 25 words.\n"
            "Use exactly this format with the bullet headings shown above."
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.5,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("DeepSeek brief failed: %s", exc)
        return ""


def _fallback_brief(regime: dict) -> str:
    """Structured text brief built entirely from regime data — no AI needed."""
    name        = regime.get("market_regime", "UNKNOWN")
    score       = regime.get("regime_score", 0.0) or 0.0
    nifty       = regime.get("nifty_price", 0.0) or 0.0
    sma50       = regime.get("sma50", 0.0) or 0.0
    sma200      = regime.get("sma200", 0.0) or 0.0
    vix         = regime.get("vix", 0.0) or 0.0
    breadth     = regime.get("breadth_label", "NEUTRAL")
    leading     = regime.get("leading_sectors", []) or []
    lagging     = regime.get("lagging_sectors", []) or []
    rotation    = regime.get("rotation_mode", "MIXED")
    playbooks   = regime.get("recommended_playbooks", []) or []
    risk_mode   = regime.get("risk_mode", "NEUTRAL")

    if sma50 > 0 and nifty > 0:
        vs50 = "above" if nifty > sma50 else "below"
        pct50 = abs(nifty - sma50) / sma50 * 100
    else:
        vs50, pct50 = "near", 0.0

    if sma200 > 0 and nifty > 0:
        vs200 = "above" if nifty > sma200 else "below"
        pct200 = abs(nifty - sma200) / sma200 * 100
    else:
        vs200, pct200 = "near", 0.0

    leading_str  = ", ".join(leading[:3])  if leading  else "none identified"
    lagging_str  = ", ".join(lagging[:3])  if lagging  else "none identified"
    pb_str       = ", ".join(playbooks[:2]) if playbooks else "WAIT"

    lines = [
        f"• **Regime & Score:** {name} — score {score:.0f}/100. Risk mode: {risk_mode}.",
        f"• **Key Levels:** Nifty {nifty:,.0f} is {vs50} SMA50 by {pct50:.1f}% "
        f"and {vs200} SMA200 by {pct200:.1f}%. Breadth is {breadth}.",
        f"• **Sector Rotation:** Leading: {leading_str}. Lagging: {lagging_str}. "
        f"Rotation theme: {rotation}.",
        f"• **Playbooks for Today:** {pb_str}.",
        f"• **Risk to Watch:** VIX at {vix:.1f}. "
        + (
            "Elevated fear — widen stops." if vix > 20 else
            "Normal range — standard sizing." if vix >= 14 else
            "Low vol — compression breakout imminent."
        ),
    ]
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio stats from journal.db
# ---------------------------------------------------------------------------

def _portfolio_stats() -> dict:
    """Read journal.db for open positions, today's PnL, streak, win rate."""
    db_path = Path("journal.db")
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Open positions (action = BUY with no corresponding SELL)
        rows = conn.execute(
            "SELECT action, price, qty, timestamp FROM journal_entries ORDER BY id"
        ).fetchall()

        open_pos = 0
        today_pnl = 0.0
        all_closed: list[dict] = []
        today_str = datetime.now().strftime("%Y-%m-%d")

        # Simple pass: count BUY minus SELL per symbol as proxy
        buys: dict[str, list] = {}
        for row in rows:
            sym = row["symbol"] if "symbol" in row.keys() else ""
            action = (row["action"] or "").upper()
            price = float(row["price"] or 0)
            qty = int(row["qty"] or 0)
            ts = row["timestamp"] or ""
            if action == "BUY":
                buys.setdefault(sym, []).append({"price": price, "qty": qty, "ts": ts})
            elif action in ("SELL", "SHORT"):
                avg_buy = buys.get(sym, [{"price": price}])[-1]["price"]
                pnl = (price - avg_buy) * qty
                all_closed.append({"pnl": pnl, "ts": ts})
                if ts.startswith(today_str):
                    today_pnl += pnl
                if sym in buys and buys[sym]:
                    buys[sym].pop()

        open_pos = sum(len(v) for v in buys.values())

        # 30-day win rate
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent = [c for c in all_closed if c["ts"] >= cutoff]
        wins_30 = sum(1 for c in recent if c["pnl"] > 0)
        win_rate_30 = (wins_30 / len(recent) * 100) if recent else 0.0

        # Current streak
        streak = 0
        if all_closed:
            last_sign = None
            for c in reversed(all_closed):
                sign = "W" if c["pnl"] > 0 else "L"
                if last_sign is None:
                    last_sign = sign
                if sign != last_sign:
                    break
                streak += 1

        conn.close()
        return {
            "open_positions": open_pos,
            "today_pnl":      today_pnl,
            "win_rate_30":    win_rate_30,
            "streak":         streak,
            "streak_sign":    last_sign if all_closed else "W",  # type: ignore[possibly-undefined]
        }
    except Exception as exc:
        logger.debug("portfolio stats failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Nifty sparkline data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def _nifty_5d_close() -> list[float]:
    """Return last 5 daily closes for Nifty via yfinance."""
    try:
        import yfinance as yf
        df = yf.download("^NSEI", period="10d", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return []
        if hasattr(df.columns, "get_level_values"):
            closes = df["Close"].squeeze()
        else:
            closes = df["Close"]
        vals = list(closes.dropna().tail(5))
        return [float(v) for v in vals]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Sub-renders
# ---------------------------------------------------------------------------

def _render_brief_column(regime: dict, refresh_key: str) -> None:
    """Left column — Today's Intelligence Brief."""
    st.markdown(
        "<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em;margin-bottom:.5rem'>Today's Intelligence Brief</div>",
        unsafe_allow_html=True,
    )

    # Refresh button
    col_ts, col_btn = st.columns([3, 1])
    with col_ts:
        now_str = datetime.now().strftime("%H:%M  %d %b %Y")
        st.markdown(
            f"<span style='font-size:.65rem;color:#4a5568'>{now_str}</span>",
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("↺ Refresh", key=f"refresh_brief_{refresh_key}",
                     help="Clear brief cache and regenerate"):
            _gen_morning_brief.clear()
            st.rerun()

    # Build regime JSON string (compact, stable for cache key)
    regime_compact = {
        k: v for k, v in regime.items()
        if k in (
            "market_regime", "regime_score", "volatility_regime",
            "breadth_label", "breadth_strength", "risk_mode",
            "nifty_price", "sma50", "sma200", "vix",
            "leading_sectors", "lagging_sectors", "rotation_mode",
            "recommended_playbooks", "institutional_activity",
        )
    }
    regime_json_str = json.dumps(regime_compact, sort_keys=True, default=str)

    # Only regenerate after market open (9:00 AM IST = 03:30 UTC)
    now_hour = datetime.now().hour  # local time
    market_open = now_hour >= 9

    with st.spinner("Generating intelligence brief…"):
        brief_md = _gen_morning_brief(regime_json_str) if market_open else ""

    if not brief_md:
        brief_md = _fallback_brief(regime)

    # Render brief inside a styled card
    st.markdown(
        f"<div style='{_CARD_STYLE}'>"
        f"<div style='font-size:.8rem;line-height:1.7;color:#c9d1d9'>"
        + brief_md.replace("\n", "<br>")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def _render_setups_column(universe: list[str], refresh_key: str) -> None:
    """Center column — Top Setups Right Now."""
    hdr_col, btn_col = st.columns([4, 1])
    with hdr_col:
        st.markdown(
            "<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
            "letter-spacing:.1em;margin-bottom:.5rem'>Top Setups Right Now</div>",
            unsafe_allow_html=True,
        )
    with btn_col:
        if st.button("⟳ Re-scan", key=f"rescan_{refresh_key}", help="Flush scan cache"):
            _run_command_scan.clear()
            st.rerun()

    universe_key = ",".join(sorted(universe))

    with st.spinner("Scanning universe…"):
        setups: list[dict] = _run_command_scan(universe_key)

    if not setups:
        st.markdown(
            f"<div style='{_CARD_STYLE};text-align:center;padding:2rem'>"
            "<span style='font-size:1.2rem'>🔍</span><br>"
            "<span style='color:#8892a4;font-size:.85rem'>"
            "No setups found — market may be choppy</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for setup in setups:
        _render_setup_card(setup)


def _render_setup_card(setup: dict) -> None:
    """Render one setup as a styled card with a View Details button."""
    symbol      = setup.get("symbol", "?")
    archetype   = setup.get("archetype", "").replace("_", " ")
    tier        = setup.get("quality_tier", "WATCHLIST")
    price       = float(setup.get("price") or 0)
    pivot       = float(setup.get("pivot_level") or 0)
    stop        = float(setup.get("stop_level") or 0)
    ev          = float(setup.get("expected_value_r") or 0)
    score       = float(setup.get("quality_score") or 0)
    tier_col    = _TIER_COLORS.get(tier, "#8892a4")
    tier_label  = _TIER_LABELS.get(tier, tier)

    # Target = pivot + (pivot - stop) * 2  (simplified 2R target)
    risk = pivot - stop if pivot > stop else 0
    target = pivot + risk * 2 if risk > 0 else pivot

    ev_str   = f"+{ev:.1f}R" if ev >= 0 else f"{ev:.1f}R"
    ev_color = "#00d4a0" if ev >= 0 else "#ff4b4b"

    # Quality bar (CSS)
    bar_pct  = min(max(score, 0), 100)
    bar_col  = tier_col

    card_html = (
        f"<div style='{_CARD_STYLE}'>"
        # Header: symbol + tier badge
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:.4rem'>"
        f"<span style='font-size:1rem;font-weight:700;color:#e6edf3;"
        f"font-family:JetBrains Mono,monospace'>{symbol}</span>"
        f"<span style='background:{tier_col}22;color:{tier_col};border:1px solid {tier_col}55;"
        f"border-radius:4px;padding:.1rem .45rem;font-size:.65rem;font-weight:700;"
        f"letter-spacing:.07em'>{tier_label}</span>"
        f"</div>"
        # Archetype
        f"<div style='font-size:.7rem;color:#8892a4;margin-bottom:.5rem'>{archetype}</div>"
        # Levels in one line
        f"<div style='font-size:.73rem;color:#c9d1d9;font-family:JetBrains Mono,monospace;"
        f"margin-bottom:.5rem'>"
        f"Price <b>₹{price:,.0f}</b> &nbsp;·&nbsp; "
        f"Pivot <b>₹{pivot:,.0f}</b> &nbsp;·&nbsp; "
        f"Stop <b>₹{stop:,.0f}</b> &nbsp;·&nbsp; "
        f"Tgt <b>₹{target:,.0f}</b>"
        f"</div>"
        # EV + quality bar
        f"<div style='display:flex;align-items:center;gap:.75rem;margin-bottom:.6rem'>"
        f"<span style='font-size:.78rem;color:{ev_color};font-weight:700'>EV {ev_str}</span>"
        f"<div style='flex:1;background:#21262d;border-radius:3px;height:5px'>"
        f"<div style='width:{bar_pct:.0f}%;background:{bar_col};"
        f"height:5px;border-radius:3px'></div>"
        f"</div>"
        f"<span style='font-size:.65rem;color:#8892a4'>{score:.0f}</span>"
        f"</div>"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if st.button(
        f"View Details — {symbol}",
        key=f"view_details_{symbol}_{setup.get('archetype','')}",
        use_container_width=True,
    ):
        st.session_state["selected_page"] = "Institutional"
        st.session_state["iq2_selected"]  = symbol
        st.session_state["sidebar_nav"]   = "🏛️  Institutional"
        st.rerun()


def _render_regime_column(regime: dict) -> None:
    """Right column — Regime Pulse + Implications."""
    st.markdown(
        "<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em;margin-bottom:.5rem'>Regime Pulse</div>",
        unsafe_allow_html=True,
    )

    name    = regime.get("market_regime", "UNKNOWN")
    score   = float(regime.get("regime_score") or 0)
    nifty   = float(regime.get("nifty_price") or 0)
    vix     = float(regime.get("vix") or 0)
    breadth = int(regime.get("breadth_strength") or 0)
    q_mult  = float(regime.get("quality_multiplier") or 1.0)
    rcol    = _regime_color(name)

    # Regime name + score (large)
    st.markdown(
        f"<div style='{_CARD_STYLE};text-align:center;padding:1.25rem .75rem'>"
        f"<div style='font-size:1.3rem;font-weight:800;color:{rcol};"
        f"font-family:JetBrains Mono,monospace;letter-spacing:.04em'>"
        f"{name.replace('_', ' ')}</div>"
        f"<div style='font-size:.75rem;color:#8892a4;margin-top:.2rem'>"
        f"Score <span style='color:{rcol};font-weight:700'>{score:.0f}</span>/100</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 2x2 quick metrics
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Nifty",   f"₹{nifty:,.0f}" if nifty else "—")
        st.metric("Breadth", f"{breadth}%"     if breadth else "—")
    with m2:
        st.metric("VIX",     f"{vix:.1f}"      if vix else "—")
        st.metric("Q-Mult",  f"{q_mult:.2f}x")

    # Nifty 5-day sparkline
    closes = _nifty_5d_close()
    if closes:
        import pandas as pd
        spark_df = pd.DataFrame({"Nifty": closes})
        st.markdown(
            "<div style='font-size:.65rem;color:#8892a4;margin-top:.4rem;"
            "margin-bottom:.2rem'>5-day Nifty</div>",
            unsafe_allow_html=True,
        )
        st.bar_chart(spark_df, height=80, use_container_width=True)

    # Implications
    implications = _get_implications(regime)
    if implications:
        st.markdown(
            f"<div style='{_CARD_STYLE}'>"
            "<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
            "letter-spacing:.1em;margin-bottom:.4rem'>Implications</div>"
            "<ul style='margin:0;padding-left:1rem'>",
            unsafe_allow_html=True,
        )
        bullets = "".join(
            f"<li style='font-size:.72rem;color:#c9d1d9;margin-bottom:.3rem'>{imp}</li>"
            for imp in implications
        )
        st.markdown(
            f"<div style='{_CARD_STYLE}'>"
            "<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
            "letter-spacing:.1em;margin-bottom:.5rem'>Implications</div>"
            f"<ul style='margin:0;padding-left:1rem;list-style:disc'>"
            f"{bullets}</ul></div>",
            unsafe_allow_html=True,
        )

    # Recent regime events timeline
    events = _get_recent_regime_events(3)
    if events:
        lines = ""
        for ev in events:
            ts   = ev.get("timestamp", "")[:16].replace("T", " ")
            frm  = ev.get("from_regime", "").replace("_", " ")
            to   = ev.get("to_regime", "").replace("_", " ")
            to_c = _regime_color(ev.get("to_regime", ""))
            lines += (
                f"<div style='font-size:.65rem;color:#8892a4;border-left:2px solid "
                f"{to_c};padding-left:.5rem;margin-bottom:.35rem'>"
                f"<span style='color:{to_c};font-weight:700'>{to}</span>"
                f"<span style='color:#4a5568'> ← {frm}</span><br>"
                f"<span style='color:#4a5568'>{ts}</span>"
                f"</div>"
            )
        st.markdown(
            f"<div style='{_CARD_STYLE}'>"
            "<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
            f"letter-spacing:.1em;margin-bottom:.5rem'>Recent Shifts</div>"
            f"{lines}</div>",
            unsafe_allow_html=True,
        )

    # Time in current regime
    time_str = _time_in_current_regime(regime)
    st.markdown(
        f"<div style='font-size:.65rem;color:#4a5568;text-align:center;margin-top:.2rem'>"
        f"In current regime: <span style='color:#8892a4'>{time_str}</span></div>",
        unsafe_allow_html=True,
    )


def _render_portfolio_pulse() -> None:
    """Bottom row — Portfolio Pulse + Today's Stats."""
    st.markdown(
        "<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em;margin:1rem 0 .4rem'>Portfolio Pulse</div>",
        unsafe_allow_html=True,
    )

    stats = _portfolio_stats()
    if not stats:
        st.info("Start trading to see your stats", icon="📊")
        return

    open_pos    = stats.get("open_positions", 0)
    today_pnl   = stats.get("today_pnl", 0.0)
    win_rate    = stats.get("win_rate_30", 0.0)
    streak      = stats.get("streak", 0)
    streak_sign = stats.get("streak_sign", "W")

    pnl_delta  = f"₹{today_pnl:+,.0f}"
    pnl_color  = "#00d4a0" if today_pnl >= 0 else "#ff4b4b"
    streak_lbl = f"{streak}W" if streak_sign == "W" else f"{streak}L"
    streak_col = "#00d4a0" if streak_sign == "W" else "#ff4b4b"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Open Positions", str(open_pos))
    with c2:
        st.markdown(
            f"<div style='font-size:.65rem;color:#8892a4;margin-bottom:.1rem'>Today's P&L</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{pnl_color};"
            f"font-family:JetBrains Mono,monospace'>{pnl_delta}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.metric("Win Rate (30d)", f"{win_rate:.0f}%")
    with c4:
        st.markdown(
            f"<div style='font-size:.65rem;color:#8892a4;margin-bottom:.1rem'>Current Streak</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{streak_col};"
            f"font-family:JetBrains Mono,monospace'>{streak_lbl}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_command_center(universe: list[str]) -> None:
    """
    Render the Smart Command Center — the proactive daily intelligence dashboard.
    Replaces the old homepage on the Dashboard page.
    """
    # Handle selected_page navigation (from setup cards on previous render)
    if st.session_state.get("selected_page"):
        target = st.session_state.pop("selected_page")
        nav_map = {
            "Institutional": "🏛️  Institutional",
            "Terminal":      "⚡  Terminal",
            "Dashboard":     "🏠  Dashboard",
        }
        if target in nav_map:
            st.session_state["sidebar_nav"] = nav_map[target]
            st.rerun()

    # Page header
    now = datetime.now()
    greeting = (
        "Good morning" if now.hour < 12 else
        "Good afternoon" if now.hour < 17 else
        "Good evening"
    )
    st.markdown(
        f"<div style='padding:.5rem 0 1.2rem'>"
        f"<span style='font-size:.7rem;color:#4a5568;text-transform:uppercase;"
        f"letter-spacing:.12em;font-family:JetBrains Mono,monospace'>{greeting}</span><br>"
        f"<span style='font-size:1.6rem;font-weight:800;color:#e6edf3;"
        f"font-family:JetBrains Mono,monospace'>What should I do today?</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Load regime once (shared across all columns)
    with st.spinner("Loading market regime…"):
        regime = _load_regime()

    if not regime:
        st.warning(
            "Regime data unavailable — check your data connection. "
            "Some panels will show limited information.",
            icon="⚠️",
        )
        regime = {}

    # Cache-buster key based on minute (ensures refresh button works within minute)
    refresh_key = now.strftime("%Y%m%d%H%M")

    # ── 3-column layout ───────────────────────────────────────────────────────
    col_left, col_center, col_right = st.columns([38, 38, 24])

    with col_left:
        _render_brief_column(regime, refresh_key)

    with col_center:
        _render_setups_column(universe, refresh_key)

    with col_right:
        _render_regime_column(regime)

    st.divider()

    # ── Bottom row: Portfolio Pulse ───────────────────────────────────────────
    _render_portfolio_pulse()
