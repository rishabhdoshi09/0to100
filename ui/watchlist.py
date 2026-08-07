"""My Watchlist — track your favourite stocks with price alerts and notes."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _quick_readiness_extended(symbols_key: str) -> dict[str, dict]:
    """
    Extended readiness check — returns dict of {symbol: {verdict, volume_ratio, week52_high, week52_low, price}}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from agents.tools import get_technical_indicators
    symbols = [s for s in symbols_key.split(",") if s]

    def _score(sym: str) -> tuple[str, dict]:
        try:
            d = get_technical_indicators(sym, days=60)
            if "error" in d:
                return sym, {"verdict": "WATCH", "volume_ratio": 1.0, "week52_high": 0.0, "week52_low": 0.0}
            score = 0
            price = d.get("ltp") or d.get("price") or 0
            ema20 = d.get("ema_20")
            rsi   = d.get("rsi_14") or 50
            vr    = float(d.get("volume_ratio") or 1.0)
            m5    = d.get("momentum_5d_pct") or 0.0
            m20   = d.get("momentum_20d_pct") or 0.0
            gc    = d.get("golden_cross", False)

            if price and ema20 and float(price) > float(ema20): score += 1
            if float(rsi) >= 50 and float(rsi) <= 72:           score += 1
            if vr >= 1.5:                                        score += 2
            elif vr >= 1.0:                                      score += 1
            if float(m5) > 2:                                    score += 1
            if float(m20) > 5:                                   score += 1
            if gc:                                               score += 1

            if score >= 5:   verdict = "READY"
            elif score >= 3: verdict = "BUILDING"
            else:            verdict = "WATCH"

            # Try to get 52W high/low
            w52_high, w52_low = 0.0, 0.0
            try:
                import yfinance as yf
                fi = yf.Ticker(f"{sym}.NS").fast_info
                w52_high = float(getattr(fi, "year_high", 0) or 0)
                w52_low  = float(getattr(fi, "year_low",  0) or 0)
            except Exception:
                pass

            return sym, {"verdict": verdict, "volume_ratio": vr,
                         "week52_high": w52_high, "week52_low": w52_low}
        except Exception:
            return sym, {"verdict": "WATCH", "volume_ratio": 1.0, "week52_high": 0.0, "week52_low": 0.0}

    result: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_score, s): s for s in symbols}
        for f in as_completed(futs):
            try:
                sym, info = f.result()
                result[sym] = info
            except Exception:
                pass
    return result


@st.cache_data(ttl=300, show_spinner=False)
def _quick_readiness(symbols_key: str) -> dict[str, str]:
    """
    Quick actionability check for watchlist stocks.
    Returns {symbol: "READY" | "BUILDING" | "WATCH"} using lightweight indicator scoring.
    No full pipeline — fast enough for watchlist rendering.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from agents.tools import get_technical_indicators
    symbols = [s for s in symbols_key.split(",") if s]

    def _score(sym: str) -> tuple[str, str]:
        try:
            d = get_technical_indicators(sym, days=60)
            if "error" in d:
                return sym, "WATCH"
            score = 0
            price = d.get("ltp") or d.get("price") or 0
            ema20 = d.get("ema_20")
            rsi   = d.get("rsi_14") or 50
            vr    = d.get("volume_ratio") or 1.0
            m5    = d.get("momentum_5d_pct") or 0.0
            m20   = d.get("momentum_20d_pct") or 0.0
            gc    = d.get("golden_cross", False)

            if price and ema20 and float(price) > float(ema20): score += 1
            if float(rsi) >= 50 and float(rsi) <= 72:           score += 1
            if float(vr) >= 1.5:                                 score += 2
            elif float(vr) >= 1.0:                               score += 1
            if float(m5) > 2:                                    score += 1
            if float(m20) > 5:                                   score += 1
            if gc:                                                score += 1

            if score >= 5:   return sym, "READY"
            elif score >= 3: return sym, "BUILDING"
            else:            return sym, "WATCH"
        except Exception:
            return sym, "WATCH"

    result: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_score, s): s for s in symbols}
        for f in as_completed(futs):
            try:
                sym, verdict = f.result()
                result[sym] = verdict
            except Exception:
                pass
    return result

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

WATCHLIST_DB = Path("logs/watchlist.db")


# ── Database helpers ──────────────────────────────────────────────────────────

def _init_db() -> None:
    WATCHLIST_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(WATCHLIST_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            added_date    TEXT NOT NULL,
            buy_zone_low  REAL,
            buy_zone_high REAL,
            target_price  REAL,
            stop_price    REAL,
            notes         TEXT,
            added_price   REAL
        )
    """)
    conn.commit()
    conn.close()


def _add_symbol(
    symbol: str,
    buy_low: Optional[float],
    buy_high: Optional[float],
    target: Optional[float],
    stop: Optional[float],
    notes: str,
    added_price: float,
) -> None:
    conn = sqlite3.connect(WATCHLIST_DB)
    conn.execute(
        """
        INSERT INTO watchlist
            (symbol, added_date, buy_zone_low, buy_zone_high, target_price, stop_price, notes, added_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol.strip().upper(),
            date.today().isoformat(),
            buy_low,
            buy_high,
            target,
            stop,
            notes.strip(),
            added_price,
        ),
    )
    conn.commit()
    conn.close()


def _remove_symbol(row_id: int) -> None:
    conn = sqlite3.connect(WATCHLIST_DB)
    conn.execute("DELETE FROM watchlist WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


def _get_watchlist() -> list[dict]:
    conn = sqlite3.connect(WATCHLIST_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM watchlist ORDER BY added_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_price(symbol: str) -> float:
    """Try Kite LTP first; fall back to yfinance; returns 0.0 on failure."""
    try:
        from data.kite_client import KiteClient
        kite = KiteClient()
        raw = kite.kite.ltp([f"NSE:{symbol}"])
        return float(raw[f"NSE:{symbol}"]["last_price"])
    except Exception:
        pass
    try:
        import yfinance as yf
        info = yf.Ticker(f"{symbol}.NS").fast_info
        price = float(getattr(info, "last_price", 0) or 0)
        if price > 0:
            return price
    except Exception:
        pass
    return 0.0


# ── Main render ───────────────────────────────────────────────────────────────

def render_watchlist() -> None:
    _init_db()

    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:4px'>"
        "MY WATCHLIST</h2>"
        "<p style='color:#4a5568;font-size:.76rem;margin-bottom:1.2rem'>"
        "Track your favourite stocks. Set buy zones, targets, and stop losses.</p>",
        unsafe_allow_html=True,
    )

    # ── Add stock form ────────────────────────────────────────────────────────
    with st.expander("+ Add a stock to my watchlist", expanded=False):
        c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])
        with c1:
            new_sym = st.text_input(
                "Stock symbol",
                placeholder="e.g. RELIANCE",
                key="wl_new_sym",
            ).strip().upper()
        with c2:
            buy_low = st.number_input(
                "Buy Zone Low (₹)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="wl_buy_low",
            )
        with c3:
            buy_high = st.number_input(
                "Buy Zone High (₹)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="wl_buy_high",
            )
        with c4:
            target = st.number_input(
                "Target Price (₹)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="wl_target",
            )
        with c5:
            stop = st.number_input(
                "Stop Loss (₹)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="wl_stop",
            )
        notes = st.text_input(
            "Notes (optional)",
            placeholder="Why are you watching this stock?",
            key="wl_notes",
        )

        if st.button("Add to Watchlist", type="primary", key="wl_add_btn"):
            if not new_sym:
                st.warning("Please enter a stock symbol.")
            else:
                live_price = _get_price(new_sym)
                _add_symbol(
                    symbol=new_sym,
                    buy_low=buy_low if buy_low > 0 else None,
                    buy_high=buy_high if buy_high > 0 else None,
                    target=target if target > 0 else None,
                    stop=stop if stop > 0 else None,
                    notes=notes,
                    added_price=live_price,
                )
                st.success(f"Added {new_sym} to your watchlist.")
                st.rerun()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Top-level scan button ─────────────────────────────────────────────────
    if st.button("🏛️ Scan my watchlist in Institutional →", key="wl_scan_institutional"):
        st.session_state["iq2_scan_watchlist"] = True
        go_to("🏛️  Institutional")

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Watchlist cards ───────────────────────────────────────────────────────
    rows = _get_watchlist()

    if not rows:
        st.markdown(
            "<div style='text-align:center;padding:48px 0;color:#4a5568;"
            "font-size:.85rem'>Your watchlist is empty.<br>"
            "<span style='font-size:.75rem;color:#374151'>"
            "Add a stock above to get started.</span></div>",
            unsafe_allow_html=True,
        )
        return

    # Auto-graduation: extended readiness check (includes volume_ratio, 52W data)
    symbols_key = ",".join(r["symbol"] for r in rows)
    readiness_ext: dict[str, dict] = {}
    try:
        with st.spinner("Checking setup readiness…"):
            readiness_ext = _quick_readiness_extended(symbols_key)
    except Exception:
        pass

    # Build simple readiness dict for backward compat
    readiness = {sym: info.get("verdict", "WATCH") for sym, info in readiness_ext.items()}

    # Sort: READY stocks float to top
    _order = {"READY": 0, "BUILDING": 1, "WATCH": 2}
    rows = sorted(rows, key=lambda r: _order.get(readiness.get(r["symbol"], "WATCH"), 2))

    # Summary line at the top
    ready_count    = sum(1 for v in readiness.values() if v == "READY")
    building_count = sum(1 for v in readiness.values() if v == "BUILDING")
    avoid_count    = sum(1 for v in readiness.values() if v == "WATCH")
    parts = []
    if ready_count:
        parts.append(f"<span style='color:#00d4a0;font-weight:700'>⚡ {ready_count} stock{'s' if ready_count>1 else ''} Ready</span>")
    if building_count:
        parts.append(f"<span style='color:#f59e0b;font-weight:700'>🔨 {building_count} Building</span>")
    if avoid_count:
        parts.append(f"<span style='color:#8892a4'>👁 {avoid_count} Watch</span>")
    if parts:
        st.markdown(
            f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:8px;"
            f"padding:8px 14px;margin-bottom:12px;font-size:.78rem;"
            f"font-family:JetBrains Mono,monospace'>"
            + "  ·  ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )

    to_remove: list[int] = []

    for item in rows:
        sym        = item["symbol"]
        added_px   = item["added_price"] or 0.0
        buy_low    = item["buy_zone_low"]
        buy_high   = item["buy_zone_high"]
        target_px  = item["target_price"]
        stop_px    = item["stop_price"]
        notes_text = item["notes"] or ""
        added_date = item["added_date"]

        current_px = _get_price(sym)

        # Extended data
        ext = readiness_ext.get(sym, {})
        vr        = ext.get("volume_ratio", 1.0)
        w52_high  = ext.get("week52_high", 0.0)
        w52_low   = ext.get("week52_low",  0.0)

        # Percentage change from added price
        if added_px and added_px > 0:
            change_pct = ((current_px - added_px) / added_px) * 100
        else:
            change_pct = 0.0

        # Determine status colour
        in_buy_zone = (
            buy_low is not None
            and buy_high is not None
            and buy_low <= current_px <= buy_high
        )
        below_stop = stop_px is not None and current_px > 0 and current_px <= stop_px
        near_target = (
            target_px is not None
            and current_px > 0
            and current_px >= target_px * 0.97
        )

        # Verdict badge (trader-friendly labels)
        verdict = readiness.get(sym, "WATCH")
        verdict_cfg = {
            "READY":    ("#00d4a0", "#00d4a022", "#00d4a066", "⚡ Ready"),
            "BUILDING": ("#f59e0b", "#f59e0b22", "#f59e0b66", "🔨 Building"),
        }.get(verdict, ("#8892a4", "#8892a422", "#8892a466", "👁 Watch"))
        vfg, vbg, vborder, vlabel = verdict_cfg
        verdict_html = (
            f"<span style='background:{vbg};border:1px solid {vborder};border-radius:4px;"
            f"padding:2px 8px;font-size:.65rem;font-weight:700;color:{vfg};"
            f"font-family:JetBrains Mono,monospace;white-space:nowrap'>{vlabel}</span>"
        )

        if below_stop:
            border_color = "#ff4466"
            status_label = "Below Stop"
            status_color = "#ff4466"
        elif verdict == "READY":
            border_color = "#00d4a0"
            status_label = "Setup Ready"
            status_color = "#00d4a0"
        elif in_buy_zone:
            border_color = "#00d4a0"
            status_label = "In Buy Zone"
            status_color = "#00d4a0"
        elif near_target:
            border_color = "#f59e0b"
            status_label = "Near Target"
            status_color = "#f59e0b"
        else:
            border_color = "#374151"
            status_label = ""
            status_color = "#8892a4"

        chg_color = "#00d4a0" if change_pct >= 0 else "#ff4466"
        chg_arrow = "▲" if change_pct >= 0 else "▼"
        chg_sign  = "+" if change_pct >= 0 else ""

        # Volume fire emoji
        vol_fire = ""
        try:
            if float(vr) >= 2.0:
                vol_fire = " 🔥🔥"
            elif float(vr) >= 1.5:
                vol_fire = " 🔥"
        except Exception:
            pass

        # Target upside / stop downside
        upside_pct   = ((target_px - current_px) / current_px * 100) if (target_px and current_px > 0) else None
        downside_pct = ((current_px - stop_px) / current_px * 100) if (stop_px and current_px > 0) else None

        # 52W position bar
        w52_bar_html = ""
        if w52_high and w52_low and current_px and w52_high > w52_low:
            try:
                pct_52 = min(max((current_px - w52_low) / (w52_high - w52_low) * 100, 0), 100)
                bar_color = "#00d4a0" if pct_52 >= 70 else ("#f59e0b" if pct_52 >= 40 else "#ff4b4b")
                w52_bar_html = (
                    f"<div style='margin-top:6px'>"
                    f"<div style='font-size:.6rem;color:#4a5568;margin-bottom:2px'>"
                    f"52W Range: ₹{w52_low:,.0f} ◄ <span style='color:{bar_color}'>{pct_52:.0f}%</span> ► ₹{w52_high:,.0f}</div>"
                    f"<div style='background:#1a2035;border-radius:3px;height:4px;width:100%;position:relative'>"
                    f"<div style='width:{pct_52:.0f}%;background:{bar_color};height:4px;border-radius:3px'></div>"
                    f"<div style='position:absolute;top:-3px;left:{pct_52:.0f}%;transform:translateX(-50%);"
                    f"width:2px;height:10px;background:#fff8;border-radius:1px'></div>"
                    f"</div></div>"
                )
            except Exception:
                pass

        with st.container():
            card_html = (
                f"<div style='background:rgba(255,255,255,.03);"
                f"border:1px solid {border_color}44;"
                f"border-left:4px solid {border_color};"
                f"border-radius:10px;padding:14px 18px;margin-bottom:10px'>"
                # Header row: symbol left, verdict badge right
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px'>"
                f"<span style='color:#e8eaf0;font-size:1rem;font-weight:700;"
                f"font-family:JetBrains Mono,monospace;letter-spacing:.03em'>{sym}</span>"
                f"{verdict_html}"
                f"</div>"
                # Price + % change — most prominent
                f"<div style='margin-bottom:4px'>"
                f"<span style='color:#e8eaf0;font-size:1.1rem;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>₹{current_px:,.2f}</span>"
                f"</div>"
                f"<div style='margin-bottom:6px'>"
                f"<span style='color:{chg_color};font-size:.9rem;font-weight:700'>"
                f"{chg_arrow} {chg_sign}{change_pct:.2f}%{vol_fire}</span>"
                + (
                    f" <span style='color:#4a5568;font-size:.65rem'>since added</span>"
                    if added_px > 0 else ""
                )
                + f"</div>"
            )

            # 52W bar
            if w52_bar_html:
                card_html += w52_bar_html

            # Buy zone bar
            if buy_low is not None and buy_high is not None:
                in_zone_bg = "#00d4a033" if in_buy_zone else "rgba(255,255,255,.05)"
                card_html += (
                    f"<div style='margin:.4rem 0;padding:.3rem .6rem;"
                    f"background:{in_zone_bg};border-radius:6px;"
                    f"font-size:.76rem;color:#8892a4'>"
                    f"Buy Zone: <span style='color:#00d4a0;font-weight:600'>"
                    f"₹{buy_low:,.2f} – ₹{buy_high:,.2f}</span>"
                    + (" ✓" if in_buy_zone else "")
                    + "</div>"
                )

            # Target and stop row
            details = []
            if target_px:
                t_color = "#f59e0b" if near_target else "#8892a4"
                up_str = f" (+{upside_pct:.1f}% to go)" if upside_pct is not None else ""
                details.append(
                    f"<span style='color:{t_color}'>Target: ₹{target_px:,.2f}{up_str}</span>"
                )
            if stop_px:
                s_color = "#ff4466" if below_stop else "#8892a4"
                dn_str = f" ({downside_pct:.1f}% buffer)" if downside_pct is not None else ""
                details.append(
                    f"<span style='color:{s_color}'>Stop: ₹{stop_px:,.2f}{dn_str}</span>"
                )
            if details:
                card_html += (
                    f"<div style='font-size:.76rem;margin:.4rem 0;display:flex;gap:1.2rem;flex-wrap:wrap'>"
                    + " ".join(details)
                    + "</div>"
                )

            # Notes
            if notes_text:
                card_html += (
                    f"<div style='font-size:.74rem;color:#6b7280;margin-top:.4rem;"
                    f"border-top:1px solid rgba(255,255,255,.06);padding-top:.4rem'>"
                    f"{notes_text}</div>"
                )

            # Added date + status badge
            card_html += (
                f"<div style='font-size:.65rem;color:#374151;margin-top:.4rem;"
                f"display:flex;justify-content:space-between;align-items:center'>"
                f"<span>Added: {added_date}"
                + (f" at ₹{added_px:,.2f}" if added_px else "")
                + "</span>"
                + (
                    f"<span style='color:{status_color};font-size:.65rem;font-weight:700'>{status_label}</span>"
                    if status_label else ""
                )
                + "</div>"
                "</div>"
            )

            st.markdown(card_html, unsafe_allow_html=True)

            # Action buttons row
            act1, act2, act3 = st.columns([3, 3, 2])
            with act1:
                nav_button(
                    "📊 View Setup",
                    "🏛️  Institutional",
                    symbol=sym,
                    key=f"wl_view_setup_{item['id']}",
                    width="stretch",
                )
            with act2:
                nav_button(
                    "🤖 Ask JARVIS",
                    "🤖  JARVIS",
                    jarvis_query=f"Analyse {sym} for me. Is it a good trade right now?",
                    key=f"wl_jarvis_{item['id']}",
                    width="stretch",
                )
            with act3:
                if st.button("Remove", key=f"wl_rm_{item['id']}", width="stretch"):
                    to_remove.append(item["id"])

    for row_id in to_remove:
        _remove_symbol(row_id)
    if to_remove:
        st.rerun()

    st.caption("Prices update on page refresh.")
