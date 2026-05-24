"""
Exit Intelligence — monitors open positions and gives EXIT / HOLD / TAKE_PARTIAL signals.

The system was 100% entry-focused. Exits are where most retail money is won or lost.
This module watches every open position and tells you exactly what to do.

Exit signals:
  EXIT_NOW     — stop triggered or distribution confirmed
  TAKE_PARTIAL — target zone reached, lock in 50%
  TRAIL_STOP   — trend intact, move stop up to protect profit
  HOLD         — setup still valid, hold with conviction
  REVIEW       — mixed signals, human judgment needed
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSignal:
    symbol: str
    action: str                  # EXIT_NOW | TAKE_PARTIAL | TRAIL_STOP | HOLD | REVIEW
    urgency: str                 # HIGH | MEDIUM | LOW
    reasons: list[str]
    suggested_stop: float        # updated ATR-based trailing stop
    profit_r: Optional[float]    # how many R in profit (None if in loss)
    current_price: float
    entry_price: float
    original_stop: float


# ── Signal computation ────────────────────────────────────────────────────────

def _analyse_position(symbol: str, entry: float, stop: float, target: float) -> PositionSignal:
    reasons: list[str] = []
    action = "HOLD"
    urgency = "LOW"

    try:
        from agents.tools import get_technical_indicators
        data = get_technical_indicators(symbol)
        if "error" in data:
            return PositionSignal(symbol, "REVIEW", "LOW", ["Data unavailable"], stop, None, entry, entry, stop)

        price = float(data.get("ltp") or data.get("price") or entry)
        rsi   = float(data.get("rsi_14") or 50)
        ema20 = data.get("ema_20")
        atr   = data.get("atr_14") or ((entry - stop) / 1.5)
        accum = data.get("accum_score", None)  # may not exist yet
        mom5  = data.get("momentum_5d_pct") or 0.0
        mom20 = data.get("momentum_20d_pct") or 0.0

        # R calculation
        r_size = abs(entry - stop) if abs(entry - stop) > 0 else entry * 0.02
        profit_r = (price - entry) / r_size if entry > 0 else None

        # ── Smart stop: volume support + round-number avoidance (not naive ATR) ──
        new_stop = stop
        stop_reason = "original stop"
        try:
            from agents.tools import _fetch_ohlcv
            from risk.smart_stop import suggest_smart_stop
            df_ohlcv = _fetch_ohlcv(symbol, days=60)
            if df_ohlcv is not None and len(df_ohlcv) >= 15:
                ss = suggest_smart_stop(symbol, df_ohlcv, price, archetype="VCP_BREAKOUT")
                smart_trail = ss["suggested_stop"]
                stop_reason = ss["reason"]
                new_stop = max(stop, smart_trail)  # trail up, never down
        except Exception:
            atr_val  = float(atr) if atr else r_size
            new_stop = max(stop, round(price - 1.5 * atr_val, 2))

        # ── Check 1: Hard stop hit
        if price <= stop:
            reasons.append(f"Price ₹{price:.1f} ≤ stop ₹{stop:.1f} — stop triggered")
            return PositionSignal(symbol, "EXIT_NOW", "HIGH", reasons, new_stop, profit_r, price, entry, stop)

        # ── Check 2: Target hit (2R or more)
        if target > entry and price >= target:
            reasons.append(f"2.5R target ₹{target:.1f} reached — lock in 50% profits")
            action, urgency = "TAKE_PARTIAL", "HIGH"

        # ── Check 3: Extreme RSI overbought (profit protection)
        if rsi > 80:
            reasons.append(f"RSI {rsi:.1f} — extremely overbought, consider taking partial")
            if action == "HOLD":
                action, urgency = "TAKE_PARTIAL", "MEDIUM"

        # ── Check 4: Price below EMA20 (trend weakening)
        if ema20 and price < float(ema20):
            reasons.append(f"Price ₹{price:.1f} below EMA20 ₹{float(ema20):.1f} — trend weakening")
            if action == "HOLD":
                action, urgency = "REVIEW", "MEDIUM"

        # ── Check 5: Distribution kicking in
        try:
            if accum is not None and float(accum) < -30:
                reasons.append(f"Distribution signal: accum score {float(accum):.0f} — smart money selling")
                if action in ("HOLD", "REVIEW"):
                    action, urgency = "EXIT_NOW", "HIGH"
        except Exception:
            pass

        # ── Check 6: Momentum reversal (5d turning negative while in profit)
        if profit_r and profit_r > 0 and float(mom5) < -3 and float(mom20) > 0:
            reasons.append(f"5d momentum {float(mom5):+.1f}% — short-term reversal in an uptrend")
            if action == "HOLD":
                action, urgency = "REVIEW", "MEDIUM"

        # ── Check 7: Trailing stop update opportunity
        if action == "HOLD" and profit_r and profit_r >= 1.0 and new_stop > stop:
            reasons.append(f"Trail stop from ₹{stop:.1f} → ₹{new_stop:.1f} ({stop_reason})")
            action, urgency = "TRAIL_STOP", "LOW"

        # ── Default: position valid, hold
        if not reasons:
            reasons.append(f"Setup intact — price ₹{price:.1f}, RSI {rsi:.1f}, {float(mom5):+.1f}% 5d momentum")

        return PositionSignal(
            symbol=symbol,
            action=action,
            urgency=urgency,
            reasons=reasons,
            suggested_stop=new_stop,
            profit_r=round(profit_r, 2) if profit_r else None,
            current_price=price,
            entry_price=entry,
            original_stop=stop,
        )
    except Exception as exc:
        return PositionSignal(symbol, "REVIEW", "LOW", [f"Analysis error: {exc}"], stop, None, entry, entry, stop)


def analyse_all_positions(positions: list[dict]) -> list[PositionSignal]:
    """
    positions: list of {symbol, entry_price, stop_price, target_price}
    Returns one PositionSignal per position, urgency-sorted.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    signals: list[PositionSignal] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(
                _analyse_position,
                p["symbol"],
                float(p.get("entry_price", 0)),
                float(p.get("stop_price", 0)),
                float(p.get("target_price", 0)),
            ): p["symbol"]
            for p in positions
        }
        for f in as_completed(futs):
            try:
                signals.append(f.result())
            except Exception:
                pass
    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return sorted(signals, key=lambda s: urgency_order.get(s.urgency, 3))


# ── Streamlit UI ─────────────────────────────────────────────────────────────

def render_exit_intelligence() -> None:
    import streamlit as st

    st.markdown(
        "<h3 style='color:#f97316;font-family:JetBrains Mono,monospace;"
        "font-size:1.1rem;letter-spacing:2px'>🚪 EXIT INTELLIGENCE</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Live monitoring of open positions — EXIT / HOLD / PARTIAL signals with reasons.")

    # ── Load positions (paper_trading or manual entry) ────────────────────────
    positions: list[dict] = []
    try:
        import sqlite3, os
        db = "paper_trading.db"
        if os.path.exists(db):
            conn = sqlite3.connect(db)
            rows = conn.execute(
                "SELECT symbol, entry_price, stop_loss, target_price FROM trades WHERE status='OPEN'"
            ).fetchall()
            conn.close()
            for r in rows:
                positions.append({
                    "symbol": r[0], "entry_price": r[1] or 0,
                    "stop_price": r[2] or 0, "target_price": r[3] or 0,
                })
    except Exception:
        pass

    # Allow manual position entry if none found
    if not positions:
        st.info("No open paper trades found. Add a position manually to monitor it.")
        with st.expander("➕ Add position to monitor"):
            c1, c2, c3, c4 = st.columns(4)
            sym    = c1.text_input("Symbol", placeholder="RELIANCE").upper().strip()
            entry  = c2.number_input("Entry ₹", min_value=0.0, value=0.0, step=1.0)
            stop   = c3.number_input("Stop ₹", min_value=0.0, value=0.0, step=1.0)
            target = c4.number_input("Target ₹", min_value=0.0, value=0.0, step=1.0)
            if st.button("Analyse", type="primary") and sym and entry > 0:
                positions = [{"symbol": sym, "entry_price": entry, "stop_price": stop, "target_price": target}]
        if not positions:
            return

    with st.spinner(f"Analysing {len(positions)} position(s)…"):
        signals = analyse_all_positions(positions)

    _ACTION_CFG = {
        "EXIT_NOW":     ("#ff4b4b", "#ff4b4b18", "🚨 EXIT NOW"),
        "TAKE_PARTIAL": ("#f59e0b", "#f59e0b18", "💰 TAKE PARTIAL"),
        "TRAIL_STOP":   ("#60a5fa", "#60a5fa18", "📈 TRAIL STOP"),
        "HOLD":         ("#00d4a0", "#00d4a018", "✅ HOLD"),
        "REVIEW":       ("#8892a4", "#8892a418", "👁 REVIEW"),
    }

    for sig in signals:
        fg, bg, label = _ACTION_CFG.get(sig.action, ("#8892a4", "#8892a418", sig.action))
        profit_txt = ""
        if sig.profit_r is not None:
            pcolor = "#00d4a0" if sig.profit_r >= 0 else "#ff4b4b"
            profit_txt = f"<span style='color:{pcolor};font-weight:700'>{sig.profit_r:+.2f}R</span>"

        reasons_html = "".join(
            f"<li style='color:#94a3b8;font-size:.72rem;margin-bottom:3px'>{r}</li>"
            for r in sig.reasons
        )
        st.markdown(
            f"""
            <div style='background:{bg};border:1px solid {fg}44;border-radius:10px;
                        padding:12px 16px;margin-bottom:8px'>
              <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px'>
                <span style='color:#e2e8f0;font-weight:700;font-size:.95rem;
                  font-family:JetBrains Mono,monospace'>{sig.symbol}</span>
                <div style='display:flex;align-items:center;gap:12px'>
                  {profit_txt}
                  <span style='background:{fg}22;border:1px solid {fg}66;border-radius:6px;
                    padding:3px 10px;font-size:.68rem;font-weight:700;color:{fg}'>{label}</span>
                </div>
              </div>
              <div style='display:flex;gap:20px;margin-bottom:8px;flex-wrap:wrap'>
                <span style='font-size:.7rem;color:#4a5568'>Entry <b style='color:#e2e8f0'>₹{sig.entry_price:.1f}</b></span>
                <span style='font-size:.7rem;color:#4a5568'>Now <b style='color:#e2e8f0'>₹{sig.current_price:.1f}</b></span>
                <span style='font-size:.7rem;color:#4a5568'>Stop <b style='color:#ff4b4b'>₹{sig.original_stop:.1f}</b></span>
                <span style='font-size:.7rem;color:#4a5568'>New stop <b style='color:#60a5fa'>₹{sig.suggested_stop:.1f}</b></span>
              </div>
              <ul style='margin:0;padding-left:18px'>{reasons_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
