"""
Options Flow Scanner — institutional signal layer.

When large players position in options before a price move, it shows up in:
  - OI spike (sudden 2-3x increase in call or put OI)
  - PCR extreme (>1.5 = bearish overhang, <0.5 = bullish)
  - IV crush or spike (implied vol moving before price)
  - Max pain divergence (stock far from max pain = likely mean reversion)

This is how institutions telegraph their intent before retail traders notice.
Scans top FNO stocks and ranks by unusual activity score.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import streamlit as st


# FNO-eligible stocks (Nifty100 core — all have liquid options chains on NSE)
_FNO_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "TCS", "KOTAKBANK",
    "AXISBANK", "SBIN", "LT", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "WIPRO", "ULTRACEMCO", "SUNPHARMA", "TATAMOTORS",
    "TITAN", "HINDUNILVR", "NESTLEIND", "POWERGRID", "NTPC",
    "TECHM", "ADANIENT", "BAJAJFINSV", "DIVISLAB", "DRREDDY",
    "CIPLA", "EICHERMOT", "HEROMOTOCO", "HINDALCO", "JSWSTEEL",
    "M&M", "ONGC", "COALINDIA", "GRASIM", "ADANIPORTS",
    "BPCL", "IOC", "INDUSINDBK", "VEDL", "TATASTEEL",
    "NIFTY", "BANKNIFTY",
]


@dataclass
class FlowSignal:
    symbol: str
    spot: float
    pcr: float                   # put/call OI ratio
    atm_iv: float                # ATM implied volatility
    iv_percentile: float         # IV rank 0-100
    max_pain: float
    max_pain_gap_pct: float      # how far spot is from max pain
    call_bias: str               # CALL_HEAVY | PUT_HEAVY | NEUTRAL
    flow_score: float            # 0-100 unusual activity score
    signal: str                  # BULLISH_FLOW | BEARISH_FLOW | NEUTRAL | IV_SPIKE
    key_strikes: dict            # resistance (call OI) + support (put OI) levels
    note: str


def _scan_one(symbol: str) -> Optional[FlowSignal]:
    """Fetch options chain and compute flow signal for one symbol."""
    try:
        from options.analytics import (
            get_option_chain, compute_pcr, compute_max_pain,
            get_atm_iv, get_oi_buildup, get_iv_percentile,
        )
        df, spot_str = get_option_chain(symbol)
        if df is None or df.empty:
            return None

        spot = float(spot_str) if spot_str else 0.0
        pcr         = compute_pcr(df)
        max_pain    = compute_max_pain(df)
        atm_iv      = get_atm_iv(df, spot)
        iv_pct      = get_iv_percentile(df)
        oi_buildup  = get_oi_buildup(df, spot)

        # ── Flow score (0-100) ────────────────────────────────────────────────
        score = 0.0
        notes: list[str] = []

        # PCR extremes = unusual positioning
        if pcr < 0.5:
            score += 35; notes.append(f"PCR {pcr:.2f} — heavy call buying (bullish)")
            call_bias = "CALL_HEAVY"
        elif pcr > 1.5:
            score += 35; notes.append(f"PCR {pcr:.2f} — heavy put buying (bearish hedge)")
            call_bias = "PUT_HEAVY"
        elif pcr < 0.7:
            score += 15; notes.append(f"PCR {pcr:.2f} — mild call dominance")
            call_bias = "CALL_HEAVY"
        elif pcr > 1.2:
            score += 15; notes.append(f"PCR {pcr:.2f} — mild put dominance")
            call_bias = "PUT_HEAVY"
        else:
            call_bias = "NEUTRAL"

        # IV spike = fear/event anticipation
        if iv_pct > 80:
            score += 25; notes.append(f"IV rank {iv_pct:.0f}% — IV spike, event expected")
        elif iv_pct > 65:
            score += 15; notes.append(f"IV rank {iv_pct:.0f}% — elevated IV")
        elif iv_pct < 20:
            score += 10; notes.append(f"IV rank {iv_pct:.0f}% — IV crush (calm before storm?)")

        # Max pain divergence — stock far from max pain = forced move likely
        if spot > 0 and max_pain > 0:
            gap = abs(spot - max_pain) / spot * 100
            if gap > 5:
                score += 20; notes.append(f"₹{gap:.1f}% from max pain ₹{max_pain:,.0f} — reversion likely")
            elif gap > 2:
                score += 10; notes.append(f"₹{gap:.1f}% from max pain ₹{max_pain:,.0f}")
        else:
            gap = 0.0

        # OI wall check — large OI at a single strike = magnet/resistance
        resistance = oi_buildup.get("resistance_levels", [])
        support    = oi_buildup.get("support_levels", [])
        if resistance:
            top_r = resistance[0]
            notes.append(f"Call OI wall at ₹{top_r['strike']:,.0f} ({top_r['ce_oi']:,.0f} contracts)")
            score += 10
        if support:
            top_s = support[0]
            notes.append(f"Put OI support at ₹{top_s['strike']:,.0f} ({top_s['pe_oi']:,.0f} contracts)")
            score += 10

        score = min(100.0, score)

        # ── Signal label ──────────────────────────────────────────────────────
        if score >= 50 and call_bias == "CALL_HEAVY" and iv_pct < 70:
            signal = "BULLISH_FLOW"
        elif score >= 50 and call_bias == "PUT_HEAVY":
            signal = "BEARISH_FLOW"
        elif iv_pct > 75:
            signal = "IV_SPIKE"
        else:
            signal = "NEUTRAL"

        return FlowSignal(
            symbol=symbol,
            spot=spot,
            pcr=round(pcr, 3),
            atm_iv=round(atm_iv, 1),
            iv_percentile=round(iv_pct, 1),
            max_pain=max_pain,
            max_pain_gap_pct=round(gap, 2),
            call_bias=call_bias,
            flow_score=round(score, 1),
            signal=signal,
            key_strikes=oi_buildup,
            note=" | ".join(notes[:3]),
        )
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _run_flow_scan(symbols_key: str) -> list[FlowSignal]:
    symbols = [s for s in symbols_key.split(",") if s]
    signals: list[FlowSignal] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_scan_one, sym): sym for sym in symbols}
        for f in as_completed(futs):
            try:
                r = f.result()
                if r is not None:
                    signals.append(r)
            except Exception:
                pass
    return sorted(signals, key=lambda s: s.flow_score, reverse=True)


def render_options_flow_scanner() -> None:
    st.markdown(
        "<h3 style='color:#a78bfa;font-family:JetBrains Mono,monospace;"
        "font-size:1.1rem;letter-spacing:2px'>🌊 OPTIONS FLOW SCANNER</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Scans NSE FNO stocks for unusual options activity — PCR extremes, IV spikes, "
        "OI walls, max pain divergence. These signals precede institutional moves."
    )

    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    custom_syms = c1.text_input(
        "Symbols to scan (comma-separated, leave blank for all FNO)",
        placeholder="RELIANCE,HDFCBANK,INFY or leave blank",
        key="flow_syms",
    )
    min_score = c2.slider("Min flow score", 20, 80, 40, 5, key="flow_min_score")
    run_btn   = c3.button("🌊 Scan Options Flow", type="primary", key="flow_run")

    if not run_btn:
        st.info("Click **Scan Options Flow** to detect unusual institutional positioning.", icon="🌊")
        return

    symbols = (
        [s.strip().upper() for s in custom_syms.split(",") if s.strip()]
        if custom_syms.strip()
        else _FNO_UNIVERSE
    )

    with st.spinner(f"Scanning options flow across {len(symbols)} FNO stocks…"):
        signals = _run_flow_scan(",".join(symbols))

    active = [s for s in signals if s.flow_score >= min_score]

    if not active:
        st.info("No unusual options flow detected above the score threshold.")
        return

    # ── Summary stats
    bullish  = [s for s in active if s.signal == "BULLISH_FLOW"]
    bearish  = [s for s in active if s.signal == "BEARISH_FLOW"]
    iv_spike = [s for s in active if s.signal == "IV_SPIKE"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scanned", len(signals))
    m2.metric("🟢 Bullish Flow", len(bullish))
    m3.metric("🔴 Bearish Flow", len(bearish))
    m4.metric("⚡ IV Spikes", len(iv_spike))

    _SIG_CFG = {
        "BULLISH_FLOW": ("#00d4a0", "🟢 BULLISH FLOW"),
        "BEARISH_FLOW": ("#ff4b4b", "🔴 BEARISH FLOW"),
        "IV_SPIKE":     ("#a78bfa", "⚡ IV SPIKE"),
        "NEUTRAL":      ("#4a5568", "⚪ NEUTRAL"),
    }

    st.markdown("---")
    for sig in active[:20]:
        fg, label = _SIG_CFG.get(sig.signal, ("#8892a4", sig.signal))

        # Bias bar
        pcr_bar = min(int(sig.pcr * 40), 100)
        bias_color = "#00d4a0" if sig.pcr < 1.0 else "#ff4b4b"

        st.markdown(
            f"""
            <div style='background:#0d1421;border:1px solid #1e293b;border-radius:8px;
                        padding:10px 14px;margin-bottom:6px'>
              <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px'>
                <div>
                  <span style='color:#e2e8f0;font-weight:700;font-size:.9rem;
                    font-family:JetBrains Mono,monospace'>{sig.symbol}</span>
                  <span style='color:#4a5568;font-size:.7rem;margin-left:8px'>₹{sig.spot:,.1f}</span>
                </div>
                <div style='display:flex;align-items:center;gap:10px'>
                  <span style='font-size:.65rem;color:#60a5fa'>Score {sig.flow_score:.0f}</span>
                  <span style='background:{fg}18;border:1px solid {fg}44;border-radius:5px;
                    padding:2px 8px;font-size:.62rem;font-weight:700;color:{fg}'>{label}</span>
                </div>
              </div>
              <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:6px'>
                <span style='font-size:.7rem;color:#4a5568'>PCR <b style='color:{bias_color}'>{sig.pcr:.2f}</b></span>
                <span style='font-size:.7rem;color:#4a5568'>ATM IV <b style='color:#e2e8f0'>{sig.atm_iv:.1f}%</b></span>
                <span style='font-size:.7rem;color:#4a5568'>IV Rank <b style='color:#a78bfa'>{sig.iv_percentile:.0f}%</b></span>
                <span style='font-size:.7rem;color:#4a5568'>Max Pain <b style='color:#f59e0b'>₹{sig.max_pain:,.0f}</b></span>
                <span style='font-size:.7rem;color:#4a5568'>Pain Gap <b style='color:#e2e8f0'>{sig.max_pain_gap_pct:.1f}%</b></span>
              </div>
              <div style='font-size:.68rem;color:#94a3b8'>{sig.note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
