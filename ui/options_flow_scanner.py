"""
Options Flow Scanner — "bade paise ka footprint" (institutional positioning).

Bade players (mutual funds, FIIs) ek stock mein move se PEHLE options mein
position lete hain. Wo footprint yahan padha jaata hai — but the RAW output
is jargon (PCR, IV rank, OI walls, max pain). Retail iska matlab nahi samajhta.

So the UI layer (below) TRANSLATES every signal into plain baat:
  • ek seedhi verdict  — "bade paise ka jhukaav UPAR / NEECHE / bada-move / none"
  • teen tradeable number — CEILING (yahan ruk sakta hai), FLOOR (yahan sambhal
    sakta hai), MAGNET (expiry tak yahan khinch sakta hai)
  • confidence dots      — signal kitna strong hai, ek nazar mein

The compute (_scan_one / analytics.*) stays jargon-native for correctness; the
plain-English + tradeable-levels layer lives in render_* so a non-options-trader
can actually USE this page. Scans top FNO stocks, ranks by unusual activity.
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


# ── Plain-English translation layer (jargon → seedhi baat) ────────────────────

def plain_verdict(sig: FlowSignal) -> tuple[str, str, str]:
    """(headline, matlab, color) — the ONE thing a non-options trader wants:
    kis taraf bada paisa jhuka hua hai, aur uska matlab kya."""
    if sig.signal == "BULLISH_FLOW":
        return ("🟢 Bade paise ka jhukaav: UPAR",
                "Options mein call-buying zyada — institutions upar ka bet le "
                "rahe. Dip pe support milne ke chance.", "#00d4a0")
    if sig.signal == "BEARISH_FLOW":
        return ("🔴 Bade paise ka jhukaav: NEECHE",
                "Put-buying zyada — institutions hedge/short kar rahe. Rally pe "
                "seller aane ke chance.", "#ff4b4b")
    if sig.signal == "IV_SPIKE":
        return ("⚡ BADA MOVE aane wala",
                "Options achanak mehenge ho gaye (IV spike) — market kisi event/"
                "bade jhatke ki taiyari mein. Direction pakka nahi, par move "
                "aayega. Option KHARIDNA mehenga, BECHNA risky.", "#a78bfa")
    return ("⚪ Koi clear jhukaav nahi",
            "Positioning balanced — is stock mein abhi institutional edge nahi "
            "dikh raha.", "#8892a4")


def tradeable_levels(sig: FlowSignal) -> tuple[float, float, float]:
    """(ceiling, floor, magnet) — the numbers you can actually trade around.
    Ceiling = nearest big CALL-OI wall above spot (yahan ruk sakta hai);
    Floor = nearest big PUT-OI wall below spot (yahan sambhal sakta hai);
    Magnet = max pain (expiry tak price idhar khinchta hai)."""
    ks = sig.key_strikes or {}

    def _pick(levels: list, above: bool) -> float:
        strikes = [float(l.get("strike") or 0) for l in (levels or [])
                   if l.get("strike")]
        if not strikes:
            return 0.0
        side = [s for s in strikes if (s >= sig.spot) == above]
        if side:
            return min(side) if above else max(side)   # nearest to spot
        return strikes[0]                                # fallback: biggest wall

    ceiling = _pick(ks.get("resistance_levels"), above=True)
    floor = _pick(ks.get("support_levels"), above=False)
    return ceiling, floor, float(sig.max_pain or 0)


def _confidence_dots(score: float) -> str:
    filled = max(1, min(5, round(score / 20)))
    return "●" * filled + "○" * (5 - filled)


def _glossary() -> None:
    with st.expander("🧠 Ye shabd matlab kya? (2-min me samajh lo)"):
        st.markdown(
            "- **Call buying / Put buying** — Call = *upar* jaane ka bet, "
            "Put = *neeche* jaane ka bet. Bade players kis taraf zyada paisa "
            "laga rahe, wahi asli signal.\n"
            "- **📈 Ceiling (resistance)** — jahan bade *call-sellers* baithe "
            "hain; price aksar wahan tak jaake ruk jaata hai.\n"
            "- **📉 Floor (support)** — jahan bade *put-sellers* baithe hain; "
            "price wahan tak girke sambhal jaata hai.\n"
            "- **🎯 Magnet (max pain)** — wo price jahan sabse zyada option-"
            "buyers ka paisa doobta hai; expiry ke paas price aksar idhar "
            "khinchta hai.\n"
            "- **IV rank** — options apni history ke hisaab se kitne mehenge "
            "hain. High = market bade move ki ummeed kar raha (aur premium "
            "mehenga).\n\n"
            "> ⚠️ Ye **context** hai, guaranteed nahi. Bade paise ka jhukaav "
            "batata hai — final trade tumhare apne setup + risk ke saath."
        )


def render_options_flow_scanner() -> None:
    st.markdown(
        "<h3 style='color:#a78bfa;font-family:JetBrains Mono,monospace;"
        "font-size:1.1rem;letter-spacing:2px'>🌊 BADE PAISE KA FOOTPRINT</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Bade players (funds/FIIs) stock ke move se PEHLE options mein position "
        "lete hain. Ye page unka jhukaav padhta hai — aur seedhi baat mein "
        "batata hai: **upar ya neeche**, aur kaunse **price levels** (ceiling / "
        "floor) pe unka bada paisa baitha hai."
    )
    _glossary()

    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    custom_syms = c1.text_input(
        "Kaun se stocks? (blank = saare bade F&O stocks)",
        placeholder="RELIANCE,HDFCBANK,INFY  ya  blank chhod do",
        key="flow_syms",
    )
    min_score = c2.slider(
        "Sirf itne strong signal", 20, 80, 40, 5, key="flow_min_score",
        help="Zyada = sirf sabse pakke signals. Kam = zyada stocks, halke "
             "signals bhi.")
    run_btn   = c3.button("🌊 Scan karo", type="primary", key="flow_run")

    if not run_btn:
        st.info("**🌊 Scan karo** dabao — main bade paise ka options footprint "
                "padh ke seedhi baat mein bataunga: kaun se stock mein upar/"
                "neeche ka bet lag raha, aur kis level pe.", icon="👆")
        return

    symbols = (
        [s.strip().upper() for s in custom_syms.split(",") if s.strip()]
        if custom_syms.strip()
        else _FNO_UNIVERSE
    )

    with st.spinner(f"{len(symbols)} F&O stocks ka options footprint padh raha…"):
        signals = _run_flow_scan(",".join(symbols))

    active = [s for s in signals if s.flow_score >= min_score]

    if not signals:
        st.warning("Options chain data nahi mila — market band ho sakta hai, ya "
                   "NSE abhi jawab nahi de raha. Market hours mein dobara try karo.")
        return
    if not active:
        st.info(f"{len(signals)} stocks dekhe — abhi koi strong footprint nahi "
                f"(is threshold pe). Slider neeche laake halke signal bhi dekh sakte ho.")
        return

    # ── Ek nazar summary — kitne upar, kitne neeche ──────────────────────────
    bullish  = [s for s in active if s.signal == "BULLISH_FLOW"]
    bearish  = [s for s in active if s.signal == "BEARISH_FLOW"]
    iv_spike = [s for s in active if s.signal == "IV_SPIKE"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks dekhe", len(signals))
    m2.metric("🟢 Upar jhukaav", len(bullish))
    m3.metric("🔴 Neeche jhukaav", len(bearish))
    m4.metric("⚡ Bada-move alert", len(iv_spike))

    st.markdown("---")
    for sig in active[:20]:
        headline, matlab, col = plain_verdict(sig)
        ceiling, floor, magnet = tradeable_levels(sig)
        dots = _confidence_dots(sig.flow_score)

        # tradeable-level chips — only show the ones we actually have
        chips = []
        if ceiling > 0:
            chips.append(f"<span style='font-size:.72rem;color:#94a3b8'>📈 Ceiling "
                         f"<b style='color:#ff6b6b'>₹{ceiling:,.0f}</b></span>")
        if floor > 0:
            chips.append(f"<span style='font-size:.72rem;color:#94a3b8'>📉 Floor "
                         f"<b style='color:#00d4a0'>₹{floor:,.0f}</b></span>")
        if magnet > 0:
            chips.append(f"<span style='font-size:.72rem;color:#94a3b8'>🎯 Magnet "
                         f"<b style='color:#f59e0b'>₹{magnet:,.0f}</b></span>")
        chips_html = "&nbsp;&nbsp;·&nbsp;&nbsp;".join(chips)

        st.markdown(
            f"""
            <div style='background:#0d1421;border:1px solid #1e293b;border-left:3px solid {col};
                        border-radius:8px;padding:11px 14px;margin-bottom:7px'>
              <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px'>
                <div>
                  <span style='color:#e2e8f0;font-weight:800;font-size:.95rem;
                    font-family:JetBrains Mono,monospace'>{sig.symbol}</span>
                  <span style='color:#64748b;font-size:.72rem;margin-left:8px'>₹{sig.spot:,.1f}</span>
                </div>
                <span style='background:{col}18;border:1px solid {col}44;border-radius:6px;
                  padding:3px 10px;font-size:.72rem;font-weight:700;color:{col}'>{headline}</span>
              </div>
              <div style='font-size:.78rem;color:#cbd5e1;margin:7px 0 8px'>{matlab}</div>
              <div style='margin-bottom:6px'>{chips_html}</div>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-size:.68rem;color:#64748b'>Confidence
                  <span style='color:{col};letter-spacing:2px'>{dots}</span></span>
                <span style='font-size:.6rem;color:#475569'>nerd stats: PCR {sig.pcr:.2f} ·
                  IV rank {sig.iv_percentile:.0f}% · ATM IV {sig.atm_iv:.0f}%</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
