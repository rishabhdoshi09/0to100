"""Options Flow Scanner — plain-language institutional positioning.

The default operational universe is derived from the current Kite NSE/NFO/BFO
instrument master. It never falls back to a hand-picked shortlist. A user may
still enter a visible custom subset for an explicit one-off scan.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Optional

import streamlit as st

from data.fno_options_universe import current_options_underlyings


@dataclass
class FlowSignal:
    symbol: str
    spot: float
    pcr: float
    atm_iv: float
    iv_percentile: float
    max_pain: float
    max_pain_gap_pct: float
    call_bias: str
    flow_score: float
    signal: str
    key_strikes: dict
    note: str


def _data_client():
    try:
        from research.intelligence.data.kite_activation import KiteDataClient

        return KiteDataClient.from_config()
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def _operational_universe(as_of_iso: str):
    return current_options_underlyings(_data_client(), as_of=date.fromisoformat(as_of_iso))


def _scan_one(symbol: str) -> Optional[FlowSignal]:
    """Fetch one option chain and compute the existing flow signal."""
    try:
        from options.analytics import (
            compute_max_pain,
            compute_pcr,
            get_atm_iv,
            get_iv_percentile,
            get_oi_buildup,
            get_option_chain,
        )

        df, spot_str = get_option_chain(symbol)
        if df is None or df.empty:
            return None
        spot = float(spot_str) if spot_str else 0.0
        pcr = compute_pcr(df)
        max_pain = compute_max_pain(df)
        atm_iv = get_atm_iv(df, spot)
        iv_pct = get_iv_percentile(df)
        oi_buildup = get_oi_buildup(df, spot)

        score = 0.0
        notes: list[str] = []
        if pcr < 0.5:
            score += 35
            notes.append(f"PCR {pcr:.2f} — heavy call buying")
            call_bias = "CALL_HEAVY"
        elif pcr > 1.5:
            score += 35
            notes.append(f"PCR {pcr:.2f} — heavy put buying")
            call_bias = "PUT_HEAVY"
        elif pcr < 0.7:
            score += 15
            call_bias = "CALL_HEAVY"
        elif pcr > 1.2:
            score += 15
            call_bias = "PUT_HEAVY"
        else:
            call_bias = "NEUTRAL"

        if iv_pct > 80:
            score += 25
            notes.append(f"IV rank {iv_pct:.0f}% — event risk elevated")
        elif iv_pct > 65:
            score += 15
        elif iv_pct < 20:
            score += 10

        if spot > 0 and max_pain > 0:
            gap = abs(spot - max_pain) / spot * 100
            if gap > 5:
                score += 20
            elif gap > 2:
                score += 10
        else:
            gap = 0.0

        resistance = oi_buildup.get("resistance_levels", [])
        support = oi_buildup.get("support_levels", [])
        if resistance:
            score += 10
        if support:
            score += 10
        score = min(100.0, score)

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
    symbols = [symbol for symbol in symbols_key.split(",") if symbol]
    signals: list[FlowSignal] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_scan_one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    signals.append(result)
            except Exception:
                pass
    return sorted(signals, key=lambda item: item.flow_score, reverse=True)


def plain_verdict(signal: FlowSignal) -> tuple[str, str, str]:
    if signal.signal == "BULLISH_FLOW":
        return (
            "🟢 Large-money bias: UP",
            "Options positioning leans upward. Treat it as context, not a guaranteed trade.",
            "#00d4a0",
        )
    if signal.signal == "BEARISH_FLOW":
        return (
            "🔴 Large-money bias: DOWN",
            "Options positioning leans defensive/downward. Treat it as context, not a guarantee.",
            "#ff4b4b",
        )
    if signal.signal == "IV_SPIKE":
        return (
            "⚡ Large move expected",
            "Options are unusually expensive. Direction is unclear and event risk is high.",
            "#a78bfa",
        )
    return (
        "⚪ No clear bias",
        "Positioning is balanced; no strong institutional options edge is visible.",
        "#8892a4",
    )


def tradeable_levels(signal: FlowSignal) -> tuple[float, float, float]:
    levels = signal.key_strikes or {}

    def pick(items: list, above: bool) -> float:
        strikes = [float(item.get("strike") or 0) for item in (items or []) if item.get("strike")]
        if not strikes:
            return 0.0
        same_side = [strike for strike in strikes if (strike >= signal.spot) == above]
        if same_side:
            return min(same_side) if above else max(same_side)
        return strikes[0]

    return (
        pick(levels.get("resistance_levels"), True),
        pick(levels.get("support_levels"), False),
        float(signal.max_pain or 0),
    )


def _confidence_dots(score: float) -> str:
    filled = max(1, min(5, round(score / 20)))
    return "●" * filled + "○" * (5 - filled)


def _glossary() -> None:
    with st.expander("What these terms mean"):
        st.markdown(
            """
- **Call/put positioning:** which direction options participants are leaning.
- **Ceiling:** a large call-open-interest level that may act as resistance.
- **Floor:** a large put-open-interest level that may act as support.
- **Magnet / max pain:** a reference expiry level, not a guaranteed destination.
- **IV rank:** how expensive options are versus their own recent history.

Options flow is context. A final decision still needs the existing setup, evidence and risk gates.
"""
        )


def render_options_flow_scanner() -> None:
    st.subheader("Options Flow")
    st.caption(
        "Reads options positioning across every currently listed valid F&O underlying from the "
        "instrument master. The default scan is never a hidden Nifty-100 shortlist."
    )
    _glossary()

    report = _operational_universe(date.today().isoformat())
    if report.source == "unavailable" or not report.underlyings:
        st.error(
            "The current F&O instrument master is unavailable. QuantTerm will not replace it with "
            "a small hard-coded list. Complete Zerodha login or refresh the instrument cache."
        )
        return

    st.caption(
        f"Universe source: {report.source} · {len(report.underlyings)} underlyings "
        f"({report.stock_count} stocks, {report.index_count} indexes) · "
        f"{report.derivative_contracts:,} derivative contracts read"
    )
    if report.cache_modified_at:
        st.caption(f"Cached instrument master modified: {report.cache_modified_at.isoformat()}")

    c1, c2, c3 = st.columns([2, 1, 1])
    custom = c1.text_input(
        "Optional visible subset",
        placeholder="Leave blank to scan the complete current F&O universe",
        key="flow_syms",
    )
    minimum = c2.slider("Minimum flow score", 20, 80, 40, 5, key="flow_min_score")
    run = c3.button("Scan options flow", type="primary", key="flow_run", width="stretch")
    if not run:
        st.info("Run the scan to read current options positioning.")
        return

    if custom.strip():
        symbols = list(dict.fromkeys(
            symbol.strip().upper() for symbol in custom.split(",") if symbol.strip()
        ))
        scope = "user-selected subset"
    else:
        symbols = report.symbols
        scope = "complete current instrument-master universe"

    with st.spinner(f"Scanning {len(symbols)} F&O underlyings…"):
        signals = _run_flow_scan(",".join(symbols))

    active = [signal for signal in signals if signal.flow_score >= minimum]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Attempted", len(symbols))
    m2.metric("Chains read", len(signals))
    m3.metric("Above threshold", len(active))
    m4.metric("Unavailable", len(symbols) - len(signals))
    st.caption(f"Scope: {scope}. Display filters do not change the attempted universe.")

    if not signals:
        st.warning("No option-chain responses were available. Market/session data may be unavailable.")
        return
    if not active:
        st.info("No underlying currently clears this flow-score threshold. This is a valid result.")
        return

    for signal in active[:50]:
        headline, meaning, colour = plain_verdict(signal)
        ceiling, floor, magnet = tradeable_levels(signal)
        parts = []
        if ceiling:
            parts.append(f"Ceiling ₹{ceiling:,.0f}")
        if floor:
            parts.append(f"Floor ₹{floor:,.0f}")
        if magnet:
            parts.append(f"Magnet ₹{magnet:,.0f}")
        st.markdown(f"### {signal.symbol} — {headline}")
        st.write(meaning)
        st.caption(
            " · ".join(parts)
            + (" · " if parts else "")
            + f"Confidence {_confidence_dots(signal.flow_score)} · score {signal.flow_score:.0f} · "
            f"PCR {signal.pcr:.2f} · IV rank {signal.iv_percentile:.0f}%"
        )
        if signal.note:
            st.caption(signal.note)
        st.divider()
