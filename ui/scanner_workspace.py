"""Unified QuantTerm scanner workspace.

Momentum, conviction, breakout, pre-breakout, F&O, long-term and avoid views are
modes of one product surface.  The page reads persisted results and queues owner
controls; it never starts a scanner or performs broker actions itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from product.conviction import build_conviction_shortlist
from product.long_term_store import load_long_term_scan
from product.market_view import current_market_view
from product.scan_store import load_scan, scan_age_hours
from product.workspace import SCANNER_MODES, scanner_rows
from research.autonomy.controls import (
    REFRESH_LONG_TERM_NOW,
    RUN_LONG_TERM_SCAN_NOW,
    RUN_SCAN_NOW,
    request_control,
)
from ui.pro_theme import evidence_panel, metric_card, page_header, section_header


@dataclass(frozen=True)
class _FallbackMarket:
    health: str = "Unavailable"
    summary: str = "Market regime is temporarily unavailable."
    trade_stance: str = "Do not assume market support."
    breadth: str = "Unavailable"
    leaders: tuple = ()
    laggards: tuple = ()


def _market():
    try:
        return current_market_view()
    except Exception:
        return _FallbackMarket()


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _score(row: Mapping[str, Any]) -> float:
    return _f(row.get("conviction_score", row.get("combined_score", row.get("score", 0.0))))


def _signals(row: Mapping[str, Any]) -> str:
    return ", ".join(str(item).replace("_", " ").title() for item in (row.get("signals") or [])[:3])


def _table(rows: list[dict], mode: str) -> pd.DataFrame:
    if mode == "Long-Term" or (rows and rows[0].get("_source") == "long_term"):
        return pd.DataFrame([
            {
                "Stock": row.get("symbol"),
                "Class": str(row.get("classification", "")).replace("_", " ").title(),
                "Score": row.get("combined_score"),
                "Fundamental": row.get("fundamental_score"),
                "Technical": row.get("technical_score"),
                "Coverage": f"{_f(row.get('fundamental_coverage')) * 100:.0f}%",
                "12m %": row.get("mom_12m_pct"),
                "From high %": row.get("from_high_pct"),
                "Timing": str(row.get("timing", "")).replace("_", " ").title(),
            }
            for row in rows
        ])
    if mode == "Conviction":
        return pd.DataFrame([
            {
                "Stock": row.get("symbol"),
                "Conviction": str(row.get("classification", "")).replace("_", " ").title(),
                "Score": row.get("conviction_score"),
                "Scanner": row.get("scanner_score"),
                "Sector": row.get("sector"),
                "Price": row.get("price"),
                "Entry": row.get("entry"),
                "Stop": row.get("stop"),
                "Target": row.get("target"),
                "Volume": f"{_f(row.get('volume_ratio')):.1f}×",
                "RSI": row.get("rsi"),
            }
            for row in rows
        ])
    return pd.DataFrame([
        {
            "Stock": row.get("symbol"),
            "Score": row.get("score"),
            "Status": row.get("status"),
            "Price": row.get("price"),
            "5d %": row.get("momentum_5d"),
            "Volume": f"{_f(row.get('volume_ratio')):.1f}×",
            "RSI": row.get("rsi"),
            "F&O": "Yes" if row.get("fno_available") else "No",
            "Signals": _signals(row),
        }
        for row in rows
    ])


def _history_chart(symbol: str) -> bool:
    try:
        from data.bhavcopy_store import get_ohlcv

        frame = get_ohlcv(symbol)
        if frame is None or len(frame) < 5:
            return False
        frame = frame.copy().tail(220)
        frame.columns = [str(col).lower() for col in frame.columns]
        if "close" not in frame.columns:
            return False
        frame["EMA 20"] = frame["close"].ewm(span=20, adjust=False).mean()
        frame["EMA 50"] = frame["close"].ewm(span=50, adjust=False).mean()
        frame["EMA 200"] = frame["close"].ewm(span=200, adjust=False).mean()
        st.line_chart(
            frame[["close", "EMA 20", "EMA 50", "EMA 200"]].rename(columns={"close": symbol}),
            height=390,
            use_container_width=True,
        )
        return True
    except Exception:
        return False


def _detail(row: Mapping[str, Any], mode: str) -> None:
    symbol = str(row.get("symbol", ""))
    source = str(row.get("_source", "market_scan"))
    score = _score(row)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Workspace Score", f"{score:.0f}", mode, tone="accent")
    with metric_cols[1]:
        metric_card("Price", f"₹{_f(row.get('price')):,.2f}", str(row.get("status", row.get("timing", ""))))
    with metric_cols[2]:
        if source == "long_term":
            metric_card("Fundamental", f"{_f(row.get('fundamental_score')):.0f}",
                        f"Coverage {_f(row.get('fundamental_coverage')) * 100:.0f}%")
        else:
            metric_card("Volume", f"{_f(row.get('volume_ratio')):.1f}×", f"RSI {_f(row.get('rsi')):.0f}")
    with metric_cols[3]:
        if source == "long_term":
            metric_card("12-Month Move", f"{_f(row.get('mom_12m_pct')):+.1f}%",
                        f"{_f(row.get('from_high_pct')):.1f}% below high")
        else:
            rr = 0.0
            entry, stop, target = _f(row.get("entry")), _f(row.get("stop")), _f(row.get("target"))
            if entry > stop and target > entry:
                rr = (target - entry) / (entry - stop)
            metric_card("Planned R:R", f"{rr:.2f}×" if rr else "Unavailable", "Predefined invalidation")

    if not _history_chart(symbol):
        st.info("Validated historical prices are not available for this stock yet.")

    if source == "long_term":
        reasons = list(row.get("quality_factors") or [])
        risks = list(row.get("risk_flags") or [])
        subtitle = (
            str(row.get("classification", "Long-term candidate")).replace("_", " ").title()
            + " · current fundamentals are a present-day snapshot, not historical PIT evidence."
        )
        plan = [
            ("Technical", f"{_f(row.get('technical_score')):.0f}"),
            ("Fundamental", f"{_f(row.get('fundamental_score')):.0f}"),
            ("Combined", f"{_f(row.get('combined_score')):.0f}"),
        ]
    else:
        reasons = list(row.get("reasons") or [])
        if not reasons and row.get("why"):
            reasons = [str(row.get("why"))]
        risks = list(row.get("risks") or [])
        if row.get("chase_risk"):
            risks.append("Price is extended; do not chase.")
        subtitle = str(row.get("classification", row.get("status", "Scanner candidate"))).replace("_", " ").title()
        plan = [
            ("Entry", f"₹{_f(row.get('entry')):,.2f}" if _f(row.get("entry")) else "Unavailable"),
            ("Stop", f"₹{_f(row.get('stop')):,.2f}" if _f(row.get("stop")) else "Unavailable"),
            ("Target", f"₹{_f(row.get('target')):,.2f}" if _f(row.get("target")) else "Unavailable"),
        ]
    evidence_panel(symbol, subtitle, reasons=reasons[:5], risks=risks[:5], plan=plan)


def render_scanner_workspace() -> None:
    scan = load_scan()
    long_term = load_long_term_scan()
    market = _market()
    conviction = build_conviction_shortlist(scan, market) if scan else []

    scan_age = scan_age_hours(scan)
    page_header(
        "Scanner",
        "One workspace for short-term strength, conviction, breakout structure, F&O liquidity and long-horizon quality.",
        eyebrow="Market Intelligence",
        badges=[
            (f"Market {market.health}", "good" if market.health == "Healthy" else "warn"),
            ((f"Scan {scan_age:.1f}h old") if scan_age is not None else "No market scan", "good" if scan_age is not None and scan_age <= 1 else "warn"),
            ("Live orders locked", "good"),
        ],
    )

    mode = st.radio("Scanner mode", SCANNER_MODES, horizontal=True, label_visibility="collapsed")

    action_cols = st.columns([1.2, 1.2, 1.4, 4.2])
    if action_cols[0].button("Run Market Scan", type="primary", width="stretch"):
        request_control(RUN_SCAN_NOW, reason=f"owner requested {mode} scan from unified Scanner")
        st.success("Market scan queued for the autonomy supervisor.")
    if action_cols[1].button("Run Long-Term Scan", width="stretch"):
        request_control(RUN_LONG_TERM_SCAN_NOW, reason="owner requested long-term scan from unified Scanner")
        st.success("Long-term scan queued for the autonomy supervisor.")
    if action_cols[2].button("Refresh Fundamentals", width="stretch"):
        request_control(REFRESH_LONG_TERM_NOW, reason="owner requested long-term fundamental refresh from unified Scanner")
        st.success("Fundamental refresh and rescan queued.")
    action_cols[3].caption(market.trade_stance)

    rows = scanner_rows(
        mode,
        scan_payload=scan,
        long_term_payload=long_term,
        conviction_rows=conviction,
    )

    filter_cols = st.columns([2.2, 1.2, 1, 1])
    search = filter_cols[0].text_input("Search stock", placeholder="Symbol or company", label_visibility="collapsed")
    minimum = filter_cols[1].slider("Minimum score", 0, 100, 0, 5)
    fno_only = filter_cols[2].checkbox("F&O only", value=False, disabled=mode == "Long-Term")
    clean_only = filter_cols[3].checkbox("Exclude chase risk", value=mode != "Avoid", disabled=mode == "Long-Term")

    filtered: list[dict] = []
    needle = search.strip().lower()
    for row in rows:
        if needle and needle not in str(row.get("symbol", "")).lower() and needle not in str(row.get("company", "")).lower():
            continue
        if _score(row) < minimum:
            continue
        if fno_only and not bool(row.get("fno_available")):
            continue
        if clean_only and bool(row.get("chase_risk")):
            continue
        filtered.append(row)

    top_score = max((_score(row) for row in filtered), default=0.0)
    source_label = "Current fundamentals + official history" if mode == "Long-Term" else "Canonical saved whole-market scan"
    summary_cols = st.columns(4)
    with summary_cols[0]:
        metric_card("Candidates", str(len(filtered)), f"{len(rows)} before filters", tone="accent")
    with summary_cols[1]:
        metric_card("Top Score", f"{top_score:.0f}", "No forced minimum when none qualify")
    with summary_cols[2]:
        metric_card("Market", market.health, market.breadth, tone="good" if market.health == "Healthy" else "warn")
    with summary_cols[3]:
        metric_card("Source", "Verified", source_label)

    section_header(f"{mode} Rankings", "Rank first, inspect evidence second, act only with a predefined invalidation.")
    if not filtered:
        st.info("No stock currently matches this scanner mode and filter combination.")
        return

    st.dataframe(_table(filtered, mode), hide_index=True, width="stretch", height=330)

    symbols = [str(row.get("symbol", "")) for row in filtered if row.get("symbol")]
    selected_symbol = st.selectbox("Inspect stock", symbols)
    selected = next(row for row in filtered if str(row.get("symbol", "")) == selected_symbol)
    section_header(f"{selected_symbol} Intelligence", "Price structure, evidence, risk and plan in one place.")
    _detail(selected, mode)
