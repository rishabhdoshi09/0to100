"""Screener — one-click buzzing-stock presets + custom filter builder."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from sq_ai.ui._api import delete as api_delete, get, post

_PRESETS_META = {
    "breakout":              ("🔥", "Breakout",        "Near 52W high · volume surge · MACD bullish"),
    "momentum":              ("📈", "Pure Momentum",   "Strong 1W + 1M + 3M returns · above SMA50"),
    "value_growth":          ("💎", "Quality Growth",  "Low P/E · high ROE · clean balance sheet"),
    "volume_surge":          ("⚡", "Volume Surge",    "ATR expanding · 2× average volume · rising"),
    "consolidation_breakout":("📊", "Base Breakout",   "RSI 50-72 · price above SMAs · building momentum"),
}

_DISPLAY_COLS = [
    "symbol", "price", "from_52w_high_pct", "rsi", "vol_ratio",
    "ret_1w", "ret_1m", "ret_3m", "macd_state", "atr_pct", "score",
]

_COL_FMT = {
    "price":              "₹{:.2f}",
    "from_52w_high_pct":  "{:.1%}",
    "rsi":                "{:.1f}",
    "vol_ratio":          "{:.1f}×",
    "ret_1w":             "{:.1%}",
    "ret_1m":             "{:.1%}",
    "ret_3m":             "{:.1%}",
    "atr_pct":            "{:.2%}",
    "score":              "{:.2f}",
}

_COL_NAMES = {
    "symbol":             "Symbol",
    "price":              "Price",
    "from_52w_high_pct":  "From 52W High",
    "rsi":                "RSI",
    "vol_ratio":          "Vol ratio",
    "ret_1w":             "1W",
    "ret_1m":             "1M",
    "ret_3m":             "3M",
    "macd_state":         "MACD",
    "atr_pct":            "ATR%",
    "score":              "Score",
}


def _render_results(results: list[dict], max_results: int = 30) -> None:
    if not isinstance(results, list):
        st.error(str(results))
        return
    if not results:
        st.info("No stocks matched. Try a different preset or relax the filters.")
        return

    st.success(f"**{len(results)} stocks found**")

    rows = []
    for r in results[:max_results]:
        row = {k: r.get(k) for k in _DISPLAY_COLS if k in r}
        rows.append(row)
    df = pd.DataFrame(rows)

    # format numeric cols
    for col, fmt in _COL_FMT.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v, f=fmt: f.format(v) if v is not None else "–"
            )

    df = df.rename(columns=_COL_NAMES)
    df = df[[c for c in _COL_NAMES.values() if c in df.columns]]

    st.dataframe(df, use_container_width=True, hide_index=True)
    raw_df = pd.DataFrame(results[:max_results])
    st.download_button(
        "⬇ Export CSV",
        data=raw_df.to_csv(index=False),
        file_name="screener_results.csv",
        mime="text/csv",
    )


def render() -> None:
    st.markdown(
        "<style>.stTabs [data-baseweb='tab']{font-size:14px}"
        " .stButton button{font-size:13px}</style>",
        unsafe_allow_html=True,
    )
    st.title("Stock screener")

    # ── Buzzing Now (one-click presets) ───────────────────────────────────────
    st.subheader("🔥 Buzzing Now — one-click ideas")
    st.caption(
        "Each preset fires a curated filter combo against the live NSE universe. "
        "Combine with the custom builder below to refine."
    )

    cols = st.columns(len(_PRESETS_META))
    clicked_preset: str | None = None
    for col, (preset_key, (icon, label, desc)) in zip(cols, _PRESETS_META.items()):
        with col:
            st.markdown(
                f"<div style='text-align:center;font-size:22px'>{icon}</div>"
                f"<div style='text-align:center;font-weight:700;font-size:13px'>{label}</div>"
                f"<div style='text-align:center;color:#aaa;font-size:11px;min-height:28px'>{desc}</div>",
                unsafe_allow_html=True,
            )
            if st.button("Run", key=f"preset_{preset_key}", use_container_width=True):
                clicked_preset = preset_key

    if clicked_preset:
        icon, label, _ = _PRESETS_META[clicked_preset]
        with st.spinner(f"scanning universe for {icon} {label} stocks…"):
            results = get("/api/screener/buzzing",
                          params={"preset": clicked_preset, "max_results": 30})
        st.subheader(f"{icon} {label} — results")
        _render_results(results or [])

    st.divider()

    # ── Custom filter builder ─────────────────────────────────────────────────
    with st.expander("⚙ Custom filter builder", expanded=False):
        presets = get("/api/screener/presets") or []
        preset_names = [p["name"] for p in presets if isinstance(p, dict)]
        sel = st.selectbox("Load saved preset", ["– new –", *preset_names])
        cur_filters: dict = {}
        if sel != "– new –":
            cur_filters = next(
                (p["filters"] for p in presets if p["name"] == sel), {}
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Technical**")
            rsi_min = st.slider("RSI min", 0, 100,
                                int((cur_filters.get("rsi") or {}).get("min", 0)))
            rsi_max = st.slider("RSI max", 0, 100,
                                int((cur_filters.get("rsi") or {}).get("max", 100)))
            pvs20 = st.selectbox("Price vs SMA20",
                                 ["any", "above", "below"], index=0)
            pvs50 = st.selectbox("Price vs SMA50",
                                 ["any", "above", "below"], index=0)
            pvs200 = st.selectbox("Price vs SMA200",
                                  ["any", "above", "below"], index=0)
        with c2:
            st.markdown("**Momentum**")
            macd_state = st.selectbox("MACD", ["any", "bullish", "bearish"])
            vol = st.selectbox("Volume", ["any", "above_avg", "below_avg"])
            ret1w = st.number_input("1W return min (%)", value=0.0) / 100
            ret1m = st.number_input("1M return min (%)", value=0.0) / 100
            ret3m = st.number_input("3M return min (%)", value=0.0) / 100
        with c3:
            st.markdown("**Fundamentals**")
            include_funda = st.checkbox("Include fundamentals (slower)", value=False)
            pe_max = st.number_input("P/E max", value=100.0)
            roe_min = st.number_input("ROE min (%)", value=0.0) / 100
            de_max = st.number_input("Debt/Equity max", value=2.0)
            mc_min = st.number_input("Mkt cap min (₹ cr)", value=0.0)

        filters: dict = {}
        if rsi_min > 0 or rsi_max < 100:
            filters["rsi"] = {"min": rsi_min, "max": rsi_max}
        if pvs20 != "any":
            filters["price_vs_sma20"] = pvs20
        if pvs50 != "any":
            filters["price_vs_sma50"] = pvs50
        if pvs200 != "any":
            filters["price_vs_sma200"] = pvs200
        if macd_state != "any":
            filters["macd"] = macd_state
        if vol != "any":
            filters["volume"] = vol
        if ret1w:
            filters["ret_1w_min"] = ret1w
        if ret1m:
            filters["ret_1m_min"] = ret1m
        if ret3m:
            filters["ret_3m_min"] = ret3m
        if include_funda:
            if pe_max < 100:
                filters["pe"] = {"min": 0, "max": pe_max}
            if roe_min > 0:
                filters["roe"] = {"min": roe_min}
            if de_max < 2:
                filters["debt_to_equity"] = {"max": de_max}
            if mc_min > 0:
                filters["market_cap_min_cr"] = mc_min

        rb_col, save_col, del_col = st.columns([2, 2, 2])
        if rb_col.button("▶ Run custom screen", type="primary"):
            with st.spinner("scanning universe…"):
                res = post("/api/screener/run", json={
                    "filters": filters,
                    "include_fundamentals": include_funda,
                    "max_results": 50,
                })
            _render_results(res if isinstance(res, list) else [])

        name_input = save_col.text_input("Preset name")
        if save_col.button("💾 Save preset", disabled=not name_input):
            post("/api/screener/presets", json={"name": name_input, "filters": filters})
            st.toast(f"saved '{name_input}'")
            st.rerun()

        if sel != "– new –" and del_col.button(f"🗑 Delete '{sel}'"):
            api_delete(f"/api/screener/presets/{sel}")
            st.toast(f"deleted '{sel}'")
            st.rerun()

        with st.expander("Filter JSON"):
            st.code(json.dumps(filters, indent=2))
