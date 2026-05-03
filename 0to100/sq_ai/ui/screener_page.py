"""Dynamic-screener Streamlit page."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from sq_ai.ui._api import delete as api_delete, get, post


def render() -> None:
    st.title("Stock screener")
    st.caption("Filter the cached universe by technical + fundamental rules.")

    presets = get("/api/screener/presets") or []
    preset_names = [p["name"] for p in presets if isinstance(p, dict)]
    sel = st.selectbox("Load preset", ["-- new --", *preset_names])
    cur_filters: dict = {}
    if sel != "-- new --":
        cur_filters = next(
            (p["filters"] for p in presets if p["name"] == sel), {}
        )

    with st.expander("Technical filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        rsi_min = c1.slider("RSI min", 0, 100,
                            int((cur_filters.get("rsi") or {}).get("min", 0)))
        rsi_max = c1.slider("RSI max", 0, 100,
                            int((cur_filters.get("rsi") or {}).get("max", 100)))
        pvs20 = c2.selectbox("Price vs SMA20",
                             ["any", "above", "below"], index=0)
        pvs50 = c2.selectbox("Price vs SMA50",
                             ["any", "above", "below"], index=0)
        pvs200 = c2.selectbox("Price vs SMA200",
                              ["any", "above", "below"], index=0)
        macd_state = c3.selectbox("MACD", ["any", "bullish", "bearish"])
        vol = c3.selectbox("Volume", ["any", "above_avg", "below_avg"])

    with st.expander("Momentum filters"):
        c1, c2, c3 = st.columns(3)
        ret1w = c1.number_input("1W return min (%)", value=0.0) / 100
        ret1m = c2.number_input("1M return min (%)", value=0.0) / 100
        ret3m = c3.number_input("3M return min (%)", value=0.0) / 100

    with st.expander("Fundamentals (slower – pulls Alpha Vantage / yfinance)"):
        include_funda = st.checkbox("Include fundamentals", value=False)
        c1, c2, c3 = st.columns(3)
        pe_max = c1.number_input("P/E max", value=100.0)
        roe_min = c2.number_input("ROE min", value=0.0)
        de_max = c3.number_input("Debt/Equity max", value=2.0)

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

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("▶ Run screener", type="primary"):
        with st.spinner("scanning universe…"):
            res = post("/api/screener/run", json={
                "filters": filters,
                "include_fundamentals": include_funda,
                "max_results": 50,
            })
        if isinstance(res, list):
            df = pd.DataFrame(res)
            st.success(f"{len(df)} matches")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Export CSV",
                data=df.to_csv(index=False),
                file_name="screener_results.csv",
                mime="text/csv",
            )
        else:
            st.error(res)

    name = c2.text_input("Preset name")
    if c2.button("💾 Save preset", disabled=not name):
        post("/api/screener/presets", json={"name": name, "filters": filters})
        st.toast(f"saved preset '{name}'")
        st.rerun()

    if sel != "-- new --" and c3.button(f"🗑 Delete preset '{sel}'"):
        api_delete(f"/api/screener/presets/{sel}")
        st.toast(f"deleted '{sel}'")
        st.rerun()

    with st.expander("Raw filter JSON"):
        st.code(json.dumps(filters, indent=2))
