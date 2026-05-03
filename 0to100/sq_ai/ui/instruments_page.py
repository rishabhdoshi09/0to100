"""NSE Instruments — full listing + live price tracking.

* Shows every instrument Kite knows about for NSE (EQ, FUT, CE/PE, INDEX).
* Search by symbol or company name (instant client-side filter).
* Filter by instrument type.
* Live price tracker: select any EQ instruments, click "Track", see their
  LTP auto-refreshing via Kite's LTP API.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from sq_ai.ui._api import get, post


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_instruments(q: str = "", exchange: str = "NSE") -> pd.DataFrame:
    params: dict = {"limit": 10000, "exchange": exchange}
    if q:
        params["q"] = q
    rows = get("/api/universe/all", params=params) or []
    if not rows or isinstance(rows, dict):
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fetch_ltp(symbols: list[str]) -> dict[str, float]:
    """Call /api/ltp with NSE:-prefixed symbols; return {symbol: price}."""
    prefixed = [s if ":" in s else f"NSE:{s}" for s in symbols]
    raw = get("/api/ltp", params={"symbols": ",".join(prefixed)}) or {}
    # Kite returns keys like "NSE:RELIANCE" — strip exchange prefix
    return {k.split(":", 1)[-1]: v for k, v in raw.items()}


# ── page ──────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("NSE Instruments")
    st.caption(
        "Source: Kite Connect SDK (if `KITE_ACCESS_TOKEN` is set) "
        "or the public `https://api.kite.trade/instruments/NSE` CSV — "
        "no credentials required for the instrument list."
    )

    # ── controls row ──────────────────────────────────────────────────────
    c_q, c_type, c_ref = st.columns([3, 2, 1])
    with c_q:
        q = st.text_input(
            "Search symbol or company", placeholder="e.g. RELIANCE or Infosys",
            label_visibility="collapsed",
        )
    with c_type:
        itype = st.selectbox(
            "Type", ["All", "EQ", "FUT", "CE", "PE", "INDEX"],
            label_visibility="collapsed",
        )
    with c_ref:
        if st.button("Refresh list", use_container_width=True):
            st.cache_data.clear()
            result = post("/api/universe/refresh") or {}
            st.toast(f"Cache refreshed — {result.get('cached', '?')} EQ rows stored.")

    # ── load & filter ─────────────────────────────────────────────────────
    df = _load_instruments(q=q)
    if df.empty:
        st.warning(
            "No instruments loaded yet.  Click **Refresh list** or wait for "
            "the 08:00 IST scheduler job to run."
        )
        return

    for col in ["instrument_type", "segment", "lot_size", "tick_size", "name"]:
        if col not in df.columns:
            df[col] = ""

    if itype != "All":
        df = df[df["instrument_type"].str.upper().fillna("") == itype]

    total = len(df)
    st.markdown(f"**{total:,} instruments**")

    # ── full listing table ────────────────────────────────────────────────
    show_cols = ["trading_symbol", "name", "instrument_type",
                 "segment", "lot_size", "tick_size", "instrument_token"]
    show_cols = [c for c in show_cols if c in df.columns]
    rename = {
        "trading_symbol":   "Symbol",
        "name":             "Company / Contract",
        "instrument_type":  "Type",
        "segment":          "Segment",
        "lot_size":         "Lot",
        "tick_size":        "Tick",
        "instrument_token": "Token",
    }
    st.dataframe(
        df[show_cols].rename(columns=rename).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Symbol":             st.column_config.TextColumn(width="medium"),
            "Company / Contract": st.column_config.TextColumn(width="large"),
            "Type":               st.column_config.TextColumn(width="small"),
            "Segment":            st.column_config.TextColumn(width="small"),
            "Lot":                st.column_config.NumberColumn(width="small", format="%d"),
            "Tick":               st.column_config.NumberColumn(width="small", format="%.2f"),
            "Token":              st.column_config.NumberColumn(width="medium"),
        },
    )

    # ── breakdown sidebar ─────────────────────────────────────────────────
    with st.expander("Breakdown by type"):
        bd = (
            df["instrument_type"].fillna("Unknown")
            .value_counts()
            .rename_axis("Type")
            .reset_index(name="Count")
        )
        st.dataframe(bd, hide_index=True, use_container_width=False)

    # ── live price tracker ────────────────────────────────────────────────
    st.divider()
    st.subheader("Live price tracker")
    st.caption(
        "Type comma-separated NSE symbols to track their LTP in real time. "
        "Requires `KITE_API_KEY` + `KITE_ACCESS_TOKEN` in `.env`."
    )

    if "tracked_symbols" not in st.session_state:
        st.session_state.tracked_symbols = ""

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        raw_input = st.text_input(
            "Symbols to track",
            value=st.session_state.tracked_symbols,
            placeholder="RELIANCE, TCS, HDFCBANK",
            label_visibility="collapsed",
        )
    with col_btn:
        track = st.button("Track", use_container_width=True)

    if track:
        st.session_state.tracked_symbols = raw_input
        st.rerun()

    tracked = [s.strip().upper() for s in st.session_state.tracked_symbols.split(",")
               if s.strip()]

    if tracked:
        auto_col, interval_col = st.columns([1, 2])
        with auto_col:
            auto_refresh = st.toggle("Auto-refresh", value=False)
        with interval_col:
            interval_s = st.slider(
                "Refresh every (s)", min_value=2, max_value=60, value=5,
                disabled=not auto_refresh,
            )

        price_placeholder = st.empty()

        def _render_prices() -> None:
            ltp = _fetch_ltp(tracked)
            if not ltp:
                price_placeholder.warning(
                    "Kite returned no prices. Check that KITE_API_KEY and "
                    "KITE_ACCESS_TOKEN are set and the market is open."
                )
                return
            rows_p = [
                {"Symbol": sym, "LTP (₹)": ltp.get(sym, "—")}
                for sym in tracked
            ]
            price_placeholder.dataframe(
                pd.DataFrame(rows_p),
                use_container_width=False,
                hide_index=True,
                column_config={
                    "Symbol":   st.column_config.TextColumn(width="medium"),
                    "LTP (₹)":  st.column_config.NumberColumn(
                        width="medium", format="₹%.2f"
                    ),
                },
            )

        _render_prices()

        if auto_refresh:
            time.sleep(interval_s)
            st.rerun()
