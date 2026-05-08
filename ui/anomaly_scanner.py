"""Real-time AI anomaly scanner — detects z-score >2 price/volume spikes across NSE universe."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

UNIVERSE_FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "WIPRO", "AXISBANK",
    "SBIN", "LT", "BAJFINANCE", "TATAMOTORS", "MARUTI", "HCLTECH", "BHARTIARTL",
    "ASIANPAINT", "NESTLEIND", "HINDUNILVR", "ULTRACEMCO", "ITC", "ONGC",
    "COALINDIA", "NTPC", "POWERGRID", "BPCL", "ADANIPORTS", "DRREDDY",
    "SUNPHARMA", "GRASIM", "JSWSTEEL", "TATASTEEL",
]


def _compute_zscore_row(symbol: str) -> dict | None:
    try:
        df = yf.download(symbol + ".NS", period="60d", interval="1d",
                         progress=False, auto_adjust=True, timeout=8)
        if df is None or len(df) < 20:
            return None
        import pandas as _pd
        if isinstance(df.columns, _pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        closes  = df["close"].squeeze()
        volumes = df["volume"].squeeze()

        ret_20 = closes.pct_change()
        price_z = (ret_20.iloc[-1] - ret_20.iloc[-21:-1].mean()) / (ret_20.iloc[-21:-1].std() + 1e-9)

        avg_vol = volumes.iloc[-21:-1].mean()
        vol_z   = (volumes.iloc[-1] - avg_vol) / (volumes.iloc[-21:-1].std() + 1e-9)

        chg_pct = ret_20.iloc[-1] * 100
        return {
            "symbol":    symbol,
            "price_z":   round(float(price_z), 2),
            "vol_z":     round(float(vol_z), 2),
            "chg_pct":   round(float(chg_pct), 2),
            "price":     round(float(closes.iloc[-1]), 2),
            "vol_spike": round(float(volumes.iloc[-1] / avg_vol), 2) if avg_vol > 0 else 0,
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def run_anomaly_scan(universe: tuple, z_threshold: float = 2.0, max_workers: int = 8) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_compute_zscore_row, sym): sym for sym in universe}
        for fut in as_completed(futures):
            r = fut.result()
            if r and (abs(r["price_z"]) >= z_threshold or abs(r["vol_z"]) >= z_threshold):
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["alert_score"] = abs(df["price_z"]) * 0.6 + abs(df["vol_z"]) * 0.4
    return df.sort_values("alert_score", ascending=False).reset_index(drop=True)


def render_anomaly_scanner(universe: list[str] | None = None):
    """Render the anomaly scanner panel."""
    st.markdown("#### 🔍 AI Anomaly Scanner")
    st.caption("Scans for statistically abnormal price/volume behaviour (z-score threshold).")

    col1, col2, col3 = st.columns(3)
    z_thresh    = col1.slider("Z-Score threshold", 1.5, 4.0, 2.0, 0.1, key="anomaly_z")
    max_symbols = col2.number_input("Max symbols", 10, 500, 30, 10, key="anomaly_max")
    col3.empty()

    uni = tuple((universe or UNIVERSE_FALLBACK)[:int(max_symbols)])

    if st.button("🔎 Scan Now", key="anomaly_scan", use_container_width=True):
        with st.spinner(f"Scanning {len(uni)} symbols for anomalies…"):
            df = run_anomaly_scan(uni, z_threshold=z_thresh)
            st.session_state["anomaly_results"] = df
            # Push to alert inbox
            if not df.empty:
                from ui.alert_inbox import push_alert
                for _, row in df.head(5).iterrows():
                    msg = (f"Price Z={row['price_z']:+.2f}, Vol Z={row['vol_z']:+.2f}, "
                           f"Chg {row['chg_pct']:+.2f}%")
                    push_alert("TECHNICAL", row["symbol"], msg, score=min(100, row["alert_score"] * 25))

    df = st.session_state.get("anomaly_results")
    if df is None:
        st.info("Run a scan to detect anomalies.")
        return

    if df.empty:
        st.success(f"No anomalies above z={z_thresh} found.")
        return

    st.markdown(f"**{len(df)} anomalies detected** — sorted by alert score")

    for _, row in df.iterrows():
        pz_color = "#00ff88" if row["price_z"] > 0 else "#ff4466"
        vz_color = "#ffb800" if abs(row["vol_z"]) > 2 else "#8892a4"
        chg_color = "#00ff88" if row["chg_pct"] >= 0 else "#ff4466"

        st.markdown(
            f"<div class='devbloom-card' style='padding:.55rem 1rem;margin-bottom:.3rem'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"  <span style='font-weight:700;color:#e8eaf0;font-size:.9rem'>{row['symbol']}</span>"
            f"  <span style='font-family:JetBrains Mono,monospace;color:{chg_color};font-size:.85rem'>"
            f"    {'+' if row['chg_pct']>=0 else ''}{row['chg_pct']:.2f}% · ₹{row['price']:,.2f}"
            f"  </span>"
            f"</div>"
            f"<div style='margin-top:.3rem;display:flex;gap:1rem'>"
            f"  <span style='font-size:.75rem'>"
            f"    <span style='color:#8892a4'>Price-Z </span>"
            f"    <span style='color:{pz_color};font-family:JetBrains Mono,monospace;font-weight:600'>{row['price_z']:+.2f}</span>"
            f"  </span>"
            f"  <span style='font-size:.75rem'>"
            f"    <span style='color:#8892a4'>Vol-Z </span>"
            f"    <span style='color:{vz_color};font-family:JetBrains Mono,monospace;font-weight:600'>{row['vol_z']:+.2f}</span>"
            f"  </span>"
            f"  <span style='font-size:.75rem;color:#8892a4'>Vol spike: "
            f"    <span style='color:#ffb800'>{row['vol_spike']:.1f}x</span>"
            f"  </span>"
            f"  <span style='font-size:.75rem;color:#8892a4'>Score: "
            f"    <span style='color:#00d4ff;font-weight:600'>{row['alert_score']:.2f}</span>"
            f"  </span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
