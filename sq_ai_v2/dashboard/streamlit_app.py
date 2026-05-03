"""
Streamlit dashboard.

Pages:
  1. Portfolio Overview — equity curve, positions, daily PnL
  2. Live Signals       — current signals per symbol, probabilities
  3. Model Insights     — feature importance, calibration curves
  4. Backtest Results   — from latest backtest run

Run: streamlit run dashboard/streamlit_app.py --server.port 8501
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage.redis_client import RedisClient

st.set_page_config(
    page_title="SimpleQuant AI v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

redis = RedisClient()
API_BASE = "http://localhost:8000"

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("SimpleQuant AI v2")
page = st.sidebar.radio(
    "Navigate",
    ["Portfolio Overview", "Live Signals", "Model Insights", "Backtest Results"],
)
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)
if auto_refresh:
    time.sleep(10)
    st.rerun()


# ── Helper: API call ──────────────────────────────────────────────────────────

def api_get(path: str, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        return r.json()
    except Exception:
        return default


# ── Page 1: Portfolio Overview ────────────────────────────────────────────────

if page == "Portfolio Overview":
    st.title("Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    portfolio = api_get("/portfolio", {})
    stats = api_get("/stats", {})

    total_val = portfolio.get("total_value", 0) or 0
    cash = portfolio.get("cash", 0) or 0
    daily_pnl = portfolio.get("daily_pnl", 0) or 0
    n_pos = len(portfolio.get("positions", {}))

    col1.metric("Total Value", f"₹{total_val:,.0f}")
    col2.metric("Cash", f"₹{cash:,.0f}")
    col3.metric("Daily PnL", f"₹{daily_pnl:,.0f}", delta=f"{daily_pnl/max(total_val, 1):.2%}")
    col4.metric("Open Positions", n_pos)

    # Positions table
    st.subheader("Open Positions")
    positions = portfolio.get("positions", {})
    if positions:
        rows = []
        for sym, p in positions.items():
            rows.append({
                "Symbol": sym,
                "Qty": p.get("quantity", 0),
                "Entry": f"₹{p.get('entry_price', 0):,.2f}",
                "Current": f"₹{p.get('current_price', 0):,.2f}",
                "Unrealised PnL": f"₹{p.get('unrealised_pnl', 0):,.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No open positions")

    # Recent trades
    st.subheader("Recent Trades")
    trades = api_get("/trades?limit=20", []) or []
    if trades:
        st.dataframe(pd.DataFrame(trades), use_container_width=True)
    else:
        st.info("No recent trades")


# ── Page 2: Live Signals ──────────────────────────────────────────────────────

elif page == "Live Signals":
    st.title("Live Signals")

    signals = api_get("/signals", {}) or {}

    if not signals:
        st.warning("No signals cached yet. Start the trading engine or force a signal via the API.")
    else:
        rows = []
        for sym, sig in signals.items():
            prob = sig.get("probability", 0.5)
            rows.append({
                "Symbol": sym,
                "Action": sig.get("action", "HOLD"),
                "Probability": prob,
                "Confidence": sig.get("confidence", 0),
                "Regime": sig.get("regime", "chop"),
                "Sentiment": sig.get("sentiment_score", 0.5),
                "Fundamental": sig.get("fundamental_score", 0.5),
                "Updated": sig.get("timestamp", ""),
            })

        df_signals = pd.DataFrame(rows)

        # Colour-code by action
        def colour_action(val):
            if val == "BUY":
                return "background-color: #d4edda"
            if val == "SELL":
                return "background-color: #f8d7da"
            return ""

        st.dataframe(
            df_signals.style.applymap(colour_action, subset=["Action"]),
            use_container_width=True,
        )

        # Probability bar chart
        fig = px.bar(
            df_signals,
            x="Symbol",
            y="Probability",
            color="Action",
            color_discrete_map={"BUY": "green", "SELL": "red", "HOLD": "gray"},
            title="Signal Probabilities",
        )
        fig.add_hline(y=0.6, line_dash="dash", line_color="green", annotation_text="BUY threshold")
        fig.add_hline(y=0.4, line_dash="dash", line_color="red", annotation_text="SELL threshold")
        st.plotly_chart(fig, use_container_width=True)

    # Force signal
    st.subheader("Force Signal Generation")
    force_sym = st.text_input("Symbol (e.g. RELIANCE)")
    if st.button("Generate Signal") and force_sym:
        try:
            r = requests.post(f"{API_BASE}/signal/force?symbol={force_sym.upper()}", timeout=5)
            st.success(r.json().get("message", "Queued"))
        except Exception as e:
            st.error(str(e))


# ── Page 3: Model Insights ────────────────────────────────────────────────────

elif page == "Model Insights":
    st.title("Model Insights")
    st.info("Train models first via 'make train' or POST /train, then refresh.")

    # Feature importance from LightGBM
    model_dir = Path("models/saved")
    lgbm_path = model_dir / "lgbm.pkl"

    if lgbm_path.exists():
        import pickle
        try:
            with open(lgbm_path, "rb") as f:
                obj = pickle.load(f)
            booster = obj.get("booster")
            if booster is not None:
                imp = booster.feature_importance(importance_type="gain")
                names = booster.feature_name()
                fi_df = pd.DataFrame({"Feature": names, "Importance": imp})
                fi_df = fi_df.sort_values("Importance", ascending=True).tail(20)

                fig = px.bar(
                    fi_df, x="Importance", y="Feature",
                    orientation="h", title="LightGBM Feature Importance (Top 20)"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load LightGBM model: {e}")
    else:
        st.warning("LightGBM model not found. Run 'make train' first.")

    # Meta-learner weights
    meta_path = model_dir / "meta_learner.pkl"
    if meta_path.exists():
        import pickle
        try:
            with open(meta_path, "rb") as f:
                obj = pickle.load(f)
            lr = obj.get("lr")
            names = obj.get("model_names", [])
            if lr is not None and names:
                weights = dict(zip(names, lr.coef_[0]))
                w_df = pd.DataFrame(list(weights.items()), columns=["Model", "Weight"])
                fig = px.bar(w_df, x="Model", y="Weight", title="Meta-Learner Model Weights")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load MetaLearner: {e}")


# ── Page 4: Backtest Results ──────────────────────────────────────────────────

elif page == "Backtest Results":
    st.title("Backtest Results")

    # Try loading cached equity curve from a parquet file
    equity_path = Path("logs/equity_curve.parquet")
    trades_path = Path("logs/trades.parquet")

    if equity_path.exists():
        eq = pd.read_parquet(equity_path)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity", line_color="royalblue"))
        fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Value (₹)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No backtest equity curve found. Run 'make backtest' first.")

    if trades_path.exists():
        trades_df = pd.read_parquet(trades_path)
        st.subheader("Trade Journal")
        st.dataframe(trades_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        if "pnl" in trades_df.columns:
            wins = trades_df[trades_df["pnl"] > 0]
            col1.metric("Win Rate", f"{len(wins)/len(trades_df):.1%}")
            col2.metric("Total PnL", f"₹{trades_df['pnl'].sum():,.0f}")
            col3.metric("Avg PnL/Trade", f"₹{trades_df['pnl'].mean():,.0f}")

        if "pnl" in trades_df.columns:
            fig2 = px.histogram(trades_df, x="pnl", title="PnL Distribution", nbins=30)
            st.plotly_chart(fig2, use_container_width=True)
