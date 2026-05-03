"""Settings Streamlit page – API key status, risk limits, paper toggle."""
from __future__ import annotations

import os

import streamlit as st


def _mask(k: str) -> str:
    if not k or "REPLACE" in k:
        return "✗ not set"
    return f"✓ {k[:6]}…{k[-4:]}" if len(k) > 12 else "✓ set"


def render() -> None:
    st.title("Settings")

    st.subheader("API keys (read-only – edit .env on disk)")
    st.write(f"- Anthropic Claude:  **{_mask(os.environ.get('ANTHROPIC_API_KEY', ''))}**")
    st.write(f"- DeepSeek:          **{_mask(os.environ.get('DEEPSEEK_API_KEY', ''))}**")
    st.write(f"- Kite API key:      **{_mask(os.environ.get('KITE_API_KEY', ''))}**")
    st.write(f"- Kite access token: **{_mask(os.environ.get('KITE_ACCESS_TOKEN', ''))}**")
    st.write(f"- NewsAPI:           **{_mask(os.environ.get('NEWSAPI_KEY', ''))}**")
    st.write(f"- Alpha Vantage:     **{_mask(os.environ.get('ALPHA_VANTAGE_KEY', ''))}**")

    st.divider()
    st.subheader("Risk limits (read-only)")
    st.write(f"- Daily loss limit: **{os.environ.get('SQ_DAILY_LOSS_LIMIT_PCT', 4.0)}%**")
    st.write(f"- Max exposure:     **{os.environ.get('SQ_MAX_EXPOSURE_PCT', 50.0)}%**")
    st.write(f"- Paper trading:    **{os.environ.get('SQ_PAPER_TRADING', 'true')}**")

    st.divider()
    st.subheader("LLM mode")
    st.selectbox(
        "Active LLM mode (affects new cycles)",
        ["Ensemble (Claude + DeepSeek)", "Claude only", "DeepSeek only"],
        index=0,
        help="Edit `config.yaml` ensemble_threshold_percent to tune the veto.",
    )
    st.caption("All knobs editable in `~/0to100/.env` and `config.yaml`. "
               "Restart the app after changes (`./run.sh`).")
