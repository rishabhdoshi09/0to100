"""
Shared navigation and context helpers.
All cross-page navigation goes through these functions so state stays consistent.
"""
from __future__ import annotations
import streamlit as st


def go_to(page: str, symbol: str | None = None, jarvis_query: str | None = None) -> None:
    """Navigate to a page, optionally pre-loading a symbol or JARVIS query."""
    _NAV_MAP = {
        "Dashboard":     "🏠  Dashboard",
        "Institutional": "🏛️  Institutional",
        "JARVIS":        "🤖  JARVIS",
        "Terminal":      "⚡  Terminal",
        "Research":      "🔬  Research",
        "AlgoLab":       "🧬  AlgoLab",
        "Tools":         "🛠️  Tools",
        "Watchlist":     "👁  My Watchlist",
        "Holdings":      "💼  My Holdings",
        "IPO":           "🚀  IPO Calendar",
        "Screener":      "🔍  Stock Screener",
    }
    if page in _NAV_MAP:
        st.session_state["_nav_pending"] = _NAV_MAP[page]
    if symbol:
        st.session_state["selected_symbol"] = symbol
        st.session_state["terminal_symbol"] = symbol
        st.session_state["iq2_selected"] = symbol
    if jarvis_query:
        st.session_state["jarvis_prefill"] = jarvis_query
    st.rerun()


def nav_button(label: str, page: str, symbol: str | None = None,
               jarvis_query: str | None = None, key: str = "", **kwargs) -> None:
    """Render a navigation button that calls go_to() on click."""
    if st.button(label, key=key or f"nav_{page}_{symbol}_{label}", **kwargs):
        go_to(page, symbol=symbol, jarvis_query=jarvis_query)


def get_selected_symbol(default: str = "RELIANCE") -> str:
    return st.session_state.get("selected_symbol", default)


def set_selected_symbol(symbol: str) -> None:
    st.session_state["selected_symbol"] = symbol
    st.session_state["terminal_symbol"] = symbol
    st.session_state["iq2_selected"] = symbol
