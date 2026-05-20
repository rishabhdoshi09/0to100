"""
Natural Language Query Layer.
Lets the user ask questions in plain English and routes to the right pipeline.

Examples:
  "Show me IT stocks in uptrend with RSI under 50"
  "VCP setups in banking sector"
  "What's my best performing playbook this month?"
  "Pharma stocks near 52-week high with low volatility"
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import streamlit as st


# ── Intent schema returned by DeepSeek ───────────────────────────────────────

@dataclass
class QueryIntent:
    intent_type: str        # SCAN | STATS | EXPLAIN | UNKNOWN
    sectors: list[str]      # e.g. ["IT", "BANK"]
    archetypes: list[str]   # e.g. ["VCP_BREAKOUT", "MOMENTUM_EXPANSION"]
    rsi_max: float | None
    rsi_min: float | None
    min_quality: str | None # ELITE_A_PLUS | A | B | WATCHLIST
    near_52w_high: bool
    uptrend_only: bool
    raw_query: str
    explanation: str        # what the system understood


_PARSE_PROMPT = """You are a trading system query parser. Parse the user's natural language query into structured JSON.

User query: "{query}"

Return ONLY valid JSON with this exact schema:
{{
  "intent_type": "SCAN",
  "sectors": [],
  "archetypes": [],
  "rsi_max": null,
  "rsi_min": null,
  "min_quality": null,
  "near_52w_high": false,
  "uptrend_only": false,
  "explanation": "one line: what you understood"
}}

Rules:
- sectors: map to one of [IT, BANK, AUTO, PHARMA, FMCG, METAL, ENERGY, REALTY, INFRA, CONS]
- archetypes: map to [VCP_BREAKOUT, MOMENTUM_EXPANSION, EARLY_LEADER, ACCUMULATION_BREAKOUT, TREND_CONTINUATION, MEAN_REVERSION, HIGH_TIGHT_FLAG]
- min_quality: map to ELITE_A_PLUS, A, B, or WATCHLIST
- near_52w_high: true if user mentions "52-week high", "all-time high", "near highs", "breakout"
- uptrend_only: true if user mentions "uptrend", "above SMA", "trending up", "momentum"
- Return only the JSON, no other text."""


def _parse_intent(query: str) -> QueryIntent:
    """Use DeepSeek to parse natural language into structured intent."""
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return _fallback_parse(query)

    import requests
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": _PARSE_PROMPT.format(query=query)}],
                "temperature": 0.0,
                "max_tokens": 300,
            },
            timeout=10,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
        return QueryIntent(
            intent_type=data.get("intent_type", "SCAN"),
            sectors=data.get("sectors", []),
            archetypes=data.get("archetypes", []),
            rsi_max=data.get("rsi_max"),
            rsi_min=data.get("rsi_min"),
            min_quality=data.get("min_quality"),
            near_52w_high=data.get("near_52w_high", False),
            uptrend_only=data.get("uptrend_only", False),
            raw_query=query,
            explanation=data.get("explanation", query),
        )
    except Exception:
        return _fallback_parse(query)


def _fallback_parse(query: str) -> QueryIntent:
    """Keyword-based fallback when DeepSeek is unavailable."""
    q = query.lower()
    sectors = []
    for s in ["it", "bank", "auto", "pharma", "fmcg", "metal", "energy", "realty"]:
        if s in q:
            sectors.append(s.upper())

    archetypes = []
    if "vcp" in q:
        archetypes.append("VCP_BREAKOUT")
    if "momentum" in q:
        archetypes.append("MOMENTUM_EXPANSION")
    if "breakout" in q and "vcp" not in q:
        archetypes.append("ACCUMULATION_BREAKOUT")

    rsi_max = None
    m = re.search(r"rsi\s*(under|below|<)\s*(\d+)", q)
    if m:
        rsi_max = float(m.group(2))

    rsi_min = None
    m = re.search(r"rsi\s*(above|over|>)\s*(\d+)", q)
    if m:
        rsi_min = float(m.group(2))

    return QueryIntent(
        intent_type="SCAN",
        sectors=sectors,
        archetypes=archetypes,
        rsi_max=rsi_max,
        rsi_min=rsi_min,
        min_quality=None,
        near_52w_high=any(x in q for x in ["52w", "52-week", "near high", "all time"]),
        uptrend_only=any(x in q for x in ["uptrend", "trending", "above sma"]),
        raw_query=query,
        explanation=f"Keyword parse: sectors={sectors}, archetypes={archetypes}",
    )


def _apply_filters(setups: list[dict], intent: QueryIntent, universe: list[str]) -> list[dict]:
    """Filter the scan results based on parsed intent."""
    result = setups

    # Sector filter
    if intent.sectors:
        _SECTOR_SYMBOLS: dict[str, list[str]] = {
            "IT":     ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "MPHASIS", "LTIM", "COFORGE"],
            "BANK":   ["HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "BANDHANBNK",
                       "INDUSINDBK", "FEDERALBNK", "AUBANK", "IDFCFIRSTB", "BAJFINANCE"],
            "AUTO":   ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
                       "TVSMOTORS", "MOTHERSON", "BOSCHLTD"],
            "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
                       "ALKEM", "IPCALAB", "GLENMARK"],
            "FMCG":   ["HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP",
                       "MARICO", "COLPAL", "EMAMILTD"],
            "METAL":  ["TATASTEEL", "HINDALCO", "JSWSTEEL", "SAIL", "HINDZINC", "NATIONALUM"],
            "ENERGY": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "POWERGRID", "NTPC"],
            "REALTY": ["DLF", "GODREJPROP", "OBEROIRLTY", "BRIGADE", "PRESTIGE"],
            "INFRA":  ["LT", "ADANIPORTS", "ADANIGREEN", "IRFC", "HAL", "BEL"],
            "CONS":   ["TITAN", "ASIANPAINT", "HAVELLS", "VOLTAS", "WHIRLPOOL"],
        }
        allowed = set()
        for sec in intent.sectors:
            allowed.update(_SECTOR_SYMBOLS.get(sec, []))
        # Also include universe stocks that match sector keywords if not in map
        result = [s for s in result if s.get("symbol", "") in allowed or not allowed]

    # Archetype filter
    if intent.archetypes:
        result = [s for s in result
                  if s.get("archetype", "") in intent.archetypes
                  or s.get("playbook_id", "") in intent.archetypes]

    # Quality tier filter
    tier_order = {"ELITE_A_PLUS": 0, "A": 1, "B": 2, "WATCHLIST": 3, "AVOID": 4}
    if intent.min_quality and intent.min_quality in tier_order:
        max_idx = tier_order[intent.min_quality]
        result = [s for s in result if tier_order.get(s.get("quality_tier", "AVOID"), 4) <= max_idx]

    return result


def _apply_price_filters(setups: list[dict], intent: QueryIntent) -> list[dict]:
    """Apply RSI and structural filters that require fetching price data."""
    if not intent.rsi_max and not intent.rsi_min and not intent.uptrend_only and not intent.near_52w_high:
        return setups

    filtered = []
    for s in setups:
        symbol = s.get("symbol", "")
        try:
            df = _fetch_recent(symbol)
            if df is None or len(df) < 20:
                filtered.append(s)  # keep if can't verify
                continue

            close = df["close"] if "close" in df.columns else df["Close"]
            rsi = _calc_rsi(close)

            if intent.rsi_max is not None and rsi > intent.rsi_max:
                continue
            if intent.rsi_min is not None and rsi < intent.rsi_min:
                continue

            if intent.uptrend_only:
                sma20 = float(close.rolling(20).mean().iloc[-1])
                sma50 = float(close.rolling(50).mean().iloc[-1])
                if not (float(close.iloc[-1]) > sma20 > sma50):
                    continue

            if intent.near_52w_high:
                high_52w = float(close.tail(252).max() if len(close) >= 252 else close.max())
                last = float(close.iloc[-1])
                if abs(last - high_52w) / high_52w > 0.05:
                    continue

            filtered.append(s)
        except Exception:
            filtered.append(s)

    return filtered


def _fetch_recent(symbol: str):
    try:
        from data.kite_client import KiteClient
        from data.instruments import InstrumentManager
        from data.historical import HistoricalDataFetcher
        from datetime import date, timedelta
        kite = KiteClient()
        if kite.is_connected():
            f = HistoricalDataFetcher(kite, InstrumentManager())
            to_dt = date.today().strftime("%Y-%m-%d")
            from_dt = (date.today() - timedelta(days=300)).strftime("%Y-%m-%d")
            return f.fetch(symbol, from_dt, to_dt, interval="day")
    except Exception:
        pass
    return None


def _calc_rsi(close, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return float((100 - 100 / (1 + rs)).iloc[-1])


# ── Streamlit component ───────────────────────────────────────────────────────

def render_nl_query(universe: list[str], current_setups: list[dict]) -> list[dict] | None:
    """
    Renders the NL query bar. Returns filtered setups if a query was run,
    None if user hasn't queried (caller should use original setups).
    """
    st.markdown(
        "<div style='background:#0d1117;border:1px solid #21262d;border-radius:10px;"
        "padding:.6rem 1rem;margin-bottom:1rem'>",
        unsafe_allow_html=True,
    )

    col_input, col_btn, col_clear = st.columns([7, 1.5, 1.2])
    with col_input:
        query = st.text_input(
            "label_hidden",
            placeholder='Ask anything: "VCP setups in IT sector" · "RSI under 45 uptrend stocks" · "Show me A-tier setups"',
            key="nl_query_input",
            label_visibility="collapsed",
        )
    with col_btn:
        run = st.button("Search", key="nl_query_run", use_container_width=True, type="primary")
    with col_clear:
        clear = st.button("Clear", key="nl_query_clear", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state.pop("nl_query_result", None)
        st.session_state.pop("nl_query_explanation", None)
        st.rerun()

    if not run or not query.strip():
        if "nl_query_result" in st.session_state:
            expl = st.session_state.get("nl_query_explanation", "")
            if expl:
                st.caption(f"Showing results for: *{expl}* — {len(st.session_state['nl_query_result'])} matches")
            return st.session_state["nl_query_result"]
        return None

    with st.spinner("Parsing query…"):
        intent = _parse_intent(query.strip())

    st.caption(f"Understood: *{intent.explanation}*")

    # Apply metadata filters first (fast, no data fetch)
    filtered = _apply_filters(current_setups, intent, universe)

    # Apply price-based filters (needs data fetch — only if small set)
    if len(filtered) <= 50 and (intent.rsi_max or intent.rsi_min or intent.uptrend_only or intent.near_52w_high):
        with st.spinner(f"Checking price structure for {len(filtered)} stocks…"):
            filtered = _apply_price_filters(filtered, intent)

    st.session_state["nl_query_result"] = filtered
    st.session_state["nl_query_explanation"] = intent.explanation

    if not filtered:
        st.info("No setups match your query in the current scan. Try a broader filter or re-run the scan.")
        return []

    return filtered
