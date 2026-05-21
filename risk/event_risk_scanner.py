"""
Event Risk Scanner — The Ronan Defense.

Ronan doesn't fight fair. He drops a bomb while you're mid-trade.

High-impact events that can gap stocks 5-15% overnight:
  - RBI Monetary Policy Committee (MPC) dates
  - Union Budget
  - SEBI board meetings (regulatory announcements)
  - NSE/BSE expiry dates (F&O rollovers cause volatility)
  - Individual stock earnings announcements
  - Index rebalancing dates (Nifty 50 / Nifty Next 50)

This module:
  1. Maintains a calendar of known high-impact dates
  2. Warns if today or tomorrow has a major event
  3. Recommends reducing position size 50% on event days
  4. Blocks overnight holds on stocks with earnings tomorrow
  5. Auto-fetches NSE earnings calendar where possible
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st

from logger import get_logger

log = get_logger(__name__)

_CACHE_FILE = Path("logs/event_calendar_cache.json")


# ── Hardcoded high-impact macro dates (FY 2025-26) ───────────────────────────
# Update annually. Format: "YYYY-MM-DD"
_MACRO_EVENTS: list[dict] = [
    # RBI MPC dates FY26
    {"date": "2025-04-09", "event": "RBI MPC Decision", "type": "RBI", "impact": "HIGH"},
    {"date": "2025-06-06", "event": "RBI MPC Decision", "type": "RBI", "impact": "HIGH"},
    {"date": "2025-08-08", "event": "RBI MPC Decision", "type": "RBI", "impact": "HIGH"},
    {"date": "2025-10-08", "event": "RBI MPC Decision", "type": "RBI", "impact": "HIGH"},
    {"date": "2025-12-05", "event": "RBI MPC Decision", "type": "RBI", "impact": "HIGH"},
    {"date": "2026-02-06", "event": "RBI MPC Decision", "type": "RBI", "impact": "HIGH"},
    # Union Budget
    {"date": "2026-02-01", "event": "Union Budget FY27", "type": "BUDGET", "impact": "EXTREME"},
    # NSE F&O Monthly Expiry (last Thursday of month) — approximate
    {"date": "2025-05-29", "event": "NSE F&O Monthly Expiry", "type": "EXPIRY", "impact": "MEDIUM"},
    {"date": "2025-06-26", "event": "NSE F&O Monthly Expiry", "type": "EXPIRY", "impact": "MEDIUM"},
    {"date": "2025-07-31", "event": "NSE F&O Monthly Expiry", "type": "EXPIRY", "impact": "MEDIUM"},
    {"date": "2025-08-28", "event": "NSE F&O Monthly Expiry", "type": "EXPIRY", "impact": "MEDIUM"},
    {"date": "2025-09-25", "event": "NSE F&O Monthly Expiry", "type": "EXPIRY", "impact": "MEDIUM"},
    {"date": "2025-10-30", "event": "NSE F&O Monthly Expiry", "type": "EXPIRY", "impact": "MEDIUM"},
    {"date": "2025-11-27", "event": "NSE F&O Monthly Expiry", "type": "EXPIRY", "impact": "MEDIUM"},
    {"date": "2025-12-25", "event": "NSE F&O Monthly Expiry", "type": "EXPIRY", "impact": "MEDIUM"},
]

_IMPACT_CONFIG = {
    "EXTREME": {"color": "#ff4b4b", "size_mult": 0.0,  "block_overnight": True,  "label": "NO TRADE"},
    "HIGH":    {"color": "#f97316", "size_mult": 0.5,  "block_overnight": True,  "label": "HALF SIZE"},
    "MEDIUM":  {"color": "#f59e0b", "size_mult": 0.75, "block_overnight": False, "label": "REDUCE SIZE"},
    "LOW":     {"color": "#8892a4", "size_mult": 1.0,  "block_overnight": False, "label": "NORMAL"},
}


@dataclass
class EventRiskReport:
    today_events: list[dict]
    tomorrow_events: list[dict]
    week_events: list[dict]
    max_impact: str                 # EXTREME | HIGH | MEDIUM | LOW | NONE
    size_multiplier: float          # 0.0 to 1.0
    block_overnight: bool
    action: str                     # plain English recommendation
    symbols_with_earnings: list[str] = field(default_factory=list)


def get_event_risk(symbols: Optional[list[str]] = None) -> EventRiskReport:
    """
    Returns event risk for today and tomorrow.
    Optionally checks earnings for specific symbols.
    """
    today     = date.today()
    tomorrow  = today + timedelta(days=1)
    week_end  = today + timedelta(days=7)

    today_events    = [e for e in _MACRO_EVENTS if e["date"] == today.isoformat()]
    tomorrow_events = [e for e in _MACRO_EVENTS if e["date"] == tomorrow.isoformat()]
    week_events     = [e for e in _MACRO_EVENTS
                       if today.isoformat() < e["date"] <= week_end.isoformat()]

    # Try fetching earnings from NSE
    earnings_today    = _fetch_earnings_dates(today)
    earnings_tomorrow = _fetch_earnings_dates(tomorrow)

    for sym, dt in earnings_today:
        today_events.append({"date": dt, "event": f"{sym} Earnings", "type": "EARNINGS", "impact": "HIGH"})
    for sym, dt in earnings_tomorrow:
        tomorrow_events.append({"date": dt, "event": f"{sym} Earnings", "type": "EARNINGS", "impact": "HIGH"})

    # Filter by universe if provided
    earnings_symbols = []
    if symbols:
        earnings_symbols = [sym for sym, _ in earnings_tomorrow if sym in symbols]

    # Determine worst impact
    all_relevant = today_events + tomorrow_events
    impacts = [e.get("impact", "LOW") for e in all_relevant]
    impact_rank = {"EXTREME": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    max_impact = max(impacts, key=lambda x: impact_rank.get(x, 0)) if impacts else "NONE"

    cfg = _IMPACT_CONFIG.get(max_impact, _IMPACT_CONFIG["LOW"])
    size_mult = cfg["size_mult"] if max_impact != "NONE" else 1.0
    block_overnight = cfg["block_overnight"] if max_impact != "NONE" else False

    # Build action string
    if max_impact == "NONE":
        action = "No major events today or tomorrow. Normal trading."
    elif max_impact == "EXTREME":
        event_names = ", ".join(e["event"] for e in all_relevant)
        action = f"🚨 EXTREME EVENT: {event_names}. NO new trades. Close all positions before EOD."
    elif max_impact == "HIGH":
        event_names = ", ".join(e["event"] for e in all_relevant)
        action = f"⚠️ HIGH IMPACT: {event_names}. Use 50% position size. No overnight holds."
    else:
        event_names = ", ".join(e["event"] for e in all_relevant)
        action = f"📅 {event_names}. Reduce size to 75%. Extra caution near EOD."

    if earnings_symbols:
        action += f" Holdings with earnings tomorrow: {', '.join(earnings_symbols)} — exit before close."

    log.info("event_risk_computed", max_impact=max_impact, today_events=len(today_events),
             tomorrow_events=len(tomorrow_events))

    return EventRiskReport(
        today_events=today_events,
        tomorrow_events=tomorrow_events,
        week_events=week_events,
        max_impact=max_impact,
        size_multiplier=size_mult,
        block_overnight=block_overnight,
        action=action,
        symbols_with_earnings=earnings_symbols,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_earnings_dates(target_date: date) -> list[tuple[str, str]]:
    """
    Fetch earnings announcements for target_date from NSE.
    Returns list of (symbol, date_str) tuples.
    """
    results = []
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                   "Referer": "https://www.nseindia.com"}
        # First get cookies
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers, timeout=5)
        resp = s.get(
            "https://www.nseindia.com/api/event-calendar",
            headers=headers, timeout=8,
        )
        data = resp.json()
        for item in data:
            if item.get("date", "")[:10] == target_date.isoformat():
                sym = item.get("symbol", "")
                purpose = str(item.get("purpose", "")).lower()
                if sym and ("result" in purpose or "dividend" in purpose or "earnings" in purpose):
                    results.append((sym.upper(), target_date.isoformat()))
    except Exception as e:
        log.debug("earnings_fetch_failed", error=str(e))
    return results


def render_event_risk_banner() -> None:
    """Render a compact event risk banner for the dashboard/scanner."""
    report = get_event_risk()
    if report.max_impact == "NONE":
        return

    cfg = _IMPACT_CONFIG.get(report.max_impact, _IMPACT_CONFIG["LOW"])
    color = cfg["color"]
    label = cfg["label"]

    st.markdown(
        f"<div style='background:{color}15;border:1px solid {color}55;border-radius:10px;"
        f"padding:.6rem 1rem;margin:.4rem 0;display:flex;justify-content:space-between;"
        f"align-items:center'>"
        f"<div style='color:{color};font-weight:700;font-size:.82rem'>"
        f"📅 EVENT RISK · {label}</div>"
        f"<div style='color:#c8cfe0;font-size:.78rem'>{report.action}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if report.week_events:
        with st.expander("📅 Upcoming events this week", expanded=False):
            for e in report.week_events:
                impact_color = _IMPACT_CONFIG.get(e.get("impact","LOW"), {}).get("color", "#8892a4")
                st.markdown(
                    f"<span style='color:{impact_color};font-weight:600'>{e['date']}</span>"
                    f" — {e['event']} "
                    f"<span style='color:{impact_color};font-size:.7rem'>[{e.get('impact','?')}]</span>",
                    unsafe_allow_html=True,
                )
