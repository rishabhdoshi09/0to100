"""QuantTerm professional command center.

This page is a read-only product projection plus durable owner controls. It does not scan, trade,
fetch fundamentals, start workers or mutate a portfolio itself. The small morning-brief helpers at
the bottom are retained as pure compatibility utilities for the older fallback and test contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st

from product import gather_product_inputs
from product.autonomy_status import read_autonomy_status
from product.long_term_store import load_long_term_scan
from product.market_view import current_market_view
from product.paper_status import read_paper_status
from product.scan_store import load_scan, scan_age_hours
from product.workspace import build_command_center_state
from research.autonomy.controls import (
    REFRESH_DATA_NOW,
    RUN_CYCLE_NOW,
    RUN_LONG_TERM_SCAN_NOW,
    RUN_SCAN_NOW,
    request_control,
)
from ui.pro_theme import insight_panel, metric_card, page_header, section_header


@dataclass(frozen=True)
class _MarketFallback:
    health: str = "Unavailable"
    summary: str = "Current market context could not be loaded."
    trade_stance: str = "Do not infer market support while the regime feed is unavailable."
    breadth: str = "Unavailable"
    leaders: tuple = ()
    laggards: tuple = ()
    nifty_change_1d: float = 0.0
    vix: float = 0.0


def _market():
    try:
        return current_market_view()
    except Exception:
        return _MarketFallback()


def _money(value: float) -> str:
    return f"₹{float(value or 0):,.0f}"


def _tone_for_market(health: str) -> str:
    health = str(health).lower()
    if health == "healthy":
        return "good"
    if health == "weak":
        return "bad"
    if health == "unavailable":
        return "warn"
    return "accent"


def _setup_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Stock": row.get("symbol"),
            "Score": row.get("score"),
            "Status": row.get("status"),
            "Price": row.get("price"),
            "Entry": row.get("entry"),
            "Stop": row.get("stop"),
            "Target": row.get("target"),
            "Volume": f"{float(row.get('volume_ratio', 0) or 0):.1f}×",
            "Why": (row.get("reasons") or [row.get("why", "")])[0],
        }
        for row in rows
    ])


def _long_term_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Stock": row.get("symbol"),
            "Class": str(row.get("classification", "")).replace("_", " ").title(),
            "Score": row.get("combined_score"),
            "Quality": row.get("fundamental_score"),
            "Technical": row.get("technical_score"),
            "Coverage": f"{float(row.get('fundamental_coverage', 0) or 0) * 100:.0f}%",
            "Timing": str(row.get("timing", "")).replace("_", " ").title(),
        }
        for row in rows
    ])


def _price_chart(symbol: str) -> bool:
    if not symbol:
        return False
    try:
        from data.bhavcopy_store import get_ohlcv

        frame = get_ohlcv(symbol)
        if frame is None or len(frame) < 5:
            return False
        frame = frame.copy().tail(180)
        frame.columns = [str(col).lower() for col in frame.columns]
        if "close" not in frame.columns:
            return False
        frame["EMA 20"] = frame["close"].ewm(span=20, adjust=False).mean()
        frame["EMA 50"] = frame["close"].ewm(span=50, adjust=False).mean()
        chart = frame[["close", "EMA 20", "EMA 50"]].rename(columns={"close": symbol})
        st.line_chart(chart, height=315, use_container_width=True)
        return True
    except Exception:
        return False


def _position_frame(rows) -> pd.DataFrame:
    output = []
    for raw in rows:
        row = dict(raw) if isinstance(raw, dict) else getattr(raw, "as_dict", lambda: {})()
        output.append({
            "Stock": row.get("symbol"),
            "Entry": row.get("entry_price", row.get("entry")),
            "Current": row.get("current_price", row.get("last_price", row.get("price"))),
            "Stop": row.get("stop_price", row.get("stop")),
            "Target": row.get("target_price", row.get("target")),
            "Risk": row.get("risk_amount"),
            "Strategy": row.get("strategy", row.get("signal_type", "")),
        })
    return pd.DataFrame(output)


def render_command_center() -> None:
    inputs = gather_product_inputs()
    scan = load_scan()
    long_term = load_long_term_scan()
    paper = read_paper_status()
    autonomy = read_autonomy_status()
    market = _market()
    state = build_command_center_state(
        scan_payload=scan,
        long_term_payload=long_term,
        paper=paper,
        autonomy=autonomy,
        market=market,
    )

    data_badge = "Data ready" if inputs.data_ready else "Data attention"
    broker_badge = "Zerodha connected" if inputs.kite_connected else "Zerodha login needed"
    auto_badge = "Supervisor online" if state["autonomy_running"] else "Supervisor offline"
    page_header(
        "Command Center",
        "A single decision surface for market posture, opportunities, portfolio risk and system operations.",
        eyebrow="QuantTerm Professional",
        badges=[
            (broker_badge, "good" if inputs.kite_connected else "warn"),
            (data_badge, "good" if inputs.data_ready else "warn"),
            (auto_badge, "good" if state["autonomy_running"] else "bad"),
        ],
    )

    action_cols = st.columns([1.2, 1.2, 1.2, 1.2, 3.2])
    if action_cols[0].button("Run Market Scan", type="primary", width="stretch"):
        request_control(RUN_SCAN_NOW, reason="owner requested market scan from Command Center")
        st.success("Market scan queued for the autonomy supervisor.")
    if action_cols[1].button("Run Long-Term Scan", width="stretch"):
        request_control(RUN_LONG_TERM_SCAN_NOW, reason="owner requested long-term scan from Command Center")
        st.success("Long-term scan queued for the autonomy supervisor.")
    if action_cols[2].button("Run Paper Cycle", width="stretch"):
        request_control(RUN_CYCLE_NOW, reason="owner requested paper cycle from Command Center")
        st.success("Paper cycle queued for the autonomy supervisor.")
    if action_cols[3].button("Refresh Data", width="stretch"):
        request_control(REFRESH_DATA_NOW, reason="owner requested data refresh from Command Center")
        st.success("Data refresh queued for the autonomy supervisor.")
    scan_age = scan_age_hours(scan)
    age_text = f"Saved scan {scan_age:.1f}h old" if scan_age is not None else "No saved market scan"
    action_cols[4].caption(f"{state['autonomy_plain_state']} · {age_text}")

    metric_cols = st.columns(5)
    with metric_cols[0]:
        metric_card(
            "Market Regime",
            state["market_health"],
            state["breadth"],
            tone=_tone_for_market(state["market_health"]),
        )
    with metric_cols[1]:
        metric_card(
            "Entry-Ready Setups",
            str(state["ready_count"]),
            f"{state['near_breakout_count']} near breakout · {state['scan_universe']:,} universe",
            tone="accent" if state["ready_count"] else "warn",
        )
    with metric_cols[2]:
        metric_card(
            "Long-Horizon Quality",
            str(state["long_term_count"]),
            f"Fundamental coverage {state['fundamental_coverage_pct']:.0f}%",
            tone="accent",
        )
    with metric_cols[3]:
        metric_card(
            "Paper Equity",
            _money(state["paper_equity"]),
            f"{state['paper_return_pct']:+.2f}% · {state['open_position_count']} open",
            tone="good" if state["paper_return_pct"] >= 0 else "bad",
        )
    with metric_cols[4]:
        metric_card(
            "Risk in Market",
            _money(state["open_risk"]),
            "PAPER only · LIVE orders locked",
            tone="warn" if state["open_risk"] else "",
        )

    section_header("Opportunity Board", "Ranked evidence on the left; selected-stock price structure on the right.")
    board_left, board_right = st.columns([1.05, 1.55], gap="large")
    setups = list(state["top_setups"])
    with board_left:
        if setups:
            st.dataframe(_setup_frame(setups), hide_index=True, width="stretch", height=322)
        else:
            st.info("No saved qualifying setup. QuantTerm will not fabricate an opportunity list.")
    with board_right:
        symbols = [str(row.get("symbol", "")) for row in setups if row.get("symbol")]
        selected = st.selectbox("Chart stock", symbols, label_visibility="collapsed") if symbols else ""
        if selected and not _price_chart(selected):
            st.info("Validated price history is not available for this stock yet.")
        elif not selected:
            st.info("Run the market scan to populate the opportunity chart.")

    lower_left, lower_mid, lower_right = st.columns([1.2, 1.05, .9], gap="large")
    with lower_left:
        section_header("Long-Term Intelligence", "Current quality and valuation coverage; separate from trading signals.")
        if state["top_long_term"]:
            st.dataframe(_long_term_frame(state["top_long_term"]), hide_index=True, width="stretch", height=280)
        else:
            st.info("No covered long-term candidate is saved yet.")
    with lower_mid:
        section_header("System Insights", "Plain-language synthesis of the current product state.")
        insight_panel("What matters now", state["insights"])
    with lower_right:
        section_header("Automation", "One supervisor owns scheduling and mutations.")
        heartbeat = state["heartbeat_ist"] or "No heartbeat"
        insight_panel(
            state["autonomy_state"].replace("_", " ").title(),
            [
                state["autonomy_plain_state"],
                f"Heartbeat: {heartbeat}",
                "Automatic paper trading is " + ("enabled." if state["paper_enabled"] else "disabled."),
                "Live broker execution remains locked.",
            ],
            kicker="Operations",
        )

    section_header("Active Paper Positions", "The system's current simulated exposure and predefined exits.")
    if state["open_positions"]:
        st.dataframe(_position_frame(state["open_positions"]), hide_index=True, width="stretch")
    else:
        st.caption("No paper position is open. That can be a valid decision when evidence is insufficient.")


# Compatibility helpers retained for the Streamlit fallback and deterministic tests.
_BIG_NEWS = (
    "trump", "fed", "rate", "rate cut", "rate hike", "inflation", "cpi",
    "tariff", "trade war", "sanction", "war", "ceasefire", "opec", "crude",
    "oil", "gulf", "china", "yuan", "dollar", "rbi", "gdp", "recession",
    "stimulus", "jobs", "payroll", "yield", "bond", "geopolit", "election",
    "budget", "fii", "dii", "downgrade", "upgrade", "default", "banking crisis",
)


def _rank_news(headlines: list[str], n: int = 2) -> list[str]:
    """Rank genuinely market-moving macro/geopolitical headlines and drop routine filler."""
    scored: list[tuple[int, str]] = []
    for headline in headlines:
        headline = (headline or "").strip()
        if len(headline) < 25:
            continue
        hits = sum(1 for keyword in _BIG_NEWS if keyword in headline.lower())
        if hits:
            scored.append((hits, headline))
    scored.sort(key=lambda item: -item[0])
    seen: set[str] = set()
    output: list[str] = []
    for _, headline in scored:
        key = headline.lower()[:40]
        if key in seen:
            continue
        seen.add(key)
        output.append(headline if len(headline) <= 140 else headline[:137] + "…")
        if len(output) >= n:
            break
    return output


@st.cache_data(ttl=900, show_spinner=False)
def _brief_cues() -> dict:
    """Optional read-only global cues; failure returns an honest empty mapping."""
    output: dict = {}
    try:
        import yfinance as yf
        for key, ticker in (
            ("sp500", "^GSPC"), ("nasdaq", "^IXIC"), ("kospi", "^KS11"),
            ("nikkei", "^N225"), ("crude", "CL=F"), ("gold", "GC=F"),
            ("btc", "BTC-USD"),
        ):
            try:
                info = yf.Ticker(ticker).fast_info
                last = float(getattr(info, "last_price", 0) or 0)
                previous = float(getattr(info, "previous_close", 0) or 0)
                if last:
                    output[key] = {
                        "price": last,
                        "chg": ((last - previous) / previous * 100) if previous else 0.0,
                    }
            except Exception:
                continue
    except Exception:
        pass
    return output


@st.cache_data(ttl=900, show_spinner=False)
def _top_market_news(n: int = 2) -> list[str]:
    try:
        from news.fetcher import NewsFetcher
        articles = NewsFetcher().fetch_all(max_age_hours=18)
    except Exception:
        return []
    return _rank_news([article.headline for article in articles], n)


def _conversational_brief(regime: dict) -> str:
    """Build a warm, deterministic market brief from available real cues."""
    cues = _brief_cues()
    hour = datetime.now().hour
    greeting = (
        "Good Morning Guys! 👋" if hour < 12
        else "Afternoon check-in! 👋" if hour < 16
        else "Evening wrap, team! 👋"
    )
    lines = [greeting, ""]

    def direction(change: float) -> str:
        if change > 0.6:
            return "nice green"
        if change > 0.05:
            return "up"
        if change >= -0.05:
            return "flat-ish"
        if change > -0.6:
            return "down"
        return "sharp red"

    sp500 = cues.get("sp500")
    if sp500:
        nasdaq_change = cues.get("nasdaq", {}).get("chg", 0)
        mood = (
            "Nice action last night from the US markets" if sp500["chg"] > 0.2
            else "US markets had a soft session last night" if sp500["chg"] < -0.2
            else "US markets were quiet last night"
        )
        line = f"{mood} — S&P {direction(sp500['chg'])} {sp500['chg']:+.2f}%"
        if nasdaq_change:
            line += f", Nasdaq {nasdaq_change:+.2f}%"
        lines.append(line + ".")

    for headline in _top_market_news(2):
        lines.append(f"📰 {headline}")

    kospi = cues.get("kospi")
    if kospi:
        if kospi["chg"] > 1:
            lines.append(
                f"Kospi is showing a sharp reversal ({kospi['chg']:+.1f}%) — "
                "let's see if that holds till the day's end."
            )
        else:
            nikkei_change = cues.get("nikkei", {}).get("chg", 0)
            line = f"Asia mixed — Kospi {kospi['chg']:+.1f}%"
            if nikkei_change:
                line += f", Nikkei {nikkei_change:+.1f}%"
            lines.append(line + ".")

    nifty = regime.get("nifty_price", 0) or 0
    nifty_change = regime.get("nifty_change_1d", regime.get("nifty_change_pct", 0)) or 0
    if nifty:
        if nifty_change < -0.2:
            lines.append(
                f"Nifty had a down day ({nifty_change:+.2f}%, {nifty:,.0f}) — "
                "some consolidation and a breakout would be ideal here!"
            )
        elif nifty_change > 0.2:
            lines.append(
                f"Nifty closed strong ({nifty_change:+.2f}%, {nifty:,.0f}) — "
                "momentum is on our side, but do not chase."
            )
        else:
            lines.append(f"Nifty flat ({nifty:,.0f}) — range-bound, patience pays.")

    crude = cues.get("crude")
    if crude:
        level = "above" if crude["price"] >= 80 else "below"
        prefix = "still " if level == "above" else ""
        lines.append(f"Crude oil is {prefix}{level} $80 (${crude['price']:.1f}).")

    gold = cues.get("gold")
    if gold:
        line = f"Gold ${gold['price']:,.0f} ({gold['chg']:+.1f}%)"
        if cues.get("btc"):
            line += f", BTC ${cues['btc']['price']:,.0f}"
        lines.append(line + ".")

    lines.extend(["", "Let's have a good day! 🙌"])
    return "\n\n".join(lines)
