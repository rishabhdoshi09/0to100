"""Reco-style desk surfaces over saved product state.

Reads the persisted whole-market scan only. Does not start scanners, workers,
or paper loops. Streamlit remains read-only versus autonomy.
"""
from __future__ import annotations

from html import escape
from typing import Any, Mapping

from product.paper_lessons import (
    BACKTEST_DOES_NOT_CHANGE,
    BACKTEST_PURPOSE,
    PAPER_TO_BACKTEST,
    paper_loss_lessons,
)
from product.scan_store import load_scan, scan_age_hours, watchlist_rows

HOW_THE_DESK_WORKS = (
    "Today shows SEPA-qualified Best Setups from the last scan, then the "
    "scanner watchlist. Paper Desk takes simulated trades. After a paper loss, "
    "Backtest is how you decide whether that style deserves another attempt."
)


def setup_badge(row: Mapping[str, Any]) -> tuple[str, str]:
    sepa_verdict = str(row.get("sepa_verdict") or "").upper()
    if sepa_verdict == "STRONG":
        return "SEPA qualified", "buy"
    if sepa_verdict == "CONSTRUCTIVE":
        return "Setup forming", "watch"
    status = str(row.get("status") or "")
    if status == "Ready to trade":
        return "Buy Setup", "buy"
    if status == "Wait for pullback":
        return "Wait", "wait"
    if status == "Watch for breakout":
        return "Watch", "watch"
    if str(row.get("verdict") or "").upper() in {"AVOID", "SELL"}:
        return "Avoid", "avoid"
    return "Watch", "watch"


def reco_card_html(row: Mapping[str, Any]) -> str:
    symbol = escape(str(row.get("symbol") or "?"))
    company = escape(str(row.get("company") or symbol))
    price = float(row.get("price") or 0.0)
    entry = float(row.get("entry") or 0.0)
    stop = float(row.get("stop") or 0.0)
    target = float(row.get("target") or 0.0)
    why = escape(str(row.get("why") or ""))
    badge, kind = setup_badge(row)
    risk = entry - stop if entry and stop and entry > stop else 0.0
    rr = ((target - entry) / risk) if risk > 0 and target else 0.0
    sepa_score = row.get("sepa_score")
    sepa_passed = row.get("sepa_passed")
    sepa_total = row.get("sepa_total")
    if entry and stop:
        levels = (
            f"Entry ₹{entry:,.0f}  ·  Stop ₹{stop:,.0f}  ·  "
            f"Target ₹{target:,.0f}" + (f"  ·  {rr:.1f}×" if rr else "")
        )
    else:
        levels = str(row.get("status") or "Watch")
    if sepa_score is not None:
        extra = f"SEPA {int(sepa_score)}/100"
        if sepa_passed is not None and sepa_total:
            extra += f"  ·  {int(sepa_passed)}/{int(sepa_total)} rules"
        levels = f"{extra}  ·  {levels}" if levels else extra
    price_html = f"₹{price:,.2f}" if price else "Price n/a"
    why_html = f"<div class='why'>{why}</div>" if why else ""
    return (
        f"<div class='reco-card'>"
        f"<div class='row'><span class='sym'>{symbol}</span>"
        f"<span class='reco-badge {kind}'>{badge}</span></div>"
        f"<div class='co'>{company}</div>"
        f"<div class='px'>{price_html}</div>"
        f"<div class='lv'>{levels}</div>"
        f"{why_html}</div>"
    )


def _money(value: float) -> str:
    return f"₹{float(value):,.0f}"


def sepa_card_row(sepa: Mapping[str, Any], scan_row: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a SEPA score onto a scan row for Reco cards. Display only."""
    from product.sepa_setup import sepa_card_fields

    card = dict(scan_row)
    card.update(sepa_card_fields(sepa))
    quote = dict(sepa.get("quote") or {})
    if quote.get("close") and not card.get("price"):
        card["price"] = quote.get("close")
    return card


def rank_sepa_from_scan(
    payload: Mapping[str, Any] | None,
    *,
    limit: int = 8,
    score_cap: int = 24,
    max_seconds: float = 8.0,
    min_score: int = 40,
) -> tuple[list[dict[str, Any]], str]:
    """Best Setups from the saved scan. Never starts a scanner."""
    from product.sepa_setup import rank_best_setups

    records = list((payload or {}).get("records") or [])
    if not records:
        return [], "No saved scan yet — SEPA ranking needs the last whole-market scan."
    cache_key = f"{payload.get('scanned_at')}:{limit}:{score_cap}:{min_score}"
    ranked, note = rank_best_setups(
        records,
        limit=limit,
        score_cap=score_cap,
        min_score=min_score,
        max_seconds=max_seconds,
        cache_key=cache_key,
    )
    cards = [sepa_card_row(sepa, row) for sepa, row in ranked]
    return cards, note


def render_sepa_best_setups(
    *,
    scan_payload: Mapping[str, Any] | None = None,
    limit: int = 8,
    score_cap: int = 24,
    max_seconds: float = 8.0,
    heading: str = "Best Setups · SEPA qualified",
) -> None:
    """Reco cards for Minervini Stage-2 names. Research overlay, not a buy list."""
    import streamlit as st

    payload = dict(scan_payload) if scan_payload is not None else load_scan()
    st.markdown(
        "<div class='qt-section'><div class='qt-eyebrow'>Top stocks</div>"
        f"<div class='t'>{escape(heading)}</div>"
        "<div class='s'>Minervini 7-rule Stage-2 template on official NSE history. "
        "A qualify is research — it does not change today's BUY list or paper autopilot.</div></div>",
        unsafe_allow_html=True,
    )
    if not payload or not payload.get("records"):
        st.info("No saved scan yet. Keep autonomy running, or open Setups → Momentum and queue a scan. "
                "SEPA ranking cannot run without that list.")
        return
    with st.spinner("Ranking last-scan names on the SEPA template…"):
        cards, note = rank_sepa_from_scan(
            payload, limit=limit, score_cap=score_cap, max_seconds=max_seconds,
        )
    st.caption(note)
    if not cards:
        return
    cols = st.columns(2)
    for idx, row in enumerate(cards):
        with cols[idx % 2]:
            st.markdown(reco_card_html(row), unsafe_allow_html=True)


def render_today_board(*, scan_payload: Mapping[str, Any] | None = None, limit: int = 6) -> None:
    """Today's Reco cards from the saved scan. Optional payload is for tests/callers."""
    import streamlit as st

    payload = dict(scan_payload) if scan_payload is not None else load_scan()
    rows = watchlist_rows(payload, limit=limit)
    st.markdown(
        "<div class='qt-section'><div class='qt-eyebrow'>Scanner watchlist</div>"
        "<div class='t'>What the momentum scan is watching</div>"
        "<div class='s'>Saved whole-market scan. Not the same as SEPA-qualified Best Setups above.</div></div>",
        unsafe_allow_html=True,
    )
    if not rows:
        st.info("No saved scan yet. Keep autonomy running, or open Setups and queue a scan. "
                "An empty list is not the same as 'no trade today'.")
        return
    age = scan_age_hours(payload)
    age_bit = f" · {age:.1f} hours old" if age is not None else ""
    st.caption(f"From the saved whole-market scan{age_bit}")
    cols = st.columns(2)
    for idx, row in enumerate(rows):
        with cols[idx % 2]:
            st.markdown(reco_card_html(row), unsafe_allow_html=True)


def render_market_strip() -> None:
    import streamlit as st

    try:
        from product.market_view import current_market_view
        view = current_market_view()
        health = view.health
        color = {"Healthy": "#34d399", "Weak": "#fb7185"}.get(health, "#fbbf24")
        nifty = f"Nifty {view.nifty_change_1d:+.2f}%"
        vix = f"VIX {view.vix:.1f}" if view.vix else "VIX n/a"
        st.markdown(
            f"<div class='reco-strip'>"
            f"<div class='row' style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='font-weight:750;color:{color}'>Market: {health}</span>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:.82rem;color:#8b94a7'>"
            f"{nifty}  ·  {vix}</span></div>"
            f"<div style='font-size:.82rem;color:#c9d1d9;margin-top:.35rem'>{view.summary}</div>"
            f"<div style='font-size:.8rem;color:#8b94a7;margin-top:.25rem'>{view.trade_stance}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.warning("Market condition is temporarily unavailable — the data feed did not respond. "
                   "This is not a 'no setups' result.")
        st.caption(f"Details: {exc}")


def render_how_the_desk_works() -> None:
    import streamlit as st

    st.markdown(
        "<div class='reco-how'><div class='qt-eyebrow'>How to use this desk</div>"
        "<ol>"
        "<li><span class='k'>Today</span> — SEPA-qualified Best Setups first, then the scanner watchlist. "
        "If SEPA is empty, the scan names did not clear the Stage-2 floor — that is not 'no scan yet'.</li>"
        "<li><span class='k'>Setups</span> — Best Setups (SEPA), Momentum, Conviction, Long-term. Do not mix them.</li>"
        "<li><span class='k'>Paper Desk</span> — the system takes simulated trades. You enable, pause, "
        "and review. You do not click broker orders here.</li>"
        "<li><span class='k'>Backtest</span> — after a paper loss, test that stock on past data. "
        f"{BACKTEST_DOES_NOT_CHANGE}</li>"
        "</ol></div>",
        unsafe_allow_html=True,
    )


def render_paper_loss_followup(closed_trades) -> None:
    import streamlit as st

    lessons = paper_loss_lessons(closed_trades, limit=4)
    if not lessons:
        return
    st.markdown(
        "<div class='qt-section'><div class='qt-eyebrow'>After paper losses</div>"
        f"<div class='t'>Next trade quality, not more clicking</div>"
        f"<div class='s'>{PAPER_TO_BACKTEST}</div></div>",
        unsafe_allow_html=True,
    )
    for lesson in lessons:
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(
                f"**{lesson['headline']}**  ·  {_money(lesson['pnl'])}"
            )
            st.caption(lesson["next_step"])
        with cols[1]:
            if st.button("Test in Backtest", key=f"qt_bt_{lesson['symbol']}", width="stretch"):
                st.session_state["qt_backtest_symbol"] = lesson["symbol"]
                st.session_state["qt_backtest_universe"] = "Selected stock"
                st.success(f"{lesson['symbol']} is filled in Backtest. Open Backtest in the sidebar.")
    st.caption(BACKTEST_PURPOSE)
