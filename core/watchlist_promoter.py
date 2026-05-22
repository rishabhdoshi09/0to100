"""
Smart Watchlist Promoter — automatically surfaces stocks when setups form.

Pipeline:
  Screener (fundamentals) → Watchlist (your radar) → Institutional (active setup)

The promoter watches your screener results and watchlist.
When a watchlist stock develops an active setup, it auto-flags it.
When a screener stock hits a setup trigger, it suggests adding to watchlist.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = Path("logs/watchlist_promoter.db")


# ── Database initialisation ───────────────────────────────────────────────────

def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS promotion_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            symbol      TEXT,
            event_type  TEXT,
            archetype   TEXT,
            tier        TEXT,
            score       REAL,
            actioned    INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _setup_key(symbol: str, archetype: str) -> str:
    """Stable key to identify a setup for deduplication."""
    return f"{symbol.upper()}::{archetype.upper()}"


def _seen_in_db(symbol: str, archetype: str, hours: int = 24) -> bool:
    """Return True if this setup was already recorded in the last `hours` hours."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT id FROM promotion_events
        WHERE symbol = ? AND archetype = ? AND timestamp >= ?
        ORDER BY id DESC LIMIT 1
        """,
        (symbol.upper(), archetype.upper(), cutoff),
    ).fetchone()
    conn.close()
    return row is not None


def _extract(item, key: str, default=None):
    """Safely get a key from a dict or an attribute from an object."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


# ── Public API ────────────────────────────────────────────────────────────────

def check_watchlist_for_setups(
    watchlist_symbols: list[str],
    setups: list[dict],
) -> list[dict]:
    """
    Cross-references watchlist symbols against current scan results (setups).

    Parameters
    ----------
    watchlist_symbols : list of NSE ticker strings on the user's watchlist
    setups            : list of setup dicts / objects from the IQ2 pipeline cache.
                        Each item should have at least: symbol, pattern/archetype,
                        tier, confidence/score.

    Returns
    -------
    list of dicts with keys:
        symbol, archetype, tier, score, is_new
    """
    _init_db()

    wl_set = {s.strip().upper() for s in watchlist_symbols if s}
    matches: list[dict] = []

    for setup in setups:
        sym       = str(_extract(setup, "symbol", "") or "").upper()
        archetype = str(
            _extract(setup, "archetype",
                     _extract(setup, "pattern", "UNKNOWN")) or "UNKNOWN"
        ).upper()
        tier      = str(_extract(setup, "tier", "UNKNOWN") or "UNKNOWN")
        raw_score = _extract(setup, "score", _extract(setup, "confidence", 0.0))
        try:
            score = float(raw_score) * 100 if float(raw_score) <= 1.0 else float(raw_score)
        except (TypeError, ValueError):
            score = 0.0

        if sym not in wl_set:
            continue

        is_new = not _seen_in_db(sym, archetype, hours=24)
        matches.append({
            "symbol":    sym,
            "archetype": archetype,
            "tier":      tier,
            "score":     score,
            "is_new":    is_new,
        })

        # Record in DB (idempotent for old ones)
        if is_new:
            record_promotion(sym, "SETUP_FORMED", archetype, tier, score)

    return matches


def check_screener_for_breakouts(
    screener_results: list[dict],
    setups: list[dict],
) -> list[dict]:
    """
    Returns screener stocks that now have active setups — candidates to add to watchlist.

    Parameters
    ----------
    screener_results : list of dicts from the fundamental screener.
                       Each must have at least a 'symbol' key.
    setups           : list of setup dicts / objects from the IQ2 pipeline cache.

    Returns
    -------
    list of dicts: symbol, archetype, tier, score, is_new
    """
    _init_db()

    # Build a lookup: symbol → list[setup]
    setup_map: dict[str, list] = {}
    for s in setups:
        sym = str(_extract(s, "symbol", "") or "").upper()
        if sym:
            setup_map.setdefault(sym, []).append(s)

    suggestions: list[dict] = []
    for result in screener_results:
        sym = str(_extract(result, "symbol", "") or "").upper()
        if not sym or sym not in setup_map:
            continue

        for setup in setup_map[sym]:
            archetype = str(
                _extract(setup, "archetype",
                         _extract(setup, "pattern", "UNKNOWN")) or "UNKNOWN"
            ).upper()
            tier      = str(_extract(setup, "tier", "UNKNOWN") or "UNKNOWN")
            raw_score = _extract(setup, "score", _extract(setup, "confidence", 0.0))
            try:
                score = float(raw_score) * 100 if float(raw_score) <= 1.0 else float(raw_score)
            except (TypeError, ValueError):
                score = 0.0

            is_new = not _seen_in_db(sym, archetype, hours=24)
            suggestions.append({
                "symbol":    sym,
                "archetype": archetype,
                "tier":      tier,
                "score":     score,
                "is_new":    is_new,
            })

            if is_new:
                record_promotion(sym, "SUGGEST_ADD", archetype, tier, score)

    return suggestions


def record_promotion(
    symbol: str,
    event_type: str,
    archetype: str,
    tier: str,
    score: float,
) -> None:
    """
    Persist a promotion event to the SQLite database.

    event_type: "SETUP_FORMED" | "SETUP_BROKEN" | "SUGGEST_ADD"
    """
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO promotion_events
            (timestamp, symbol, event_type, archetype, tier, score, actioned)
        VALUES (?, ?, ?, ?, ?, ?, 0)
        """,
        (
            datetime.utcnow().isoformat(),
            symbol.upper(),
            event_type,
            archetype.upper(),
            tier,
            float(score),
        ),
    )
    conn.commit()
    conn.close()


def mark_actioned(event_id: int) -> None:
    """Mark an event as actioned (user clicked a CTA button)."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE promotion_events SET actioned = 1 WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()


def get_pending_suggestions() -> list[dict]:
    """
    Return unactioned suggestions from the last 24 hours, newest first.
    """
    _init_db()
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT * FROM promotion_events
        WHERE actioned = 0 AND timestamp >= ?
        ORDER BY id DESC
        """,
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Streamlit UI component ────────────────────────────────────────────────────

def render_promotion_alerts() -> None:
    """
    Streamlit component showing pending promotion alerts as dismissable cards.

    Displays:
    - SETUP_FORMED  → watchlist stock now has an active institutional setup
    - SUGGEST_ADD   → screener stock just formed a setup (candidate for watchlist)
    """
    try:
        import streamlit as st
    except ImportError:
        return

    suggestions = get_pending_suggestions()
    if not suggestions:
        return

    st.markdown(
        "<div style='font-size:.72rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.07em;margin-bottom:.5rem'>Smart Promoter Alerts</div>",
        unsafe_allow_html=True,
    )

    to_dismiss: list[int] = []

    for item in suggestions:
        ev_id     = item["id"]
        sym       = item["symbol"]
        etype     = item["event_type"]
        archetype = item["archetype"].replace("_", " ").title()
        score     = float(item["score"])

        if etype == "SETUP_FORMED":
            icon        = "🟢"
            headline    = (
                f"{icon} **{sym}** is now showing a **{archetype}** "
                f"(Score: {score:.0f}) — it's on your watchlist!"
            )
            cta_primary = ("View in Institutional", "Institutional")
        else:  # SUGGEST_ADD
            icon        = "💡"
            headline    = (
                f"{icon} **{sym}** (from your screener) just formed a "
                f"**{archetype}** (Score: {score:.0f})"
            )
            cta_primary = ("Add to Watchlist", None)

        with st.container():
            col_text, col_actions = st.columns([5, 3])

            with col_text:
                st.markdown(headline)

            with col_actions:
                btn_cols = st.columns(3)

                # Primary CTA
                with btn_cols[0]:
                    if etype == "SETUP_FORMED":
                        if st.button(
                            "View →",
                            key=f"promo_view_{ev_id}",
                            use_container_width=True,
                        ):
                            try:
                                from ui.context import go_to
                                go_to("🏛️  Institutional", symbol=sym)
                            except Exception:
                                st.session_state["selected_symbol"] = sym
                                st.session_state["_nav_pending"] = "🏛️  Institutional"
                                st.rerun()
                    else:
                        if st.button(
                            "Add →",
                            key=f"promo_add_{ev_id}",
                            use_container_width=True,
                        ):
                            # Auto-add to watchlist DB
                            try:
                                from ui.watchlist import _add_symbol, _get_price
                                live_px = _get_price(sym)
                                _add_symbol(
                                    symbol=sym,
                                    buy_low=None,
                                    buy_high=None,
                                    target=None,
                                    stop=None,
                                    notes=f"Auto-added by Promoter: {archetype}",
                                    added_price=live_px,
                                )
                            except Exception:
                                pass
                            mark_actioned(ev_id)
                            st.success(f"Added {sym} to watchlist.")
                            st.rerun()

                # Setup view
                with btn_cols[1]:
                    if st.button(
                        "Setup →",
                        key=f"promo_setup_{ev_id}",
                        use_container_width=True,
                    ):
                        try:
                            from ui.context import go_to
                            go_to("🏛️  Institutional", symbol=sym)
                        except Exception:
                            st.session_state["selected_symbol"] = sym
                            st.session_state["_nav_pending"] = "🏛️  Institutional"
                            st.rerun()

                # Dismiss
                with btn_cols[2]:
                    if st.button(
                        "Dismiss",
                        key=f"promo_dismiss_{ev_id}",
                        use_container_width=True,
                    ):
                        mark_actioned(ev_id)
                        to_dismiss.append(ev_id)

        st.markdown(
            "<div style='height:.25rem;border-bottom:1px solid rgba(255,255,255,.05);"
            "margin-bottom:.5rem'></div>",
            unsafe_allow_html=True,
        )

    if to_dismiss:
        st.rerun()
