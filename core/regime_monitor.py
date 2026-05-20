"""
core/regime_monitor.py — Regime history tracker and shift detector.

Tracks every regime reading in a lightweight SQLite database, detects
when the market regime changes, and provides human-readable implications
and playbook guidance for the current environment.

Public API:
    RegimeEvent        — dataclass describing a regime transition
    RegimeMonitor      — main class; importable without Streamlit
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Playbook mappings per regime
# ---------------------------------------------------------------------------

_REGIME_PLAYBOOKS: dict[str, list[str]] = {
    "TRENDING_BULL": ["VCP_BREAKOUT", "MOMENTUM_EXPANSION", "EARLY_LEADER", "SECTOR_MOMENTUM"],
    "EXPANSION":     ["MOMENTUM_EXPANSION", "SECTOR_BREAKOUT", "VCP_BREAKOUT", "BROAD_MARKET_LONG"],
    "CHOPPY":        ["RANGE_FADE", "MEAN_REVERSION"],
    "COMPRESSION":   ["ACCUMULATION_BREAKOUT", "VCP_BREAKOUT"],
    "DISTRIBUTION":  ["DEFENSIVE_POSITIONING", "CASH_CONSERVATION"],
    "TRENDING_BEAR": ["FAILED_BREAKOUT_REVERSAL", "MEAN_REVERSION"],
}

_REGIME_AVOID: dict[str, list[str]] = {
    "TRENDING_BULL": ["MEAN_REVERSION", "COUNTER_TREND_SHORT"],
    "EXPANSION":     ["MEAN_REVERSION"],
    "CHOPPY":        ["MOMENTUM_EXPANSION", "BREAKOUT_BUY"],
    "COMPRESSION":   ["MOMENTUM_CHASING", "LATE_MOMENTUM"],
    "DISTRIBUTION":  ["BREAKOUT_BUY", "VCP_BREAKOUT", "EARLY_LEADER"],
    "TRENDING_BEAR": ["BREAKOUT_BUY", "MOMENTUM_EXPANSION", "VCP_BREAKOUT"],
}

_REGIME_IMPLICATIONS: dict[str, list[str]] = {
    "TRENDING_BULL": [
        "TRENDING_BULL: Favour VCP_BREAKOUT and EARLY_LEADER setups",
        "TRENDING_BULL: Increase position sizing to 1.15–1.25x normal",
        "TRENDING_BULL: Trail stops aggressively — let winners run",
        "TRENDING_BULL: Focus on stocks near 52-week highs with tight bases",
    ],
    "EXPANSION": [
        "EXPANSION: Maximum aggression — all breakout plays valid",
        "EXPANSION: MOMENTUM_EXPANSION and SECTOR_BREAKOUT are highest probability",
        "EXPANSION: Increase position sizing to 1.10x; breadth supportive",
        "EXPANSION: Move fast — expansion phases are brief; act on breakouts same day",
    ],
    "CHOPPY": [
        "CHOPPY: Reduce position size to 0.7x — mean reversion only",
        "CHOPPY: Avoid breakout entries; they whipsaw in this regime",
        "CHOPPY: Widen stops to avoid noise-driven exits",
        "CHOPPY: Prefer fading extremes (RANGE_FADE) over trend following",
    ],
    "COMPRESSION": [
        "COMPRESSION: Wait mode — small size, no new breakouts yet",
        "COMPRESSION: Accumulate stocks building bases quietly",
        "COMPRESSION: Set alerts for volume expansion — the move is coming",
        "COMPRESSION: Keep dry powder; breakout plays will arrive soon",
    ],
    "DISTRIBUTION": [
        "DISTRIBUTION: Reduce longs — institutions are selling into strength",
        "DISTRIBUTION: Look for failed breakouts as short signals",
        "DISTRIBUTION: Tighten stops on all open positions",
        "DISTRIBUTION: Avoid adding to positions; protect capital",
    ],
    "TRENDING_BEAR": [
        "TRENDING_BEAR: Cash or short bias — strict stop discipline required",
        "TRENDING_BEAR: No new long breakout trades",
        "TRENDING_BEAR: FAILED_BREAKOUT_REVERSAL setups have highest edge",
        "TRENDING_BEAR: Reduce max position count; preserve capital",
    ],
}


def _vix_implications(vix: float) -> list[str]:
    """Return 1-2 actionable VIX-based implications."""
    if vix <= 0:
        return []
    if vix > 25:
        return [
            f"VIX {vix:.1f}: Extreme fear — no new longs, full risk-off mode",
            "VIX >25: Consider hedges or reduce gross exposure significantly",
        ]
    if vix > 20:
        return [
            f"VIX {vix:.1f}: Elevated — widen stops, reduce position size",
            "VIX >20: Avoid initiating breakout trades until volatility cools",
        ]
    if vix >= 14:
        return [
            f"VIX {vix:.1f}: Normal range — standard sizing and stop distances",
        ]
    # vix < 14
    return [
        f"VIX {vix:.1f}: Low volatility — tighten stops, targets easier to hit",
        "Low VIX: Trending setups work well; compression breakouts imminent",
    ]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RegimeEvent:
    timestamp: str
    from_regime: str
    to_regime: str
    regime_score: float
    vix: float
    nifty_price: float
    implications: list[str]
    playbooks_to_use: list[str]
    playbooks_to_avoid: list[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "from_regime": self.from_regime,
            "to_regime": self.to_regime,
            "regime_score": self.regime_score,
            "vix": self.vix,
            "nifty_price": self.nifty_price,
            "implications": self.implications,
            "playbooks_to_use": self.playbooks_to_use,
            "playbooks_to_avoid": self.playbooks_to_avoid,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegimeMonitor:
    """
    Tracks regime history in a SQLite database and detects regime shifts.

    Every call to check_and_record() stores the current reading. When the
    market_regime changes from the last stored value, a RegimeEvent is
    returned and persisted to the regime_shifts table.
    """

    def __init__(self, db_path: str = "logs/regime_history.db"):
        self.db_path = Path(db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create DB and tables if they don't exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS regime_readings (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    market_regime   TEXT NOT NULL,
                    volatility_regime TEXT,
                    breadth_label   TEXT,
                    risk_mode       TEXT,
                    regime_score    REAL,
                    vix             REAL,
                    nifty_price     REAL
                );

                CREATE TABLE IF NOT EXISTS regime_shifts (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    from_regime     TEXT NOT NULL,
                    to_regime       TEXT NOT NULL,
                    regime_score    REAL,
                    vix             REAL,
                    nifty_price     REAL
                );
            """)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("RegimeMonitor._ensure_schema failed: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _last_recorded_regime(self) -> Optional[str]:
        """Return the market_regime from the most recent reading, or None."""
        try:
            conn = self._connect()
            row = conn.execute(
                "SELECT market_regime FROM regime_readings ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return row["market_regime"] if row else None
        except Exception as exc:
            logger.warning("RegimeMonitor._last_recorded_regime failed: %s", exc)
            return None

    def _store_reading(self, regime_state) -> None:
        """Persist every regime reading (regardless of shift)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT INTO regime_readings
                    (timestamp, market_regime, volatility_regime,
                     breadth_label, risk_mode, regime_score, vix, nifty_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    getattr(regime_state, "market_regime", "UNKNOWN"),
                    getattr(regime_state, "volatility_regime", None),
                    getattr(regime_state, "breadth_label", None),
                    getattr(regime_state, "risk_mode", None),
                    getattr(regime_state, "regime_score", None),
                    getattr(regime_state, "vix", None),
                    getattr(regime_state, "nifty_price", None),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("RegimeMonitor._store_reading failed: %s", exc)

    def _store_shift(
        self,
        from_regime: str,
        to_regime: str,
        regime_score: float,
        vix: float,
        nifty_price: float,
        ts: str,
    ) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT INTO regime_shifts
                    (timestamp, from_regime, to_regime, regime_score, vix, nifty_price)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, from_regime, to_regime, regime_score, vix, nifty_price),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("RegimeMonitor._store_shift failed: %s", exc)

    def _build_event(
        self,
        from_regime: str,
        to_regime: str,
        regime_score: float,
        vix: float,
        nifty_price: float,
        ts: str,
    ) -> RegimeEvent:
        implications = self._implications_for(to_regime, vix)
        playbooks_use = list(_REGIME_PLAYBOOKS.get(to_regime, []))
        playbooks_avoid = list(_REGIME_AVOID.get(to_regime, []))
        return RegimeEvent(
            timestamp=ts,
            from_regime=from_regime,
            to_regime=to_regime,
            regime_score=regime_score,
            vix=vix,
            nifty_price=nifty_price,
            implications=implications,
            playbooks_to_use=playbooks_use,
            playbooks_to_avoid=playbooks_avoid,
        )

    def _implications_for(self, regime: str, vix: float) -> list[str]:
        """Combine regime-level and VIX-level implications."""
        regime_lines = list(_REGIME_IMPLICATIONS.get(regime, [
            f"{regime}: No specific playbook guidance available"
        ]))
        vix_lines = _vix_implications(vix)
        # Return at most 5 lines total, regime lines take priority
        combined = regime_lines + vix_lines
        return combined[:5]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_record(self, regime_state) -> Optional[RegimeEvent]:
        """
        Call with the current RegimeState object.

        Stores every regime reading (for trend analysis).
        Returns a RegimeEvent if market_regime changed from the last recorded
        value, else returns None.
        """
        current_regime = getattr(regime_state, "market_regime", "UNKNOWN")
        vix = float(getattr(regime_state, "vix", 0.0) or 0.0)
        nifty_price = float(getattr(regime_state, "nifty_price", 0.0) or 0.0)
        regime_score = float(getattr(regime_state, "regime_score", 0.0) or 0.0)
        ts = datetime.now(timezone.utc).isoformat()

        # Store every reading
        self._store_reading(regime_state)

        # Check for shift
        prev_regime = self._last_recorded_regime()

        # Note: _last_recorded_regime now returns the one we JUST inserted,
        # so we need to check the SECOND-to-last for the "previous" value.
        # Re-fetch the prior regime (before our insert) by querying 2nd-last row.
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT market_regime FROM regime_readings ORDER BY id DESC LIMIT 2"
            ).fetchall()
            conn.close()
            if len(rows) < 2:
                # First ever reading — no prior state to compare
                return None
            prev_regime = rows[1]["market_regime"]
        except Exception as exc:
            logger.warning("RegimeMonitor shift-check query failed: %s", exc)
            return None

        if prev_regime == current_regime:
            return None

        # Regime changed — persist and return event
        logger.info(
            "Regime shift detected: %s -> %s (score=%.1f, vix=%.1f)",
            prev_regime, current_regime, regime_score, vix,
        )
        self._store_shift(prev_regime, current_regime, regime_score, vix, nifty_price, ts)
        event = self._build_event(
            from_regime=prev_regime,
            to_regime=current_regime,
            regime_score=regime_score,
            vix=vix,
            nifty_price=nifty_price,
            ts=ts,
        )
        return event

    def get_recent_events(self, n: int = 10) -> list[RegimeEvent]:
        """
        Returns the last N regime shift events, most recent first.
        """
        try:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT timestamp, from_regime, to_regime, regime_score, vix, nifty_price
                FROM regime_shifts
                ORDER BY id DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("RegimeMonitor.get_recent_events failed: %s", exc)
            return []

        events: list[RegimeEvent] = []
        for row in rows:
            vix_val = float(row["vix"] or 0.0)
            event = self._build_event(
                from_regime=row["from_regime"],
                to_regime=row["to_regime"],
                regime_score=float(row["regime_score"] or 0.0),
                vix=vix_val,
                nifty_price=float(row["nifty_price"] or 0.0),
                ts=row["timestamp"],
            )
            events.append(event)
        return events

    def get_current_implications(self, regime_state) -> list[str]:
        """
        Returns 3-5 actionable implications for the current regime,
        combining regime-level guidance with VIX-level sizing adjustments
        and sector rotation context.

        Example output:
            - 'TRENDING_BULL: Favour breakout setups, increase position sizing to 1.15x'
            - 'VIX 13.4: Low volatility — tighten stops, targets likely to be hit'
            - 'BANK leading: Overweight banking setups vs market'
        """
        market_regime = getattr(regime_state, "market_regime", "UNKNOWN")
        vix = float(getattr(regime_state, "vix", 0.0) or 0.0)
        leading_sectors: list[str] = list(getattr(regime_state, "leading_sectors", []) or [])

        implications = self._implications_for(market_regime, vix)

        # Add sector leadership line if meaningful
        if leading_sectors:
            top_sector = leading_sectors[0]
            sector_line = (
                f"{top_sector} leading: Overweight {top_sector.lower()} setups vs broader market"
            )
            # Insert after the first regime line
            if len(implications) >= 2:
                implications.insert(1, sector_line)
            else:
                implications.append(sector_line)

        # Trim to 5 total
        return implications[:5]

    def time_in_current_regime(self) -> str:
        """
        Returns a human-readable string describing how long the market
        has been in the current regime since the last detected shift.

        Examples: '3 days', '12 hours', '45 minutes', 'just started'
        """
        try:
            conn = self._connect()
            row = conn.execute(
                "SELECT timestamp FROM regime_shifts ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
        except Exception as exc:
            logger.warning("RegimeMonitor.time_in_current_regime failed: %s", exc)
            return "unknown"

        if row is None:
            # No shifts recorded — use first reading timestamp
            try:
                conn = self._connect()
                row = conn.execute(
                    "SELECT timestamp FROM regime_readings ORDER BY id ASC LIMIT 1"
                ).fetchone()
                conn.close()
            except Exception:
                return "unknown"

        if row is None:
            return "unknown"

        try:
            shift_ts = datetime.fromisoformat(row["timestamp"])
            # Make timezone-aware if naive
            if shift_ts.tzinfo is None:
                shift_ts = shift_ts.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - shift_ts
            total_secs = int(delta.total_seconds())

            if total_secs < 120:
                return "just started"
            if total_secs < 3600:
                mins = total_secs // 60
                return f"{mins} minute{'s' if mins != 1 else ''}"
            if total_secs < 86400:
                hours = total_secs // 3600
                return f"{hours} hour{'s' if hours != 1 else ''}"
            days = total_secs // 86400
            return f"{days} day{'s' if days != 1 else ''}"
        except Exception as exc:
            logger.warning("RegimeMonitor time parsing failed: %s", exc)
            return "unknown"
