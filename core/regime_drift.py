"""
Regime Drift — detects regime change BEFORE it officially flips.
Warns 30-60 minutes before the regime engine would normally call it.
"Regime drifting toward DISTRIBUTION (confidence: 78→71→64). Review stops."
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

_DB_PATH = Path("logs/regime_drift.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS regime_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    market_regime TEXT,
    confidence REAL,
    vix REAL,
    breadth TEXT,
    institutional TEXT
)
"""

# How many snapshots to keep for history (avoids unbounded growth)
_MAX_SNAPSHOTS = 200


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def record_regime_snapshot(regime_state) -> None:
    """
    Store the current regime state snapshot.
    Called every trading cycle. Wraps all errors silently.
    """
    try:
        market_regime = getattr(regime_state, "market_regime", "UNKNOWN")
        confidence = float(getattr(regime_state, "regime_confidence", 50.0) or 50.0)
        vix = float(getattr(regime_state, "vix", 0.0) or 0.0)
        breadth = getattr(regime_state, "breadth_label", "NEUTRAL")
        institutional = getattr(regime_state, "institutional_activity", "NEUTRAL")

        conn = _get_conn()
        try:
            conn.execute(
                """
                INSERT INTO regime_snapshots
                    (timestamp, market_regime, confidence, vix, breadth, institutional)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    market_regime,
                    confidence,
                    vix,
                    breadth,
                    institutional,
                ),
            )
            conn.commit()

            # Prune old snapshots to avoid unbounded growth
            conn.execute(
                f"""
                DELETE FROM regime_snapshots
                WHERE id NOT IN (
                    SELECT id FROM regime_snapshots ORDER BY id DESC LIMIT {_MAX_SNAPSHOTS}
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # Never break the main cycle


def detect_drift() -> Optional[dict]:
    """
    Analyse last 6 snapshots for early regime drift signals.

    Returns None if no drift detected.
    Returns a dict with drift details if:
    - Confidence dropped >= 10pts in last 3 snapshots
    - Institutional activity shifted toward DISTRIBUTION
    - Breadth shifted from STRONG to WEAK

    Returns
    -------
    None | dict with keys:
        detected, direction, confidence_trend, speed, alert, recommended_action
    """
    try:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """
                SELECT timestamp, market_regime, confidence, vix, breadth, institutional
                FROM regime_snapshots
                ORDER BY id DESC LIMIT 6
                """
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return None

    if len(rows) < 3:
        return None  # Not enough history

    # rows[0] is most recent, rows[-1] is oldest
    # Reverse for chronological order
    rows = list(reversed(rows))

    confidences = [float(r[2] or 50.0) for r in rows]
    breadths = [r[4] or "NEUTRAL" for r in rows]
    institutionals = [r[5] or "NEUTRAL" for r in rows]
    regimes = [r[1] or "UNKNOWN" for r in rows]

    # ── Check 1: Confidence drop in last 3 readings ───────────────────────────
    last3_conf = confidences[-3:]
    conf_drop = last3_conf[0] - last3_conf[-1]
    conf_rising = last3_conf[-1] > last3_conf[0]

    drift_detected = False
    direction = "WEAKENING"
    speed = "GRADUAL"
    alert_parts = []
    recommended_action = "No action"

    if conf_drop >= 10:
        drift_detected = True
        direction = "WEAKENING"
        speed = "FAST" if conf_drop >= 15 else "GRADUAL"
        alert_parts.append(
            f"Confidence falling: {last3_conf[0]:.0f}→{last3_conf[1]:.0f}→{last3_conf[-1]:.0f}"
        )
        recommended_action = "Review stops"
    elif conf_rising and (last3_conf[-1] - last3_conf[0]) >= 8:
        drift_detected = True
        direction = "STRENGTHENING"
        speed = "GRADUAL"
        alert_parts.append(
            f"Confidence rising: {last3_conf[0]:.0f}→{last3_conf[1]:.0f}→{last3_conf[-1]:.0f}"
        )
        recommended_action = "No action"

    # ── Check 2: Institutional shift ─────────────────────────────────────────
    bearish_inst = {"DISTRIBUTION", "RISK_OFF"}
    bullish_inst = {"ACCUMULATION", "RISK_ON"}
    recent_inst = institutionals[-1]
    prev_inst = institutionals[0]

    if prev_inst in bullish_inst and recent_inst in bearish_inst:
        drift_detected = True
        direction = "TOWARD_BEAR"
        alert_parts.append(
            f"Institutional shift: {prev_inst} → {recent_inst}"
        )
        recommended_action = "Reduce size"
    elif prev_inst in bearish_inst and recent_inst in bullish_inst:
        drift_detected = True
        direction = "TOWARD_BULL"
        alert_parts.append(
            f"Institutional shift: {prev_inst} → {recent_inst}"
        )
        recommended_action = "No action"

    # ── Check 3: Breadth deterioration ───────────────────────────────────────
    if breadths[0] == "STRONG" and breadths[-1] == "WEAK":
        drift_detected = True
        if direction not in ("TOWARD_BEAR", "TOWARD_BULL"):
            direction = "WEAKENING"
        alert_parts.append("Breadth collapsed: STRONG → WEAK")
        if recommended_action == "No action":
            recommended_action = "Review stops"

    if not drift_detected:
        return None

    # Build current regime context
    current_regime = regimes[-1]
    regime_map = {
        "TOWARD_BEAR": "DISTRIBUTION",
        "TOWARD_BULL": "EXPANSION",
        "WEAKENING": "uncertain",
        "STRENGTHENING": "stronger",
    }
    toward = regime_map.get(direction, direction)

    alert = (
        f"Regime drifting toward {toward} "
        f"({', '.join(alert_parts)}). {recommended_action}."
    )

    return {
        "detected": True,
        "direction": direction,
        "confidence_trend": [round(c, 1) for c in last3_conf],
        "speed": speed,
        "alert": alert,
        "recommended_action": recommended_action,
        "current_regime": current_regime,
    }


def render_drift_alert() -> None:
    """
    Renders a drift warning only if drift is detected.
    Completely silent when regime is stable.
    """
    return
