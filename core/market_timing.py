"""
Market Microstructure Timing — scores trade quality by time of day.
NSE has predictable intraday patterns. Enter at wrong time = unnecessary risk.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

# (start_min, end_min, score, label, reason)
_TIMING_WINDOWS = [
    (555, 570,  20, "DANGER",  "Opening 15min — gap traps, wide spreads, avoid"),
    (570, 600,  55, "CAUTION", "9:30-10:00 — settling, use limit orders"),
    (600, 660,  85, "GOOD",    "10:00-11:00 — trend established, good entries"),
    (660, 690, 100, "OPTIMAL", "11:00-11:30 — direction setter, highest quality"),
    (690, 750,  70, "GOOD",    "11:30-12:30 — solid trend continuation"),
    (750, 840,  40, "WEAK",    "12:30-14:00 — low volume drift, avoid new entries"),
    (840, 900,  75, "GOOD",    "14:00-15:00 — afternoon trend resumes"),
    (900, 930,  30, "CAUTION", "15:00-15:30 — expiry/close positioning, unpredictable"),
    (930, 960,  10, "CLOSED",  "After market close"),
]
# Times are minutes since midnight


def get_timing_score() -> dict:
    """
    Returns current time quality for entering a trade.

    Returns a dict with keys:
      score  : 0-100 (higher = better time to trade)
      label  : OPTIMAL | GOOD | CAUTION | WEAK | DANGER | PRE-MARKET | CLOSED
      reason : human-readable explanation
      time   : current IST time string
    """
    now_ist = datetime.now(IST)
    mins = now_ist.hour * 60 + now_ist.minute
    time_str = now_ist.strftime("%H:%M IST")

    for start, end, score, label, reason in _TIMING_WINDOWS:
        if start <= mins < end:
            return {"score": score, "label": label, "reason": reason, "time": time_str}

    # Market closed / pre-market
    if mins < 555:
        return {"score": 0, "label": "PRE-MARKET", "reason": "Market not open yet", "time": time_str}
    return {"score": 0, "label": "CLOSED", "reason": "Market closed", "time": time_str}


def render_timing_badge() -> str:
    """
    Returns an HTML badge showing current timing score.
    Green if OPTIMAL/GOOD, yellow if CAUTION/WEAK, red if DANGER/CLOSED/PRE-MARKET.
    """
    t = get_timing_score()
    score = t["score"]
    label = t["label"]
    reason = t["reason"]
    time_str = t["time"]

    if label in ("OPTIMAL", "GOOD"):
        bg     = "#001a12"
        border = "#00d4a0"
        color  = "#00d4a0"
        icon   = "🟢"
    elif label in ("CAUTION", "WEAK"):
        bg     = "#1f1500"
        border = "#f59e0b"
        color  = "#f59e0b"
        icon   = "🟡"
    else:  # DANGER, CLOSED, PRE-MARKET
        bg     = "#1a0505"
        border = "#ff4b4b"
        color  = "#ff4b4b"
        icon   = "🔴"

    return (
        f"<span style='display:inline-flex;align-items:center;gap:5px;"
        f"background:{bg};border:1px solid {border}44;border-radius:6px;"
        f"padding:3px 8px;font-size:.65rem' title='{reason}'>"
        f"{icon} <span style='color:{color};font-weight:700'>{label}</span>"
        f"<span style='color:#4a5568;margin-left:2px'>{score}/100</span>"
        f"<span style='color:#4a5568;margin-left:4px'>{time_str}</span>"
        f"</span>"
    )
