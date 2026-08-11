"""
🍃 Eco mode — shared-machine ke liye QuantTerm ki CPU bhookh kam karo.

MacBook (fanless!) pe business software ke saath chalana ho toh full-power
scanning garmi banati hai. QT_ECO=1 *or* QT_LOW_POWER=1 pe:

  • off-hours mein market scan BILKUL band (raat ko 3 baje scan ka koi
    matlab nahi tha — data hi nahi badalta)
  • scan threads aadhe-se-kam (8 → 2, 6 → 2): kaam wahi, spikes nahi
  • market-hours cadence relaxed (min 30 min; sniper phir bhi instant
    breakouts pakadta hai — woh websocket hai, CPU nahi khaata)

Kya NAHI badalta: signals, gates, risk, autopilot logic, market-scan
bootstrap that feeds autopilot — sirf KAB aur KITNI-TEZI se compute hota hai.

Enable: .env mein QT_ECO=1, or launch via scripts/run_quantterm_low_power.sh
(which sets QT_LOW_POWER=1 and starts the same complete stack).
"""
from __future__ import annotations

import os

from logger import get_logger

log = get_logger(__name__)

_ECO_WORKERS = 2
_ECO_MIN_INTERVAL_S = 1800        # market hours: at most every 30 min


def eco_on() -> bool:
    """True when QT_ECO=1 or QT_LOW_POWER=1 (Mac low-power launcher)."""
    for key in ("QT_ECO", "QT_LOW_POWER"):
        if str(os.getenv(key, "")).strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def workers(default: int) -> int:
    """Thread-pool size for scans: eco clamps to 2 (never raises)."""
    return min(_ECO_WORKERS, default) if eco_on() else default


def scan_interval(configured_s: int) -> int:
    """Market-hours scan cadence: eco enforces a floor of 30 min."""
    return max(_ECO_MIN_INTERVAL_S, configured_s) if eco_on() else configured_s


def should_scan_now(market_open: bool) -> bool:
    """Full-market scan abhi chale? Eco mein sirf market hours — off-hours
    scan pure heat hai (data badalta hi nahi). Normal mode: hamesha (purana
    behaviour, dedicated servers ke liye)."""
    return market_open if eco_on() else True
