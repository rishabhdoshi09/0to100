"""
🍃 Eco mode — shared-machine ke liye QuantTerm ki CPU bhookh kam karo.

MacBook (fanless!) pe business software ke saath chalana ho toh full-power
scanning garmi banati hai. QT_ECO=1 pe:

  • off-hours mein market scan BILKUL band (raat ko 3 baje scan ka koi
    matlab nahi tha — data hi nahi badalta)
  • scan threads aadhe-se-kam (8 → 2, 6 → 2): kaam wahi, spikes nahi
  • market-hours cadence relaxed (min 30 min; sniper phir bhi instant
    breakouts pakadta hai — woh websocket hai, CPU nahi khaata)

Kya NAHI badalta: signals, gates, risk, autopilot logic — sirf KAB aur
KITNI-TEZI se compute hota hai. Nightly backtest + outcome updates bhi
chalte rehte hain (din mein ek baar, evidence rukna nahi chahiye).

Enable: .env mein QT_ECO=1 (ya launchd/systemd env). Default: OFF —
dedicated server pe full power.
"""
from __future__ import annotations

import os

from logger import get_logger

log = get_logger(__name__)

_ECO_WORKERS = 2
_ECO_MIN_INTERVAL_S = 1800        # market hours: at most every 30 min


def eco_on() -> bool:
    return str(os.getenv("QT_ECO", "")).strip().lower() in ("1", "true", "yes")


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
