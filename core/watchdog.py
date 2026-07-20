"""
🐕 Watchdog — daemon mare toh Telegram khud bataye, Diagnostics kholne ka
wait nahi.

24/7 unattended system ka pehla operational rule: silence suspicious hai.
Ye module core.health ke heartbeats padhta hai; koi daemon DEAD dikhe toh
ek Telegram alert (per daemon per day — spam nahi).

Do jagah se piggyback hota hai (apna thread nahi — jitne kam moving parts,
utna sane):
  • auto_scan worker loop — har cycle
  • autopilot fast-exit monitor — har ~5 min (throttled), taaki scan worker
    KHUD mar jaye toh doosra daemon uski maut ki khabar de

Imandaar seema: process-level death (poora app crash) in-process watchdog
nahi pakad sakta — woh launchd/systemd ke auto-restart ka kaam hai (jo
deploy scripts pehle se lagate hain). Ye layer un daemons ke liye hai jo
process ke andar chupchap mar jate hain.
"""
from __future__ import annotations

import time

from logger import get_logger

log = get_logger(__name__)

_THROTTLE_S = 300              # piggyback callers ke liye min gap
_last_check = 0.0
_alerted: dict[str, str] = {}  # {daemon: YYYY-MM-DD} — once per day per daemon


def check(force: bool = False) -> list[str]:
    """Throttled dead-daemon sweep. Returns daemons alerted this call."""
    global _last_check
    now = time.time()
    if not force and now - _last_check < _THROTTLE_S:
        return []
    _last_check = now
    try:
        from core.health import pulse
        daemons = pulse().get("daemons", {})
    except Exception:
        return []
    try:
        from core.market_clock import today_ist
        today = today_ist().isoformat()
    except Exception:
        from datetime import date
        today = date.today().isoformat()

    fired: list[str] = []
    for name, d in daemons.items():
        status = d.get("status") if isinstance(d, dict) else str(d)
        if status != "DEAD":
            continue
        if _alerted.get(name) == today:
            continue                     # aaj bata chuke — spam nahi
        _alerted[name] = today
        fired.append(name)
        age_min = int(float(d.get("age_s", 0)) / 60) if isinstance(d, dict) else 0
        msg = (f"🐕 <b>Watchdog</b>: daemon <b>{name}</b> DEAD hai "
               f"(~{age_min} min se koi heartbeat nahi).\n"
               f"Scans/alerts ruk sakte hain — app restart karo:\n"
               f"<code>launchctl kickstart -k gui/$(id -u)/com.quantterm.app</code>\n"
               f"(server: <code>systemctl restart quantterm</code>)")
        try:
            from alerts.telegram_alerts import AlertEngine
            eng = AlertEngine()
            if eng.is_configured():
                eng.send(msg)
        except Exception:
            pass
        log.warning("watchdog_dead_daemon", daemon=name)
    return fired
