"""
🗄️ Evidence backup — system ki asli poonji uske outcomes hain, unka bima.

trades.db, decisions.db, signal_outcomes.db, autopilot state — ye sab kuch
mahino ki MEHNAT hai (calibration, EV, edge sab isi se). SD-card corruption
ya galat rm ek raat mein sab mita sakta hai.

Roz (off-hours, auto_scan housekeeping se) logs/backup/YYYY-MM-DD/ mein
snapshot; 7 din ki rotation. SQLite ke liye proper backup API (WAL ke
beech copy karna corrupt snapshot deta hai — file-copy nahi karte).
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

_LOGS = Path(__file__).resolve().parent.parent / "logs"
_BACKUP_ROOT = _LOGS / "backup"
KEEP_DAYS = 7

# (filename, is_sqlite)
_TARGETS = [
    ("trades.db", True),
    ("decisions.db", True),
    ("signal_outcomes.db", True),
    ("autopilot.json", False),
    ("us_autopilot.json", False),
]


def snapshot(day: str | None = None) -> dict:
    """One dated snapshot + rotation. Returns {copied: [...], pruned: [...]}.
    Missing files skip silently (fresh installs), failures never raise —
    backup kabhi trading path ko nahi girata."""
    if day is None:
        try:
            from core.market_clock import today_ist
            day = today_ist().isoformat()
        except Exception:
            from datetime import date
            day = date.today().isoformat()
    dest = _BACKUP_ROOT / day
    copied: list[str] = []
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for name, is_sqlite in _TARGETS:
            src = _LOGS / name
            if not src.exists():
                continue
            try:
                if is_sqlite:
                    # consistent snapshot even mid-write (WAL-safe)
                    with sqlite3.connect(src) as s_conn, \
                            sqlite3.connect(dest / name) as d_conn:
                        s_conn.backup(d_conn)
                else:
                    shutil.copy2(src, dest / name)
                copied.append(name)
            except Exception as exc:
                log.debug("backup_item_failed", item=name, error=str(exc))
    except Exception as exc:
        log.debug("backup_failed", error=str(exc))
        return {"copied": copied, "pruned": []}

    # rotation — sirf KEEP_DAYS sabse naye din
    pruned: list[str] = []
    try:
        days = sorted(d.name for d in _BACKUP_ROOT.iterdir() if d.is_dir())
        for old in days[:-KEEP_DAYS]:
            shutil.rmtree(_BACKUP_ROOT / old, ignore_errors=True)
            pruned.append(old)
    except Exception:
        pass
    if copied:
        log.info("evidence_backup", day=day, files=len(copied),
                 pruned=len(pruned))
    return {"copied": copied, "pruned": pruned}
