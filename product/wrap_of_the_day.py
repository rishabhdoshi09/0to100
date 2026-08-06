"""Wrap of the Day — user-authored daily market wrap.

Honesty rules (hard):
  • Never invent wrap bullets, prices, or signals.
  • Only surfaces text the user saved (or a committed seed for that date).
  • Research narrative only — not a buy ticket; never places orders.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.market_clock import IST

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "wrap_of_the_day.json"
SEED_DIR = ROOT / "content" / "wraps"
SCHEMA_VERSION = 1


def wrap_path(path: Path | None = None) -> Path:
    env = os.environ.get("QT_WRAP_FILE", "").strip()
    if path is not None:
        return Path(path)
    if env:
        return Path(env)
    return DEFAULT_PATH


def today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def parse_wrap_text(text: str) -> list[str]:
    """Parse numbered / bulleted wrap lines into clean bullets (user text only)."""
    raw = str(text or "").strip()
    if not raw:
        return []
    # Drop a leading title line like "Here's the Wrap of the Day:"
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines and re.search(r"wrap of the day", lines[0], re.I) and len(lines) > 1:
        lines = lines[1:]
    bullets: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^\s*(?:\d+[\)\.]|[-•*])\s*", "", line).strip()
        cleaned = cleaned.strip("`").strip()
        if cleaned:
            bullets.append(cleaned)
    return bullets


def empty_wrap(*, message: str = "", date: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "date": date or today_ist(),
        "title": "Wrap of the Day",
        "bullets": [],
        "source": "",
        "updated_at": "",
        "message": message
        or "No Wrap of the Day saved yet. Paste today's wrap on Daily Pulse.",
        "places_orders": False,
        "honesty": (
            "Wrap of the Day is user-authored only. QuantTerm never invents wrap bullets."
        ),
    }


def _normalize(
    bullets: Sequence[str],
    *,
    date: str | None = None,
    source: str = "paste",
    raw_text: str = "",
) -> dict[str, Any]:
    clean = [str(b).strip() for b in bullets if str(b).strip()]
    day = date or today_ist()
    if not clean:
        return empty_wrap(date=day, message="Wrap saved empty — paste numbered bullets.")
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "date": day,
        "title": "Wrap of the Day",
        "bullets": clean,
        "source": source,
        "raw_text": raw_text or "\n".join(f"{i}) {b}" for i, b in enumerate(clean, 1)),
        "updated_at": datetime.now(IST).isoformat(),
        "message": f"{len(clean)} wrap bullet(s) for {day}.",
        "places_orders": False,
        "honesty": (
            "User-authored Wrap of the Day. Research narrative only — not a buy ticket."
        ),
    }


def save_wrap(
    bullets: Sequence[str] | None = None,
    *,
    text: str = "",
    date: str | None = None,
    source: str = "paste",
    path: Path | None = None,
) -> dict[str, Any]:
    parsed = list(bullets or [])
    if not parsed and text:
        parsed = parse_wrap_text(text)
    payload = _normalize(parsed, date=date, source=source, raw_text=text)
    target = wrap_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
    return payload


def _load_seed(date: str) -> dict[str, Any] | None:
    seed = SEED_DIR / f"{date}.md"
    if not seed.exists():
        return None
    try:
        text = seed.read_text(encoding="utf-8")
    except Exception:
        return None
    bullets = parse_wrap_text(text)
    if not bullets:
        return None
    return _normalize(bullets, date=date, source="seed", raw_text=text)


def load_wrap(path: Path | None = None, *, date: str | None = None) -> dict[str, Any]:
    day = date or today_ist()
    target = wrap_path(path)
    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("available"):
                # Prefer saved wrap when it matches requested/today date.
                saved_day = str(payload.get("date") or "")
                if not date or saved_day == day:
                    return payload
        except Exception:
            pass
    seeded = _load_seed(day)
    if seeded:
        return seeded
    return empty_wrap(date=day)


def wrap_telegram_message(wrap: Mapping[str, Any] | None = None) -> str:
    payload = dict(wrap or load_wrap())
    bullets = list(payload.get("bullets") or [])
    day = payload.get("date") or today_ist()
    if not bullets:
        return (
            f"<b>Wrap of the Day</b> — {day}\n"
            "No wrap saved yet.\n"
            "<i>User-authored only · not a buy ticket</i>"
        )
    lines = [f"<b>Wrap of the Day</b> — {day}", ""]
    for i, bullet in enumerate(bullets[:12], 1):
        lines.append(f"{i}) {bullet}")
    if len(bullets) > 12:
        lines.append(f"… +{len(bullets) - 12} more")
    lines.append("\n<i>User-authored research wrap · not a buy ticket · paper-first</i>")
    return "\n".join(lines)


def notify_wrap_telegram(wrap: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(wrap or load_wrap())
    try:
        from alerts.telegram_alerts import AlertEngine

        engine = AlertEngine()
        if not engine.is_configured():
            return {
                "sent": False,
                "reason": "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)",
            }
    except Exception as exc:
        return {"sent": False, "reason": str(exc)}
    ok = bool(engine.send(wrap_telegram_message(payload)))
    return {
        "sent": ok,
        "count": len(payload.get("bullets") or []),
        "date": payload.get("date") or today_ist(),
    }
