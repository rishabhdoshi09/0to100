"""Wrap of the Day — system-composed daily market wrap.

Honesty rules (hard):
  • Compose only from real Pulse stores (indices, sectors, scan, news, global cues).
  • Never invent prices, company narratives, CA, or fills.
  • Missing stores stay missing / disclosed.
  • Optional user paste is an override, not the default path.
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
SCHEMA_VERSION = 2
MANUAL_SOURCES = frozenset({"paste", "manual", "override", "seed"})


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
    """Parse numbered / bulleted wrap lines into clean bullets."""
    raw = str(text or "").strip()
    if not raw:
        return []
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
        "auto": True,
        "override": False,
        "updated_at": "",
        "gaps": [],
        "message": message
        or (
            "Wrap of the Day rebuilds from scan, bhav, options, news and global cues. "
            "Run a market scan or Rebuild pulse when stores are empty."
        ),
        "places_orders": False,
        "honesty": (
            "System-composed Wrap of the Day from QuantTerm stores. "
            "Missing evidence stays missing — never invented."
        ),
    }


def _normalize(
    bullets: Sequence[str],
    *,
    date: str | None = None,
    source: str = "auto",
    raw_text: str = "",
    gaps: Sequence[str] | None = None,
) -> dict[str, Any]:
    clean = [str(b).strip() for b in bullets if str(b).strip()]
    day = date or today_ist()
    src = str(source or "auto").strip().lower() or "auto"
    if not clean:
        return empty_wrap(
            date=day,
            message="No wrap bullets available from current stores yet.",
        )
    is_manual = src in MANUAL_SOURCES
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "date": day,
        "title": "Wrap of the Day",
        "bullets": clean[:8],
        "source": src,
        "auto": not is_manual,
        "override": is_manual and src != "seed",
        "raw_text": raw_text or "\n".join(f"{i}) {b}" for i, b in enumerate(clean[:8], 1)),
        "updated_at": datetime.now(IST).isoformat(),
        "gaps": [str(g) for g in (gaps or []) if str(g).strip()][:8],
        "message": (
            f"{len(clean[:8])} wrap bullet(s) for {day} · "
            + ("user override" if is_manual and src != "seed" else f"source={src}")
        ),
        "places_orders": False,
        "honesty": (
            "User override Wrap of the Day. Research narrative only — not a buy ticket."
            if is_manual and src != "seed"
            else (
                "System-composed Wrap of the Day from scan, bhav, options, news and "
                "global cues. Missing stays missing — not a buy ticket."
            )
        ),
    }


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def compose_from_pulse(pulse: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build newsletter-style wrap bullets from an assembled Street Pulse payload."""
    payload = dict(pulse or {})
    bullets: list[str] = []
    gaps: list[str] = []

    snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), Mapping) else {}
    indices = list(snapshot.get("indices") or [])
    if indices:
        parts: list[str] = []
        for idx in indices[:3]:
            name = str(idx.get("name") or "").strip()
            chg = idx.get("chg_pct")
            price = idx.get("price")
            if not name or chg is None:
                continue
            bit = f"{name} {_f(chg):+.2f}%"
            if price is not None:
                bit += f" at {_f(price):,.0f}"
            parts.append(bit)
        if parts:
            bullets.append("Indian benchmarks: " + "; ".join(parts) + ".")
    else:
        gaps.append("Index quotes unavailable")

    stance = ((snapshot.get("options_stance") or {}) if isinstance(snapshot, Mapping) else {}).get("stance")
    if stance:
        bullets.append(
            f"NIFTY options positioning reads {stance} — research context only, not a buy stance."
        )

    if snapshot.get("commentary"):
        commentary = str(snapshot.get("commentary") or "").strip()
        if commentary and commentary not in bullets:
            bullets.append(commentary if commentary.endswith(".") else f"{commentary}.")

    sectors = payload.get("sectors") if isinstance(payload.get("sectors"), Mapping) else {}
    leaders = list(sectors.get("leaders") or [])[:2]
    laggards = list(sectors.get("laggards") or [])[:1]
    if leaders:
        lead_txt = ", ".join(
            f"{r.get('sector')} ({_f(r.get('chg_1d')):+.1f}%)" for r in leaders if r.get("sector")
        )
        if lead_txt:
            bullets.append(f"Sector heat leaders: {lead_txt}.")
    elif not sectors.get("available"):
        gaps.append("Sector heat unavailable")
    # Keep laggard only when the wrap is still short — prefer news/global later.
    if laggards and laggards[0].get("sector") and len(bullets) < 4:
        bullets.append(
            f"Sector heat laggard: {laggards[0].get('sector')} ({_f(laggards[0].get('chg_1d')):+.1f}%)."
        )

    buzzing = payload.get("buzzing") if isinstance(payload.get("buzzing"), Mapping) else None
    gainers = [r for r in (payload.get("gainers") or []) if isinstance(r, Mapping)]
    if buzzing and buzzing.get("symbol"):
        note = str(buzzing.get("note") or buzzing.get("why") or "").strip()
        chg = buzzing.get("change_pct")
        vol = buzzing.get("volume_ratio")
        core = f"{buzzing.get('symbol')} is buzzing"
        extras: list[str] = []
        if chg is not None:
            extras.append(f"{_f(chg):+.1f}%")
        if vol is not None:
            extras.append(f"{_f(vol):.1f}× volume")
        if extras:
            core += f" ({', '.join(extras)})"
        if note:
            core += f" — {note.rstrip('.')}"
        bullets.append(core + ".")
    elif gainers and gainers[0].get("symbol"):
        g = gainers[0]
        bullets.append(
            f"{g.get('symbol')} led liquid gainers ({_f(g.get('chg_pct')):+.1f}%)."
        )

    headlines = [str(h).strip() for h in (payload.get("headlines") or []) if str(h).strip()]
    for head in headlines[:1]:
        bullets.append(f"In the news: {head.rstrip('.')}.")
    if not headlines:
        gaps.append("No fresh headlines in curator/fetcher")

    cues = [r for r in (payload.get("global_cues") or []) if isinstance(r, Mapping)]
    if cues:
        cue_bits = []
        for row in cues[:3]:
            name = str(row.get("name") or "").strip()
            chg = row.get("chg_pct")
            if not name or chg is None:
                continue
            cue_bits.append(f"{name} {_f(chg):+.2f}%")
        if cue_bits:
            bullets.append("Global cues: " + "; ".join(cue_bits) + ".")
    else:
        gaps.append("Global/US cues unavailable")

    breakouts = [r for r in (payload.get("breakouts_today") or []) if isinstance(r, Mapping)]
    strength = payload.get("strength") if isinstance(payload.get("strength"), Mapping) else None
    if breakouts:
        syms = ", ".join(str(r.get("symbol")) for r in breakouts[:3] if r.get("symbol"))
        if syms:
            bullets.append(f"Breakouts in focus: {syms} (research labels, not buy tickets).")
    elif strength and strength.get("symbol"):
        dist = strength.get("pivot_distance_pct")
        dist_txt = (
            f"{_f(dist):.1f}% below breakout level"
            if dist is not None
            else "near breakout level"
        )
        bullets.append(f"Gaining strength: {strength.get('symbol')} — {dist_txt}.")

    weak = payload.get("weak") if isinstance(payload.get("weak"), Mapping) else None
    if weak and weak.get("symbol"):
        bullets.append(f"Losing momentum: {weak.get('symbol')} — stay selective.")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for b in bullets:
        key = b.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(b)

    if not unique:
        # Surface pulse gaps honestly instead of inventing a wrap.
        pulse_gaps = [str(g) for g in (payload.get("gaps") or []) if str(g).strip()]
        return empty_wrap(
            message=(
                "Could not compose Wrap of the Day from current stores. "
                + (pulse_gaps[0] if pulse_gaps else "Run a market scan, then Rebuild pulse.")
            )
        )

    return _normalize(unique[:8], source="auto", gaps=gaps)


def compose_wrap(*, persist: bool = True) -> dict[str, Any]:
    """Compose from live stores via Street Pulse assembly (no invented fills)."""
    try:
        from reports.street_pulse import build_pulse

        pulse = build_pulse(persist=False)
    except Exception as exc:
        return empty_wrap(message=f"Wrap compose failed: {exc}")
    # Prefer any manual override for today over freshly composed auto text.
    manual = load_manual_override()
    if manual:
        return manual
    wrap = compose_from_pulse(pulse)
    if persist and wrap.get("available"):
        save_wrap(
            wrap.get("bullets") or [],
            date=str(wrap.get("date") or today_ist()),
            source="auto",
            gaps=list(wrap.get("gaps") or []),
        )
    return wrap


def save_wrap(
    bullets: Sequence[str] | None = None,
    *,
    text: str = "",
    date: str | None = None,
    source: str = "auto",
    gaps: Sequence[str] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    parsed = list(bullets or [])
    if not parsed and text:
        parsed = parse_wrap_text(text)
    payload = _normalize(
        parsed,
        date=date,
        source=source,
        raw_text=text,
        gaps=gaps,
    )
    target = wrap_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
    return payload


def clear_override(path: Path | None = None) -> dict[str, Any]:
    """Remove a user override and recompose from stores."""
    target = wrap_path(path)
    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and str(payload.get("source") or "").lower() in {
                "paste",
                "manual",
                "override",
            }:
                target.unlink(missing_ok=True)
        except Exception:
            target.unlink(missing_ok=True)
    return compose_wrap(persist=True)


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


def _read_saved(path: Path | None = None) -> dict[str, Any] | None:
    target = wrap_path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_manual_override(path: Path | None = None, *, date: str | None = None) -> dict[str, Any] | None:
    day = date or today_ist()
    payload = _read_saved(path)
    if not payload or not payload.get("available"):
        return None
    src = str(payload.get("source") or "").lower()
    if src not in {"paste", "manual", "override"}:
        return None
    saved_day = str(payload.get("date") or "")
    if saved_day and saved_day != day:
        return None
    payload = dict(payload)
    payload["override"] = True
    payload["auto"] = False
    return payload


def load_wrap(
    path: Path | None = None,
    *,
    date: str | None = None,
    compose: bool = True,
) -> dict[str, Any]:
    """Return today's wrap: manual override > auto compose > historical seed."""
    day = date or today_ist()
    manual = load_manual_override(path, date=day)
    if manual:
        return manual

    saved = _read_saved(path)
    if (
        saved
        and saved.get("available")
        and str(saved.get("source") or "").lower() == "auto"
        and str(saved.get("date") or "") == day
        and not compose
    ):
        return saved

    # Auto-compose for today from persisted pulse (no pulse_api_payload — avoids recursion).
    if compose and day == today_ist():
        try:
            from reports.street_pulse import build_pulse, load_pulse

            pulse = load_pulse()
            if not pulse or not pulse.get("available"):
                pulse = build_pulse(persist=True)
            wrap = compose_from_pulse(pulse)
            if wrap.get("available"):
                save_wrap(
                    wrap.get("bullets") or [],
                    date=day,
                    source="auto",
                    gaps=list(wrap.get("gaps") or []),
                    path=path,
                )
                return wrap
            empty = wrap if wrap.get("message") else empty_wrap(date=day)
        except Exception as exc:
            empty = empty_wrap(date=day, message=f"Wrap compose failed: {exc}")
    else:
        empty = empty_wrap(date=day)

    # Historical seed only — never invent; never override today's auto path unless empty.
    seeded = _load_seed(day)
    if seeded and (day != today_ist() or not empty.get("available")):
        # For today, seed is last-resort archive only when stores produced nothing.
        if day == today_ist() and empty.get("available") is False:
            return seeded
        if day != today_ist():
            return seeded

    if saved and saved.get("available") and str(saved.get("date") or "") == day:
        return saved
    return empty


def wrap_telegram_message(wrap: Mapping[str, Any] | None = None) -> str:
    payload = dict(wrap or load_wrap())
    bullets = list(payload.get("bullets") or [])
    day = payload.get("date") or today_ist()
    source = str(payload.get("source") or "auto")
    if not bullets:
        return (
            f"<b>Wrap of the Day</b> — {day}\n"
            "No wrap available from current stores yet.\n"
            "<i>System-composed · not a buy ticket</i>"
        )
    label = "user override" if payload.get("override") else f"source {source}"
    lines = [f"<b>Wrap of the Day</b> — {day}", f"<i>{label}</i>", ""]
    for i, bullet in enumerate(bullets[:12], 1):
        lines.append(f"{i}) {bullet}")
    if len(bullets) > 12:
        lines.append(f"… +{len(bullets) - 12} more")
    lines.append("\n<i>Research wrap · not a buy ticket · paper-first</i>")
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
        "source": payload.get("source") or "",
    }
