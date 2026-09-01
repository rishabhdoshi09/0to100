"""Durable paper-autopilot decision blotter.

Every candidate the autopilot sees is recorded with a machine-readable reason.
This is how the desk answers “why did QuantTerm not take a trade today?” without
reading source. Missing stays missing — an empty blotter is a valid day.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "paper_autopilot_journal.json"
SCHEMA_VERSION = 1


def journal_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_PAPER_AUTOPILOT_JOURNAL")
    if override:
        return Path(override)
    return DEFAULT_PATH


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_journal(path: str | Path | None = None) -> dict[str, Any]:
    target = journal_path(path)
    payload = _read(target)
    if not payload:
        return {
            "schema_version": SCHEMA_VERSION,
            "cycles": [],
            "latest": {},
        }
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("cycles", [])
    payload.setdefault("latest", {})
    return payload


def record_cycle(cycle: Mapping[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    target = journal_path(path)
    payload = load_journal(target)
    row = dict(cycle)
    row.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    cycles = list(payload.get("cycles") or [])
    cycles.append(row)
    payload["cycles"] = cycles[-120:]
    payload["latest"] = row
    payload["schema_version"] = SCHEMA_VERSION
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return payload


def why_no_trade(path: str | Path | None = None) -> dict[str, Any]:
    """Operator-facing summary of the latest autopilot cycle."""
    payload = load_journal(path)
    latest = dict(payload.get("latest") or {})
    if not latest:
        return {
            "available": False,
            "headline": "No paper-autopilot cycle has been recorded yet.",
            "decision": "UNKNOWN",
            "reasons": ["NO_CYCLE_RECORDED"],
            "rejections": [],
            "taken": [],
            "as_of": "",
        }
    taken = list(latest.get("taken") or [])
    rejections = list(latest.get("rejections") or [])
    waits = list(latest.get("waits") or [])
    decision = str(latest.get("final_decision") or "")
    reasons = [str(x) for x in (latest.get("cycle_reasons") or []) if x]
    if taken:
        headline = (
            f"Took {len(taken)} paper trade(s): "
            + ", ".join(str(t.get('symbol') or "") for t in taken[:8])
        )
    elif not reasons and rejections:
        top = {}
        for row in rejections:
            code = str(row.get("reason_code") or "REJECTED")
            top[code] = top.get(code, 0) + 1
        ranked = sorted(top.items(), key=lambda kv: (-kv[1], kv[0]))
        reasons = [code for code, _ in ranked[:8]]
        headline = (
            "No paper trade taken. "
            + "; ".join(f"{n}× {code}" for code, n in ranked[:6])
        )
        decision = decision or "NO_TRADE"
    elif reasons:
        headline = "No paper trade taken: " + "; ".join(reasons[:6])
        decision = decision or "NO_TRADE"
    else:
        headline = str(latest.get("summary") or "No paper trade taken.")
        decision = decision or "NO_TRADE"
    return {
        "available": True,
        "headline": headline,
        "decision": decision,
        "reasons": reasons,
        "rejections": rejections[-40:],
        "waits": waits[-20:],
        "taken": taken,
        "as_of": str(latest.get("as_of") or latest.get("recorded_at") or ""),
        "entries_allowed": bool(latest.get("entries_allowed")),
        "paper_enabled": bool(latest.get("paper_enabled")),
        "candidates_seen": int(latest.get("candidates_seen") or 0),
        "eligible_count": int(latest.get("eligible_count") or 0),
    }


def flatten_cards(workspace: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Pull recommendation cards from the saved desk payload. Empty is valid."""
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cat in list((workspace or {}).get("categories") or []):
        for card in list((cat or {}).get("cards") or []):
            if not isinstance(card, Mapping):
                continue
            symbol = str(card.get("symbol") or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            row = dict(card)
            row["symbol"] = symbol
            cards.append(row)
    return cards
