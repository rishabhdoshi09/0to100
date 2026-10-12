"""Durable paper-autopilot decision blotter.

Every candidate the autopilot sees is recorded with a machine-readable reason.
This is how the desk answers “why did QuantTerm not take a trade today?” without
reading source. Missing stays missing — an empty blotter is a valid day.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from core.runtime_paths import _require_existing_runtime, logs_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = logs_dir() / "product" / "paper_autopilot_journal.json"
SCHEMA_VERSION = 1


def journal_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_PAPER_AUTOPILOT_JOURNAL")
    if override:
        return Path(override)
    # A module-import-time default can point at a disconnected checkout after
    # host migration. Resolve the canonical persistent runtime on every call.
    return logs_dir() / "product" / "paper_autopilot_journal.json"


def _require_safe_journal_destination(path: Path) -> None:
    """Never recreate an unmounted installed runtime or escape its logs tree."""
    if _require_existing_runtime():
        expected = logs_dir()
        try:
            path.resolve(strict=False).relative_to(expected)
        except ValueError as exc:
            raise RuntimeError(
                "paper journal destination is outside the installed persistent runtime"
            ) from exc


@contextmanager
def _journal_write_lock(path: Path):
    """Lock the entire read-modify-write, including across worker processes."""
    _require_safe_journal_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    _require_safe_journal_destination(lock)
    fd = os.open(
        str(lock), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read(path: Path, *, strict: bool = False) -> dict[str, Any]:
    if strict:
        _require_safe_journal_destination(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        if strict:
            raise RuntimeError("paper journal is unreadable or corrupt") from exc
        return {}
    if not isinstance(payload, dict):
        if strict:
            raise RuntimeError("paper journal root must be a JSON object")
        return {}
    if strict and (not isinstance(payload.get("cycles", []), list)
                   or not isinstance(payload.get("latest", {}), dict)):
        raise RuntimeError("paper journal has an invalid cycle structure")
    return payload


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
    with _journal_write_lock(target):
        # Strict under the lock: a malformed journal must not be silently
        # replaced with an empty history by the next paper cycle.
        payload = _read(target, strict=True)
        row = dict(cycle)
        row.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        cycles = list(payload.get("cycles") or [])
        cycles.append(row)
        payload["cycles"] = cycles[-120:]
        payload["latest"] = row
        payload["schema_version"] = SCHEMA_VERSION
        fd, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        tmp = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            _require_safe_journal_destination(target)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
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
