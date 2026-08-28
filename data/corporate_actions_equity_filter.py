"""Equity-price corporate-action safety gate.

The exchange corporate-actions feeds can label distributions of *other securities*
as "bonus" (for example redeemable preference shares issued to equity holders).
Those are genuine corporate actions, but they do not multiply the outstanding
ordinary-equity share count and therefore must never be converted into an equity
OHLC adjustment factor.

This module installs an idempotent guard around the canonical resilient parser,
cleans any previously misclassified adjustment rows, and makes unresolved factor
conflicts part of adjustment-readiness. It does not guess an adjustment from a
price gap or silently resolve conflicting official factors.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from data import corporate_actions_resilient as CAR

_FILTER_VERSION = 1

# Security types that can be distributed to equity shareholders without changing
# the count of ordinary equity shares. Keep the patterns explicit and conservative.
_NON_EQUITY_PATTERNS = (
    re.compile(r"\bpreference\s+shares?\b", re.I),
    re.compile(r"\bpref(?:erence)?\.?\s+shares?\b", re.I),
    re.compile(r"\bredeemable\s+preference\b", re.I),
    re.compile(r"\bnon[-\s]?convertible\s+redeemable\b", re.I),
    re.compile(r"\bNCRPS?\b", re.I),
    re.compile(r"\bdebentures?\b", re.I),
    re.compile(r"\bbonds?\b", re.I),
    re.compile(r"\bwarrants?\b", re.I),
)


def is_non_equity_distribution(subject: Any) -> bool:
    """True when the official purpose describes a non-ordinary-equity security."""
    text = " ".join(str(subject or "").replace("\xa0", " ").split())
    return bool(text and any(pattern.search(text) for pattern in _NON_EQUITY_PATTERNS))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sanitize_persisted_adjustments(*, events_path: Path | None = None,
                                   coverage_path: Path | None = None) -> dict[str, int]:
    """Remove historical rows that were misclassified as equity adjustments.

    The source feed remains authoritative; the next resilient refresh will fetch
    the same genuine corporate actions again, but the guarded parser will classify
    non-equity distributions as non-adjusting and will rebuild the conflict set.
    """
    events_path = Path(events_path or CAR.DEFAULT_EVENTS_PATH)
    coverage_path = Path(coverage_path or CAR.DEFAULT_COVERAGE_PATH)
    raw = _safe_json(events_path, [])
    rows = list(raw) if isinstance(raw, list) else []
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if is_non_equity_distribution(row.get("subject")):
            removed += 1
            continue
        kept.append(row)

    if removed:
        _atomic_json(events_path, kept)
        try:
            from data.bhavcopy_store import reload_corporate_actions
            reload_corporate_actions()
        except Exception:
            pass

    coverage = _safe_json(coverage_path, {})
    if not isinstance(coverage, dict):
        coverage = {}
    previous_version = int(coverage.get("equity_security_filter_version") or 0)
    if removed or previous_version < _FILTER_VERSION:
        # Force one normal background refresh so conflicts/source mix are rebuilt
        # from official data under the corrected classifier.
        coverage["equity_security_filter_version"] = _FILTER_VERSION
        coverage["last_refresh_at"] = ""
        coverage["security_filter_last_sanitized_at"] = CAR._now_iso()
        coverage["security_filter_removed_adjustments"] = int(removed)
        _atomic_json(coverage_path, coverage)
    return {"removed": removed, "kept": len(kept)}


def install() -> None:
    """Install the security-type guard and conflict-aware readiness exactly once."""
    if getattr(CAR, "_equity_security_filter_installed", False):
        return

    base_parse = CAR.parse_share_count_action
    base_status = CAR.coverage_status

    def guarded_parse(subject: str):
        if is_non_equity_distribution(subject):
            return None
        return base_parse(subject)

    def guarded_status(*args, **kwargs):
        status = dict(base_status(*args, **kwargs) or {})
        conflicts = list(status.get("conflicts") or [])
        status["unresolved_conflicts"] = len(conflicts)
        # Window coverage and adjustment readiness are distinct. A complete
        # download with an unresolved factor disagreement is not safe to apply.
        status["window_coverage_complete"] = bool(status.get("coverage_complete"))
        status["coverage_complete"] = bool(status.get("coverage_complete")) and not conflicts
        status["adjustment_ready"] = bool(status["coverage_complete"])
        return status

    CAR.parse_share_count_action = guarded_parse
    CAR.coverage_status = guarded_status
    CAR.ledger_status = guarded_status
    CAR._equity_security_filter_installed = True
    sanitize_persisted_adjustments()
