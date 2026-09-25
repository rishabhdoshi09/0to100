"""Persisted startup trade-discovery projection.

The whole-market scan is the expensive trigger. Once it finishes, QuantTerm
projects the canonical decision board exactly once and stores it here. The
Decision Simulation gate then remains a cheap read path instead of rebuilding
recommendations/rankings inside an HTTP request.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1


def _path() -> Path:
    return logs_path("product/startup_trade_discovery.json")


def _long_term_fingerprint() -> str:
    """Fingerprint decision-relevant long-term content, excluding run timestamp.

    A cache-only long-term refresh can rewrite ``scanned_at`` while producing the
    exact same evidence. That wall-clock change must not invalidate an otherwise
    identical startup decision projection. Conversely, any material long-term
    payload change must still fail closed and require a new projection.
    """
    try:
        from product.long_term_store import load_long_term_scan

        payload = dict(load_long_term_scan() or {})
    except Exception:
        return ""
    if not payload:
        return ""
    payload.pop("scanned_at", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save(
    board: Mapping[str, Any],
    *,
    scan_scanned_at: str,
    long_term_scanned_at: str,
    thesis_hash: str,
) -> Path:
    target = _path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scan_scanned_at": str(scan_scanned_at or ""),
        "long_term_scanned_at": str(long_term_scanned_at or ""),
        "long_term_fingerprint": _long_term_fingerprint(),
        "thesis_hash": str(thesis_hash or ""),
        "board": dict(board),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load(
    *,
    scan_scanned_at: str,
    long_term_scanned_at: str,
    thesis_hash: str,
) -> dict[str, Any] | None:
    target = _path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("schema_version") or 0) != SCHEMA_VERSION:
        return None
    if str(payload.get("scan_scanned_at") or "") != str(scan_scanned_at or ""):
        return None
    stored_lt_at = str(payload.get("long_term_scanned_at") or "")
    current_lt_at = str(long_term_scanned_at or "")
    if stored_lt_at != current_lt_at:
        # Timestamp-only refreshes are harmless only when the complete
        # decision-relevant long-term payload is byte-deterministically equal.
        # Legacy rows without a content fingerprint remain strict/fail-closed.
        stored_fp = str(payload.get("long_term_fingerprint") or "")
        current_fp = _long_term_fingerprint()
        if not stored_fp or not current_fp or stored_fp != current_fp:
            return None
    if str(payload.get("thesis_hash") or "") != str(thesis_hash or ""):
        return None
    board = payload.get("board")
    return dict(board) if isinstance(board, dict) else None


def load_current() -> dict[str, Any] | None:
    """Load the discovery board only when it matches current canonical identity."""
    try:
        from product.scan_store import load_scan
        from product.long_term_store import load_long_term_scan
        from product.trading_thesis import manifest

        scan = dict(load_scan() or {})
        long_term = dict(load_long_term_scan() or {})
        thesis = dict(manifest() or {})
        return load(
            scan_scanned_at=str(scan.get("scanned_at") or ""),
            long_term_scanned_at=str(long_term.get("scanned_at") or ""),
            thesis_hash=str(thesis.get("thesis_hash") or ""),
        )
    except Exception:
        return None
