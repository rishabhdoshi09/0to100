"""Atomic persistence for the latest current long-term shortlist.

The payload is a present-day decision aid, not point-in-time research evidence.
Historical research continues to use the audited snapshot/evidence stores.

The durable invariant is:

    len(payload["records"]) == payload["summary"]["candidates"]

Summary counts are always recomputed from the records that are actually about
to be written. An earlier pre-filter / pre-dedup / pre-validation count is
never trusted. Writes are atomic: a failed serialization cannot replace a
previous good artifact with a partial file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
from core.runtime_paths import logs_dir, logs_path

DEFAULT_LONG_TERM_PATH = logs_path("product/latest_long_term_scan.json")

_CLASSIFICATIONS = (
    "QUALITY_COMPOUNDER",
    "GARP_CANDIDATE",
    "QUALITY_BUT_EXPENSIVE",
    "LONG_TERM_WATCH",
    "NEEDS_FUNDAMENTALS",
    "AVOID_REVIEW",
)


class ArtifactIntegrityError(ValueError):
    """Durable long-term artifact would violate the records/candidates invariant."""


def _clean_records(records: Any) -> list[dict[str, Any]]:
    """Keep only mapping records. Invalid rows are dropped, never counted."""
    if not isinstance(records, list):
        return []
    return [dict(row) for row in records if isinstance(row, Mapping)]


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first record per non-empty symbol. Blank-symbol rows stay unique."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            out.append(row)
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(row)
    return out


def summary_from_records(
    records: list[Mapping[str, Any]],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build summary counts from the records that will actually be persisted."""
    summary = dict(previous or {})
    rows = [row for row in records if isinstance(row, Mapping)]
    for name in _CLASSIFICATIONS:
        summary[name.lower()] = sum(1 for row in rows if str(row.get("classification") or "") == name)
    covered = [row for row in rows if float(row.get("fundamental_coverage", 0) or 0) >= 0.50]
    summary["candidates"] = len(rows)
    summary["fundamentally_covered"] = len(covered)
    summary["fundamental_errors"] = sum(1 for row in rows if row.get("fundamental_error"))
    summary["coverage_pct"] = round(len(covered) / len(rows) * 100, 1) if rows else 0.0
    return summary


def reconcile_long_term_payload(
    payload: Mapping[str, Any] | None,
    *,
    dedupe: bool = True,
) -> dict[str, Any]:
    """Return a payload whose summary.candidates equals len(records)."""
    data = dict(payload or {})
    records = _clean_records(data.get("records"))
    if dedupe:
        records = _dedupe_records(records)
    data["records"] = records
    data["summary"] = summary_from_records(records, data.get("summary") if isinstance(data.get("summary"), Mapping) else None)
    claimed = data["summary"]["candidates"]
    if len(records) != claimed:
        raise ArtifactIntegrityError(
            f"long-term artifact invariant failed: len(records)={len(records)} "
            f"summary.candidates={claimed}"
        )
    return data


def save_long_term_scan(payload: Mapping[str, Any], path: str | Path = DEFAULT_LONG_TERM_PATH) -> Path:
    """Atomically persist a reconciled long-term artifact.

    Summary counts are recomputed from the final records. A previous durable
    file is left untouched if serialization or the invariant check fails.
    """
    data = reconcile_long_term_payload(payload)
    if len(data["records"]) != int(data["summary"]["candidates"]):
        raise ArtifactIntegrityError(
            f"refusing to persist inconsistent long-term artifact: "
            f"len(records)={len(data['records'])} candidates={data['summary']['candidates']}"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, indent=2, default=str)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(encoded, encoding="utf-8")
        os.replace(tmp, target)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise
    return target


def load_long_term_scan(path: str | Path = DEFAULT_LONG_TERM_PATH) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1 or not isinstance(payload.get("records"), list):
            return None
        return payload
    except Exception:
        return None
