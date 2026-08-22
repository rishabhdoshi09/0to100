"""Versioned symbol → sector / industry map with honest PIT status.

A static modern classification may be used descriptively. It must never
masquerade as historical truth. Frozen snapshot files are immutable.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAP_VERSION = "sector_map.v2"
PIT_SECTOR_STRONG = "PIT_SECTOR_STRONG"
STATIC_BACKFILL = "STATIC_BACKFILL"
UNKNOWN = "UNKNOWN"

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "logs" / "sector_maps"


def _iso_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _official_industry_from_cache() -> dict[str, str]:
    """Current NSE index-constituent Industry column, if ingested. Not history."""
    import csv
    import io
    d = Path(__file__).resolve().parent.parent / "logs" / "acquisition" / "raw" / "sector"
    if not d.exists():
        return {}
    out: dict[str, str] = {}
    for p in sorted(d.glob("*.csv")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            for r in reader:
                r = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
                sym = str(r.get("Symbol") or r.get("SYMBOL") or "").strip().upper()
                industry = str(r.get("Industry") or r.get("industry") or "").strip()
                if sym and industry and sym not in out:
                    out[sym] = industry
        except Exception:
            continue
    return out


def build_static_map() -> dict[str, Any]:
    """Reuse SEPA-003 comment parser + optional official current industry.

    Official industry raises coverage. It does not upgrade PIT class.
    """
    from research.sepa003.sector import load_sector_map_v1

    v1 = load_sector_map_v1()
    mapping = dict(v1.get("map") or {})
    official = _official_industry_from_cache()
    for sym, industry in official.items():
        mapping.setdefault(sym, industry)
    rows = []
    for sym, sec in sorted(mapping.items()):
        industry = official.get(sym) or sec
        rows.append({
            "symbol": sym,
            "sector": sec,
            "industry": industry,
            "classification_source": (
                "nse_index_constituent_industry" if sym in official else v1.get("source")
            ),
            "valid_from": None,
            "valid_to": None,
            "source_timestamp": None,
            "mapping_version": MAP_VERSION,
            "pit_status": STATIC_BACKFILL,
        })
    blob = json.dumps({"version": MAP_VERSION, "rows": rows}, sort_keys=True).encode()
    return {
        "version": MAP_VERSION,
        "pit_class": STATIC_BACKFILL,
        "sector_identity_pit": False,
        "n_mapped": len(rows),
        "n_official_industry": len(official),
        "n_unknown_policy": "unmapped stays UNKNOWN",
        "source": (
            "sepa003+nse_index_constituent_industry" if official else v1.get("source")
        ),
        "never_projects_silently": True,
        "content_hash": hashlib.sha256(blob).hexdigest(),
        "rows": rows,
        "map": mapping,
    }


def freeze_snapshot(
    *,
    dest_dir: Path | None = None,
    as_of: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> Path:
    """Write an immutable versioned snapshot. Later rebuilds use a new file."""
    payload = mapping or build_static_map()
    as_of = as_of or _iso_today()
    d = dest_dir or _DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    name = f"{payload['version']}_{as_of}_{payload['content_hash'][:12]}.json"
    path = d / name
    if path.exists():
        return path
    body = {
        **{k: v for k, v in payload.items() if k != "map"},
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "as_of_label": as_of,
        "immutable": True,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_snapshot(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sector snapshot must be an object")
    return raw


def sector_of(
    symbol: str,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snap = snapshot or build_static_map()
    rows = {r["symbol"]: r for r in snap.get("rows") or []}
    sym = str(symbol or "").upper()
    row = rows.get(sym)
    if not row:
        # fallback map dict
        sec = (snap.get("map") or {}).get(sym)
        if not sec:
            return {
                "symbol": sym,
                "sector": UNKNOWN,
                "industry": UNKNOWN,
                "pit_status": UNKNOWN,
                "mapping_version": snap.get("version") or MAP_VERSION,
            }
        return {
            "symbol": sym,
            "sector": sec,
            "industry": sec,
            "pit_status": snap.get("pit_class") or STATIC_BACKFILL,
            "mapping_version": snap.get("version") or MAP_VERSION,
        }
    return dict(row)


def coverage(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    snap = snapshot or build_static_map()
    return {
        "version": snap.get("version"),
        "n_mapped": snap.get("n_mapped") or len(snap.get("rows") or []),
        "pit_class": STATIC_BACKFILL,
        "status": "RESEARCH_READY_WITH_LIMITATIONS",
        "content_hash": snap.get("content_hash"),
        "note": (
            "Static modern / comment-derived classification. Descriptive use "
            "only. Not PIT_SECTOR_STRONG. Do not back-project as historical truth."
        ),
    }
