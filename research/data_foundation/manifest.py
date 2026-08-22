"""Immutable evidence snapshot manifest. Version changes change the hash."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _sha_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, timeout=5,
        )
        return out.strip()
    except Exception:
        return None


def _cfg_hash(config: dict[str, Any] | None) -> str:
    blob = json.dumps(config or {}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build_manifest(
    *,
    as_of: str,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    bind_live_stores: bool = True,
) -> dict[str, Any]:
    from data.benchmarks import file_coverage
    from data.corporate_actions import events_path as ca_events_path
    from data.pit_events import content_hash as events_hash
    from data.pit_events import ledger_path as earnings_path
    from data.pit_fundamentals import content_hash as fund_hash
    from data.pit_fundamentals import ledger_path as fund_path
    from data.sector_map import MAP_VERSION, build_static_map
    from data.universe_history import history_path

    bhav_pkl = ROOT / "logs" / "bhav" / "bhav_store.pkl"
    if not bhav_pkl.exists():
        cand = list((ROOT / "logs" / "bhav").glob("*.pkl")) if (ROOT / "logs" / "bhav").exists() else []
        bhav_pkl = cand[0] if cand else bhav_pkl

    sector = build_static_map()
    bench = file_coverage() if bind_live_stores else {"n_files": 0, "files_hash": None, "first": None, "last": None}
    body = {
        "schema_version": 1,
        "as_of": as_of,
        "price_store": {
            "path": str(bhav_pkl) if (bind_live_stores and bhav_pkl.exists()) else None,
            "hash": _sha_file(bhav_pkl) if (bind_live_stores and bhav_pkl.exists()) else None,
        },
        "ca_ledger": {
            "path": str(ca_events_path()) if bind_live_stores else None,
            "hash": _sha_file(Path(ca_events_path())) if bind_live_stores else None,
        },
        "universe_history": {
            "path": str(history_path()) if bind_live_stores else None,
            "hash": _sha_file(Path(history_path())) if bind_live_stores else None,
        },
        "fundamentals": {
            "path": str(fund_path()) if bind_live_stores else None,
            "hash": fund_hash() if bind_live_stores else None,
            "version": "pit_fundamentals.v1",
        },
        "earnings_events": {
            "path": str(earnings_path()) if bind_live_stores else None,
            "hash": events_hash() if bind_live_stores else None,
            "version": "pit_events.v1",
        },
        "sector_map": {
            "version": MAP_VERSION,
            "hash": sector.get("content_hash"),
            "pit_class": sector.get("pit_class"),
        },
        "benchmarks": {
            "source": "nse_ind_close_all_local",
            "files_hash": bench.get("files_hash"),
            "first": bench.get("first"),
            "last": bench.get("last"),
        },
        "code_sha": _code_sha(),
        "experiment_config_hash": _cfg_hash(config),
        "extra": extra or {},
    }
    ident = json.dumps(body, sort_keys=True, default=str).encode()
    body["snapshot_hash"] = hashlib.sha256(ident).hexdigest()
    return body


def snapshot_hash(manifest: dict[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k != "snapshot_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
