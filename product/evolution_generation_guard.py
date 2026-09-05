"""Generation guard for autonomous historical evidence.

Historical reproduction is valid only for the exact policy/data generation that
produced it. If decision rules, PIT versions, or the warehouse fingerprint move,
old derived replay artifacts are invalidated before PAPER selection can use them.

Only derived autonomous-evolution artifacts are removed. Source market data,
forward paper evidence, and live state are never touched.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDENTITY = ROOT / "logs" / "product" / "autonomous_evolution_identity.json"
_lock = threading.Lock()


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, sort_keys=True, default=str)
        return value
    except Exception:
        return str(value)


def current_generation() -> dict[str, Any]:
    versions: Mapping[str, Any] | Any = {}
    warehouse: Any = {}
    champion: Mapping[str, Any] | Any = {}
    try:
        from product.pit_versions import current_versions

        raw = current_versions()
        versions = raw.as_dict() if hasattr(raw, "as_dict") else dict(raw or {})
    except Exception as exc:
        versions = {"error": str(exc)[:160]}
    try:
        from product.pit_warehouse import warehouse_fingerprint

        warehouse = warehouse_fingerprint()
    except Exception as exc:
        warehouse = {"error": str(exc)[:160]}
    try:
        from product.strategy_catalog import ensemble_identity

        champion = ensemble_identity() or {}
    except Exception as exc:
        champion = {"error": str(exc)[:160]}

    payload = {
        "versions": _json_safe(versions),
        "warehouse": _json_safe(warehouse),
        "champion": _json_safe(champion),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return {
        **payload,
        "fingerprint": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20],
    }


def identity_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_AUTONOMOUS_EVOLUTION_IDENTITY")
    return Path(override) if override else DEFAULT_IDENTITY


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["recorded_at"] = datetime.now(timezone.utc).isoformat()
    data["live_locked"] = True
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def ensure_current_generation(*, identity: str | Path | None = None) -> dict[str, Any]:
    """Invalidate stale derived history and return the current generation proof."""
    from product.autonomous_evolution import run_dir, state_path

    target = identity_path(identity)
    current = current_generation()
    with _lock:
        previous = _read(target)
        previous_fp = str(previous.get("fingerprint") or "")
        current_fp = str(current.get("fingerprint") or "")
        state_exists = state_path().exists()
        # A pre-guard state has no identity proof. Treat it as stale instead of
        # grandfathering historical evidence produced under unknown rules.
        changed = bool(
            (previous_fp and previous_fp != current_fp)
            or (not previous_fp and state_exists)
        )
        invalidated: list[str] = []
        if changed:
            state = state_path()
            if state.exists():
                try:
                    state.unlink()
                    invalidated.append(str(state))
                except OSError:
                    pass
            derived = run_dir()
            if derived.exists():
                try:
                    shutil.rmtree(derived)
                    invalidated.append(str(derived))
                except OSError:
                    pass
        _write(target, current)
    return {
        **current,
        "changed": changed,
        "previous_fingerprint": previous_fp,
        "invalidated": invalidated,
        "historical_replay_required": bool(changed or not state_path().exists()),
        "live_locked": True,
    }
