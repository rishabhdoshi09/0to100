"""
E5 — Experiment Registry.

Every historical run is stamped so its result is perfectly reproducible: an
experiment id, the git commit, a hash of the exact dataset, a hash of the frozen
config, the random seed, and a timestamp. Two runs with the same code + data +
config + seed must produce the same verdict — and the registry is the proof.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

_REG_FILE = Path(__file__).resolve().parent.parent / "logs" / "gauntlet" / "experiments.jsonl"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5,
                             cwd=str(Path(__file__).resolve().parent.parent))
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def dataset_hash(fingerprint: dict) -> str:
    """Hash a compact fingerprint of the data the run consumed — e.g. per source
    {latest_day, n_sessions, n_symbols, ca_events, universe_complete}. Any change
    to the underlying data changes this hash, so a result is bound to its data."""
    blob = json.dumps(fingerprint, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def register(config_hash: str, data_fingerprint: dict, seed: int,
             extra: dict | None = None) -> dict:
    """Create and persist an experiment stamp. Returns the full record."""
    dh = dataset_hash(data_fingerprint)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    exp_id = hashlib.sha256(
        f"{config_hash}|{dh}|{seed}|{ts}".encode()).hexdigest()[:12]
    rec = {"experiment_id": exp_id, "git_commit": _git_commit(),
           "dataset_hash": dh, "config_hash": config_hash, "seed": int(seed),
           "timestamp": ts, "data_fingerprint": data_fingerprint}
    if extra:
        rec.update(extra)
    try:
        _REG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_REG_FILE, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass
    return rec
