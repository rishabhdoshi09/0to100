"""Guard against F&O learning becoming self-reinforcing.

A ranking that reads real evidence has an obvious failure mode: a setup that
gets promoted trades more, which grows its sample faster, which can make it
look even stronger even if its true edge is mediocre -- while a setup that
never gets picked (because it ranks lower, or capacity/underlying-already-
open caps hold it back) never accumulates enough sample to prove itself
either way. This module makes that loop visible rather than letting it run
silently:

  - ``scans_seen``   -- how often a context appeared in a ranked cycle at all
  - ``times_top5``   -- how often it was highly ranked
  - ``times_selected`` -- how often a paper position was actually opened for it
  - ``selection_rate`` -- selected / seen, the number a self-reinforcing loop
    would drive toward 1.0 for a promoted context and toward 0.0 for a
    demoted one

This is diagnostic instrumentation only: it never feeds
product.fno_evidence_fusion, never changes a ranking, and never gates a
trade. It is a separate, durable counter store so a promoted context's
growing selection rate is something a human (or a future guardrail) can
actually see, instead of only inferring it from a rising sample count.

It also makes explicit what is already true elsewhere in the desk and is
easy to lose sight of: rejected/not-selected candidates are NOT dropped from
learning. product.fno_historical_walkforward grades every historical
candidate this same context could produce, taken or not
(classification_counts_by_score_bucket splits exactly on that "taken"
line) -- this module's job is only to show the live, forward-cycle version
of the same exposure/selection split.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1
DEFAULT_PATH = logs_path("product", "fno_exposure_tracking.json")
TOP_N = 5

#: A context needs at least this many scan appearances before its selection
#: rate means anything -- one lucky/unlucky cycle is not a pattern.
MIN_SCANS_FOR_A_READING = 10
#: Selection rate above this, with enough sample, flags possible
#: self-reinforcement (this context crowds out almost everything else it
#: competes with).
HIGH_SELECTION_RATE = 0.80
#: Selection rate at/below this despite frequent top-5 appearances flags
#: exploration starvation (capacity or ties keep starving a plausible setup
#: of the sample it would need to ever move ranking).
LOW_SELECTION_RATE_DESPITE_TOP5 = 0.10


def store_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_FNO_EXPOSURE_TRACKING")
    return Path(override) if override else DEFAULT_PATH


def _load(path: str | Path | None) -> dict[str, Any]:
    target = store_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or not isinstance(payload.get("contexts"), dict):
        payload = {"schema_version": SCHEMA_VERSION, "contexts": {}}
    return payload


def _save(payload: Mapping[str, Any], path: str | Path | None) -> None:
    target = store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)


def record_scan_exposure(
    ranked_rows: Sequence[Mapping[str, Any]],
    *,
    opened_underlyings: set[str] | frozenset[str],
    now_iso: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Record one ranking cycle's exposure/selection for every usable context.

    ``ranked_rows`` is product.fno_ranking.rank_fno_candidates's output
    (already in rank order); ``opened_underlyings`` is the set of underlying
    symbols a paper position was actually opened for THIS cycle. A row
    without a usable ranking context (no direction on the setup) is skipped
    -- there is nothing to attribute exposure to.
    """
    from product.fno_evidence import fno_context_key

    payload = _load(path)
    contexts = payload.setdefault("contexts", {})
    for index, row in enumerate(ranked_rows):
        setup = row.get("setup") if isinstance(row.get("setup"), Mapping) else {}
        key = fno_context_key(setup)
        if not key:
            continue
        symbol = str(row.get("symbol") or "").upper()
        entry = contexts.setdefault(key, {
            "scans_seen": 0, "times_top5": 0, "times_selected": 0, "last_seen_at": "",
        })
        entry["scans_seen"] = int(entry.get("scans_seen") or 0) + 1
        if index < TOP_N:
            entry["times_top5"] = int(entry.get("times_top5") or 0) + 1
        if symbol and symbol in opened_underlyings:
            entry["times_selected"] = int(entry.get("times_selected") or 0) + 1
        if now_iso:
            entry["last_seen_at"] = now_iso
    payload["schema_version"] = SCHEMA_VERSION
    _save(payload, path)
    return payload


def exposure_report(path: str | Path | None = None) -> dict[str, Any]:
    """Read-only summary: per-context exposure/selection, plus the two bias
    flags this module exists to surface. Never mutates the store.
    """
    payload = _load(path)
    contexts = dict(payload.get("contexts") or {})
    rows: list[dict[str, Any]] = []
    over_selected: list[dict[str, Any]] = []
    under_explored: list[dict[str, Any]] = []
    for key, entry in contexts.items():
        seen = int(entry.get("scans_seen") or 0)
        selected = int(entry.get("times_selected") or 0)
        top5 = int(entry.get("times_top5") or 0)
        rate = (selected / seen) if seen else 0.0
        row = {
            "context_key": key,
            "scans_seen": seen,
            "times_top5": top5,
            "times_selected": selected,
            "selection_rate": round(rate, 4),
            "last_seen_at": entry.get("last_seen_at") or "",
        }
        rows.append(row)
        if seen >= MIN_SCANS_FOR_A_READING and rate >= HIGH_SELECTION_RATE:
            over_selected.append(row)
        if seen >= MIN_SCANS_FOR_A_READING and top5 >= MIN_SCANS_FOR_A_READING and rate <= LOW_SELECTION_RATE_DESPITE_TOP5:
            under_explored.append(row)
    rows.sort(key=lambda r: -int(r.get("scans_seen") or 0))
    return {
        "schema_version": SCHEMA_VERSION,
        "contexts_tracked": len(rows),
        "min_scans_for_a_reading": MIN_SCANS_FOR_A_READING,
        "high_selection_rate_threshold": HIGH_SELECTION_RATE,
        "low_selection_rate_threshold": LOW_SELECTION_RATE_DESPITE_TOP5,
        "possibly_self_reinforcing": over_selected[:10],
        "possibly_exploration_starved": under_explored[:10],
        "top_contexts": rows[:10],
        "note": (
            "Diagnostic only -- never feeds ranking or gates a trade. A context "
            "flagged here still learns exactly as product.fno_evidence_fusion "
            "already governs; this only makes the exposure pattern visible."
        ),
    }
