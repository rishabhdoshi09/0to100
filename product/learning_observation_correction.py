"""Find and quarantine learning observations built from a store that had lost official history.

The store integrity audit (docs/research/store_integrity_2026-11-22.md) found
50 young listings whose first weeks of official history were missing from the
OHLCV store. A DECISION observation frozen for one of them carries point-in-time
technical features (product.pit_features) computed from that truncated history.
Most are unaffected, but a long window such as the 252-bar
``dist_from_high_pct`` is wrong when the true high sat in the lost sessions
(VEDPOWER on 2026-11-20: stored 8.68%, true 29.79%).

The frozen vector is the decision-time record and is never rewritten. A proven
defect is QUARANTINED (research.feature_store.quarantine_batch), so learning
and evidence consumers skip it.

* ``plan`` (read-only; COPIES only, ``/Volumes`` refused) is bound to the
  store repair's manifest: the current pickle must be its ``target_sha256`` and
  the repaired pickle its ``new_sha256``, and both must cover the same sessions.
  It recomputes every PIT fill of every changed-history DECISION observation
  twice, each frame read the way ``get_ohlcv`` serves it (corporate actions
  from ``--ca-events``):
    - on the current store, where it must reproduce the stored value, or the
      row is ``unexplained`` and the plan incomplete;
    - on the repaired store, where a value that moves by more than ``ABS_TOL``
      AND more than ``REL_TOL``, or disappears, is a proven defect.
  Cross-sectional fills (breadth, RS percentile) depend on every symbol, so ANY
  observation carrying one, and a changed symbol's own delivery fill, is
  counted under ``not_rederived`` and the plan is incomplete: it never claims
  more than it re-derived (GPT QT-03).

  Features the decision carried itself (the scanner's RSI, quality score, the
  setup and its levels) came from the scanner's history (this store, or the
  active Kite snapshot while Kite is authoritative: scan.bulk_fetcher.get_cached)
  and cannot be recomputed here. So the plan also lists, as ``exposed``, every
  observation of any kind whose INPUT PREFIX (every bar up to its cutoff, in
  the order the producer read it: an EMA is seeded at the first bar) differs
  between the two stores (GPT QT-09). Exposed rows are evidence of risk, not
  proof of a wrong value.

  With ``--shadow-db`` (a copy of ``logs/feature002/shadow.db``) the plan also
  re-derives that research ledger's trend features (research.feature001) for
  the same symbols and REPORTS the rows that differ. Nothing in that ledger is
  ever written by this tool.
* ``apply`` quarantines exactly the planned rows in ONE SQLite transaction
  (every vector re-verified inside it), under a batch id derived from the plan,
  so a rerun after an interruption completes the same batch and ``rollback``
  releases exactly that batch (GPT QT-05). It needs the owner's approval in
  words, a complete plan, a verified repair binding and a stopped desk for a
  live target. ``--include-exposed`` also quarantines the exposed rows.

    python -m product.learning_observation_correction plan --features-db <copy> \\
        --current-pickle <copy> --repaired-pickle <planned pickle> --ca-events <copy> \\
        --store-manifest <store_manifest.json> --out <manifest.json>
    python -m product.learning_observation_correction apply --manifest <manifest.json> \\
        --features-db <db> --owner-approval "<the owner's words>"
    python -m product.learning_observation_correction rollback --manifest <manifest.json> --features-db <db> \\
        --owner-approval "<the owner's words>"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

#: A recomputed fill must differ by more than both to count as a defect: the
#: 20-EMA moves by ~0.001 pp with a longer seed, which is not a defect.
ABS_TOL = 0.05
REL_TOL = 0.01
REASON = "TRUNCATED_HISTORY_PIT_FILL"
EXPOSED_REASON = "TRUNCATED_HISTORY_INPUT_WINDOW"
#: Fills computed across the whole market: a changed name moves every peer.
CROSS_SECTIONAL = ("breadth_pct_above_50dma", "rs_percentile")
#: Fills computed from the symbol's own frame but not by technical_features.
OWN_FRAME_OTHER = ("delivery_pct",)
#: Market-wide values the SCANNER writes straight into a vector (non-event
#: rows, cards): they carry no PIT-fill marker (GPT R6, cycle 6).
CARRIED_CROSS = ("breadth_pct_above_50dma", "rs_percentile", "sector_rank_percentile",
                 "sector_strength", "volatility_percentile")


#: Paths on the production runtime volume. Copy mode (--target-is-copy) can
#: never authorize a path under one of these (Cycle 7).
PRODUCTION_PREFIXES = ("/Volumes/",)


def _is_production(path: str | Path) -> bool:
    """Case-insensitively: the Mac's volumes are case-insensitive and realpath keeps
    the spelling given, so '/volumes/...' is the same production path (cycle-7 review)."""
    p = str(Path(path).resolve()).lower()
    return p.startswith(tuple(x.lower() for x in PRODUCTION_PREFIXES))


def _not_live(path: str | Path, what: str) -> Path:
    p = Path(path).resolve()
    if _is_production(p):
        raise SystemExit(f"refusing {what} under /Volumes: {p} (work on a copy)")
    return p


def _entry(path: str | Path) -> str:
    """The directory ENTRY a repair writes: the parent resolved, the final name
    kept. bhav_store_rebuild swaps with os.replace(candidate, target), which
    replaces that entry (a final-component symlink included), so resolving the
    final component would make a successful repair invalidate its own
    correction plan (GPT C7-04, cycle 7)."""
    p = Path(path)
    return str(p.parent.resolve() / p.name)


def _repair_identity(sm: dict) -> dict:
    """The IMMUTABLE identity of a store repair: which file it replaces, from what
    to what. Its state, backup path and timestamps legitimately change as the
    repair is applied; these three never do (Cycle 7)."""
    target = str(sm.get("target_pickle") or "")
    return {"target_pickle": _entry(target) if target else "",
            "target_sha256": str(sm.get("target_sha256") or ""),
            "new_sha256": str(sm.get("new_sha256") or "")}


def _identity_sha(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _vector_sha(features_json: str) -> str:
    from research.feature_store import vector_sha256

    return vector_sha256(features_json)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _payload(path: Path) -> dict:
    with open(path, "rb") as fh:
        return dict(pickle.load(fh) or {})


def _last_day(payload: dict):
    import pandas as pd

    day = payload.get("last_day")
    if day is not None:
        return pd.Timestamp(day).normalize()
    ends = [pd.DatetimeIndex(f.index).max() for f in (payload.get("store") or {}).values()
            if f is not None and len(f)]
    return pd.Timestamp(max(ends)).normalize() if ends else None


def _write_json(path: Path, obj: dict) -> None:
    """Atomic: a reader sees the old manifest or the new one, never a torn file."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(obj, indent=2, default=str))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _same_window(a, b) -> bool:
    """Same dates in the same ORDER and the same VALUES (dtype-blind: a full build
    stores volume as int64 where appends left float64, which is not a change)."""
    import numpy as np

    if a is None or b is None:
        return a is None and b is None
    if len(a) != len(b) or not _index_equal(a, b):
        return False
    cols = set(map(str, a.columns))
    if cols != set(map(str, b.columns)):
        return False
    for c in sorted(cols):
        x = a[c].to_numpy(dtype=float, na_value=np.nan)
        y = b[c].to_numpy(dtype=float, na_value=np.nan)
        if not np.array_equal(x, y, equal_nan=True):
            return False
    return True


def _index_equal(a, b) -> bool:
    import numpy as np
    import pandas as pd

    ia, ib = pd.DatetimeIndex(a.index), pd.DatetimeIndex(b.index)
    ia = ia.tz_localize(None) if ia.tz is not None else ia
    ib = ib.tz_localize(None) if ib.tz is not None else ib
    return np.array_equal(ia.normalize().as_unit("ns").asi8, ib.normalize().as_unit("ns").asi8)


def _changed_symbols(current: dict, repaired: dict) -> set:
    """Symbols whose rows differ in any date, value OR order.

    A corrected archive changes values only, and a re-sorted frame changes
    neither: the desk's rolling windows read the frame in its stored order, so
    an out-of-order frame is examined like any other changed history.
    """
    out = set()
    for sym in set(current) | set(repaired):
        if not _same_window(current.get(sym), repaired.get(sym)):
            out.add(sym)
    return out


def _served(frame, events):
    """The frame as data.bhavcopy_store.get_ohlcv serves it (CA-adjusted on read)."""
    if frame is None or not events:
        return frame
    from data.corporate_actions import adjust_frame

    return adjust_frame(frame, events, copy=True)


def _prefix(frame, cut):
    """Every bar up to ``cut`` (inclusive), in STORED order: what a producer read."""
    import pandas as pd

    if frame is None or cut is None:
        return None
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return frame[idx.normalize() <= pd.Timestamp(cut)]


def _material(stored: float, repaired: float) -> bool:
    diff = abs(float(repaired) - float(stored))
    return diff > ABS_TOL and diff > REL_TOL * max(abs(float(stored)), 1e-12)


def _shadow_report(shadow_db: Path, current: dict, repaired: dict, events: dict, changed: set) -> dict:
    """feature002 rows whose trend features differ on the repaired store (report only)."""
    import pandas as pd
    from research.feature001.trend_features import compute_trend_features

    con = sqlite3.connect(f"file:{shadow_db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT event_id, candidate_set_id, symbol, session_date, feature_snapshot, "
                           "eligible_primary FROM observations").fetchall()
    finally:
        con.close()
    rep: dict[str, Any] = {"examined": 0, "affected": [], "exposed_only": [], "unexplained": []}
    sets = set()
    for eid, cset, sym, sess, snap, elig in rows:
        if sym not in changed:
            continue
        snap = json.loads(snap or "{}")
        trend = snap.get("trend") or {}
        cut = pd.Timestamp(str(snap.get("hist_as_of") or sess)[:10])
        ev = events.get(str(sym or "").upper())
        a = _prefix(_served(current.get(sym), ev), cut)
        b = _prefix(_served(repaired.get(sym), ev), cut)
        if _same_window(a, b):
            continue
        rep["examined"] += 1
        now = compute_trend_features(a) if a is not None and len(a) else {}
        fixed = compute_trend_features(b) if b is not None and len(b) else {}
        diffs, odd = {}, []
        for k, stored in trend.items():
            if k == "version":
                continue
            x, y = now.get(k), fixed.get(k)
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (stored, x))
            if numeric:
                if abs(float(x) - float(stored)) > 1e-3:
                    odd.append(k)
                elif y is None or isinstance(y, bool) or _material(stored, y):
                    diffs[k] = {"stored": stored, "repaired": y}
            elif stored != x:
                odd.append(k)
            elif x != y:
                diffs[k] = {"stored": stored, "repaired": y}
        row = {"event_id": eid, "symbol": sym, "session_date": sess, "candidate_set_id": cset,
               "eligible_primary": bool(elig)}
        if odd:
            rep["unexplained"].append(dict(row, features=odd))
        elif diffs:
            rep["affected"].append(dict(row, features=diffs))
            sets.add(cset)
        else:
            rep["exposed_only"].append(row)
    rep["candidate_sets_with_affected_rows"] = len(sets)
    return rep


def _archive_sessions(store_manifest) -> list | None:
    """Verified sessions named by the repair manifest's archives (DDMMYYYY.csv), or None."""
    import pandas as pd

    if store_manifest is None:
        return None
    try:
        names = (json.loads(Path(store_manifest).read_text()).get("source") or {}).get("archives") or {}
        return sorted(pd.Timestamp(f"{n[4:8]}-{n[2:4]}-{n[0:2]}") for n in names if len(n) >= 12)
    except Exception:
        return None


def _unseen_session(last, cut, sessions, holidays=None) -> bool:
    """Could a decision with this cutoff have read a session the store lacks?

    The expected sessions come from the VERIFIED exchange calendar
    (research.intelligence.data.nse_calendar), not from the archives the store
    happens to hold: a missing Monday archive is a missing session, not a gap in
    the calendar (GPT R2, cycle 6). A year the calendar does not cover fails
    closed. An archive the calendar does not list (a special session) counts too.
    """
    import pandas as pd
    from research.intelligence.data import nse_calendar as NC

    if cut is None or last is None:
        return False
    cut, last = pd.Timestamp(cut).normalize(), pd.Timestamp(last).normalize()
    if cut <= last:
        return False
    if sessions is not None and any(last < d <= cut for d in sessions):
        return True
    hol = NC.load_holidays() if holidays is None else holidays
    if NC.calendar_gap((last + pd.Timedelta(days=1)).date(), cut.date(), hol):
        return True                                     # unknown coverage: fail closed
    day = last + pd.Timedelta(days=1)
    while day <= cut:
        if NC.is_session(day.date(), hol):
            return True
        day += pd.Timedelta(days=1)
    return False


def _carries_cross(features: dict, filled) -> list:
    """Market-wide values in a vector: PIT fills, or written by the scanner."""
    out = [k for k in (filled or []) if k in CROSS_SECTIONAL]
    out += [k for k in CARRIED_CROSS if k not in out and features.get(k) is not None]
    return out


#: The columns the plan's judgment depends on, in this order (the digest's projection).
SCOPE_COLUMNS = "observation_id, symbol, ts, kind, features, meta"


def _scope_digest(rows, changed) -> tuple[str, int]:
    """Identity of every observation the plan judged: (sha256, count).

    Each row of a changed symbol, or carrying a market-wide value, contributes
    EVERY immutable input the planner used: id, symbol, decision time, kind,
    the vector and its PIT-fill provenance. Mutable settlement columns (outcome)
    are not part of it. apply recomputes it before writing and again inside its
    transaction (GPT R2, cycle 6; C7V-01, cycle 7: a row with the same id and
    vector but no PIT-fill marker was judged differently and must not match).
    """
    items = []
    for oid, sym, ts, kind, feats, meta in rows:
        f = json.loads(feats or "{}")
        pit = json.loads(meta or "{}").get("pit_features") or {}
        filled = pit.get("filled") or []
        if sym in changed or _carries_cross(f, filled):
            items.append(json.dumps({"id": oid, "symbol": sym, "ts": ts, "kind": kind,
                                     "vector": _vector_sha(feats), "pit": pit},
                                    sort_keys=True, default=str))
    items.sort()
    return hashlib.sha256("\n".join(items).encode()).hexdigest(), len(items)


def _breadth_by_session(frames: dict, sessions) -> dict:
    """{session: % of names above their 50-DMA} on one store (product.pit_features' market pass)."""
    from product.pit_features import BREADTH_MIN_N, _market_pass

    if not sessions:
        return {}
    syms = sorted(frames)
    market = _market_pass(tuple(d.date() for d in sessions), syms, frames.get)
    return {d: (100.0 * up / n) for d, (up, n) in market.items() if n >= BREADTH_MIN_N}


def _binding(store_manifest, cur_p: Path, rep_p: Path) -> dict:
    """Tie the plan to the store repair it assumes (GPT QT-02)."""
    if store_manifest is None:
        return {"verified": False, "reason": "no store manifest given"}
    sm_path = Path(store_manifest).resolve()
    sm = json.loads(sm_path.read_text())
    cur_sha, rep_sha = _file_sha(cur_p), _file_sha(rep_p)
    identity = _repair_identity(sm)
    ok = (cur_sha == identity["target_sha256"] and rep_sha == identity["new_sha256"]
          and bool(identity["target_pickle"]) and bool(sm.get("ready")))
    return {"verified": ok, "store_manifest": str(sm_path), "store_manifest_sha256": _file_sha(sm_path),
            "current_sha256": cur_sha, "repaired_sha256": rep_sha,
            "target_pickle": identity["target_pickle"],
            "repair_identity": identity, "repair_identity_sha256": _identity_sha(identity),
            "expected_current_sha256": sm.get("target_sha256"), "expected_repaired_sha256": sm.get("new_sha256"),
            "repair_ready": bool(sm.get("ready"))}


def plan(features_db: str | Path, current_pickle: str | Path, repaired_pickle: str | Path,
         out: str | Path, ca_events: str | Path | None = None,
         shadow_db: str | Path | None = None, store_manifest: str | Path | None = None) -> dict[str, Any]:
    from data.corporate_actions import load_events_with_digest
    from product.pit_features import PRODUCED, cutoff_session, technical_features

    db = _not_live(features_db, "the feature store")
    if not db.is_file():
        raise SystemExit(f"refusing: no feature store at {db}")
    cur_p, rep_p = _not_live(current_pickle, "the current pickle"), _not_live(repaired_pickle, "the repaired pickle")
    out = _not_live(out, "the manifest")
    ca_p = None if ca_events is None else _not_live(ca_events, "the corporate-action table")
    events, ca_digest = load_events_with_digest(ca_p) if ca_p is not None else ({}, "")
    cur_payload, rep_payload = _payload(cur_p), _payload(rep_p)
    current, repaired = dict(cur_payload.get("store") or {}), dict(rep_payload.get("store") or {})
    cur_last, rep_last = _last_day(cur_payload), _last_day(rep_payload)
    refusals = []
    if cur_last is None or rep_last is None or cur_last != rep_last:
        # A repaired store that ends on another session cannot judge a decision
        # the other one could see (GPT QT-02): a stale candidate is not proof.
        refusals.append({"reason": "COVERAGE_MISMATCH", "current_last_day": str(cur_last),
                         "repaired_last_day": str(rep_last)})
    changed = _changed_symbols(current, repaired)
    sessions = _archive_sessions(store_manifest)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT observation_id, symbol, ts, features, meta, outcome, kind "
                           "FROM observations").fetchall()
    finally:
        con.close()
    import pandas as pd

    affected, unexplained, examined, not_rederived, exposed = [], [], 0, {}, []
    # Scanner-carried breadth is bounded by recomputing breadth on BOTH stores:
    # only the changed names can move it, so |repaired - current| per session is
    # the most the store defect can have shifted any carried value (GPT R6).
    breadth_cur = breadth_rep = None
    breadth_bound = {"sessions_checked": 0, "max_abs_delta_pp": 0.0, "material_sessions": []}
    needs_breadth = bool(changed) and any(
        json.loads(f or "{}").get("breadth_pct_above_50dma") is not None for _o, _s, _t, f, _m, _x, _k in rows)
    if needs_breadth:
        cal = sessions if sessions is not None else sorted(
            {pd.Timestamp(d).normalize() for fr in repaired.values() if fr is not None
             for d in pd.DatetimeIndex(fr.index)})
        breadth_cur, breadth_rep = _breadth_by_session(current, cal), _breadth_by_session(repaired, cal)
    archive_days = sessions
    for oid, sym, ts, feats, meta, outcome, kind in rows:
        all_filled = (json.loads(meta or "{}").get("pit_features") or {}).get("filled") or []
        vec = json.loads(feats or "{}")
        for k in all_filled:
            # A cross-sectional fill moves when ANY symbol's history changes.
            if k in CROSS_SECTIONAL and changed:
                not_rederived[k] = not_rederived.get(k, 0) + 1
        for k in CARRIED_CROSS:
            if k in all_filled or vec.get(k) is None or not changed:
                continue
            if k != "breadth_pct_above_50dma" or breadth_cur is None:
                not_rederived[f"{k}[carried]"] = not_rederived.get(f"{k}[carried]", 0) + 1
                continue
            cut = cutoff_session(ts)
            day = None
            if cut is not None:
                pool = archive_days if archive_days is not None else sorted(breadth_rep)
                prior = [d for d in pool if pd.Timestamp(d).normalize() <= pd.Timestamp(cut)]
                day = pd.Timestamp(prior[-1]).date() if prior else None
            a, b = breadth_cur.get(day), breadth_rep.get(day)
            if a is None or b is None:
                not_rederived[f"{k}[carried]"] = not_rederived.get(f"{k}[carried]", 0) + 1
                continue
            breadth_bound["sessions_checked"] += 1
            delta = abs(b - a)
            breadth_bound["max_abs_delta_pp"] = max(breadth_bound["max_abs_delta_pp"], round(delta, 6))
            if delta > ABS_TOL:
                not_rederived[f"{k}[carried]"] = not_rederived.get(f"{k}[carried]", 0) + 1
                if str(day) not in breadth_bound["material_sessions"]:
                    breadth_bound["material_sessions"].append(str(day))
        if sym not in changed:
            continue
        for k in all_filled:
            if k in OWN_FRAME_OTHER or (k not in PRODUCED and k not in CROSS_SECTIONAL
                                        and k not in ("vix", "index_trend")):
                not_rederived[k] = not_rederived.get(k, 0) + 1
        cut = cutoff_session(ts)
        if _unseen_session(rep_last, cut, sessions):
            unexplained.append({"observation_id": oid, "reason": "AFTER_STORE_COVERAGE", "cutoff": str(cut)})
            continue
        ev = events.get(str(sym or "").upper())
        cur_f, rep_f = _served(current.get(sym), ev), _served(repaired.get(sym), ev)
        if not _same_window(_prefix(cur_f, cut), _prefix(rep_f, cut)):
            exposed.append({"observation_id": oid, "symbol": sym, "ts": ts, "kind": kind,
                            "settled": outcome is not None, "vector_sha256": _vector_sha(feats)})
        if kind != "DECISION":
            continue
        filled = [k for k in all_filled if k in PRODUCED]
        if not filled:
            continue
        examined += 1
        stored = json.loads(feats or "{}")
        now = technical_features(sym, ts, frame=cur_f)
        fixed = technical_features(sym, ts, frame=rep_f)
        diffs = {}
        for k in filled:
            s, a, b = stored.get(k), now.get(k), fixed.get(k)
            if s is None:
                continue
            if a is None or abs(float(a) - float(s)) > 1e-6:
                unexplained.append({"observation_id": oid, "feature": k, "stored": s, "current_recompute": a})
                continue
            if b is None:
                # The value cannot exist on complete history (GPT QT-03): the
                # stored number is not a property of the official record.
                diffs[k] = {"stored": s, "repaired": None}
            elif _material(s, b):
                diffs[k] = {"stored": s, "repaired": round(float(b), 6)}
        if diffs:
            affected.append({"observation_id": oid, "symbol": sym, "ts": ts, "settled": outcome is not None,
                             "features": diffs, "vector_sha256": _vector_sha(feats)})
    proven = {a["observation_id"] for a in affected}
    binding = _binding(store_manifest, cur_p, rep_p)
    digest, scope_n = _scope_digest([(r[0], r[1], r[2], r[6], r[3], r[4]) for r in rows], changed)
    manifest = {
        "schema": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": REASON,
        "features_db": str(db),
        "features_db_sha256": _file_sha(db),
        "current_pickle": str(cur_p),
        "repaired_pickle": str(rep_p),
        "store_binding": binding,
        "coverage": {"current_last_day": str(cur_last), "repaired_last_day": str(rep_last)},
        "ca_events": None if ca_p is None else str(ca_p),
        "ca_events_sha256": ca_digest,
        "tolerance": {"abs": ABS_TOL, "rel": REL_TOL},
        "symbols_with_changed_history": len(changed),
        "observations_examined": examined,
        "affected": affected,
        "exposed": [r for r in exposed if r["observation_id"] not in proven],
        "exposure_rule": "input prefix up to the cutoff differs, in stored order",
        "unexplained": unexplained,
        "not_rederived": not_rederived,
        "carried_breadth_bound": breadth_bound,
        "refusals": refusals,
        "changed_symbols": sorted(changed),
        "scope": {"sha256": digest, "observations": scope_n},
        "complete": not unexplained and not not_rederived and not refusals,
        "state": "planned",
    }
    if shadow_db is not None:
        sdb = _not_live(shadow_db, "the feature002 shadow ledger")
        manifest["shadow_feature002"] = dict(_shadow_report(sdb, current, repaired, events, changed),
                                             path=str(sdb), sha256=_file_sha(sdb), written=False)
    # Unique per plan instance: two plans never share a batch, so rolling one
    # back can never release the other's rows (GPT R5, cycle 6).
    import uuid

    manifest["batch_id"] = f"loc-{uuid.uuid4().hex[:20]}"
    _write_json(Path(out), manifest)
    return manifest


def _check_copy_mode(db: Path, target_is_copy: bool) -> None:
    if target_is_copy and _is_production(db):
        raise SystemExit(f"refusing: copy mode cannot authorize a production path ({db})")


def _require_scope(m: dict) -> None:
    """The judged observation set must be pinned; an absent digest is a refusal,
    never a skipped check (Cycle 7)."""
    scope = m.get("scope") or {}
    if not scope.get("sha256") or "changed_symbols" not in m or not isinstance(m.get("changed_symbols"), list):
        raise SystemExit("refusing: the plan does not pin the observation set it judged (scope / changed_symbols)")


def _require_installed_repair(binding: dict) -> None:
    """The repair INSTALLED must be exactly the one the plan was bound to.

    The binding's own recorded identity is the authority: the repair manifest is
    re-read only to confirm its immutable identity (target path, both shas) is
    unchanged, so a manifest replaced by another repair, or a target redirected
    to an identical-looking file, refuses. Its mutable fields (state, backup,
    timestamps) may change as the repair is applied (Cycle 7).
    """
    if not binding.get("verified"):
        raise SystemExit("refusing: the plan is not bound to a verified store repair manifest (--store-manifest)")
    target, repaired = str(binding.get("target_pickle") or ""), str(binding.get("repaired_sha256") or "")
    recorded = binding.get("repair_identity") or {}
    if not target or not repaired or not recorded or binding.get("repair_identity_sha256") != _identity_sha(recorded):
        raise SystemExit("refusing: the plan's repair binding is incomplete (re-plan with --store-manifest)")
    try:
        current = _repair_identity(json.loads(Path(binding["store_manifest"]).read_text()))
    except Exception as exc:
        raise SystemExit(f"refusing: the bound repair manifest cannot be read ({type(exc).__name__})")
    if current != recorded:
        raise SystemExit("refusing: the repair manifest no longer describes the repair the plan was bound to")
    live = Path(target)
    if _entry(live) != recorded["target_pickle"] or recorded["new_sha256"] != repaired:
        raise SystemExit("refusing: the bound target is not the repair's target")
    if not live.is_file() or _file_sha(live) != repaired:
        raise SystemExit("refusing: the live store is not the repaired pickle the plan is bound to "
                         "(apply the store repair first)")


def apply(manifest_path: str | Path, features_db: str | Path, owner_approval: str, *,
          target_is_copy: bool = False, include_exposed: bool = False) -> dict[str, Any]:
    from product.bhav_store_rebuild import _check_desk
    from research import feature_store as FS

    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if not str(owner_approval or "").strip():
        raise SystemExit("refusing: apply needs the owner's approval in words (--owner-approval)")
    if m.get("state") not in {"planned", "applying"}:
        raise SystemExit(f"refusing: the manifest is in state {m.get('state')!r}, not 'planned'")
    if not m.get("complete"):
        raise SystemExit("refusing: the plan is incomplete (unexplained or not re-derived fills, or a "
                         "coverage refusal); investigate before quarantining anything")
    db = Path(features_db).resolve()
    _check_copy_mode(db, target_is_copy)
    _require_scope(m)
    binding = m.get("store_binding") or {}
    if not target_is_copy:
        _require_installed_repair(binding)
    _check_desk(db, target_is_copy)
    entries = [{"observation_id": r["observation_id"], "reason": m["reason"],
                "vector_sha256": r["vector_sha256"],
                "evidence": {"features": r["features"], "plan": str(manifest_path),
                             "repaired_pickle": m["repaired_pickle"],
                             "source": "docs/research/store_integrity_2026-11-22.md"}}
               for r in m["affected"]]
    if include_exposed:
        entries += [{"observation_id": r["observation_id"], "reason": EXPOSED_REASON,
                     "vector_sha256": r["vector_sha256"],
                     "evidence": {"plan": str(manifest_path), "rule": m.get("exposure_rule"),
                                  "repaired_pickle": m["repaired_pickle"]}}
                    for r in m.get("exposed", [])]
    scope = {"ids": sorted(e["observation_id"] for e in entries), "include_exposed": bool(include_exposed)}
    if m.get("state") == "applying" and m.get("approved_scope") != scope:
        # A retry resumes exactly the transaction it is completing (GPT R5).
        raise SystemExit("refusing: a retry must use the same scope as the interrupted apply "
                         f"(include_exposed={m.get('approved_scope', {}).get('include_exposed')})")
    # Read-only pre-check BEFORE any write: a stale plan refuses without even
    # marking the manifest 'applying' (the in-transaction check below stays the
    # authority against a race) (GPT cycle-7 review, refusal ordering).
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            pre, _n = _scope_digest(con.execute(
                f"SELECT {SCOPE_COLUMNS} FROM observations").fetchall(),
                set(m.get("changed_symbols") or []))
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise SystemExit(f"refusing: the feature store cannot be read ({type(exc).__name__})")
    if pre != (m.get("scope") or {}).get("sha256"):
        raise SystemExit("refusing: the observation set changed since the plan; nothing was written (re-plan)")
    # The transaction record goes first: an interruption after the commit
    # leaves a manifest whose batch id drives a rerun or the rollback.
    m.update({"state": "applying", "owner_approval": owner_approval.strip(), "approved_scope": scope,
              "included_exposed": include_exposed, "apply_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    _write_json(manifest_path, m)
    changed = set(m.get("changed_symbols") or [])
    want = (m.get("scope") or {}).get("sha256")

    def same_observations(con) -> None:
        # Inside the write transaction: the store holds exactly the observations
        # the plan judged, none added or changed since (GPT R2, cycle 6).
        got, _n = _scope_digest(con.execute(
            f"SELECT {SCOPE_COLUMNS} FROM observations").fetchall(), changed)
        if got != want:
            raise FS.QuarantineRefused("the observation set changed since the plan")

    saved = FS._DB_PATH
    FS._DB_PATH = db
    try:
        try:
            results = FS.quarantine_batch(entries, batch_id=m["batch_id"], precondition=same_observations)
        except FS.QuarantineRefused as exc:
            raise SystemExit(f"refusing: {exc}; nothing was quarantined (re-plan)")
    finally:
        FS._DB_PATH = saved
    m.update({"state": "applied", "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": results})
    _write_json(manifest_path, m)
    return m


def rollback(manifest_path: str | Path, features_db: str | Path, *, target_is_copy: bool = False,
             owner_approval: str = "") -> dict[str, Any]:
    """Release exactly the rows this plan's batch inserted (never a prior quarantine).

    Releasing rows changes what learning reads, so a live rollback needs the
    owner's words like apply does; copy mode cannot touch a production path."""
    from product.bhav_store_rebuild import _check_desk
    from research import feature_store as FS

    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    db = Path(features_db).resolve()
    _check_copy_mode(db, target_is_copy)
    if m.get("state") not in {"applying", "applied"}:
        raise SystemExit(f"nothing to roll back: the manifest is in state {m.get('state')!r}")
    if not target_is_copy and not str(owner_approval or "").strip():
        raise SystemExit("refusing: a live rollback needs the owner's approval in words (--owner-approval)")
    _check_desk(db, target_is_copy)
    saved = FS._DB_PATH
    FS._DB_PATH = db
    try:
        released = FS.undo_quarantine_batch(m["batch_id"])
    finally:
        FS._DB_PATH = saved
    m.update({"state": "rolled_back", "rolled_back_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "released": released, "rollback_approval": str(owner_approval or "").strip()})
    _write_json(manifest_path, m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--features-db", required=True)
    p.add_argument("--current-pickle", required=True)
    p.add_argument("--repaired-pickle", required=True)
    p.add_argument("--ca-events", required=True, help="a copy of logs/ca_events.json (absent file: no CA)")
    p.add_argument("--store-manifest", required=True, help="the store repair's manifest (bhav_store_rebuild plan)")
    p.add_argument("--shadow-db", default=None, help="a copy of logs/feature002/shadow.db (report only)")
    p.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--manifest", required=True)
    a.add_argument("--features-db", required=True)
    a.add_argument("--owner-approval", required=True)
    a.add_argument("--target-is-copy", action="store_true")
    a.add_argument("--include-exposed", action="store_true",
                   help="also quarantine the exposed (unproven) rows")
    r = sub.add_parser("rollback")
    r.add_argument("--manifest", required=True)
    r.add_argument("--features-db", required=True)
    r.add_argument("--target-is-copy", action="store_true")
    r.add_argument("--owner-approval", default="", help="required for a live rollback")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        out = plan(args.features_db, args.current_pickle, args.repaired_pickle, args.out, args.ca_events,
                   args.shadow_db, args.store_manifest)
        print(json.dumps({k: out[k] for k in ("symbols_with_changed_history", "observations_examined",
                                              "unexplained", "not_rederived", "refusals", "complete")}
                         | {"affected": len(out["affected"]), "exposed": len(out["exposed"]),
                            "store_binding_verified": out["store_binding"]["verified"],
                            "batch_id": out["batch_id"]},
                         indent=1, default=str))
        for row in out["affected"]:
            print(row["observation_id"], row["symbol"], row["ts"], json.dumps(row["features"]))
        if "shadow_feature002" in out:
            sh = out["shadow_feature002"]
            print("feature002:", json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in sh.items()}))
    elif args.cmd == "apply":
        out = apply(args.manifest, args.features_db, args.owner_approval, target_is_copy=args.target_is_copy,
                    include_exposed=args.include_exposed)
        print(json.dumps({"batch_id": out["batch_id"], "results": out["results"]}, indent=1))
    else:
        out = rollback(args.manifest, args.features_db, target_is_copy=args.target_is_copy,
                       owner_approval=args.owner_approval)
        print(json.dumps({"batch_id": out["batch_id"], "released": out["released"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
