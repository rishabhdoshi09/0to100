"""Find and quarantine learning observations built from a store that had lost official history.

The store integrity audit (docs/research/store_integrity_2026-11-22.md) found
50 young listings whose first weeks of official history were missing from the
OHLCV store. A DECISION observation frozen for one of them carries point-in-time
technical features (product.pit_features) computed from that truncated history.
Most are unaffected, because their windows are short, but a 252-bar feature
such as ``dist_from_high_pct`` is wrong when the true high sat in the lost
sessions (VEDPOWER on 2026-11-20: stored 8.68%, true 29.79%).

The frozen vector is the decision-time record and is never rewritten. A proven
defect is QUARANTINED (research.feature_store.quarantine_observation), so the
learning loaders skip it.

* ``plan`` (read-only; COPIES only, ``/Volumes`` refused) recomputes every PIT
  fill of every affected-symbol DECISION observation twice: on the current
  store (it must reproduce the stored value, or the row is not explained by the
  store defect and is left alone) and on the repaired store
  (product.bhav_store_rebuild's planned pickle), each read the way
  ``get_ohlcv`` serves it (corporate actions applied from ``--ca-events``). A
  row is affected when a repaired value differs from the stored one by more
  than ``ABS_TOL`` AND more than ``REL_TOL`` of it. The manifest records each
  row's vector sha256. Fills this tool cannot re-derive per symbol (market
  context, cross-sectional RS) are counted under ``not_rederived``; a plan with
  any is flagged incomplete rather than silently passed.

  Features the decision carried itself (the scanner's RSI, quality score, the
  setup and its levels) were computed from the same history but cannot be
  recomputed here. So the plan also lists, as ``exposed``, every observation of
  any kind whose input window (the last ``LOOKBACK`` bars up to its cutoff, the
  longest window any feature reads) differs between the two stores. Exposed
  rows are evidence of risk, not proof of a wrong value.

  With ``--shadow-db`` (a copy of ``logs/feature002/shadow.db``) the plan also
  re-derives that research ledger's trend features (research.feature001) for
  the same symbols and REPORTS the rows that differ. Its ranks are
  cross-sectional, so a row cannot be corrected in place; nothing in that ledger
  is ever written by this tool.
* ``apply`` quarantines exactly the planned rows. It needs the owner's
  approval in words, a stopped desk for a live target, and an unchanged vector
  sha256 for every row. ``--include-exposed`` also quarantines the exposed
  rows. Nothing else in the store is written.

    python -m product.learning_observation_correction plan --features-db <copy> \\
        --current-pickle <copy> --repaired-pickle <planned pickle> --ca-events <copy> \\
        --out <manifest.json>
    python -m product.learning_observation_correction apply --manifest <manifest.json> \\
        --features-db <db> --owner-approval "<the owner's words>"
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
#: The longest window any frozen feature reads: 252 bars (dist_from_high_pct,
#: the scanner's 52-week checks) plus the bar before it.
LOOKBACK = 253


def _not_live(path: str | Path, what: str) -> Path:
    p = Path(path).resolve()
    if str(p).startswith("/Volumes/"):
        raise SystemExit(f"refusing {what} under /Volumes: {p} (work on a copy)")
    return p


def _vector_sha(features_json: str) -> str:
    canonical = json.dumps(json.loads(features_json or "{}"), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frames(path: Path) -> dict:
    with open(path, "rb") as fh:
        return dict((pickle.load(fh) or {}).get("store") or {})


def _sorted(frame):
    return frame if frame is None or frame.index.is_monotonic_increasing else frame.sort_index(kind="stable")


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


def _window(frame, ts):
    """The last LOOKBACK bars a decision at ``ts`` could read (None when no cutoff)."""
    import pandas as pd
    from product.pit_features import cutoff_session

    cut = cutoff_session(ts)
    if cut is None or frame is None:
        return None
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    keep = idx.normalize() <= pd.Timestamp(cut)
    return _sorted(frame[keep]).iloc[-LOOKBACK:]


def _same_window(a, b) -> bool:
    """Same dates in the same order and the same VALUES (dtype-blind: a full build
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

    def upto(frame, cut):
        if frame is None:
            return None
        idx = pd.DatetimeIndex(frame.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        return _sorted(frame[idx.normalize() <= cut])

    for eid, cset, sym, sess, snap, elig in rows:
        if sym not in changed:
            continue
        snap = json.loads(snap or "{}")
        trend = snap.get("trend") or {}
        cut = pd.Timestamp(str(snap.get("hist_as_of") or sess)[:10])
        ev = events.get(str(sym or "").upper())
        a = upto(_served(current.get(sym), ev), cut)
        b = upto(_served(repaired.get(sym), ev), cut)
        if _same_window(None if a is None else a.iloc[-LOOKBACK:], None if b is None else b.iloc[-LOOKBACK:]):
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


def plan(features_db: str | Path, current_pickle: str | Path, repaired_pickle: str | Path,
         out: str | Path, ca_events: str | Path | None = None,
         shadow_db: str | Path | None = None) -> dict[str, Any]:
    from data.corporate_actions import load_events_with_digest
    from product.pit_features import PRODUCED, technical_features

    db = _not_live(features_db, "the feature store")
    if not db.is_file():
        raise SystemExit(f"refusing: no feature store at {db}")
    cur_p, rep_p = _not_live(current_pickle, "the current pickle"), _not_live(repaired_pickle, "the repaired pickle")
    out = _not_live(out, "the manifest")
    ca_p = None if ca_events is None else _not_live(ca_events, "the corporate-action table")
    events, ca_digest = load_events_with_digest(ca_p) if ca_p is not None else ({}, "")
    current, repaired = _frames(cur_p), _frames(rep_p)
    changed = _changed_symbols(current, repaired)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT observation_id, symbol, ts, features, meta, outcome, kind "
                           "FROM observations").fetchall()
    finally:
        con.close()
    affected, unexplained, examined, not_rederived, exposed = [], [], 0, {}, []
    for oid, sym, ts, feats, meta, outcome, kind in rows:
        if sym not in changed:
            continue
        ev = events.get(str(sym or "").upper())
        cur_f, rep_f = _served(current.get(sym), ev), _served(repaired.get(sym), ev)
        if not _same_window(_window(cur_f, ts), _window(rep_f, ts)):
            exposed.append({"observation_id": oid, "symbol": sym, "ts": ts, "kind": kind,
                            "settled": outcome is not None, "vector_sha256": _vector_sha(feats)})
        if kind != "DECISION":
            continue
        all_filled = (json.loads(meta or "{}").get("pit_features") or {}).get("filled") or []
        for k in all_filled:
            if k not in PRODUCED:
                not_rederived[k] = not_rederived.get(k, 0) + 1
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
            if s is None or a is None or b is None:
                continue
            if abs(float(a) - float(s)) > 1e-6:
                unexplained.append({"observation_id": oid, "feature": k, "stored": s, "current_recompute": a})
                continue
            if _material(s, b):
                diffs[k] = {"stored": s, "repaired": round(float(b), 6)}
        if diffs:
            affected.append({"observation_id": oid, "symbol": sym, "ts": ts, "settled": outcome is not None,
                             "features": diffs, "vector_sha256": _vector_sha(feats)})
    manifest = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": REASON,
        "features_db": str(db),
        "current_pickle": str(cur_p),
        "repaired_pickle": str(rep_p),
        "ca_events": None if ca_p is None else str(ca_p),
        "ca_events_sha256": ca_digest,
        "features_db_sha256": _file_sha(db),
        "tolerance": {"abs": ABS_TOL, "rel": REL_TOL},
        "symbols_with_changed_history": len(changed),
        "observations_examined": examined,
        "affected": affected,
        "exposed": [r for r in exposed if r["observation_id"] not in {a["observation_id"] for a in affected}],
        "lookback_bars": LOOKBACK,
        "unexplained": unexplained,
        "not_rederived": not_rederived,
        "complete": not unexplained and not not_rederived,
        "state": "planned",
    }
    if shadow_db is not None:
        sdb = _not_live(shadow_db, "the feature002 shadow ledger")
        manifest["shadow_feature002"] = dict(_shadow_report(sdb, current, repaired, events, changed),
                                             path=str(sdb), sha256=_file_sha(sdb), written=False)
    Path(out).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def apply(manifest_path: str | Path, features_db: str | Path, owner_approval: str, *,
          target_is_copy: bool = False, include_exposed: bool = False) -> dict[str, Any]:
    from product.bhav_store_rebuild import _check_desk
    from research import feature_store as FS

    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if not str(owner_approval or "").strip():
        raise SystemExit("refusing: apply needs the owner's approval in words (--owner-approval)")
    if m.get("state") != "planned":
        raise SystemExit(f"refusing: the manifest is in state {m.get('state')!r}, not 'planned'")
    if not m.get("complete"):
        raise SystemExit("refusing: the plan is incomplete (unexplained or not re-derived fills); "
                         "investigate before quarantining anything")
    db = Path(features_db).resolve()
    _check_desk(db, target_is_copy)
    chosen = [(row, m["reason"]) for row in m["affected"]]
    if include_exposed:
        chosen += [(row, EXPOSED_REASON) for row in m.get("exposed", [])]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for row, _reason in chosen:
            got = con.execute("SELECT features FROM observations WHERE observation_id=?",
                              (row["observation_id"],)).fetchone()
            if got is None or _vector_sha(got[0]) != row["vector_sha256"]:
                raise SystemExit(f"refusing: {row['observation_id']} is missing or changed since the plan")
    finally:
        con.close()
    saved = FS._DB_PATH
    FS._DB_PATH = db
    results = {}
    try:
        for row, reason in chosen:
            evidence = {"plan": str(manifest_path), "repaired_pickle": m["repaired_pickle"],
                        "source": "docs/research/store_integrity_2026-11-22.md"}
            if "features" in row:
                evidence["features"] = row["features"]
            else:
                evidence["input_window_bars"] = m.get("lookback_bars")
            results[row["observation_id"]] = FS.quarantine_observation(row["observation_id"], reason, evidence)
    finally:
        FS._DB_PATH = saved
    m.update({"state": "applied", "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "owner_approval": owner_approval.strip(), "included_exposed": include_exposed,
              "results": results})
    manifest_path.write_text(json.dumps(m, indent=2, default=str))
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--features-db", required=True)
    p.add_argument("--current-pickle", required=True)
    p.add_argument("--repaired-pickle", required=True)
    p.add_argument("--ca-events", required=True, help="a copy of logs/ca_events.json (absent file: no CA)")
    p.add_argument("--shadow-db", default=None, help="a copy of logs/feature002/shadow.db (report only)")
    p.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--manifest", required=True)
    a.add_argument("--features-db", required=True)
    a.add_argument("--owner-approval", required=True)
    a.add_argument("--target-is-copy", action="store_true")
    a.add_argument("--include-exposed", action="store_true",
                   help="also quarantine the exposed (unproven) rows")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        out = plan(args.features_db, args.current_pickle, args.repaired_pickle, args.out, args.ca_events,
                   args.shadow_db)
        print(json.dumps({k: out[k] for k in ("symbols_with_changed_history", "observations_examined",
                                              "unexplained", "not_rederived", "complete")}
                         | {"affected": len(out["affected"]), "exposed": len(out["exposed"])},
                         indent=1, default=str))
        for row in out["affected"]:
            print(row["observation_id"], row["symbol"], row["ts"], json.dumps(row["features"]))
        if "shadow_feature002" in out:
            sh = out["shadow_feature002"]
            print("feature002:", json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in sh.items()}))
    else:
        out = apply(args.manifest, args.features_db, args.owner_approval, target_is_copy=args.target_is_copy,
                    include_exposed=args.include_exposed)
        print(json.dumps(out["results"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
