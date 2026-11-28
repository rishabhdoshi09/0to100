"""Extend a scoped Gauntlet checkpoint with provenance-bound feature observations."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from core.live_volume import on_live_volume
from data.corporate_actions import parse_events
from gauntlet.validator import scoped_equity_frames
from research.autonomy import outcome_discovery as base

HORIZONS = (3, 20)
MAX_ARCHIVE_BYTES = 384 * 1024 * 1024


def observations(frame, days, anchors):
    import numpy as np

    ordinals = {day: i for i, day in enumerate(days)}
    by_day = {ordinals[t.date().isoformat()]: tuple(row)
              for t, row in frame[["open", "close", "volume", "deliv_per", "share_factor"]].iterrows()}
    for anchor in anchors:
        i = ordinals[anchor]
        history = [by_day.get(j) for j in range(i-20, i+1)]
        if any(bar is None for bar in history):
            continue
        start = history[0]
        previous, overnight, intraday, delivered, turnover = start, [], [], [], []
        for bar in history[1:]:
            op, close, volume, delivery, factor = bar
            overnight.append(math.log(op * factor / previous[1]))
            intraday.append(math.log(close / op))
            delivered.append(delivery)
            turnover.append(close * volume)
            previous = bar
        if np.mean(turnover) < base.MIN_TURNOVER:
            continue
        features = {
            "momentum": sum(overnight) + sum(intraday),
            "overnight_share": sum(overnight) - sum(intraday),
            "delivery_change": float(np.mean(delivered[-5:]) - np.mean(delivered[:-5]))
                               if all(math.isfinite(x) for x in delivered) else None,
        }
        outcomes = {}
        for horizon in HORIZONS:
            # Preserve the exclusive reserved boundary and every intervening session.
            if i + horizon >= len(days) - 1:
                continue
            forward = [by_day.get(j) for j in range(i+1, i+horizon+1)]
            if any(bar is None for bar in forward):
                continue
            shares = math.prod(bar[4] for bar in forward[1:])
            outcomes[str(horizon)] = forward[-1][1] * shares / forward[0][0] - 1
        yield anchor, json.dumps(features, allow_nan=False), json.dumps(outcomes, allow_nan=False)


def build(checkpoint: Path, archive: Path):
    """Read verified archived inputs once and commit one append-only Gauntlet slice per symbol."""
    from core.runtime_paths import runtime_root
    from product.learning_observations import writer_lock

    source = base._isolated_checkpoint(checkpoint)
    root = runtime_root().resolve()
    if on_live_volume(archive):
        raise ValueError("Live archive refused")
    archive = archive.resolve(strict=True)
    if not archive.is_relative_to(Path("/tmp").resolve()) or os.statvfs(archive).f_flag & os.ST_RDONLY:
        raise ValueError("Use the isolated writable archive copy")
    target = root / "gauntlet-frontier.sqlite"
    sidecars = [target.with_name(target.name + suffix) for suffix in ("-wal", "-shm", "-journal")]
    lock = root / "gauntlet-frontier.lock"
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in [target, lock, *sidecars]):
        raise ValueError("Gauntlet output escapes scratch runtime")
    stat = base._identity(source)
    if target == source or (target.exists() and base._identity(target)[:2] == stat[:2]):
        raise ValueError("Frontier output aliases the original checkpoint")
    if stat[-1] > base.WINDOW["budget"]["input_bytes"]:
        raise ValueError("Unbounded source checkpoint")
    digest = base._digest(source, stat)
    with closing(sqlite3.connect(source.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
        _, end, _, _, spec = json.loads(db.execute("SELECT spec FROM geometry_config WHERE id=1").fetchone()[0])
    days = spec["sessions"]
    hashes = spec["archive_sha256"]
    if (end != base.END or days[-1] != base.END or days != sorted(set(days))
            or set(hashes) != set(days[:-1]) or not spec.get("exploratory") or len(days) > 500
            or len(spec["symbols"]) > 400 or base._digest(source, stat) != digest):
        raise ValueError("Invalid scoped Gauntlet archive binding")
    code = hashlib.sha256(Path(__file__).read_bytes()
                          + Path(__file__).with_name("validator.py").read_bytes()).hexdigest()
    binding = json.dumps([digest, code, HORIZONS], sort_keys=True)
    anchors = days[20:-1]
    with writer_lock(lock.with_suffix("")), closing(sqlite3.connect(target)) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS frontier_binding (id INTEGER PRIMARY KEY, spec TEXT, report TEXT);
            CREATE TABLE IF NOT EXISTS frontier_done (symbol TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS frontier_observations (
                symbol TEXT, day TEXT, feature TEXT, outcome TEXT, PRIMARY KEY(symbol,day)
            );
        """)
        with db:
            db.execute("INSERT OR IGNORE INTO frontier_binding VALUES (1,?,NULL)", (binding,))
        previous, stored_report = db.execute("SELECT spec,report FROM frontier_binding WHERE id=1").fetchone()
        if previous != binding:
            raise ValueError("Gauntlet frontier inputs changed")
        done = {r[0] for r in db.execute("SELECT symbol FROM frontier_done")}
        symbols = set(spec["symbols"])
        recorded = {r[0] for r in db.execute("SELECT DISTINCT symbol FROM frontier_observations")}
        if not recorded <= done <= symbols:
            raise ValueError("Frontier checkpoint symbol membership changed")
        pending = symbols - done
        if not pending and stored_report:
            return target, days, json.loads(stored_report)
        ca = archive / "ca_events.json"
        if not ca.resolve().is_relative_to(archive):
            raise ValueError("Action ledger escapes archive")
        event_bytes = ca.read_bytes()
        if hashlib.sha256(event_bytes).hexdigest() != spec["actions_sha256"]:
            raise ValueError("Corporate-action source mismatch")
        official = [e for e in json.loads(event_bytes) if e.get("source_tier") == 1
                    and str(e.get("source_url", "")).startswith(("https://www.nseindia.com/", "https://nsearchives.nseindia.com/"))]
        actions = parse_events(official)
        # Keep all known action dates quarantined, even when the action is unsuitable to bridge.
        for symbol, events in parse_events(json.loads(event_bytes)).items():
            verified = actions.setdefault(symbol, [])
            verified.extend({**e, "type": "unverified"} for e in events if e not in verified)

        def packets():
            used = 0
            for day in days[:-1]:
                stamp = date.fromisoformat(day)
                path = archive / "bhav" / stamp.strftime("%d%m%Y.csv")
                if not path.resolve().is_relative_to(archive):
                    raise ValueError("Bhavcopy escapes archive")
                payload = None
                if hashes[day]:
                    size = path.stat().st_size
                    used += size
                    if size > 3 * 1024 * 1024 or used > MAX_ARCHIVE_BYTES:
                        raise ValueError("Archive diagnostic budget exceeded")
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != hashes[day]:
                        raise ValueError(f"Archive mismatch on {day}")
                yield stamp, payload

        # One bounded pass per resume; completed symbols never regenerate observations.
        frames, report = scoped_equity_frames(packets(), spec["symbols"], actions, bridge_share_actions=True)
        for symbol in sorted(pending):
            frame = frames.get(symbol)
            with db:
                if frame is not None:
                    db.executemany("INSERT INTO frontier_observations VALUES (?,?,?,?)",
                                   ((symbol, *row) for row in observations(frame, days, anchors)))
                db.execute("INSERT INTO frontier_done VALUES (?)", (symbol,))
        with db:
            db.execute("UPDATE frontier_binding SET report=? WHERE id=1", (json.dumps(report),))
        return target, days, report
