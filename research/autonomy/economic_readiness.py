"""Audit publication-time eligibility without spending a predictive research look."""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import date, datetime, time
from itertools import groupby
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

from core.atomic_io import atomic_write_text
from core.runtime_paths import logs_path, runtime_root
from research.autonomy import outcome_discovery as scope
from research.intelligence.data.nse_calendar import documented_special_sessions

IST = ZoneInfo("Asia/Kolkata")
HORIZON = 20
MIN_INDEPENDENT_WINDOWS = 20
MAX_RECORDS = 200
OUTPUT_NAME = "economic-research-readiness.json"


def _instant(value):
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("Exchange dissemination timestamp must include time and offset")
    instant = datetime.fromisoformat(value)
    if instant.utcoffset() is None:
        raise ValueError("Exchange dissemination timestamp lacks a UTC offset")
    return instant.astimezone(IST)


def first_daily_entry(row, sessions, special):
    source = urlsplit(row["source_url"])
    if (source.scheme != "https" or source.netloc != "www.nseindia.com"
            or source.path != "/companies-listing/corporate-filings-financial-results"
            or parse_qs(source.query).get("symbol") != [row["symbol"]]):
        raise ValueError("Unsupported official filing metadata source")
    received, public = _instant(row["received_at"]), _instant(row["disseminated_at"])
    day = public.date().isoformat()
    period = date.fromisoformat(row["period_end"]).isoformat()
    if received > public or period >= day or day >= scope.END:
        raise ValueError("Invalid or reserved publication chronology")
    # Daily-bar convention uses the full following session. A same-day open may
    # technically be actionable for pre-open news but is outside this design.
    i = bisect.bisect_right(sessions, day)
    if i == 0 or i >= len(sessions) or sessions[i] >= scope.END:
        return {"earliest_actionable_daily_session": None, "disseminated_at": public.isoformat(),
                "exclusion": "OUTSIDE_VERIFIED_SESSION_WINDOW"}
    if sessions[i] in special:
        return {"earliest_actionable_daily_session": None, "disseminated_at": public.isoformat(),
                "exclusion": "NEXT_SESSION_SPECIAL_HOURS_UNVERIFIED"}
    anchor = sessions[i-1]
    return {
        "earliest_actionable_daily_session": sessions[i],
        "formation_session": anchor,
        "reprice_gap_observable": anchor == day and public.time() > time(15, 30) and day not in special,
        "disseminated_at": public.isoformat(),
    }


def _audit_price_windows(archive, binding, dates, events):
    from core.live_volume import on_live_volume
    from data.corporate_actions import load_events_with_digest
    from gauntlet.validator import scoped_equity_frames

    if on_live_volume(archive):
        raise ValueError("Live archive refused")
    archive = archive.resolve(strict=True)
    if not archive.is_relative_to(Path("/tmp").resolve()) or os.statvfs(archive).f_flag & os.ST_RDONLY:
        raise ValueError("Use an isolated archive copy")
    action_file = archive / "ca_events.json"
    if not action_file.resolve().is_relative_to(archive):
        raise ValueError("Action ledger escapes archive")
    actions, digest = load_events_with_digest(action_file)
    if digest != binding["actions_sha256"]:
        raise ValueError("Corporate-action source mismatch")
    ordinal = {day: i for i, day in enumerate(dates)}
    windows = {}
    for key, event in events.items():
        entry = event["earliest_actionable_daily_session"]
        if entry:
            i = ordinal[entry]
            windows[key] = dates[i - 1:i + HORIZON]
    needed = sorted({day for window in windows.values() for day in window})
    if len(needed) > 160:
        raise ValueError("Price availability exceeds 160-session audit budget")

    def packets():
        for day in needed:
            stamp = date.fromisoformat(day)
            path = archive / "bhav" / stamp.strftime("%d%m%Y.csv")
            if not path.resolve().is_relative_to(archive):
                raise ValueError("Bhavcopy escapes archive")
            expected = binding["archive_sha256"][day]
            payload = None
            if expected:
                if path.stat().st_size > 3 * 1024 * 1024:
                    raise ValueError("Unbounded archive payload")
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError("Price audit archive hash mismatch")
            yield stamp, payload

    symbols = sorted({symbol for symbol, _ in events})
    admitted = {symbol: set() for symbol in symbols}
    # The strict validator carries a previous close within a contiguous packet
    # set. Reset it across uninspected periods instead of comparing distant bars.
    for _, group in groupby(enumerate(packets()), key=lambda item: ordinal[item[1][0].isoformat()] - item[0]):
        frames, _ = scoped_equity_frames((packet for _, packet in group), symbols, actions)
        for symbol, frame in frames.items():
            admitted[symbol].update(stamp.date().isoformat() for stamp in frame.index)
    for key, event in events.items():
        window = windows.get(key, ())
        event["strict_archive_price_window"] = len(window) == HORIZON + 1 and set(window) <= admitted.get(key[0], set())
        event["overnight_gap_prices_available"] = (
            event.get("reprice_gap_observable", False) and len(window) >= 2
            and set(window[:2]) <= admitted.get(key[0], set())
        )


def audit(checkpoint: Path, frontier: Path, receipt: Path, *, archive: Path | None = None):
    base = scope._isolated_checkpoint(checkpoint)
    derived = scope._isolated_checkpoint(frontier)
    record = scope._isolated_checkpoint(receipt)
    target = logs_path(OUTPUT_NAME)
    if (target.is_symlink() or target.resolve() in (base, derived, record)
            or not target.resolve().is_relative_to(runtime_root().resolve())):
        raise ValueError("Audit output escapes isolated runtime")
    if record.stat().st_size > 128_000 or base.stat().st_size > 64 * 1024**2 or derived.stat().st_size > 64 * 1024**2:
        raise ValueError("Economic audit exceeds its bounded input size")
    paths = (base, derived, record)
    identities = [scope._identity(path) for path in paths]
    digests = [scope._digest(path, identity) for path, identity in zip(paths, identities)]
    with closing(sqlite3.connect(base.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        _, end, _, _, binding = json.loads(db.execute("SELECT spec FROM geometry_config WHERE id=1").fetchone()[0])
    sessions = binding["sessions"]
    if end != scope.END or sessions[-1] != end or sessions != sorted(set(sessions)) or not binding["exploratory"]:
        raise ValueError("Unaudited or reserved Gauntlet scope")
    dates = sessions[:-1]
    ordinals = {day: index for index, day in enumerate(dates)}
    special = documented_special_sessions(binding["calendar_manifest"])
    rows = json.loads(record.read_bytes())
    if not isinstance(rows, list) or len(rows) > MAX_RECORDS:
        raise ValueError("Invalid filing metadata receipt")
    events = {}
    for row in rows:
        event = first_daily_entry(row, dates, special)
        key = (row["symbol"], date.fromisoformat(row["period_end"]).isoformat())
        prior = events.get(key)
        if prior is None or event["disseminated_at"] < prior["disseminated_at"]:
            events[key] = {**row, **event, "period_end": key[1]}
    # Check outcome availability only. No return values are read into the study.
    with closing(sqlite3.connect(derived.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        spec, report = db.execute("SELECT spec,report FROM frontier_binding WHERE id=1").fetchone()
        if json.loads(spec)[0] != digests[0] or not report:
            raise ValueError("Frontier is incomplete or bound to another archive")
        for event in events.values():
            anchor = event.get("formation_session")
            found = anchor is not None and ordinals[anchor] + HORIZON < len(dates) and db.execute(
                """SELECT 1 FROM frontier_observations WHERE symbol=? AND day=?
                   AND json_type(outcome,'$."20"') IN ('integer','real')""",
                (event["symbol"], anchor),
            ).fetchone()
            event["liquidity_qualified_frontier_window"] = bool(found)
    if archive is not None:
        _audit_price_windows(archive, binding, dates, events)
    blocks, last = 0, -HORIZON
    usable = {ordinals[e["earliest_actionable_daily_session"]] for e in events.values() if e.get("strict_archive_price_window", e["liquidity_qualified_frontier_window"])}
    for i in sorted(usable):
        if i >= last + HORIZON:
            blocks += 1
            last = i
    if [scope._digest(path, identity) for path, identity in zip(paths, identities)] != digests:
        raise ValueError("Economic audit input changed")
    result = {
        "status": "BLOCKED", "stage": "DATA_ACQUISITION", "evidence_class": "HISTORICAL_REPLAY",
        "predictive_looks_used": 0, "receipt_rows": len(rows), "filing_events": len(events),
        "events_with_daily_entry": sum(e["earliest_actionable_daily_session"] is not None for e in events.values()),
        "timing_eligible_overnight_gaps": sum(e.get("reprice_gap_observable", False) for e in events.values()),
        "strict_archive_price_windows": (sum(e.get("strict_archive_price_window", False) for e in events.values())
                                         if archive is not None else None),
        "verified_overnight_gap_price_pairs": (sum(e.get("overnight_gap_prices_available", False) for e in events.values())
                                               if archive is not None else None),
        "liquidity_qualified_frontier_windows": sum(e["liquidity_qualified_frontier_window"] for e in events.values()),
        "nonoverlapping_price_windows": blocks, "minimum_independent_windows": MIN_INDEPENDENT_WINDOWS,
        "base_sha256": digests[0], "frontier_sha256": digests[1], "receipt_sha256": digests[2],
        "next": "ACQUIRE_ORIGINAL_QUARTERLY_DISCLOSURES_AND_DATED_CONTROLS",
        "dependencies": {
            "original_quarterly_facts": "Verify versions of original XBRL/PDF artifacts independently of release timestamps.",
            "matched_controls": "Verify event coverage for control names, with dated sector, liquidity and risk eligibility.",
            "validation": "Preregister separately before accessing an untouched evaluation period.",
        },
        "events": sorted(events.values(), key=lambda row: (row["disseminated_at"], row["symbol"])),
        "promotion_eligible": False,
    }
    atomic_write_text(target, json.dumps(result, sort_keys=True, indent=2) + "\n", durable=True)
    return result


def configured_job(checkpoint):
    """One metadata audit job per immutable scratch input generation."""
    from research.autonomy.schedules import ECONOMIC_READINESS_PREFIX

    receipt = os.environ.get("QT_ECONOMIC_RECEIPT")
    frontier = os.environ.get("QT_ECONOMIC_FRONTIER")
    if not receipt or not frontier:
        return None
    paths = [scope._isolated_checkpoint(Path(item)) for item in (checkpoint, frontier, receipt)]
    archive = os.environ.get("QT_FRONTIER_ARCHIVE")
    identity = json.dumps([[(str(p), scope._identity(p)) for p in paths], archive])
    key = ECONOMIC_READINESS_PREFIX + hashlib.sha256(identity.encode()).hexdigest()[:20]
    return key, paths, Path(archive) if archive else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "frontier", "receipt"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.checkpoint, args.frontier, args.receipt, archive=args.archive), allow_nan=False))
