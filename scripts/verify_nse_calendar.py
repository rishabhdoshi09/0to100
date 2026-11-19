#!/usr/bin/env python3
"""Verify the bundled NSE holiday calendar against NSE's own published list.

Reads NSE's trading-holiday master (segment CM) from ``--input`` (a saved
JSON response of https://www.nseindia.com/api/holiday-master?type=trading) or
fetches it with ``--fetch``. For each requested year it compares the weekday
holidays with data/nse_holidays.json and, only when they match exactly,
records a VERIFIED_COMPLETE manifest entry (source, retrieval time, count,
SHA-256, weekend-dated and asterisked entries). ``--write-holidays`` first
replaces that year's dates in data/nse_holidays.json with NSE's list.

Nothing here is used at runtime; evidence code reads only the manifest. A
mismatch exits non-zero and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.intelligence.data.nse_calendar import (  # noqa: E402
    MANIFEST_PATH, VERIFIED_COMPLETE, load_manifest, weekday_holiday_digest)

SOURCE = "https://www.nseindia.com/api/holiday-master?type=trading (segment CM)"
HOLIDAYS_PATH = ROOT / "data" / "nse_holidays.json"


def parse_cm(payload: dict) -> list[dict]:
    rows = []
    for item in payload.get("CM") or []:
        day = datetime.strptime(str(item["tradingDate"]), "%d-%b-%Y").date()
        rows.append({"date": day, "description": str(item.get("description") or "")})
    return rows


def fetch() -> dict:
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    s.get("https://www.nseindia.com/", timeout=15)
    r = s.get("https://www.nseindia.com/api/holiday-master?type=trading", timeout=15,
              headers={"Referer": "https://www.nseindia.com/resources/exchange-communication-holidays"})
    r.raise_for_status()
    return r.json()


def verify(payload: dict, year: int, holidays: list[str], *, retrieved_at: str) -> tuple[bool, dict]:
    rows = [r for r in parse_cm(payload) if r["date"].year == year]
    official = sorted({r["date"].isoformat() for r in rows if r["date"].weekday() < 5})
    bundled = sorted({d for d in holidays if d.startswith(str(year))
                      and date.fromisoformat(d).weekday() < 5})
    if not rows or official != bundled:
        return False, {"year": year, "official_only": sorted(set(official) - set(bundled)),
                       "bundled_only": sorted(set(bundled) - set(official)), "official_rows": len(rows)}
    count, digest = weekday_holiday_digest(official, year)
    return True, {
        "status": VERIFIED_COMPLETE, "source": SOURCE, "retrieved_at": retrieved_at,
        "weekday_holidays": count, "weekday_holidays_sha256": digest,
        "weekend_dated_entries": sorted(r["date"].isoformat() for r in rows if r["date"].weekday() >= 5),
        "special_session_markers": sorted(f"{r['date'].isoformat()} {r['description']}"
                                          for r in rows if "*" in r["description"]),
        "note": ("Weekday CM trading holidays as published by NSE. Asterisked entries are exchange-notified "
                 "special sessions (e.g. Muhurat); they are not regular sessions and are not counted in "
                 "10-session outcome windows."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path)
    src.add_argument("--fetch", action="store_true")
    ap.add_argument("--year", type=int, action="append", required=True)
    ap.add_argument("--write-holidays", action="store_true")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--holidays", type=Path, default=HOLIDAYS_PATH)
    args = ap.parse_args(argv)
    payload = json.loads(args.input.read_text()) if args.input else fetch()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    holidays = sorted(set(json.loads(args.holidays.read_text()))) if args.holidays.exists() else []
    if args.write_holidays:
        years = set(args.year)
        keep = [d for d in holidays if int(d[:4]) not in years]
        add = [r["date"].isoformat() for r in parse_cm(payload) if r["date"].year in years and r["date"].weekday() < 5]
        holidays = sorted(set(keep) | set(add))
    manifest = load_manifest(args.manifest) or {"schema_version": 1, "segment": "CM", "years": {}}
    for year in args.year:
        ok, entry = verify(payload, year, holidays, retrieved_at=retrieved_at)
        if not ok:
            print(json.dumps({"verified": False, **entry}, indent=2))
            return 2
        manifest.setdefault("years", {})[str(year)] = entry
    if args.write_holidays:
        args.holidays.write_text(json.dumps(holidays, indent=2) + "\n")
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified": True, "years": args.year, "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
