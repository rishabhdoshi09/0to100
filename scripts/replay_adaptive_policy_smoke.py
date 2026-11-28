"""Bounded discovery-only smoke test using a read-only Gauntlet source binding."""

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import date
from hashlib import sha256
from pathlib import Path

from core.live_volume import on_live_volume
from core.runtime_paths import REPO_ROOT, runtime_root
from data.corporate_actions import load_events
from gauntlet.validator import scoped_equity_frames
from product.historical_paper_loop import compare_historical_policies
from product.historical_replay import decide_session, scan_session

COHORT = ("BHARTIARTL", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "LT", "RELIANCE", "SBIN", "TCS", "TITAN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_logs", type=Path)
    parser.add_argument("geometry_checkpoint", type=Path)
    args = parser.parse_args()
    root = runtime_root().resolve(strict=True)
    archive = args.archive_logs.resolve(strict=True)
    checkpoint = args.geometry_checkpoint.resolve(strict=True)
    if root == REPO_ROOT or any(on_live_volume(path) for path in (root, archive, checkpoint)):
        parser.error("Use an isolated runtime and copied archive and checkpoint")

    with closing(sqlite3.connect(checkpoint.as_uri() + "?mode=ro", uri=True)) as db:
        binding = json.loads(db.execute("SELECT spec FROM geometry_config").fetchone()[0])[-1]
    hashes = binding["archive_sha256"]
    days = sorted(hashes)
    start = days.index("2025-04-01")
    formation = days[start - 105:start + 32]
    if len(formation) != 137 or formation[-1] >= "2025-11-01":
        raise ValueError("Incomplete smoke window or overlap with reserve")
    ordinal = {day: index for index, day in enumerate(formation)}

    def packets():
        for session in formation:
            day = date.fromisoformat(session)
            payload = (archive / "bhav" / day.strftime("%d%m%Y.csv")).read_bytes()
            if sha256(payload).hexdigest() != hashes[session]:
                raise ValueError(f"Archive changed on {session}")
            yield day, payload

    actions = archive / "ca_events.json"
    if sha256(actions.read_bytes()).hexdigest() != binding["actions_sha256"]:
        raise ValueError("Corporate-action archive changed")
    frames, report = scoped_equity_frames(packets(), COHORT, load_events(actions))
    if report["market_wide_excluded_sessions"]:
        raise ValueError(report["market_wide_excluded_sessions"])

    decisions = []
    scans = []
    for day in days[start:start + 12]:
        def prefix(symbol):
            frame = frames[symbol].loc[:day]
            if frame.empty or frame.index[-1].date().isoformat() != day:
                return None
            offsets = [ordinal[index.date().isoformat()] for index in frame.index]
            begin = max((i for i in range(1, len(offsets)) if offsets[i] != offsets[i - 1] + 1), default=0)
            return frame.iloc[begin:]

        scan = scan_session(day, COHORT, ohlcv_fn=prefix)
        if scan["errors"]:
            raise ValueError(scan["errors"])
        decisions.extend(decide_session(day, scan, use_committee=False, bind_selection_card=True))
        scans.append({"day": day, "scanned": scan["scanned"]})

    sessions = days[start:start + 32]
    bars = {day: {} for day in sessions}
    for symbol, frame in frames.items():
        for timestamp, row in frame.loc[sessions[0]:sessions[-1]].iterrows():
            bars[timestamp.date().isoformat()][symbol] = tuple(float(row[k]) for k in ("open", "high", "low", "close"))
    frozen = [row for row in decisions if row.get("freeze_id") and (row.get("pit") or {}).get("max_bar_date")]
    result = compare_historical_policies(frozen, official_sessions=sessions, bars=bars, directory=root / "comparison")
    print(json.dumps({"raw_decisions": len(decisions), "frozen_decisions": len(frozen), "scans": scans, "result": result}))


if __name__ == "__main__":
    main()
