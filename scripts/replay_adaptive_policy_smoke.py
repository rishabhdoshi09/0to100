"""Bounded discovery-only smoke test using a read-only Gauntlet source binding."""

import argparse
import json
import os
import sys
import sqlite3
from contextlib import closing, redirect_stdout
from datetime import date
from hashlib import sha256
from math import prod
from pathlib import Path

from core.live_volume import on_live_volume
from core.runtime_paths import REPO_ROOT, logs_dir, runtime_root
from data.corporate_actions import load_events
from gauntlet.validator import scoped_equity_frames
from product.decision_freeze import freeze_path
from product.historical_paper_loop import compare_historical_policies
from product.historical_replay import decide_session, scan_session
from research.auto_research.costs import india_cash_costs
from research.intelligence.data.nse_calendar import documented_special_sessions

COHORT = ("BHARTIARTL", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "LT", "RELIANCE", "SBIN", "TCS", "TITAN")


BENCHMARK = ("ICICIBANK", "INFY", "TCS")


def benchmarks(bars, result):
    """Idle cash, passive buy-and-hold and lagged equity-exposure comparators."""
    sessions = list(bars)
    missing = [day for day, prices in bars.items() if any(s not in prices for s in BENCHMARK)]
    if missing:
        return {"cash_pct": 0.0, "basket": BENCHMARK, "unavailable_sessions": missing}
    slip = 0.0003
    basis = {s: bars[sessions[0]][s][0] * (1 + slip) for s in BENCHMARK}
    previous = 1.0
    passive = []
    for day in sessions:
        marks = []
        for symbol in BENCHMARK:
            end = bars[day][symbol][3]
            fees = 0
            if day == sessions[-1]:
                end *= 1 - slip
                fees = india_cash_costs(basis[symbol], end, 1)
            marks.append((end - fees) / basis[symbol])
        value = sum(marks) / len(marks)
        passive.append(value / previous - 1)
        previous = value
    matched = {}
    for name, branch in result["branches"].items():
        weights = [0.0] + [min(1.0, row["deployed"] / row["equity"]) if row["equity"] > 0 else 0.0
                           for row in branch["daily"][:-1]]
        matched[name] = 100 * (prod(1 + w * r for w, r in zip(weights, passive)) - 1)
    return {"cash_pct": 0.0, "basket": BENCHMARK, "passive_pct": 100 * (previous - 1),
            "prior_close_exposure_matched_pct": matched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_logs", type=Path)
    parser.add_argument("geometry_checkpoint", type=Path)
    parser.add_argument("--development", action="store_true", help="Use the fixed weekly discovery window")
    args = parser.parse_args()
    if not os.environ.get("QT_RUNTIME_ROOT", "").strip():
        parser.error("Set QT_RUNTIME_ROOT to a dedicated scratch directory")
    root = runtime_root().resolve(strict=True)
    archive = args.archive_logs.resolve(strict=True)
    checkpoint = args.geometry_checkpoint.resolve(strict=True)
    if (root == REPO_ROOT or any(on_live_volume(path) for path in (root, archive, checkpoint))
            or logs_dir().is_symlink() or os.statvfs(root).f_flag & os.ST_RDONLY):
        parser.error("Use an isolated runtime and copied archive and checkpoint")

    if (not freeze_path().resolve().is_relative_to(root)
            or any(os.environ.get(key) and not Path(os.environ[key]).resolve().is_relative_to(root)
                   for key in ("LOG_DIR", "QT_LEARNING_POLICIES"))):
        parser.error("Replay paths must stay inside QT_RUNTIME_ROOT")

    with closing(sqlite3.connect(checkpoint.as_uri() + "?mode=ro", uri=True)) as db:
        binding = json.loads(db.execute("SELECT spec FROM geometry_config").fetchone()[0])[-1]
    hashes = binding["archive_sha256"]
    days = sorted(hashes)
    start = days.index("2024-10-21" if args.development else "2025-04-01")
    sessions = days[start:] if args.development else days[start:start + 32]
    formation = days[start - 105:start] + sessions
    if len(formation) != 105 + len(sessions) or sessions[-1] >= "2025-11-01":
        raise ValueError("Incomplete replay window or overlap with reserve")
    decision_days = sessions[:-20]
    if args.development:
        special = documented_special_sessions(binding["calendar_manifest"])
        weekly = {}
        for day in decision_days:
            if day not in special:
                weekly.setdefault(date.fromisoformat(day).isocalendar()[:2], day)
        decision_days = list(weekly.values())
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
    for day in decision_days:
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

    bars = {day: {} for day in sessions}
    for symbol, frame in frames.items():
        for timestamp, row in frame.loc[sessions[0]:sessions[-1]].iterrows():
            bars[timestamp.date().isoformat()][symbol] = tuple(float(row[k]) for k in ("open", "high", "low", "close"))
    frozen = [row for row in decisions if row.get("freeze_id") and (row.get("pit") or {}).get("max_bar_date")]
    result = compare_historical_policies(frozen, official_sessions=sessions, bars=bars, directory=root / "comparison")
    return {"raw_decisions": len(decisions), "frozen_decisions": len(frozen), "scans": scans, "benchmarks": benchmarks(bars, result), "result": result}


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        result = main()
    print(json.dumps(result))
