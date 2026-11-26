"""Research-only execution geometry over the existing scanner walk-forward."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
from bisect import bisect_left
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from core.costs import cost_in_r, round_trip_cost_pct
from core.evaluation_spec import BREAKEVEN_TRIGGER_PCT, HORIZON_SESSIONS, simulate_trade

# Entry buffer and minimum stop distance are fractions of entry. A None
# target keeps the scanner level. Horizons count sessions from the signal.
GRID = {
    "scanner": (0.0, 0.0, None, HORIZON_SESSIONS),
    "entry_buffer": (0.005, 0.0, None, HORIZON_SESSIONS),
    "stop_floor": (0.0, 0.03, None, HORIZON_SESSIONS),
    "target_3pct": (0.0, 0.0, 0.03, HORIZON_SESSIONS),
    "target_5pct": (0.0, 0.0, 0.05, HORIZON_SESSIONS),
    "short": (0.0, 0.0, None, 5),
    "long": (0.0, 0.0, None, 20),
    "buffer_floor_target": (0.005, 0.03, 0.05, 20),
}
MAX_HORIZON = max(spec[3] for spec in GRID.values())


class GeometryRun:
    def __init__(self, db, *, cutoff, end, binding):
        self.cutoff = date.fromisoformat(cutoff).isoformat()
        self.end = date.fromisoformat(end).isoformat()
        if self.end <= self.cutoff:
            raise ValueError("Evaluation end must follow the development cutoff")
        self.db = db
        self.boundaries = None
        config = json.dumps([self.cutoff, self.end, GRID, BREAKEVEN_TRIGGER_PCT, binding], sort_keys=True)
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS geometry_config (id INTEGER PRIMARY KEY, spec TEXT)")
            db.execute("INSERT OR IGNORE INTO geometry_config VALUES (1, ?)", (config,))
            if db.execute("SELECT spec FROM geometry_config WHERE id=1").fetchone()[0] != config:
                raise ValueError("Checkpoint belongs to a different snapshot or experiment")
            db.execute("CREATE TABLE IF NOT EXISTS geometry_done (symbol TEXT PRIMARY KEY)")
            db.execute("""
                CREATE TABLE IF NOT EXISTS geometry_outcomes (
                    symbol TEXT, split TEXT, day TEXT, variant TEXT, outcome TEXT,
                    net_r REAL, net_pct REAL, holding INTEGER
                )
            """)

    def completed(self, symbol):
        return self.db.execute("SELECT 1 FROM geometry_done WHERE symbol=?", (symbol,)).fetchone() is not None

    def save(self, symbol, rows):
        # Commit a complete symbol once, even if two workers overlap.
        with self.db:
            inserted = self.db.execute("INSERT OR IGNORE INTO geometry_done VALUES (?)", (symbol,))
            if inserted.rowcount:
                self.db.executemany("INSERT INTO geometry_outcomes VALUES (?,?,?,?,?,?,?,?)",
                                    ((symbol, *row) for row in rows))

    @staticmethod
    def rescale(frame, multiplier):
        frame = frame.copy()
        columns = ["open", "high", "low", "close"]
        frame[columns] *= multiplier
        frame["volume"] /= multiplier
        frame.attrs.pop("ca_events", None)
        return frame

    def includes(self, day, sessions):
        boundary = self.cutoff if day < self.cutoff else self.end
        if day >= boundary:
            return False
        if self.boundaries is None:
            days = sorted(sessions)
            self.boundaries = {
                d: sessions[days[i]] if (i := bisect_left(days, d)) < len(days) else None
                for d in (self.cutoff, self.end)
            }
        boundary_ordinal = self.boundaries[boundary]
        return boundary_ordinal is not None and sessions[day] + MAX_HORIZON < boundary_ordinal

    def measure(self, day, setup, fwd, sessions):
        if not self.includes(day, sessions):
            return
        split = "development" if day < self.cutoff else "evaluation"
        if not all(math.isfinite(x) for x in (setup.stop, setup.entry, setup.target)):
            return
        if not 0 < setup.stop < setup.entry < setup.target:
            return
        bars = {k: fwd[k].to_numpy() for k in ("open", "high", "low", "close")}
        for variant, (buffer, stop_floor, target_pct, horizon) in GRID.items():
            entry = setup.entry * (1 + buffer)
            stop = min(setup.stop, entry * (1 - stop_floor))
            target = setup.target if target_pct is None else entry * (1 + target_pct)
            if target <= entry:
                yield split, day, variant, "INVALID", None, None, None
                continue
            sim = simulate_trade(entry, stop, target, bars["high"][:horizon], bars["low"][:horizon],
                                 bars["close"][:horizon], opens=bars["open"][:horizon])
            if len(fwd) < horizon and sim.outcome in {"FLAT", "NO_FILL"}:
                yield split, day, variant, "CENSORED", None, None, None
            elif not sim.filled:
                yield split, day, variant, "NO_FILL", None, None, None
            else:
                risk_fraction = (sim.fill_price - stop) / sim.fill_price
                net_r = sim.r - cost_in_r(risk_fraction, "CNC")
                yield (split, day, variant, sim.outcome, net_r, 100 * net_r * risk_fraction,
                       1 + sim.exit_offset - sim.entry_offset)

    def _summary(self, split, variant):
        total, n, wins, censored, net_r, net_pct, holding, sessions = self.db.execute("""
            SELECT count(*), count(net_r), sum(net_r > 0), sum(outcome='CENSORED'),
                   avg(net_r), avg(net_pct), avg(holding),
                   count(DISTINCT CASE WHEN net_r IS NOT NULL THEN day END)
            FROM geometry_outcomes WHERE split=? AND variant=?
        """, (split, variant)).fetchone()
        return {
            "setups": total, "filled": n, "sessions": sessions, "censored": censored or 0,
            "positive_net_trades": wins or 0,
            "probability_positive_net": (wins + 0.5) / (n + 1) if n else None,
            "mean_net_R": net_r, "mean_net_return_pct": net_pct,
            "mean_holding_sessions": holding,
            "net_return_pct_per_resolved_setup": (
                net_pct * n / (total - censored) if n else None
            ),
        }

    def report(self, symbols, official_index):
        with self.db:
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS geometry_by_split_variant
                ON geometry_outcomes (split, variant, day, symbol)
            """)
        development = {v: self._summary("development", v) for v in GRID}
        eligible = [v for v, s in development.items() if s["filled"] >= 100 and s["sessions"] >= 25]
        selected = max(eligible, key=lambda v: development[v]["net_return_pct_per_resolved_setup"],
                       default=None)
        from scan.signal_backtest import _evidence_fields

        evaluation = {}
        for variant in dict.fromkeys(v for v in ("scanner", selected) if v):
            result = self._summary("evaluation", variant)
            p = development[variant]["probability_positive_net"]
            baseline_p = development["scanner"]["probability_positive_net"]
            n, wins = result["filled"], result["positive_net_trades"]
            daily = self.db.execute("""
                SELECT day, sum(coalesce(net_pct, 0)), count(*)
                FROM geometry_outcomes WHERE split='evaluation' AND variant=? AND outcome!='CENSORED'
                GROUP BY day
            """, (variant,)).fetchall()
            clusters = {day: (total, count) for day, total, count in daily}
            inference = _evidence_fields({"clusters": clusters}, official_index,
                                         horizon=MAX_HORIZON)
            result["net_return_pct_per_setup_lower_95"] = inference["lower_r"]
            result["realized_positive_net_rate"] = wins / n if n else None
            result.pop("probability_positive_net")
            result["development_probability_positive_net"] = p
            for label, prediction in (("brier", p), ("baseline_brier", baseline_p)):
                result[label] = (
                    (wins * (1 - prediction)**2 + (n - wins) * prediction**2) / n
                    if n and prediction is not None else None
                )
            evaluation[variant] = result
        comparison = None
        if selected and selected != "scanner":
            paired = self.db.execute("""
                SELECT candidate.day,
                       sum(coalesce(candidate.net_pct, 0) - coalesce(baseline.net_pct, 0)), count(*)
                FROM geometry_outcomes AS candidate JOIN geometry_outcomes AS baseline
                  ON candidate.symbol=baseline.symbol AND candidate.day=baseline.day
                WHERE candidate.split='evaluation' AND baseline.split='evaluation'
                  AND candidate.variant=? AND baseline.variant='scanner'
                  AND candidate.outcome!='CENSORED' AND baseline.outcome!='CENSORED'
                GROUP BY candidate.day
            """, (selected,)).fetchall()
            if paired:
                clusters = {day: (total, count) for day, total, count in paired}
                bound = _evidence_fields({"clusters": clusters}, official_index, horizon=MAX_HORIZON)
                comparison = {"paired_setups": sum(count for _, _, count in paired),
                              "mean_net_return_pct_difference": bound["expectancy_r_precise"],
                              "lower_95_pct_difference": bound["lower_r"]}
        return {"evidence": "historical_research", "symbols": len(symbols),
                "development_before": self.cutoff, "evaluation_before": self.end,
                "grid": GRID, "development": development, "selected": selected,
                "evaluation": evaluation, "paired_comparison": comparison}


def run(checkpoint: Path, *, cutoff: str, end: str):
    """Resume on a frozen local archive using the existing dataset gate."""
    from core.live_volume import on_live_volume

    root = os.environ.get("QT_RUNTIME_ROOT")
    if not root or on_live_volume(root) or on_live_volume(checkpoint):
        raise ValueError("Geometry requires an explicit isolated runtime and checkpoint")
    from data import bhavcopy_store, index_store
    from data.corporate_actions import _events_path
    from data.nse_universe import universe_history_path
    from core.runtime_paths import logs_dir, logs_path

    root_path = Path(root).resolve()
    metadata = (bhavcopy_store._PKL, index_store._PKL, _events_path(),
                universe_history_path())
    instrument_names = logs_path("instruments_cache.csv")
    paths = (*metadata, logs_dir(), bhavcopy_store._BHAV_DIR, instrument_names,
             logs_dir() / "nse_holidays.json")
    if any(on_live_volume(path) or not path.resolve().is_relative_to(root_path) for path in paths):
        raise ValueError("Geometry input paths must stay inside the isolated runtime")
    from gauntlet.validator import validate
    from scan.signal_backtest import GRID_SESSION_ANCHORED, _instrument_names, _run_backtest_inner, _stock_symbols
    from scan.unified_scanner import UnifiedScanner

    # Never bootstrap missing data over the network. The snapshot owner first
    # prepares the existing bhavcopy, membership and index stores.
    missing = [path.name for path in metadata if not path.is_file()]
    if missing:
        return {"status": "ABORTED", "validation": {"ok": False,
                "failed": ["missing_frozen_inputs"], "missing": missing}}
    if not bhavcopy_store._load_pkl():
        return {"status": "ABORTED", "validation": {"ok": False,
                "failed": ["verified_bhavcopy_store_unavailable"]}}
    if not index_store.load_index_store_from_cache():
        return {"status": "ABORTED", "validation": {"ok": False,
                "failed": ["verified_index_store_unavailable"]}}
    validation = validate(sample=len(bhavcopy_store.store_symbols()))
    if not validation["ok"]:
        return {"status": "ABORTED", "validation": validation}

    # Bind the checkpoint to code (including local edits), runtime dependencies,
    # and the input files rather than trusting the checkpoint's filename.
    repo = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    files = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=repo)
    for name in sorted(set(files.decode().split("\0"))):
        path = repo / name
        if path.suffix == ".py" and path.is_file():
            code.update(name.encode() + b"\0" + path.read_bytes())
    from core.session_window import official_session_index

    inputs = {}
    input_files = (*metadata, instrument_names) if instrument_names.is_file() else metadata
    for path in input_files:
        with path.open("rb") as handle:
            inputs[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    verified_days = bhavcopy_store._dates_on_disk()
    if not verified_days or date.fromisoformat(end) > verified_days[-1]:
        raise ValueError("Evaluation end must be within verified NSE history")
    from research.intelligence.data.nse_calendar import covered_years, load_holidays

    years = covered_years(load_holidays())
    if not set(range(verified_days[0].year, date.fromisoformat(end).year + 1)) <= years:
        return {"status": "ABORTED", "validation": {"failed": ["incomplete_nse_calendar_coverage"]}}
    sessions = official_session_index(d.isoformat() for d in verified_days)
    if not math.isfinite(round_trip_cost_pct("CNC")) or round_trip_cost_pct("CNC") <= 0:
        raise ValueError("Geometry research requires positive delivery trading costs")
    # QT overrides can change the scanner, session handling and execution rules.
    env = json.dumps({k: v for k, v in os.environ.items() if k.startswith("QT_")}, sort_keys=True)
    binding = {"code": code.hexdigest(), "inputs": inputs, "pandas": pd.__version__,
               "numpy": np.__version__, "python": sys.version,
               "env_sha256": hashlib.sha256(env.encode()).hexdigest(),
               "calendar_sha256": hashlib.sha256(json.dumps(sessions, sort_keys=True).encode()).hexdigest(),
               "cnc_cost_pct": round_trip_cost_pct("CNC"),
               "verified_days": hashlib.sha256(str(verified_days).encode()).hexdigest()}
    names = _instrument_names()
    symbols = sorted(_stock_symbols(bhavcopy_store.store_symbols(), names))
    binding["symbols_sha256"] = hashlib.sha256(json.dumps(symbols).encode()).hexdigest()
    with closing(sqlite3.connect(checkpoint, timeout=30)) as db:
        geometry = GeometryRun(db, cutoff=cutoff, end=end, binding=binding)
        scanner = UnifiedScanner(evidence_gate=False, calibration_snapshot={"multipliers": {}})
        result = _run_backtest_inner(scanner, {}, symbols,
                                   5, len(verified_days), MAX_HORIZON, 0,
                                   decision_grid=GRID_SESSION_ANCHORED, geometry=geometry)
        return {**result, "validation": validation, "binding": binding}
