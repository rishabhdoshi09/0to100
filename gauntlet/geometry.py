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
from datetime import date, timedelta
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
EXIT_CLASSES = ("target_first", "stop_first", "time_expiry")
DISCOVERY_END = "2025-11-01"


class GeometryRun:
    def __init__(self, db, *, cutoff, end, binding):
        self.cutoff = date.fromisoformat(cutoff).isoformat()
        self.end = date.fromisoformat(end).isoformat()
        if self.end <= self.cutoff:
            raise ValueError("Evaluation end must follow the development cutoff")
        self.db = db
        self.boundaries = None
        self.scoped_sessions = None
        self.scoped_frames = None
        self.exploratory = bool(binding.get("exploratory"))
        self.observations = bool(binding.get("observations"))
        self.config = json.dumps([self.cutoff, self.end, GRID, BREAKEVEN_TRIGGER_PCT, binding], sort_keys=True)
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS geometry_config (id INTEGER PRIMARY KEY, spec TEXT)")
            db.execute("INSERT OR IGNORE INTO geometry_config VALUES (1, ?)", (self.config,))
            previous = db.execute("SELECT spec FROM geometry_config WHERE id=1").fetchone()[0]
            db.execute("CREATE TABLE IF NOT EXISTS geometry_done (symbol TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS geometry_validation (id INTEGER PRIMARY KEY, report TEXT)")
            db.execute("""
                CREATE TABLE IF NOT EXISTS geometry_outcomes (
                    symbol TEXT, split TEXT, day TEXT, variant TEXT, outcome TEXT,
                    net_r REAL, net_pct REAL, holding INTEGER, exit_day TEXT
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(geometry_outcomes)")}
            if "exit_day" not in columns:
                db.execute("ALTER TABLE geometry_outcomes ADD COLUMN exit_day TEXT")
            db.execute("""
                CREATE INDEX IF NOT EXISTS geometry_by_split_variant
                ON geometry_outcomes (split, variant, day, symbol)
            """)
            if self.observations:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS geometry_observations (
                        symbol TEXT, day TEXT, features TEXT NOT NULL,
                        outcome TEXT NOT NULL, PRIMARY KEY (symbol, day)
                    ) WITHOUT ROWID
                """)
                db.execute("CREATE TABLE IF NOT EXISTS geometry_resume (symbol TEXT PRIMARY KEY, first_day TEXT NOT NULL)")
            if previous != self.config:
                self._extend(previous)

    def _extend(self, previous):
        cutoff, end, grid, breakeven, old = json.loads(previous)
        new_cutoff, new_end, new_grid, new_breakeven, new = json.loads(self.config)
        mutable = {"symbols", "sessions", "archive_sha256"}
        compatible = (
            self.observations and old.get("observations") and old.get("exploratory")
            and (cutoff, grid, breakeven) == (new_cutoff, new_grid, new_breakeven) and new_end >= end
            and {k: v for k, v in old.items() if k not in mutable}
            == {k: v for k, v in new.items() if k not in mutable}
            and mutable <= old.keys() and mutable <= new.keys()
            and set(old["symbols"]) <= set(new["symbols"])
            and old["sessions"][:-1] == new["sessions"][:len(old["sessions"]) - 1]
        )
        if not compatible or any(new["archive_sha256"].get(day) != digest and digest is not None
                                 for day, digest in old["archive_sha256"].items()):
            raise ValueError("Checkpoint belongs to a different snapshot or experiment")
        done = {row[0] for row in self.db.execute("SELECT symbol FROM geometry_done")}
        if done != set(old["symbols"]):
            raise ValueError("Complete the previous checkpoint scope before extending it")

        changed = [i for i, day in enumerate(old["sessions"][:-1])
                   if old["archive_sha256"][day] != new["archive_sha256"][day]]
        if new_end > end:
            changed.append(len(old["sessions"]) - 1)
        if changed:
            affected = new["sessions"][max(0, min(changed) - MAX_HORIZON)]
            self.db.execute("DELETE FROM geometry_outcomes WHERE day >= ?", (affected,))
            self.db.execute("DELETE FROM geometry_observations WHERE day >= ?", (affected,))
            self.db.executemany("INSERT OR REPLACE INTO geometry_resume VALUES (?, ?)",
                                ((symbol, affected) for symbol in done))
            self.db.execute("DELETE FROM geometry_done")
        self.db.execute("DELETE FROM geometry_validation")
        self.db.execute("UPDATE geometry_config SET spec=? WHERE id=1", (self.config,))

    def resume_day(self, symbol):
        if not self.observations:
            return ""
        row = self.db.execute("SELECT first_day FROM geometry_resume WHERE symbol=?", (symbol,)).fetchone()
        return row[0] if row else ""

    def validate_once(self, validate):
        """Reuse the successful preflight for the exact bound snapshot."""
        cached = self.db.execute("SELECT report FROM geometry_validation WHERE id=1").fetchone()
        if cached:
            return json.loads(cached[0])
        report = validate()
        if report["ok"]:
            with self.db:
                self.db.execute("""
                    INSERT OR IGNORE INTO geometry_validation SELECT 1, ?
                    WHERE (SELECT spec FROM geometry_config WHERE id=1)=?
                """, (json.dumps(report, sort_keys=True), self.config))
        return report

    def completed(self, symbol):
        return self.db.execute("SELECT 1 FROM geometry_done WHERE symbol=?", (symbol,)).fetchone() is not None

    def save(self, symbol, rows, observations=()):
        # Commit a complete symbol once, even if two workers overlap.
        with self.db:
            inserted = self.db.execute("""
                INSERT OR IGNORE INTO geometry_done SELECT ?
                WHERE (SELECT spec FROM geometry_config WHERE id=1)=?
            """, (symbol, self.config))
            if inserted.rowcount:
                self.db.executemany("INSERT INTO geometry_outcomes VALUES (?,?,?,?,?,?,?,?,?)",
                                    ((symbol, *row) for row in rows))
                if self.observations:
                    self.db.executemany("INSERT INTO geometry_observations VALUES (?,?,?,?)",
                                        ((symbol, *row) for row in observations))
                    self.db.execute("DELETE FROM geometry_resume WHERE symbol=?", (symbol,))

    @staticmethod
    def observe(day, history, forward, setup):
        """Store decision features separately from future measurements."""
        close = history.close.to_numpy()
        returns = close[-20:] / close[-21:-1] - 1
        features = {
            "close": float(close[-1]),
            "momentum_20": float(close[-1] / close[-21] - 1),
            "volatility_20": float(np.std(returns)),
            "turnover_20": float((history.close * history.volume).tail(20).mean()),
            "signals": sorted(setup.signals) if setup else [],
            "score": float(setup.score) if setup else None,
            "verdict": setup.verdict if setup else None,
        }
        # A missing next bar is a censoring event, not a zero gap or a no-fill.
        outcome = {"next_open_gap": None, "close_return": None, "open_to_close_return": None}
        if len(forward):
            entry = float(forward.open.iloc[0])
            outcome["next_open_gap"] = entry / float(close[-1]) - 1
            if len(forward) >= HORIZON_SESSIONS:
                end = float(forward.close.iloc[HORIZON_SESSIONS - 1])
                outcome.update(close_return=end / float(close[-1]) - 1,
                               open_to_close_return=end / entry - 1)
        return day, json.dumps(features, sort_keys=True, allow_nan=False), json.dumps(outcome, sort_keys=True, allow_nan=False)

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
                yield split, day, variant, "INVALID", None, None, None, None
                continue
            sim = simulate_trade(entry, stop, target, bars["high"][:horizon], bars["low"][:horizon],
                                 bars["close"][:horizon], opens=bars["open"][:horizon])
            if len(fwd) < horizon and sim.outcome in {"FLAT", "NO_FILL"}:
                yield split, day, variant, "CENSORED", None, None, None, None
            elif not sim.filled:
                yield split, day, variant, "NO_FILL", None, None, None, None
            else:
                risk_fraction = (sim.fill_price - stop) / sim.fill_price
                net_r = sim.r - cost_in_r(risk_fraction, "CNC")
                outcome = sim.outcome
                if outcome == "SCRATCH" and sim.fill_price >= target:
                    # The canonical simulator scratches a gap fill beyond the target.
                    # The exit was triggered by the target, even though gross P&L was zero.
                    outcome = "TARGET_GAP_SCRATCH"
                yield (split, day, variant, outcome, net_r, 100 * net_r * risk_fraction,
                       1 + sim.exit_offset - sim.entry_offset,
                       fwd.index[sim.exit_offset].strftime("%Y-%m-%d"))

    def _drawdown(self, split, variant):
        equity = peak = drawdown = 0.0
        for (daily_r,) in self.db.execute("""
            SELECT sum(net_r) FROM geometry_outcomes
            WHERE split=? AND variant=? AND net_r IS NOT NULL AND exit_day IS NOT NULL
            GROUP BY exit_day ORDER BY exit_day
        """, (split, variant)):
            equity += daily_r
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        return drawdown

    def _summary(self, split, variant):
        total, n, wins, censored, net_r, net_pct, holding, sessions, targets, stops, expired = self.db.execute("""
            SELECT count(*), count(net_r), sum(net_r > 0), sum(outcome='CENSORED'),
                   avg(net_r), avg(net_pct), avg(holding),
                   count(DISTINCT CASE WHEN net_r IS NOT NULL THEN day END),
                   sum(outcome IN ('WIN','TARGET_GAP_SCRATCH')), sum(outcome IN ('LOSS','SCRATCH')), sum(outcome='FLAT')
            FROM geometry_outcomes WHERE split=? AND variant=?
        """, (split, variant)).fetchone()
        return {
            "setups": total, "filled": n, "sessions": sessions, "censored": censored or 0,
            "positive_net_trades": wins or 0,
            "exit_counts": dict(zip(EXIT_CLASSES, (targets or 0, stops or 0, expired or 0))),
            "probability_positive_net": (wins + 0.5) / (n + 1) if n else None,
            "mean_net_R": net_r, "mean_net_return_pct": net_pct,
            "mean_holding_sessions": holding,
            "settled_daily_fixed_risk_max_drawdown_R": self._drawdown(split, variant) if n else None,
            "net_return_pct_per_resolved_setup": (
                net_pct * n / (total - censored) if n else None
            ),
        }

    def _exit_calibration(self, variant, development, evaluation, official_index):
        from scan.signal_backtest import _evidence_fields

        n_dev, n_eval = development["filled"], evaluation["filled"]
        if not n_dev or not n_eval:
            return {}
        daily = self.db.execute("""
            SELECT day, count(*), sum(outcome IN ('WIN','TARGET_GAP_SCRATCH')),
                   sum(outcome IN ('LOSS','SCRATCH')), sum(outcome='FLAT')
            FROM geometry_outcomes WHERE split='evaluation' AND variant=? AND net_r IS NOT NULL
            GROUP BY day
        """, (variant,)).fetchall()
        result = {}
        for offset, label in enumerate(EXIT_CLASSES, 2):
            # A joint Dirichlet(0.5, 0.5, 0.5) prior keeps the three forecasts normalized.
            p = (development["exit_counts"][label] + 0.5) / (n_dev + 1.5)
            rate = evaluation["exit_counts"][label] / n_eval
            clusters = {row[0]: (row[offset], row[1]) for row in daily}
            complement = {day: (n - k, n) for day, (k, n) in clusters.items()}
            low = _evidence_fields({"clusters": clusters}, official_index, horizon=MAX_HORIZON)["lower_r"]
            inverse = _evidence_fields({"clusters": complement}, official_index, horizon=MAX_HORIZON)["lower_r"]
            result[label] = {
                "development_probability": p,
                "evaluation_rate": rate,
                "calibration_error": rate - p,
                "brier": rate * (1 - p)**2 + (1 - rate) * p**2,
                "evaluation_lower_95_one_sided": max(0, low) if low is not None else None,
                "evaluation_upper_95_one_sided": min(1, 1 - inverse) if inverse is not None else None,
            }
        return result

    def report(self, symbols, official_index):
        if self.db.in_transaction:
            return self._report(symbols, official_index)
        with self.db:
            self.db.execute("BEGIN")
            return self._report(symbols, official_index)

    def _report(self, symbols, official_index):
        if self.db.execute("SELECT spec FROM geometry_config WHERE id=1").fetchone()[0] != self.config:
            raise ValueError("Checkpoint scope changed before reporting")
        development = {v: self._summary("development", v) for v in GRID}
        eligible = [v for v, s in development.items() if s["filled"] >= 100 and s["sessions"] >= 25]
        selected = max(eligible, key=lambda v: development[v]["net_return_pct_per_resolved_setup"],
                       default=None)
        from scan.signal_backtest import _evidence_fields

        evaluation = {}
        for variant in GRID:
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
            result["exit_calibration"] = self._exit_calibration(
                variant, development[variant], result, official_index)
            if result["exit_calibration"]:
                result["multiclass_brier"] = sum(
                    item["brier"] for item in result["exit_calibration"].values())
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
        return {"evidence": "exploratory_historical_research" if self.exploratory else "historical_research",
                "symbols": len(symbols),
                "development_before": self.cutoff, "evaluation_before": self.end,
                "grid": GRID, "development": development, "selected": selected,
                "evaluation_policy": ("Both periods are exploratory. The later period is a temporal split, not unseen evidence."
                                      if self.exploratory else
                                      "Confirmatory for the development selection only. Other results are diagnostic."),
                "drawdown_definition": "Peak-to-trough cumulative realized net R by exit day, with one risk unit per fill. No portfolio or mark-to-market claim.",
                "evaluation": evaluation, "paired_comparison": comparison}


def _code_digest():
    repo = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    files = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=repo)
    for name in sorted(set(files.decode().split("\0"))):
        path = repo / name
        if path.suffix == ".py" and path.is_file():
            code.update(name.encode() + b"\0" + path.read_bytes())
    return code.hexdigest()


def run(checkpoint: Path, *, cutoff: str, end: str, research_start=None, stocks=None,
        observations=False, sample_step=1, max_symbols=None):
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
    if research_start is not None or stocks is not None:
        return _run_scoped(checkpoint, cutoff=cutoff, end=end, start=research_start, stocks=stocks,
                           root=root_path, observations=observations, sample_step=sample_step,
                           max_symbols=max_symbols)
    if observations or sample_step != 1 or max_symbols is not None:
        raise ValueError("Observation and batching options require scoped research")
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
    if not bhavcopy_store._load_pkl(remove_invalid=False):
        return {"status": "ABORTED", "validation": {"ok": False,
                "failed": ["verified_bhavcopy_store_unavailable"]}}
    if not index_store.load_index_store_from_cache():
        return {"status": "ABORTED", "validation": {"ok": False,
                "failed": ["verified_index_store_unavailable"]}}

    # Bind the checkpoint to code (including local edits), runtime dependencies,
    # and the input files rather than trusting the checkpoint's filename.
    from core.session_window import official_session_index

    inputs = {}
    input_files = (*metadata, instrument_names) if instrument_names.is_file() else metadata
    for path in input_files:
        with path.open("rb") as handle:
            inputs[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    verified_days = bhavcopy_store._dates_on_disk()
    if not verified_days or date.fromisoformat(end) > verified_days[-1]:
        raise ValueError("Evaluation end must be within verified NSE history")
    from research.intelligence.data.nse_calendar import MANIFEST_PATH, covered_years, load_holidays

    years = covered_years(load_holidays())
    uncovered = sorted(set(range(verified_days[0].year, date.fromisoformat(end).year + 1)) - years)
    inputs["calendar_manifest"] = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    sessions = official_session_index(d.isoformat() for d in verified_days)
    if not math.isfinite(round_trip_cost_pct("CNC")) or round_trip_cost_pct("CNC") <= 0:
        raise ValueError("Geometry research requires positive delivery trading costs")
    # QT overrides can change the scanner, session handling and execution rules.
    env = json.dumps({k: v for k, v in os.environ.items() if k.startswith("QT_")}, sort_keys=True)
    binding = {"code": _code_digest(), "inputs": inputs, "pandas": pd.__version__,
               "numpy": np.__version__, "python": sys.version,
               "env_sha256": hashlib.sha256(env.encode()).hexdigest(),
               "calendar_sha256": hashlib.sha256(json.dumps(sessions, sort_keys=True).encode()).hexdigest(),
               "cnc_cost_pct": round_trip_cost_pct("CNC"),
               "verified_days": hashlib.sha256(str(verified_days).encode()).hexdigest()}
    names = _instrument_names()
    symbols = sorted(_stock_symbols(bhavcopy_store.store_symbols(), names))
    binding["symbols_sha256"] = hashlib.sha256(json.dumps(symbols).encode()).hexdigest()

    def preflight():
        report = validate(sample=len(bhavcopy_store.store_symbols()))
        report["checks"].append({"check": "complete_nse_calendar_coverage", "ok": not uncovered,
                                 "detail": f"Unverified years: {uncovered}" if uncovered else "All years verified"})
        if uncovered:
            report["failed"].append("complete_nse_calendar_coverage")
            report["ok"] = False
        return report

    if not checkpoint.exists():
        # An invalid snapshot must not reserve the path for a future study.
        validation = preflight()
        if not validation["ok"]:
            return {"status": "ABORTED", "validation": validation}
    else:
        validation = None
    with closing(sqlite3.connect(checkpoint, timeout=30)) as db:
        geometry = GeometryRun(db, cutoff=cutoff, end=end, binding=binding)
        validation = geometry.validate_once(preflight if validation is None else lambda: validation)
        if not validation["ok"]:
            return {"status": "ABORTED", "validation": validation}
        scanner = UnifiedScanner(evidence_gate=False, calibration_snapshot={"multipliers": {}})
        result = _run_backtest_inner(scanner, {}, symbols,
                                   5, len(verified_days), MAX_HORIZON, 0,
                                   decision_grid=GRID_SESSION_ANCHORED, geometry=geometry)
        return {**result, "validation": validation, "binding": binding}


def _run_scoped(checkpoint, *, cutoff, end, start, stocks, root, observations=False,
                sample_step=1, max_symbols=None):
    from core.runtime_paths import logs_path
    from data import bhavcopy_store
    from data.corporate_actions import events_path, load_events
    from data.nse_universe import _is_non_stock_fund
    from gauntlet.validator import scoped_equity_frames
    from research.intelligence.data.nse_calendar import (
        MANIFEST_PATH, covered_years, documented_special_sessions, is_session, load_holidays, load_manifest,
    )
    from scan.signal_backtest import GRID_SESSION_ANCHORED, _run_backtest_inner
    from scan.unified_scanner import UnifiedScanner

    first, split, last = map(date.fromisoformat, (start, cutoff, end))
    if not first < split < last <= date.today():
        raise ValueError("Use ordered historical start, split and end dates")
    if (sample_step < 1 or (max_symbols is not None and max_symbols < 1)
            or (observations and last > date.fromisoformat(DISCOVERY_END))):
        raise ValueError("Use positive batching parameters and keep observations outside the reserved period")
    symbols = sorted({symbol.strip().upper() for symbol in stocks if symbol.strip()})
    if not symbols or any(_is_non_stock_fund(symbol) for symbol in symbols):
        raise ValueError("Supply explicit operating-company NSE symbols")
    action_path = events_path().resolve()
    archive_dir = bhavcopy_store._BHAV_DIR.resolve()
    if not all(path.is_relative_to(root) for path in (action_path, archive_dir)) or not action_path.is_file():
        raise ValueError("Scoped research requires isolated archive and corporate-action inputs")
    action_bytes = action_path.read_bytes()
    actions = load_events(action_path)
    if not actions or action_bytes != action_path.read_bytes():
        raise ValueError("Scoped research requires a valid, frozen corporate-action ledger")
    research_manifest = logs_path("nse_calendar_manifest.json")
    if any(not path.resolve().is_relative_to(root)
           for path in (research_manifest, logs_path("nse_holidays.json"))):
        raise ValueError("The research calendar must stay inside the isolated runtime")
    manifest = load_manifest(research_manifest) if research_manifest.exists() else load_manifest(MANIFEST_PATH)
    holidays = load_holidays()
    if not set(range(first.year, last.year + 1)) <= covered_years(holidays, manifest):
        raise ValueError("Every year in the scoped window must have verified NSE calendar coverage")
    special_sessions = documented_special_sessions(manifest)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    expected = [day for day in days if day < last
                and (is_session(day, holidays) or day.isoformat() in special_sessions)]
    archives = {}
    digests = {}
    for day in expected:
        path = bhavcopy_store._day_path(day)
        if not path.resolve().is_relative_to(archive_dir):
            raise ValueError(f"Archive escapes the isolated directory: {day}")
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            payload = None
        archives[day] = payload
        digests[day.isoformat()] = hashlib.sha256(payload).hexdigest() if payload is not None else None
    # The exclusive boundary needs an ordinal even on a weekend/holiday.
    sessions = {day.isoformat(): i for i, day in enumerate(expected)}
    sessions[last.isoformat()] = len(sessions)
    if not math.isfinite(round_trip_cost_pct("CNC")) or round_trip_cost_pct("CNC") <= 0:
        raise ValueError("Geometry research requires positive delivery trading costs")
    binding = {"exploratory": True, "start": first.isoformat(), "symbols": symbols,
               "code": _code_digest(), "archive_sha256": digests,
               "actions_sha256": hashlib.sha256(action_bytes).hexdigest(),
               "sessions": list(sessions), "calendar_manifest": manifest, "sample_step": sample_step,
               "observations": observations, "cnc_cost_pct": round_trip_cost_pct("CNC"),
               "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version,
               "settings_sha256": hashlib.sha256(json.dumps(
                   {k: v for k, v in os.environ.items() if k.startswith("QT_")}, sort_keys=True
               ).encode()).hexdigest()}
    with closing(sqlite3.connect(checkpoint, timeout=30)) as db:
        geometry = GeometryRun(db, cutoff=cutoff, end=end, binding=binding)
        geometry.scoped_sessions = sessions
        pending = [symbol for symbol in symbols if not geometry.completed(symbol)]
        reused = len(symbols) - len(pending)
        reused_observations = (db.execute("SELECT count(*) FROM geometry_observations").fetchone()[0]
                               if observations else None)
        pending = pending[:max_symbols]
        geometry.scoped_frames = {}

        def validate():
            geometry.scoped_frames, report = scoped_equity_frames(archives.items(), symbols, actions)
            report["documented_special_sessions"] = sorted(set(sessions) & special_sessions - {end})
            return report

        validation = geometry.validate_once(validate)
        if not validation["ok"]:
            return {"status": "ABORTED", "validation": validation}
        if pending and not geometry.scoped_frames:
            geometry.scoped_frames, _ = scoped_equity_frames(archives.items(), pending, actions)
        if pending:
            scanner = UnifiedScanner(evidence_gate=False, calibration_snapshot={"multipliers": {}})
            _run_backtest_inner(scanner, {}, pending, sample_step, len(days), MAX_HORIZON, 0,
                                decision_grid=GRID_SESSION_ANCHORED, geometry=geometry)
        db.execute("BEGIN")
        completed = sum(geometry.completed(symbol) for symbol in symbols)
        observation_count, observation_symbols = (db.execute(
            "SELECT count(*), count(DISTINCT symbol) FROM geometry_observations"
        ).fetchone() if observations else (None, None))
        return {**geometry.report(symbols, sessions), "validation": validation,
                "status": "COMPLETE" if completed == len(symbols) else "PARTIAL",
                "checkpoint_progress": {"reused_symbols": reused, "completed_symbols": completed,
                                        "pending_symbols": len(symbols) - completed,
                                        "observations": observation_count,
                                        "reused_observations": reused_observations,
                                        "symbols_with_observations": observation_symbols},
                "research_start": start, "stock_scope": symbols,
                "selection_bias": ("Manually named stock cohort. Censored windows and uninterrupted EQ warm-up "
                                   "select for listings with continuous trading. Population inference is unsupported."),
                "checkpoint_binding": hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()}
