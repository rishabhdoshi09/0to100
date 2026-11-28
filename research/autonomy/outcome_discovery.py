"""One bounded, preregistered outcome-first look on an isolated Gauntlet snapshot."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from statistics import mean, median
from time import monotonic

from core.live_volume import on_live_volume
from core.runtime_paths import REPO_ROOT, logs_path, runtime_root
from research import registry
from research.overlap_inference import overlap_robust_test

NAME = "Liquid cross-sectional 20-session reversal over 10 sessions"
START, END = "2024-08-01", "2025-11-01"
HORIZON, MIN_NAMES, MIN_TURNOVER = 10, 50, 100_000_000
TRIAL_BUDGET, ROW_BUDGET = 1, 30_000
CRITERIA = {"lower_95_one_sided": {">=": 0.0035}, "net_return_stress": {">": 0}}
WINDOW = {
    "start": START, "end_exclusive": END, "stage": "DISCOVERY",
    "hypothesis": "Temporary liquidity-driven selling reverses in heavily traded cash equities.",
    "motivation": "Scanner score lacked predictive rank information; test short-horizon reversal without pattern filters.",
    "eligibility": "Gauntlet scoped EQ cohort, trailing mean turnover >= INR 100m and >= 50 names per decision date.",
    "formation": "Sort by trailing 20-session return at decision close; nominate bottom quintile, symbol breaks ties.",
    "target": "Next official open to tenth official close; complete paired cross-section required.",
    "benchmarks": "Same eligible equal-weight basket and idle cash.",
    "costs": "32 bps round trip, 52 bps stress; arithmetic cohort payoff, no portfolio-fill claim.",
    "diagnostics": "Five fixed quintiles, cross-sectional Spearman, next-open gap, positive/negative median formation return.",
    "inference": "Equal weight per session; overlap-aware 10-session intervals; regime splits descriptive.",
    "budget": {"looks": TRIAL_BUDGET, "rows": ROW_BUDGET, "input_bytes": 64 * 1024 * 1024},
    "confirmation": "Separate authorized, pre-registered prospective shadow with untouched evidence; no automatic promotion.",
}


def _series(connection, ordinals, symbols):
    grouped = defaultdict(list)
    rows = connection.execute(
        """SELECT symbol, day, features, outcome FROM geometry_observations
           WHERE day >= ? AND day < ? ORDER BY day, symbol LIMIT ?""",
        (START, END, ROW_BUDGET + 1),
    )
    row_count = 0
    for row_count, (symbol, day, features_json, outcome_json) in enumerate(rows, 1):
        if row_count > ROW_BUDGET:
            raise ValueError("Discovery row budget exceeded")
        if symbol not in symbols or day not in ordinals or ordinals[END] - ordinals[day] <= HORIZON:
            raise ValueError("Observation could cross the discovery boundary")
        features = json.loads(features_json)
        turnover, momentum = features["turnover_20"], features["momentum_20"]
        if not math.isfinite(turnover) or not math.isfinite(momentum):
            raise ValueError("Invalid formation observation")
        if turnover >= MIN_TURNOVER:
            outcome = json.loads(outcome_json)
            grouped[day].append((momentum, symbol, outcome["open_to_close_return"],
                                 outcome["next_open_gap"], bool(features["signals"])))
    return grouped, row_count


def _measure(connection, ordinals, symbols):
    from scipy.stats import spearmanr

    grouped, row_count = _series(connection, ordinals, symbols)
    paired, absolute, stress, gaps, rank = {}, {}, {}, [], []
    regimes = defaultdict(list)
    quintiles = [[] for _ in range(5)]
    sparse = incomplete = names = signal_free = 0
    missing_next_open = incomplete_followthrough = 0
    for day, observations in grouped.items():
        if len(observations) < MIN_NAMES:
            sparse += 1
            continue
        ranked = sorted(observations, key=lambda row: (row[0], row[1]))
        returns = [row[2] for row in ranked]
        if any(value is None for value in returns):
            incomplete += 1
            missing_next_open += sum(row[2] is None and row[3] is None for row in ranked)
            incomplete_followthrough += sum(row[2] is None and row[3] is not None for row in ranked)
            continue
        if not all(math.isfinite(value) for value in returns):
            raise ValueError("Invalid forward observation")
        n = len(ranked)
        buckets = [mean(returns[n*i//5:n*(i+1)//5]) for i in range(5)]
        for values, payoff in zip(quintiles, buckets):
            values.append(payoff - 0.0032)
        excess = buckets[0] - mean(returns)
        ordinal = ordinals[day]
        paired[ordinal] = (excess, 1)
        absolute[ordinal] = buckets[0] - 0.0032
        stress[ordinal] = buckets[0] - 0.0052
        formation = [row[0] for row in ranked]
        regimes["NONNEGATIVE" if median(formation) >= 0 else "NEGATIVE"].append(excess)
        if len(set(formation)) > 1 and len(set(returns)) > 1:
            rank.append(float(spearmanr(formation, returns).statistic))
        gaps.extend(row[3] for row in ranked[:n//5] if row[3] is not None and math.isfinite(row[3]))
        names += n
        signal_free += sum(not row[4] for row in ranked)

    inference = overlap_robust_test(paired, HORIZON)
    power = 0.0 if not inference["se"] else 2.487 * inference["se"]
    return {
        "observations_read": row_count, "eligible_observations_measured": names,
        "without_scanner_signals": signal_free, "decision_sessions": len(paired),
        "first_admitted_date": min((day for day, i in ordinals.items() if i in paired), default=None),
        "last_admitted_date": max((day for day, i in ordinals.items() if i in paired), default=None),
        "below_minimum_names": sparse, "incomplete_outcome_sessions": incomplete,
        "unavailable_next_opens": missing_next_open, "partly_observed_forward_windows": incomplete_followthrough,
        "expected_net_return": mean(absolute.values()) if absolute else None,
        "net_return_stress": mean(stress.values()) if stress else None,
        "equal_weight_net_return": mean(absolute[k] - paired[k][0] for k in paired) if paired else None,
        "quintile_net_returns": [mean(values) if values else None for values in quintiles],
        "selected_mean_next_open_gap": mean(gaps) if gaps else None,
        "formation_return_rank_correlation": mean(rank) if rank else None,
        "formation_regimes": {k: {"sessions": len(v), "mean_excess": mean(v)} for k, v in regimes.items()},
        "excess_inference": inference, "detectable_excess_80pct_power_approx": power or None,
        "lower_95_one_sided": inference["lower_95_one_sided"],
    }


def _isolated_checkpoint(checkpoint: Path) -> Path:
    raw_root = os.environ.get("QT_RUNTIME_ROOT", "").strip()
    if not raw_root or on_live_volume(raw_root) or on_live_volume(checkpoint):
        raise ValueError("Discovery requires an isolated development runtime")
    root = runtime_root().resolve(strict=True)
    temp = Path("/tmp").resolve()
    if (root == REPO_ROOT or root == temp or not root.is_relative_to(temp)
            or os.statvfs(root).f_flag & os.ST_RDONLY):
        raise ValueError("Discovery requires a writable scratch runtime")
    source = checkpoint.resolve(strict=True)
    destination = registry._DB_PATH
    lock = logs_path("outcome_discovery.lock")
    writes = (destination, lock, *(destination.with_name(destination.name + suffix)
                                   for suffix in ("-wal", "-shm", "-journal")))
    if (not source.is_relative_to(root) or source == destination.resolve()
            or any(path.is_symlink() or not path.resolve().is_relative_to(root) for path in writes)):
        raise ValueError("Discovery inputs and registry must stay within the scratch runtime")
    return source


def configured_checkpoint() -> Path | None:
    raw = os.environ.get("QT_OUTCOME_DISCOVERY_CHECKPOINT")
    return _isolated_checkpoint(Path(raw)) if raw else None


def _digest(path):
    with path.open("rb") as snapshot:
        return hashlib.file_digest(snapshot, "sha256").hexdigest()


def run(checkpoint: Path) -> dict:
    from product.learning_observations import writer_lock

    source = _isolated_checkpoint(checkpoint)
    if source.stat().st_size > WINDOW["budget"]["input_bytes"] or any(
        source.with_name(source.name + suffix).exists() for suffix in ("-wal", "-journal", "-shm")
    ):
        raise ValueError("Discovery requires a bounded, sealed checkpoint without sidecars")
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with writer_lock(logs_path("outcome_discovery")):
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
            _, end, grid, _, binding = json.loads(
                connection.execute("SELECT spec FROM geometry_config WHERE id=1").fetchone()[0]
            )
            days = binding["sessions"]
            symbols = set(binding["symbols"])
            done = {row[0] for row in connection.execute("SELECT symbol FROM geometry_done")}
            if (end != END or not binding.get("exploratory") or not binding.get("observations")
                    or grid["scanner"][-1] != HORIZON or days != sorted(set(days))
                    or binding["start"] > START or days[0] > START
                    or days[-1] != END or set(binding["archive_sha256"]) != set(days[:-1])
                    or symbols != done):
                raise ValueError("Requires the complete, discovery-only Gauntlet observation checkpoint")
            if connection.execute("SELECT 1 FROM geometry_observations WHERE day >= ? LIMIT 1", (END,)).fetchone():
                raise ValueError("Checkpoint contains reserved observation keys")
            source_hash = _digest(source)
            description = f"{WINDOW['hypothesis']} Gauntlet SHA-256 {source_hash}."
            hid = registry.register_hypothesis(NAME, CRITERIA, WINDOW, description, code_hash=code_hash)
            existing = registry.get_experiment(hid)
            if existing["result"]:
                return {"decision": "DISCOVERY_BUDGET_EXHAUSTED", "hypothesis_id": hid, **json.loads(existing["result"])}
            if existing["code_hash"] != code_hash or existing["description"] != description:
                raise ValueError("Pending discovery was registered by a different implementation")
            start = monotonic()
            metrics = _measure(connection, {day: index for index, day in enumerate(days)}, symbols)
        if _digest(source) != source_hash:
            raise ValueError("Gauntlet snapshot changed during discovery")
        if metrics["excess_inference"]["status"] != "OK":
            status = "INCONCLUSIVE"
        elif registry.meets_criteria(metrics, CRITERIA):
            status = "PROMISING"
        elif metrics["expected_net_return"] <= 0 and metrics["excess_inference"]["mean"] <= 0:
            status = "REJECTED"
        else:
            status = "INCONCLUSIVE"
        result = {"status": status, "source_sha256": source_hash, "code_sha256": code_hash,
                  "evidence_class": "HISTORICAL_REPLAY", "promotion_eligible": False,
                  "elapsed_seconds": round(monotonic() - start, 3), "metrics": metrics,
                  "next": ("REGISTER_INDEPENDENT_SHADOW_PROTOCOL" if status == "PROMISING"
                           else "AUDIT_MISSING_PAIRED_OUTCOMES" if metrics["incomplete_outcome_sessions"]
                           else "AWAIT_DISTINCT_POINT_IN_TIME_OBSERVATIONS"),
                  }
        registry.record_discovery_result(hid, result)
        return {"decision": "DISCOVERY_COMPLETE", "hypothesis_id": hid, **result}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    print(json.dumps(run(parser.parse_args().checkpoint), allow_nan=False))
