"""Bounded feature discovery using the existing Gauntlet and research registry."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import monotonic

from research import registry
from research.autonomy import outcome_discovery as base
from research.overlap_inference import overlap_robust_test

PROGRAMME = "nse-outcome-information-v1"
BUDGET = 3
MIN_COVERAGE = 0.98
COST, STRESS_COST = 0.0032, 0.0052
# Bonferroni across all three diagnostics, including the non-tradable gap.
CRITERIA = {"economic_lower_bound": {">=": 0.0035}, "net_return_stress": {">": 0}}


@dataclass(frozen=True)
class Specification:
    family: str
    feature: str
    target: str
    rationale: str
    requires: str | None = None


def hypotheses(legacy, results):
    """Generate distinct tests of explanations supported by available facts."""
    prior = legacy["metrics"]
    momentum_rank = prior.get("formation_return_rank_correlation") or 0
    liquidity = results.get("liquidity", {}).get("metrics", {}).get("excess_inference", {}).get("mean")
    continuation = results.get("continuation", {}).get("status")
    candidates = [
        (4 if momentum_rank > 0 else 2, Specification(
            "continuation", "risk_adjusted_momentum", "holding_return",
            "Positive historical momentum rank association after failed reversal suggests persistence; normalize by prior volatility.",
        )),
        (3, Specification(
            "liquidity", "turnover", "holding_return",
            "Persistent turnover may capture participation separately from momentum; measure within-date association and stress friction.",
        )),
        (2 + int(liquidity is not None and liquidity <= 0) + int(continuation == "REJECTED"), Specification(
            "overnight", "momentum", "next_open_gap",
            "A weak executable payoff may coexist with overnight price discovery before the first actionable entry.",
        )),
        (1, Specification(
            "sector", "sector_relative_momentum", "holding_return",
            "Peer-relative performance requires dated sector assignments.", "historical_sector_membership",
        )),
    ]
    return sorted(candidates, key=lambda item: (-item[0], item[1].family))


def rows():
    entries = []
    for row in registry.list_experiments():
        window = json.loads(row["data_window"] or "{}")
        if window.get("programme") == PROGRAMME:
            entries.append((row, window))
    entries.sort(key=lambda item: item[1]["slot"])
    if (len(entries) > BUDGET or [w["slot"] for _, w in entries] != list(range(1, len(entries) + 1))
            or len({w["family"] for _, w in entries}) != len(entries)
            or any(row["result"] is None for row, _ in entries[:-1])):
        raise ValueError("Programme registry contains conflicting spend records")
    return entries


def next_slot():
    entries = rows()
    pending = [window["slot"] for row, window in entries if row["result"] is None]
    slot = min(pending) if pending else len(entries) + 1
    return slot if slot <= BUDGET else None


def scheduler_key():
    from research.autonomy import schedules

    legacy = registry.get_experiment(registry.hypothesis_id(base.NAME, base.CRITERIA, base.WINDOW))
    if not legacy or not legacy["result"]:
        return schedules.OUTCOME_DISCOVERY_KEY
    slot = next_slot()
    return f"{schedules.OUTCOME_PROGRAMME_PREFIX}{slot}" if slot is not None else None


def _measure(connection, days, symbols, spec):
    from scipy.stats import spearmanr, t

    ordinals = {day: i for i, day in enumerate(days)}
    by_day = defaultdict(list)
    count = 0
    for count, (symbol, day, raw_features, raw_outcome) in enumerate(connection.execute(
        "SELECT symbol, day, features, outcome FROM geometry_observations WHERE day >= ? AND day < ? "
        "ORDER BY day,symbol LIMIT ?", (base.START, base.END, base.ROW_BUDGET + 1),
    ), 1):
        if (count > base.ROW_BUDGET or symbol not in symbols or day not in ordinals
                or ordinals[base.END] - ordinals[day] <= base.HORIZON):
            raise ValueError("Discovery observation outside its frozen scope")
        f, o = json.loads(raw_features), json.loads(raw_outcome)
        turnover, momentum, volatility = (float(f[k]) for k in ("turnover_20", "momentum_20", "volatility_20"))
        if not all(math.isfinite(x) for x in (turnover, momentum, volatility)):
            raise ValueError("Nonfinite formation feature")
        if turnover < base.MIN_TURNOVER:
            continue
        score = {"turnover": turnover, "momentum": momentum, "risk_adjusted_momentum":
                 momentum / max(volatility, 0.001)}[spec.feature]
        target = o["next_open_gap" if spec.target == "next_open_gap" else "open_to_close_return"]
        if target is not None and not math.isfinite(target):
            raise ValueError("Nonfinite observed payoff")
        by_day[day].append((score, symbol, target, momentum, bool(f["signals"])))

    gap = spec.target == "next_open_gap"
    cells, sensitivity, payoff, benchmark, correlation = {}, {}, [], [], []
    quantiles = [[] for _ in range(5)]
    regimes, periods = defaultdict(list), defaultdict(list)
    names = no_signal = excluded = missing = 0
    for day, entries in by_day.items():
        ranked = sorted(entries, key=lambda row: (row[0], row[1]))
        n = len(ranked)
        if n < base.MIN_NAMES:
            excluded += 1
            continue
        selected = ranked[4*n//5:]
        other = ranked[:4*n//5]
        measured = [r for r in ranked if r[2] is not None]
        picked = [r[2] for r in selected if r[2] is not None]
        missing += n - len(measured)
        if len(measured) / n < MIN_COVERAGE or len(picked) / len(selected) < MIN_COVERAGE:
            excluded += 1
            continue
        best, basket = mean(picked), mean(r[2] for r in measured)
        ordinal = ordinals[day]
        excess = best - basket
        cells[ordinal] = (excess, 1)
        # Explicit unfavorable scenario, not a bound or recovered price.
        selected_stress = (sum(picked) - 0.20 * (len(selected) - len(picked))) / len(selected)
        basket_stress = (selected_stress * len(selected) + sum(r[2] if r[2] is not None else 0.20 for r in other)) / n
        sensitivity[ordinal] = (selected_stress - basket_stress, 1)
        payoff.append(best if gap else best - COST)
        benchmark.append(basket if gap else basket - COST)
        for i, bucket in enumerate(quantiles):
            values = [r[2] for r in ranked[n*i//5:n*(i+1)//5] if r[2] is not None]
            if values:
                bucket.append(mean(values) - (0 if gap else COST))
        scores, targets = [r[0] for r in measured], [r[2] for r in measured]
        if len(set(scores)) > 1 and len(set(targets)) > 1:
            correlation.append(float(spearmanr(scores, targets).statistic))
        regimes["nonnegative" if median(r[3] for r in ranked) >= 0 else "negative"].append(excess)
        periods[day[:4]].append(excess)
        names += len(measured)
        no_signal += sum(not r[4] for r in measured)

    horizon = 1 if gap else base.HORIZON
    inference = overlap_robust_test(cells, horizon)
    conservative = overlap_robust_test(sensitivity, horizon)
    lower = None
    if inference["status"] == conservative["status"] == "OK":
        lower = min(item["mean"] - float(t.ppf(1 - 0.05 / BUDGET, item["df"])) * item["se"]
                    for item in (inference, conservative))
    return {
        "observations_read": count, "eligible_observations_measured": names, "without_scanner_signals": no_signal,
        "decision_sessions": len(cells), "excluded_dates": excluded, "missing_outcomes": missing,
        "expected_net_return": mean(payoff) if payoff and not gap else None,
        "equal_weight_net_return": mean(benchmark) if benchmark and not gap else None,
        "net_return_stress": mean(payoff) + COST - STRESS_COST if payoff and not gap else None,
        "quintiles": [mean(values) if values else None for values in quantiles],
        "formation_return_rank_correlation": mean(correlation) if correlation else None,
        "regime_excess": {k: {"sessions": len(v), "mean_excess": mean(v)} for k, v in regimes.items()},
        "year_excess": {k: {"sessions": len(v), "mean_excess": mean(v)} for k, v in periods.items()},
        "excess_inference": inference, "adverse_censoring_scenario": conservative,
        "economic_lower_bound": lower, "detectable_excess_80pct_power_approx":
            2.97 * inference["se"] if inference["se"] else None,
    }


def _advance(slot, connection, days, symbols, legacy_row, source_hash, verify_source):
    """Plan and commit one step while the caller holds the scratch writer lock."""
    if not 1 <= slot <= BUDGET:
        raise ValueError("Discovery slot outside the cumulative budget")
    legacy = json.loads(legacy_row["result"])
    if source_hash != legacy["source_sha256"]:
        raise ValueError("Programme requires the preserved reversal snapshot")
    entries = rows()
    complete = {w["family"]: json.loads(r["result"]) for r, w in entries if r["result"]}
    existing = next(((r, w) for r, w in entries if w["slot"] == slot), None)
    if existing and existing[0]["result"]:
        return {"decision": "DISCOVERY_ALREADY_COMPLETED", **json.loads(existing[0]["result"])}
    if slot != next_slot():
        raise ValueError("Programme requires the earliest unspent slot")
    options = hypotheses(legacy, complete)
    blocked = {spec.family: spec.requires for _, spec in options if spec.requires}
    if existing:
        row, window = existing
        hid = row["hypothesis_id"]
        if window.get("source_sha256") != source_hash:
            raise ValueError("Registered programme snapshot changed")
        spec = Specification(**window["specification"])
    else:
        spec = next(spec for _, spec in options if not spec.requires and spec.family not in complete)
        ordinals = {day: i for i, day in enumerate(days)}
        horizon = 1 if spec.target == "next_open_gap" else base.HORIZON
        decision_days = [ordinals[row[0]] for row in connection.execute(
            "SELECT DISTINCT day FROM geometry_observations WHERE day >= ? AND day < ? ORDER BY day",
            (base.START, base.END),
        )]
        blocks = 0
        last = -horizon
        for ordinal in decision_days:
            if ordinal >= last + horizon:
                blocks += 1
                last = ordinal
        window = {
            "source_sha256": source_hash, "exposure_horizon": horizon,
            "potential_nonoverlapping_blocks": blocks,
            "stage": "DISCOVERY", "programme": PROGRAMME, "family": spec.family, "slot": slot,
            "specification": asdict(spec), "start": base.START, "end_exclusive": base.END,
            "minimum_coverage": MIN_COVERAGE, "benchmark": "same eligible basket and idle cash",
            "formation": "top quintile at close; no ranking on realized availability",
            "net_cost": COST, "stress_cost": STRESS_COST, "row_budget": base.ROW_BUDGET,
            "family_looks": 1, "cumulative_looks": BUDGET,
        }
        hid = registry.register_hypothesis(f"Outcome discovery · {spec.family}", CRITERIA, window,
                                          spec.rationale, code_hash=_code_hash())
        row = registry.get_experiment(hid)
    if row["code_hash"] != _code_hash():
        raise ValueError("Pending specification belongs to a different implementation")
    registry.start_discovery(hid)
    started = monotonic()
    try:
        metrics = _measure(connection, days, symbols, spec)
    except (ArithmeticError, KeyError, ValueError) as exc:
        if not verify_source():
            raise ValueError("Gauntlet snapshot changed during discovery") from exc
        result = {
            "hypothesis_id": hid, "status": "INCONCLUSIVE", "failure": str(exc),
            "source_sha256": source_hash, "code_sha256": _code_hash(),
            "evidence_class": "HISTORICAL_REPLAY", "promotion_eligible": False,
            "budget_used": slot, "budget_total": BUDGET,
            "blocked_dependencies": blocked, "next": "SELECT_NEXT_FEASIBLE_HYPOTHESIS" if slot < BUDGET
            else "AWAIT_DISTINCT_POINT_IN_TIME_EVIDENCE",
        }
        registry.record_discovery_result(hid, result)
        return {"decision": "DISCOVERY_FAILED", **result}
    if not verify_source():
        raise ValueError("Gauntlet snapshot changed during discovery")
    gap = spec.target == "next_open_gap"
    if gap or metrics["excess_inference"]["status"] != "OK":
        status = "INCONCLUSIVE"
    elif registry.meets_criteria(metrics, CRITERIA):
        status = "PROMISING"
    elif metrics["expected_net_return"] <= 0 and metrics["excess_inference"]["mean"] <= 0:
        status = "REJECTED"
    else:
        status = "INCONCLUSIVE"
    remaining = slot < BUDGET
    result = {
        "hypothesis_id": hid, "status": status, "source_sha256": source_hash, "code_sha256": _code_hash(),
        "evidence_class": "HISTORICAL_REPLAY", "promotion_eligible": False, "metrics": metrics,
        "elapsed_seconds": round(monotonic() - started, 3), "budget_used": slot, "budget_total": BUDGET,
        "blocked_dependencies": blocked,
        "next": "SELECT_NEXT_FEASIBLE_HYPOTHESIS" if remaining else "AWAIT_DISTINCT_POINT_IN_TIME_EVIDENCE",
        "validation": {"status": "INDEPENDENT_EVIDENCE_REQUIRED", "specification": window,
                       "protocol": "Pre-register next eligible prospective sessions, matched basket and cash, fixed costs and minimum 20 independent outcome blocks. Separate approval required."}
                      if status == "PROMISING" else None,
    }
    registry.record_discovery_result(hid, result)
    return {"decision": "DISCOVERY_COMPLETE", **result}


def _code_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + Path(base.__file__).read_bytes()).hexdigest()
