"""Learning Effectiveness Scorecard and graduation verdicts.

One read-only answer to: is QuantTerm making better decisions, and what stands
between it and real money? It composes evidence the system already records:

* tracked BUY outcomes (``signal_log``, the scanner's surfaced BUYs, settled on
  official bars by the one rulebook, ``core.evaluation_spec``);
* the forward ledger (paper TAKEN trades and settled counterfactual rejections,
  counted as independent decisions, ``forward_evidence.independent_decisions``);
* the learning-policy ladder (what, if anything, now changes selection);
* the champion/challenger and ML challenger stores;
* the live interlock and the signal-outcome resolution report.

Statistical unit. Signals logged on the same session share one market path, so
each session's mean net R is ONE observation; the time-ordered series of
session means goes to ``research.harness.block_bootstrap_mean_ci`` (stationary
bootstrap, serial dependence). No edge claim is made below ``MIN_SESSIONS`` or
``MIN_N``. Costs are ``core.costs`` (delivery taxes/fees + slippage) in R.

Nothing here writes to a ledger or changes a decision. Live stays LOCKED.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1

INSUFFICIENT = "INSUFFICIENT EVIDENCE"
CREDIBLE_POSITIVE = "CREDIBLE_POSITIVE"
CREDIBLE_NEGATIVE = "CREDIBLE_NEGATIVE"
NOT_DISTINGUISHABLE = "NOT_DISTINGUISHABLE_FROM_ZERO"

# <30 = no claim is the system's standing evidence invariant (CLAUDE.md #6).
MIN_N = 30
# Same reasoning as learning_policy_store.MIN_SESSIONS_ELIGIBLE: ~4
# non-overlapping 10-session horizons before any direction is called.
MIN_SESSIONS = 20
# Improvement/deterioration needs two independent halves that each clear the floor.
MIN_SESSIONS_TREND = 2 * MIN_SESSIONS
# Forward-paper graduation: matches forward_soak live-readiness (100 trades /
# 40 trading days) and is then judged by the CI, not the count alone.
FORWARD_MIN_TRADES = 100
FORWARD_MIN_SESSIONS = 40
# Calibration requires the prediction to be recorded with the outcome.
MIN_CALIBRATION_N = 100
_Z_ALPHA_ONE_SIDED, _Z_POWER = 1.6449, 0.8416  # alpha 0.05 one-sided, power 0.80
# R is defined only for stops >= 1% of entry. Round-trip cost is ~0.32%
# (core.costs), so a tighter stop spends > 0.32R on friction alone and its R is
# a division artefact: in July 2026, 94 liquid/money-market ETF "BUYs" with
# ~0.1% stops turned 0.5% moves into -5R. They stay in the net-% series and are
# counted and named, never silently dropped.
MIN_RISK_FRAC_FOR_R = 0.01
# Maturation. A signal counts only once its whole outcome horizon has passed;
# otherwise the newest sessions include only trades that resolved EARLY (fast
# stops/targets) - a maturation bias. "Passed" is measured in real NSE sessions
# (exchange calendar, holidays included) AND requires a verified official
# bhavcopy on disk for each of them; anything unprovable is NOT_YET_VERIFIED.
MATURED, IMMATURE, NOT_YET_VERIFIED = "MATURED", "IMMATURE", "NOT_YET_VERIFIED"


# ── small statistics ─────────────────────────────────────────────────────────

def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _std(xs: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """P(a random winner scores above a random loser). 0.5 = no skill."""
    pos = sum(1 for y in labels if y)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _rank(list(scores))
    rank_sum = sum(r + 1 for r, y in zip(ranks, labels) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def session_series_verdict(session_means: Sequence[float], n_obs: int) -> dict[str, Any]:
    """CI for the mean of per-session means, respecting serial dependence."""
    sessions = len(session_means)
    out: dict[str, Any] = {
        "sessions": sessions,
        "n": int(n_obs),
        "mean_of_session_means_R": None if not sessions else round(sum(session_means) / sessions, 4),
        "ci95_R": None,
        "verdict": INSUFFICIENT,
        "minimum_detectable_edge_R": None,
        "sessions_needed_for_0_10R": None,
    }
    if sessions < 2:
        return out
    from research.harness import block_bootstrap_mean_ci, min_samples_for_edge

    ci = block_bootstrap_mean_ci(list(session_means), n_boot=4000, seed=20261118)
    out["ci95_R"] = [round(float(ci["ci_lower"]), 4), round(float(ci["ci_upper"]), 4)]
    out["effective_sessions"] = round(float(ci.get("n_eff") or sessions), 1)
    sd = _std(session_means) or 0.0
    if sd > 0:
        out["minimum_detectable_edge_R"] = round((_Z_ALPHA_ONE_SIDED + _Z_POWER) * sd / math.sqrt(sessions), 4)
        out["sessions_needed_for_0_10R"] = int(min_samples_for_edge(0.10, sd))
    if sessions < MIN_SESSIONS or n_obs < MIN_N:
        return out
    if ci["ci_lower"] > 0:
        out["verdict"] = CREDIBLE_POSITIVE
    elif ci["ci_upper"] < 0:
        out["verdict"] = CREDIBLE_NEGATIVE
    else:
        out["verdict"] = NOT_DISTINGUISHABLE
    return out


# ── tracked BUY outcomes (signal_log) ────────────────────────────────────────

def load_signal_rows(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    if db_path is None:
        from core.signal_outcome_tracker import _DB_PATH as db_path  # type: ignore[no-redef]
    path = Path(str(db_path))
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM signal_log")]
    finally:
        conn.close()


def _settled_trade(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """A filled, settled BUY with usable risk geometry, in gross and net R."""
    from core.costs import net_r

    if row.get("worked") not in (0, 1) or row.get("outcome_pct") is None:
        return None
    try:
        entry = float(row.get("entry_price") or 0)
        stop = float(row.get("stop_price") or 0)
        pct = float(row["outcome_pct"])
    except (TypeError, ValueError):
        return None
    if entry <= 0 or stop <= 0 or stop >= entry:
        return None
    from core.costs import round_trip_cost_pct

    risk_frac = (entry - stop) / entry
    gross = (pct / 100.0) / risk_frac
    return {
        "risk_frac": risk_frac,
        "r_defined": risk_frac >= MIN_RISK_FRAC_FOR_R,
        "net_pct": pct - round_trip_cost_pct("CNC"),
        "session": str(row.get("logged_at") or "")[:10],
        "symbol": str(row.get("symbol") or ""),
        "archetype": str(row.get("archetype") or "") or "UNLABELLED",
        "regime": str(row.get("regime") or "") or "UNLABELLED",
        "quality_score": row.get("quality_score"),
        "won": int(row["worked"]) == 1,
        "gross_R": gross,
        "net_R": net_r(gross, risk_frac),
        "pct": pct,
    }


def _group_stats(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Net-R statistics over trades with defined R (see MIN_RISK_FRAC_FOR_R)."""
    trades = [t for t in trades if t.get("r_defined", True)]
    by_session: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_session[t["session"]].append(float(t["net_R"]))
    sessions = sorted(by_session)
    means = [sum(by_session[s]) / len(by_session[s]) for s in sessions]
    out = session_series_verdict(means, len(trades))
    wins = sum(1 for t in trades if t["won"])
    out["hit_rate"] = round(wins / len(trades), 4) if trades else None
    out["hit_rate_ci95"] = wilson_interval(wins, len(trades))
    out["mean_gross_R_naive"] = None if not trades else round(_mean([t["gross_R"] for t in trades]), 4)
    out["mean_net_R_naive"] = None if not trades else round(_mean([t["net_R"] for t in trades]), 4)
    pos = sum(t["net_R"] for t in trades if t["net_R"] > 0)
    neg = -sum(t["net_R"] for t in trades if t["net_R"] < 0)
    out["profit_factor_net"] = None if neg <= 0 else round(pos / neg, 3)
    eq = peak = dd = 0.0
    for m in means:  # equal-weight book: one unit of risk per session
        eq += m
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    out["max_drawdown_R_equal_weight_sessions"] = round(dd, 4) if means else None
    out["first_session"] = sessions[0] if sessions else None
    out["last_session"] = sessions[-1] if sessions else None
    return out


def maturity_checker(*, verified_sessions=None, holidays=None, last_completed=None, horizon: int | None = None):
    """fn(logged_day 'YYYY-MM-DD') -> MATURED | IMMATURE | NOT_YET_VERIFIED.

    MATURED: the first ``horizon`` NSE sessions after the signal day have all
    completed AND each has a verified official bhavcopy on disk. IMMATURE: the
    calendar says they have not all completed yet. NOT_YET_VERIFIED: they
    should have completed but a verified bar is missing, or the calendar /
    verified-dates source is unavailable - never assumed matured.
    """
    from datetime import date as _date, timedelta

    try:
        if horizon is None:
            from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS as horizon  # type: ignore[no-redef]
        from research.intelligence.data import nse_calendar as CAL

        if holidays is None:
            holidays = CAL.load_holidays()
        if last_completed is None:
            last_completed = CAL.completed_session(CAL._now_ist(), holidays)
        if verified_sessions is None:
            from data.bhavcopy_store import _dates_on_disk
            verified_sessions = _dates_on_disk()
        verified = {d if isinstance(d, _date) else _date.fromisoformat(str(d)[:10]) for d in verified_sessions}
        is_session = CAL.is_session
    except Exception:
        return lambda _day: NOT_YET_VERIFIED
    if not verified or last_completed is None:
        return lambda _day: NOT_YET_VERIFIED
    cache: dict[str, str] = {}

    def check(day: str) -> str:
        key = str(day)[:10]
        if key in cache:
            return cache[key]
        try:
            cur = _date.fromisoformat(key)
        except ValueError:
            cache[key] = NOT_YET_VERIFIED
            return NOT_YET_VERIFIED
        window: list = []
        for _ in range(int(horizon) * 3 + 30):       # bounded walk over the calendar
            cur += timedelta(days=1)
            if is_session(cur, holidays):
                window.append(cur)
                if len(window) >= int(horizon):
                    break
        if len(window) < int(horizon) or window[-1] > last_completed:
            status = IMMATURE
        elif all(d in verified for d in window):
            status = MATURED
        else:
            status = NOT_YET_VERIFIED
        cache[key] = status
        return status

    return check


def tracked_signal_evidence(rows: Iterable[Mapping[str, Any]], *, maturity=None) -> dict[str, Any]:
    rows = list(rows)
    total_rows = len(rows)
    check = maturity or maturity_checker()
    statuses = [check(str(r.get("logged_at") or "")[:10]) for r in rows]
    counts = {k: statuses.count(k) for k in (MATURED, IMMATURE, NOT_YET_VERIFIED)}
    rows = [r for r, st in zip(rows, statuses) if st == MATURED]
    trades = [t for t in (_settled_trade(r) for r in rows) if t is not None]
    never_filled = sum(1 for r in rows if r.get("worked") == -1)
    open_rows = sum(1 for r in rows if r.get("worked") is None)
    overall = _group_stats(trades)
    tight = [t for t in trades if not t["r_defined"]]
    tight_symbols: dict[str, int] = defaultdict(int)
    for t in tight:
        tight_symbols[t["symbol"]] += 1
    pct_by_session: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        pct_by_session[t["session"]].append(float(t["net_pct"]))
    pct_sessions = sorted(pct_by_session)
    net_pct = session_series_verdict(
        [sum(pct_by_session[s]) / len(pct_by_session[s]) for s in pct_sessions], len(trades),
    )
    net_pct = {("mean_of_session_means_pct" if k == "mean_of_session_means_R" else
                "ci95_pct" if k == "ci95_R" else k): v
               for k, v in net_pct.items()
               if k not in {"minimum_detectable_edge_R", "sessions_needed_for_0_10R"}}
    breakdown: dict[str, dict[str, Any]] = {}
    for dim in ("archetype", "regime"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in trades:
            groups[t[dim]].append(t)
        breakdown[dim] = {
            key: {k: v for k, v in _group_stats(ts).items() if k in {
                "n", "sessions", "mean_of_session_means_R", "ci95_R", "verdict", "hit_rate",
            }}
            for key, ts in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:15]
        }
    unlabelled_regime = sum(1 for t in trades if t["regime"] == "UNLABELLED")
    return {
        "source": "signal_log (surfaced BUYs, official-bar settlement, core.evaluation_spec)",
        "rows_logged": total_rows,
        "rows_matured": len(rows),
        "maturity": counts,
        "maturity_rule": ("first horizon NSE sessions after the signal completed AND each has a verified "
                          "official bhavcopy; otherwise IMMATURE or NOT_YET_VERIFIED (excluded)"),
        "settled_filled": len(trades),
        "never_filled": never_filled,
        "open_or_unresolved": open_rows,
        "fill_rate": None if not (len(trades) + never_filled) else round(len(trades) / (len(trades) + never_filled), 4),
        "overall_net": overall,
        "overall_net_pct_all_filled": net_pct,
        "excluded_from_R_tight_stop": {
            "n": len(tight),
            "rule": f"stop < {MIN_RISK_FRAC_FOR_R:.0%} of entry: costs alone exceed ~0.32R",
            "symbols": dict(sorted(tight_symbols.items(), key=lambda kv: -kv[1])[:15]),
        },
        "by_archetype": breakdown["archetype"],
        "by_regime": breakdown["regime"],
        "regime_label_coverage": None if not trades else round(1 - unlabelled_regime / len(trades), 4),
        "cost_model": "core.costs round_trip_cost_pct('CNC') in R per trade",
        "_trades": trades,
    }


def score_calibration(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Does quality_score rank winners above losers, and is it a probability?"""
    pairs = [(float(t["quality_score"]), 1 if t["won"] else 0) for t in trades if t.get("quality_score") is not None]
    out: dict[str, Any] = {
        "n": len(pairs),
        "auc": None,
        "brier_score_as_probability": None,
        "brier_base_rate": None,
        "brier_skill": None,
        "reliability": [],
        "verdict": INSUFFICIENT,
        "note": (
            "quality_score is a heuristic 0-100 setup score. A score of 85 is not an 85% "
            "win probability unless this table says so."
        ),
    }
    if len(pairs) < MIN_CALIBRATION_N:
        return out
    scores = [s for s, _ in pairs]
    labels = [y for _, y in pairs]
    base = sum(labels) / len(labels)
    probs = [min(1.0, max(0.0, s / 100.0)) for s in scores]
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(pairs)
    brier_ref = sum((base - y) ** 2 for y in labels) / len(pairs)
    a = auc(scores, labels)
    out.update({
        "auc": None if a is None else round(a, 4),
        "base_rate": round(base, 4),
        "brier_score_as_probability": round(brier, 4),
        "brier_base_rate": round(brier_ref, 4),
        "brier_skill": None if brier_ref <= 0 else round(1 - brier / brier_ref, 4),
    })
    ordered = sorted(pairs)
    bins = 5
    for i in range(bins):
        seg = ordered[i * len(ordered) // bins:(i + 1) * len(ordered) // bins]
        if not seg:
            continue
        k = sum(y for _, y in seg)
        out["reliability"].append({
            "score_range": [round(seg[0][0], 1), round(seg[-1][0], 1)],
            "n": len(seg),
            "mean_score_as_prob": round(sum(s for s, _ in seg) / len(seg) / 100.0, 4),
            "realized_win_rate": round(k / len(seg), 4),
            "win_rate_ci95": wilson_interval(k, len(seg)),
        })
    if a is not None and a < 0.55:
        out["verdict"] = "NO_USEFUL_DISCRIMINATION"
    elif out["brier_skill"] is not None and out["brier_skill"] < 0:
        out["verdict"] = "RANKS_BUT_NOT_A_PROBABILITY"
    else:
        out["verdict"] = "MEASURED"
    return out


def trend(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_session: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_session[t["session"]].append(float(t["net_R"]))
    sessions = sorted(by_session)
    out: dict[str, Any] = {"sessions": len(sessions), "verdict": INSUFFICIENT,
                           "required_sessions": MIN_SESSIONS_TREND}
    if len(sessions) < MIN_SESSIONS_TREND:
        return out
    half = len(sessions) // 2
    early = [sum(by_session[s]) / len(by_session[s]) for s in sessions[:half]]
    late = [sum(by_session[s]) / len(by_session[s]) for s in sessions[half:]]
    diff = _mean(late) - _mean(early)
    se = math.sqrt((_std(early) or 0) ** 2 / len(early) + (_std(late) or 0) ** 2 / len(late))
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    out.update({
        "early_mean_R": round(_mean(early), 4), "late_mean_R": round(_mean(late), 4),
        "difference_R": round(diff, 4), "difference_ci95_R": [round(lo, 4), round(hi, 4)],
        "verdict": "IMPROVING" if lo > 0 else "DETERIORATING" if hi < 0 else "NO_CREDIBLE_CHANGE",
    })
    return out


# ── forward ledger: paper trades + counterfactual rejections ─────────────────

def forward_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from product.forward_evidence import REAL_FORWARD_MARKET, independent_decisions

    real = [dict(r) for r in rows if str(r.get("provenance") or "") == REAL_FORWARD_MARKET]
    units = independent_decisions(real)
    taken = [r for r in units if r.get("entered")]
    settled_taken = [r for r in taken if r.get("gross_R") is not None or r.get("execution_adjusted_R") is not None]
    rejected = [r for r in units if not r.get("entered")]
    settled_rej = [r for r in rejected if r.get("counterfactual_classification")]
    by_reason: dict[str, dict[str, Any]] = {}
    for r in settled_rej:
        reason = str(r.get("reason_code") or "UNSPECIFIED")
        b = by_reason.setdefault(reason, {"n": 0, "missed_winner": 0, "avoided_or_correct": 0,
                                          "forward_returns": [], "sessions": set()})
        b["n"] += 1
        cls = str(r.get("counterfactual_classification") or "")
        if cls == "MISSED_WINNER":
            b["missed_winner"] += 1
        if cls in {"AVOIDED_LOSER", "CORRECT_REJECTION"}:
            b["avoided_or_correct"] += 1
        fr = (r.get("later_outcome") or {}).get("forward_return_pct")
        if fr is not None:
            b["forward_returns"].append(float(fr))
        b["sessions"].add(str(r.get("market_timestamp") or "")[:10])
    gates = {}
    for reason, b in sorted(by_reason.items(), key=lambda kv: -kv[1]["n"]):
        fr = b["forward_returns"]
        sessions = len(b["sessions"])
        gates[reason] = {
            "independent_n": b["n"],
            "sessions": sessions,
            "missed_winner_rate": round(b["missed_winner"] / b["n"], 4),
            "missed_winner_rate_ci95": wilson_interval(b["missed_winner"], b["n"]),
            "avoided_or_correct_rate": round(b["avoided_or_correct"] / b["n"], 4),
            "mean_forward_return_pct_5s": None if not fr else round(_mean(fr), 3),
            "evidence": INSUFFICIENT if (b["n"] < MIN_N or sessions < MIN_SESSIONS) else "MEASURED",
        }
    gross = [float(r["gross_R"]) for r in settled_taken if r.get("gross_R") is not None]
    adj = [float(r["execution_adjusted_R"]) for r in settled_taken if r.get("execution_adjusted_R") is not None]
    trade_sessions = {str(r.get("market_timestamp") or "")[:10] for r in settled_taken}
    return {
        "ledger_rows": len(real),
        "independent_decisions": len(units),
        "paper_trades_taken": len(taken),
        "paper_trades_settled": len(settled_taken),
        "paper_trade_sessions": len(trade_sessions),
        "paper_mean_gross_R": None if not gross else round(_mean(gross), 4),
        "paper_mean_execution_adjusted_R": None if not adj else round(_mean(adj), 4),
        "execution_adjusted_coverage": None if not settled_taken else round(len(adj) / len(settled_taken), 4),
        "paper_evidence": INSUFFICIENT if len(settled_taken) < MIN_N else "MEASURED",
        "rejections_settled_independent": len(settled_rej),
        "rejection_gates": gates,
        "counterfactual_note": "Rejections are hypothetical, not P&L; 5-session close-to-close.",
    }


# ── learning influence & challengers ─────────────────────────────────────────

def learning_influence(policies: Sequence[Mapping[str, Any]], forward_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from product.forward_evidence import independent_decisions

    by_status: dict[str, int] = defaultdict(int)
    influencing = []
    for p in policies:
        status = str(p.get("production_status") or "OBSERVING")
        by_status[status] += 1
        if status in {"ELIGIBLE", "ACTIVE"} and p.get("affects_selection") is not False:
            influencing.append({
                "policy_id": p.get("policy_id"), "status": status,
                "n": p.get("sample_size"), "sessions": p.get("sessions_observed"),
                "edge_R": p.get("expectancy_difference_R"),
            })
    effects: dict[str, int] = defaultdict(int)
    for r in independent_decisions(forward_rows):
        effects[str(r.get("policy_effect") or "NEUTRAL")] += 1
    changed = sum(v for k, v in effects.items() if k not in {"NEUTRAL", ""})
    return {
        "policies_by_status": dict(by_status),
        "policies_influencing_selection": influencing,
        "decisions_by_policy_effect": dict(effects),
        "decisions_changed_by_learning": changed,
        "answer": (
            "YES" if changed and influencing else
            "NO: no learned policy currently clears the sample, edge and session floors"
            if not influencing else "POLICIES ELIGIBLE BUT NO DECISION CHANGED YET"
        ),
    }


def challenger_evidence() -> dict[str, Any]:
    out: dict[str, Any] = {"champion_challenger": None, "ml_challenger": None}
    try:
        from product.champion_challenger import load_store

        store = load_store()
        ch = list(store.get("challengers") or [])
        out["champion_challenger"] = {
            "champion": (store.get("champion") or {}).get("rules_hash") or (store.get("champion") or {}).get("name"),
            "challengers": len(ch),
            "with_forward_evidence": sum(1 for c in ch if int(c.get("forward_n") or 0) >= MIN_N),
        }
    except Exception as exc:
        out["champion_challenger"] = {"error": type(exc).__name__}
    try:
        from core.runtime_paths import logs_path

        cur = (json.loads(logs_path("product", "challenger_learning.json").read_text()) or {}).get("current") or {}
        medians = list(cur.get("medians") or [])
        scales = list(cur.get("scales") or [])
        coef = list(cur.get("coef") or [])
        dead = sum(1 for m, s, c in zip(medians, scales, coef) if float(m) == 0.0 and float(s) == 1.0 and float(c) == 0.0)
        fwd = cur.get("forward_validation") or {}
        out["ml_challenger"] = {
            "model_version": cur.get("model_version"),
            "status": cur.get("status"),
            "historical_n": cur.get("historical_n"),
            "oof_brier": (cur.get("validation") or {}).get("challenger_brier"),
            "features": len(coef),
            "features_constant_zero": dead,
            "real_forward_n": int(cur.get("real_forward_n") or 0),
            "beats_champion_forward": fwd.get("improvement_lower_95") is not None and float(fwd["improvement_lower_95"]) > 0,
            "verdict": INSUFFICIENT if int(cur.get("real_forward_n") or 0) < MIN_N else "MEASURED",
        }
    except Exception:
        out["ml_challenger"] = None
    return out


# ── graduation ───────────────────────────────────────────────────────────────

def graduation(card: Mapping[str, Any], *, institutional: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Five verdicts, each with the exact unmet conditions. Nothing is READY on vibes."""
    sig = (card.get("tracked_signals") or {}).get("overall_net") or {}
    fwd = card.get("forward") or {}
    cal = card.get("calibration") or {}
    live = card.get("live_interlock") or {}
    domains = {d.get("key"): d for d in (institutional or {}).get("domains", []) or []}

    def domain_ok(key: str) -> bool | None:
        if not domains:
            return None
        return str((domains.get(key) or {}).get("status") or "") == "READY"

    research_unmet = []
    if sig.get("verdict") != CREDIBLE_POSITIVE:
        research_unmet.append(
            f"tracked BUYs: net edge {sig.get('verdict')} over {sig.get('sessions')} sessions "
            f"(mean {sig.get('mean_of_session_means_R')}R, CI {sig.get('ci95_R')}); needs CI lower bound > 0 "
            f"with >= {MIN_SESSIONS} sessions"
        )
    if domain_ok("data") is not True:
        research_unmet.append("point-in-time data controls not certified (institutional 'data' domain)"
                              if domains else "point-in-time data controls: not in the nightly snapshot (live: /api/institutional-readiness)")
    if cal.get("verdict") not in {"MEASURED"}:
        research_unmet.append(f"setup score calibration: {cal.get('verdict')} (AUC {cal.get('auc')})")

    forward_unmet = []
    if int(fwd.get("paper_trades_settled") or 0) < FORWARD_MIN_TRADES:
        forward_unmet.append(f"settled paper trades {fwd.get('paper_trades_settled')} < {FORWARD_MIN_TRADES}")
    if int(fwd.get("paper_trade_sessions") or 0) < FORWARD_MIN_SESSIONS:
        forward_unmet.append(f"paper trade sessions {fwd.get('paper_trade_sessions')} < {FORWARD_MIN_SESSIONS}")
    if fwd.get("execution_adjusted_coverage") is None or float(fwd.get("execution_adjusted_coverage") or 0) < 0.8:
        forward_unmet.append("execution-adjusted outcome coverage < 80%")
    if research_unmet:
        forward_unmet.append("Stage 1 (research validated) not met")

    ops_unmet = []
    for key in ("operations", "reconciliation", "execution", "protection", "risk"):
        ok = domain_ok(key)
        if ok is not True:
            ops_unmet.append(
                f"institutional '{key}' domain not READY" if ok is False else
                f"institutional '{key}' domain not in the nightly snapshot (live: /api/institutional-readiness)"
            )
    pipeline = card.get("outcome_pipeline") or {}
    if pipeline.get("stuck_past_horizon"):
        ops_unmet.append(f"{pipeline['stuck_past_horizon']} BUY outcomes stuck past horizon")
    ops_unmet.append("host soak / fault-injection / restore drill: BLOCKED (requires host evidence, not CI)")

    sup_unmet = ["Stages 1-3 not met"] if (research_unmet or forward_unmet or ops_unmet) else []
    sup_unmet.append("separate explicit owner authorization for supervised micro-capital")
    if live.get("locked") is not False:
        sup_unmet.append("canonical live interlock is LOCKED (by design)")
    auto_unmet = ["supervised micro-capital stage not completed", "real-fill analysis absent",
                  "second explicit owner authorization"]

    verdicts = {
        "RESEARCH READY": "READY" if not research_unmet else "NOT READY",
        "FORWARD PAPER READY": "READY" if not forward_unmet else "NOT READY",
        "OPERATIONALLY READY": "READY" if not ops_unmet else "NOT READY",
        "SUPERVISED LIVE ELIGIBLE": "NOT ELIGIBLE",
        "AUTONOMOUS LIVE ELIGIBLE": "NOT ELIGIBLE",
    }
    stage = 0
    for unmet in (research_unmet, forward_unmet, ops_unmet):
        if unmet:
            break
        stage += 1
    return {
        "current_stage": stage,
        "stage_names": {
            1: "Research Validated", 2: "Forward Paper Validated", 3: "Operationally Certified",
            4: "Supervised Micro-Capital Eligible", 5: "Limited Autonomous Live Eligible",
        },
        "verdicts": verdicts,
        "unmet": {
            "RESEARCH READY": research_unmet,
            "FORWARD PAPER READY": forward_unmet,
            "OPERATIONALLY READY": ops_unmet,
            "SUPERVISED LIVE ELIGIBLE": sup_unmet,
            "AUTONOMOUS LIVE ELIGIBLE": auto_unmet,
        },
        "criteria_rationale": {
            "sessions_not_rows": "Same-session signals share one market path; each session mean is one observation.",
            "min_sessions": f"{MIN_SESSIONS} sessions ~ 4 non-overlapping 10-session horizons before any direction is called.",
            "forward_paper": (
                f"{FORWARD_MIN_TRADES} trades / {FORWARD_MIN_SESSIONS} sessions match the forward-soak contract; "
                "the gate is the CI lower bound > 0, so a small edge needs more trades (see minimum_detectable_edge_R)."
            ),
            "ci_rule": "Promotion requires the dependence-aware CI to exclude zero, never a point estimate.",
        },
        "live_trading": "LOCKED",
    }


# ── assembly ─────────────────────────────────────────────────────────────────

def build_scorecard(
    *,
    signal_rows: Iterable[Mapping[str, Any]] | None = None,
    forward_rows: Sequence[Mapping[str, Any]] | None = None,
    policies: Sequence[Mapping[str, Any]] | None = None,
    institutional: Mapping[str, Any] | None = None,
    outcome_pipeline: Mapping[str, Any] | None = None,
    include_challengers: bool = True,
    maturity=None,
) -> dict[str, Any]:
    if signal_rows is None:
        signal_rows = load_signal_rows()
    if forward_rows is None:
        from product.forward_evidence import load_ledger
        forward_rows = load_ledger()
    if policies is None:
        from product.learning_policy_store import load_policies
        policies = list(load_policies().get("policies") or [])
    if outcome_pipeline is None:
        try:
            from core.runtime_paths import logs_path
            from core.signal_outcome_tracker import RESOLUTION_REPORT
            outcome_pipeline = json.loads(logs_path("product", RESOLUTION_REPORT).read_text())
        except Exception:
            outcome_pipeline = {"status": "NOT_YET_REPORTED"}
    tracked = tracked_signal_evidence(signal_rows, maturity=maturity)
    trades = tracked.pop("_trades")
    try:
        from product.live_execution_interlock import get_live_execution_state
        live = get_live_execution_state().as_dict()
    except Exception as exc:
        live = {"locked": True, "status": "UNVERIFIED", "error": type(exc).__name__}
    card: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracked_signals": tracked,
        "calibration": score_calibration(trades),
        "trend": trend(trades),
        "forward": forward_evidence(list(forward_rows)),
        "learning_influence": learning_influence(list(policies), list(forward_rows)),
        "challengers": challenger_evidence() if include_challengers else {},
        "outcome_pipeline": dict(outcome_pipeline or {}),
        "live_interlock": {"locked": live.get("locked", True), "status": live.get("status")},
    }
    sig = tracked["overall_net"]
    card["answers"] = {
        "1_matured_independent_decisions": {
            "tracked_buys_settled": tracked["settled_filled"], "tracked_buy_sessions": sig.get("sessions"),
            "paper_trades_settled": card["forward"]["paper_trades_settled"],
            "counterfactual_rejections_settled": card["forward"]["rejections_settled_independent"],
        },
        "2_accuracy_and_calibration": card["calibration"]["verdict"],
        "3_strategy_regime": "see tracked_signals.by_archetype / by_regime (each labelled)",
        "4_beat_champion_forward": (
            "YES" if (card["challengers"].get("ml_challenger") or {}).get("beats_champion_forward") else INSUFFICIENT
        ),
        "5_net_of_costs": sig.get("verdict"),
        "6_statistically_credible": sig.get("verdict") in {CREDIBLE_POSITIVE, CREDIBLE_NEGATIVE},
        "7_learning_changed_decisions": card["learning_influence"]["answer"],
        "8_improving_or_deteriorating": card["trend"]["verdict"],
    }
    card["graduation"] = graduation(card, institutional=institutional)
    return card


SNAPSHOT_FILE = "learning_scorecard.json"
HISTORY_FILE = "learning_scorecard_history.jsonl"


def compact(card: Mapping[str, Any], session: str) -> dict[str, Any]:
    """One history line per session: enough to plot whether decisions improve."""
    sig = (card.get("tracked_signals") or {}).get("overall_net") or {}
    pct = (card.get("tracked_signals") or {}).get("overall_net_pct_all_filled") or {}
    cal = card.get("calibration") or {}
    fwd = card.get("forward") or {}
    inf = card.get("learning_influence") or {}
    grad = card.get("graduation") or {}
    return {
        "session": str(session)[:10],
        "generated_at": card.get("generated_at"),
        "tracked_n": sig.get("n"),
        "tracked_sessions": sig.get("sessions"),
        "net_R_session_mean": sig.get("mean_of_session_means_R"),
        "net_R_ci95": sig.get("ci95_R"),
        "net_R_verdict": sig.get("verdict"),
        "net_pct_session_mean": pct.get("mean_of_session_means_pct"),
        "score_auc": cal.get("auc"),
        "score_brier_skill": cal.get("brier_skill"),
        "paper_trades_settled": fwd.get("paper_trades_settled"),
        "rejections_settled_independent": fwd.get("rejections_settled_independent"),
        "policies_influencing": len(inf.get("policies_influencing_selection") or []),
        "decisions_changed_by_learning": inf.get("decisions_changed_by_learning"),
        "stage": grad.get("current_stage"),
        "verdicts": grad.get("verdicts"),
    }


def persist_snapshot(card: Mapping[str, Any], session: str) -> dict[str, Any]:
    """Write the latest card and append one history line per session (idempotent)."""
    from core.atomic_io import atomic_write_text
    from core.runtime_paths import logs_path

    snap = logs_path("product", SNAPSHOT_FILE)
    snap.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(snap, json.dumps(card, indent=2, default=str))
    hist = logs_path("product", HISTORY_FILE)
    line = compact(card, session)
    seen = set()
    if hist.exists():
        for raw in hist.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(str(json.loads(raw).get("session")))
            except Exception:
                continue
    appended = line["session"] not in seen
    if appended:
        with hist.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
    return {"snapshot": str(snap), "history_appended": appended, "summary": line}


def load_snapshot() -> dict[str, Any] | None:
    from core.runtime_paths import logs_path

    try:
        return json.loads(logs_path("product", SNAPSHOT_FILE).read_text(encoding="utf-8"))
    except Exception:
        return None


def load_history(limit: int = 60) -> list[dict[str, Any]]:
    from core.runtime_paths import logs_path

    hist = logs_path("product", HISTORY_FILE)
    if not hist.exists():
        return []
    rows = []
    for raw in hist.read_text(encoding="utf-8").splitlines()[-int(limit):]:
        try:
            rows.append(json.loads(raw))
        except Exception:
            continue
    return rows


def home_summary(card: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Small Home section: never a green badge over unproven evidence."""
    card = card if card is not None else load_snapshot()
    if not card:
        return {
            "available": False,
            "headline": "Scorecard not computed yet: it runs after the next nightly learning cycle.",
            "live_trading": "LOCKED",
        }
    sig = (card.get("tracked_signals") or {}).get("overall_net") or {}
    cal = card.get("calibration") or {}
    fwd = card.get("forward") or {}
    inf = card.get("learning_influence") or {}
    grad = card.get("graduation") or {}
    top = (cal.get("reliability") or [{}])[-1] if cal.get("reliability") else {}
    verdict = sig.get("verdict") or INSUFFICIENT
    headline = (
        f"No proven edge yet: tracked BUYs {sig.get('mean_of_session_means_R')}R per session over "
        f"{sig.get('sessions')} sessions (95% CI {sig.get('ci95_R')}) - {verdict.lower()}."
        if verdict != CREDIBLE_POSITIVE
        else f"Tracked BUYs net positive: {sig.get('mean_of_session_means_R')}R per session (CI {sig.get('ci95_R')})."
    )
    return {
        "available": True,
        "generated_at": card.get("generated_at"),
        "headline": headline,
        "net_edge": {
            "verdict": verdict, "sessions": sig.get("sessions"), "n": sig.get("n"),
            "mean_R_per_session": sig.get("mean_of_session_means_R"), "ci95_R": sig.get("ci95_R"),
            "sessions_needed_for_0_10R": sig.get("sessions_needed_for_0_10R"),
        },
        "calibration": {
            "verdict": cal.get("verdict"), "auc": cal.get("auc"), "brier_skill": cal.get("brier_skill"),
            "top_bucket_score_as_prob": top.get("mean_score_as_prob"),
            "top_bucket_realized_win_rate": top.get("realized_win_rate"),
        },
        "paper_trades_settled": fwd.get("paper_trades_settled"),
        "decisions_changed_by_learning": inf.get("decisions_changed_by_learning"),
        "stage": grad.get("current_stage"),
        "verdicts": grad.get("verdicts") or {},
        "blockers": {k: list(v or [])[:2] for k, v in (grad.get("unmet") or {}).items()},
        "live_trading": "LOCKED",
    }


def refresh_snapshot(session: str) -> dict[str, Any]:
    """Nightly: rebuild from the ledgers and persist. Read-only on every ledger."""
    card = build_scorecard()
    return persist_snapshot(card, session)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Print the learning effectiveness scorecard as JSON.")
    ap.add_argument("--signal-db", default=None)
    args = ap.parse_args(argv)
    card = build_scorecard(signal_rows=load_signal_rows(args.signal_db) if args.signal_db else None)
    print(json.dumps(card, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
