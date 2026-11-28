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
# 20 distinct sessions is only 2 non-overlapping 10-session horizons, so it is a
# coverage floor, not an independence claim: a direction is called only by the
# overlap-robust interval (research.overlap_inference) on the session series.
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


def session_series_verdict(session_means: Sequence[float], n_obs: int,
                           sessions: Sequence[str] | None = None, *,
                           horizon: int | None = None) -> dict[str, Any]:
    """CI for the mean of per-session means, respecting OVERLAPPING outcome windows.

    Each session's mean is a 10-session outcome, so sessions fewer than ten
    apart share outcome days and consecutive blocks still share their
    boundary. Inference is research.overlap_inference on the sessions' NSE
    ordinals (strict calendar: an uncovered year is no verdict). A claim
    needs >= MIN_SESSIONS sessions, >= MIN_N observations AND >= 20
    independent 10-session blocks; anything less is INSUFFICIENT EVIDENCE.
    """
    count = len(session_means)
    out: dict[str, Any] = {
        "sessions": count,
        "n": int(n_obs),
        "mean_of_session_means_R": None if not count else round(sum(session_means) / count, 4),
        "ci95_R": None,
        "verdict": INSUFFICIENT,
        "minimum_detectable_edge_R": None,
        "sessions_needed_for_0_10R": None,
        "independent_blocks": 0,
    }
    if count < 2:
        return out
    if sessions is None or len(sessions) != count:
        out["reason"] = "SESSION_DATES_UNKNOWN"
        return out
    if horizon is None:
        from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS as horizon  # type: ignore[no-redef]
    from research.overlap_inference import METHOD, OK, cells_from_pairs, overlap_robust_test
    from research.intelligence.data.nse_calendar import load_holidays, strict_session_ordinals

    try:
        ordinals = strict_session_ordinals(list(sessions), load_holidays())
    except Exception:
        ordinals = None
    if ordinals is None:
        out["reason"] = "EXCHANGE_CALENDAR_UNAVAILABLE_OR_NOT_COVERED"
        return out
    rep = overlap_robust_test(
        cells_from_pairs((ordinals[str(d)[:10]], float(m)) for d, m in zip(sessions, session_means)),
        int(horizon))
    out.update({"independent_blocks": rep["independent_sessions"], "inference": METHOD,
                "variance_inflation": rep["variance_inflation"],
                "residual_dependence": rep["residual_dependence"]})
    if rep["ci95"] is not None:
        out["ci95_R"] = [round(float(rep["ci95"][0]), 4), round(float(rep["ci95"][1]), 4)]
    se = rep["se"]
    if se:
        mde = (_Z_ALPHA_ONE_SIDED + _Z_POWER) * float(se)
        out["minimum_detectable_edge_R"] = round(mde, 4)
        out["sessions_needed_for_0_10R"] = int(math.ceil(count * (mde / 0.10) ** 2))
    if count < MIN_SESSIONS or n_obs < MIN_N:
        out["reason"] = f"{count} sessions / {int(n_obs)} observations below the floor"
        return out
    if rep["status"] != OK:
        out["reason"] = rep["reason"]
        return out
    lo, hi = rep["ci95"]
    if lo > 0:
        out["verdict"] = CREDIBLE_POSITIVE
    elif hi < 0:
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
    out = session_series_verdict(means, len(trades), sessions)
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
    completed AND each has a verified official bhavcopy on disk. IMMATURE: not
    even a weekdays-only count could have completed them yet. NOT_YET_VERIFIED:
    everything else that is not provably matured - a verified bar is missing,
    the verified-dates source is unavailable, or the exchange holiday calendar
    is unavailable, malformed or does not cover every year the window touches
    (nse_calendar.calendar_gap). The calendar is never assumed: treating an
    uncovered year's weekdays as sessions would count holidays as outcome days.
    ``check.reasons`` counts why rows were NOT_YET_VERIFIED.
    """
    from collections import Counter
    from datetime import date as _date, timedelta

    reasons: Counter = Counter()

    def _closed(reason: str):
        def check(_day: str) -> str:
            reasons[reason] += 1
            return NOT_YET_VERIFIED
        check.reasons = reasons  # type: ignore[attr-defined]
        return check

    try:
        if horizon is None:
            from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS as horizon  # type: ignore[no-redef]
        from research.intelligence.data import nse_calendar as CAL

        if holidays is None:
            holidays = CAL.load_holidays()
        holidays = {str(x)[:10] for x in (holidays or ())}
        if last_completed is None:
            last_completed = CAL.completed_session(CAL._now_ist(), holidays)
        if verified_sessions is None:
            from data.bhavcopy_store import _dates_on_disk
            verified_sessions = _dates_on_disk()
        verified = {d if isinstance(d, _date) else _date.fromisoformat(str(d)[:10]) for d in verified_sessions}
    except Exception:
        return _closed("SOURCE_UNAVAILABLE")
    if not holidays:
        return _closed("EXCHANGE_CALENDAR_UNAVAILABLE")
    if not verified or last_completed is None:
        return _closed("VERIFIED_SESSIONS_UNAVAILABLE")
    # Official bars on days the calendar calls closed (weekends, holidays)
    # mean the calendar missed a session: unless the manifest documents it as
    # a special session (e.g. Muhurat), a window spanning it is not counted.
    import bisect
    try:
        special = CAL.documented_special_sessions()
    except Exception:
        special = set()
    extras = sorted(d for d in verified if not CAL.is_session(d, holidays) and d.isoformat() not in special)
    cache: dict[str, str] = {}

    def check(day: str) -> str:
        key = str(day)[:10]
        if key in cache:
            if cache[key] == NOT_YET_VERIFIED:
                reasons[_why[key]] += 1
            return cache[key]
        try:
            start = _date.fromisoformat(key)
        except ValueError:
            status, why = NOT_YET_VERIFIED, "UNPARSEABLE_DATE"
        else:
            earliest, weekdays = start, 0         # weekdays-only: the earliest the window can end
            while weekdays < int(horizon):
                earliest += timedelta(days=1)
                weekdays += earliest.weekday() < 5
            window = CAL.forward_sessions(start, int(horizon), holidays)
            if earliest > last_completed:
                status, why = IMMATURE, ""
            elif window is None:
                status, why = NOT_YET_VERIFIED, (CAL.calendar_gap(start, earliest, holidays)
                                                 or "EXCHANGE_CALENDAR_NOT_COVERED")
            elif window[-1] > last_completed:
                status, why = IMMATURE, ""
            elif not all(d in verified for d in window):
                status, why = NOT_YET_VERIFIED, "VERIFIED_BAR_MISSING"
            else:
                i = bisect.bisect_right(extras, start)
                if i < len(extras) and extras[i] <= window[-1]:
                    status, why = NOT_YET_VERIFIED, "CALENDAR_CONTRADICTED_BY_OFFICIAL_BARS"
                else:
                    status, why = MATURED, ""
        cache[key] = status
        if status == NOT_YET_VERIFIED:
            _why[key] = why
            reasons[why] += 1
        return status

    _why: dict[str, str] = {}
    check.reasons = reasons  # type: ignore[attr-defined]
    return check


def dedupe_decisions(rows: Iterable[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int]:
    """One tracked decision per (session, symbol): the earliest logged row.

    The same symbol surfaced twice on one session shares one outcome window;
    counting both would double that observation's weight in the evidence.
    """
    best: dict[tuple[str, str], Mapping[str, Any]] = {}
    order: list[tuple[str, str]] = []
    dropped = 0
    for r in rows:
        key = (str(r.get("logged_at") or "")[:10], str(r.get("symbol") or "").upper())
        if not key[1]:
            order.append(("", str(id(r))))
            best[order[-1]] = r
            continue
        if key in best:
            dropped += 1
            if str(r.get("logged_at") or "") < str(best[key].get("logged_at") or ""):
                best[key] = r
            continue
        best[key] = r
        order.append(key)
    return [best[k] for k in order], dropped


def _calendar_provenance() -> dict[str, Any]:
    try:
        from research.intelligence.data.nse_calendar import provenance_summary
        return provenance_summary()
    except Exception as exc:
        return {"error": type(exc).__name__}


def tracked_signal_evidence(rows: Iterable[Mapping[str, Any]], *, maturity=None) -> dict[str, Any]:
    rows, duplicates = dedupe_decisions(list(rows))
    total_rows = len(rows) + duplicates
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
        [sum(pct_by_session[s]) / len(pct_by_session[s]) for s in pct_sessions], len(trades), pct_sessions,
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
        "duplicate_decision_rows_dropped": duplicates,
        "rows_matured": len(rows),
        "maturity": counts,
        "maturity_not_verified_reasons": dict(getattr(check, "reasons", {}) or {}),
        "calendar_provenance": _calendar_provenance(),
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
    # Each half's error is overlap-robust (10-session windows); each half
    # needs half of the 20-block floor, so the comparison as a whole rests on
    # >= 20 independent blocks like every other claim.
    from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS
    from research.intelligence.data.nse_calendar import load_holidays, strict_session_ordinals
    from research.overlap_inference import MIN_INDEPENDENT_BLOCKS, _t_ppf, cells_from_pairs, overlap_robust_test

    ordinals = strict_session_ordinals(sessions, load_holidays())
    if ordinals is None:
        out["reason"] = "EXCHANGE_CALENDAR_UNAVAILABLE_OR_NOT_COVERED"
        return out
    half = len(sessions) // 2
    reps = []
    for part in (sessions[:half], sessions[half:]):
        reps.append(overlap_robust_test(
            cells_from_pairs((ordinals[s], sum(by_session[s]) / len(by_session[s])) for s in part),
            OUTCOME_HORIZON_SESSIONS, min_blocks=MIN_INDEPENDENT_BLOCKS // 2))
    early, late = reps
    out["independent_blocks"] = [early["independent_sessions"], late["independent_sessions"]]
    if early["mean"] is None or late["mean"] is None:
        return out
    diff = late["mean"] - early["mean"]
    out.update({"early_mean_R": round(early["mean"], 4), "late_mean_R": round(late["mean"], 4),
                "difference_R": round(diff, 4)})
    if early["status"] != "OK" or late["status"] != "OK":
        out["reason"] = early["reason"] or late["reason"]
        return out
    se = math.sqrt(early["se"] ** 2 + late["se"] ** 2)
    half_width = _t_ppf(0.975, min(early["df"], late["df"])) * se
    lo, hi = diff - half_width, diff + half_width
    out.update({
        "difference_ci95_R": [round(lo, 4), round(hi, 4)],
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
    from core.evaluation_spec import spec_tag
    from product.counterfactual_learning import is_rulebook
    all_settled_rej = [r for r in rejected if r.get("counterfactual_classification")]
    # One rulebook: only rejections judged exactly as taken trades are judged
    # (same horizon, stop/target path, costs) are comparable with them. Legacy
    # 5-session close-to-close rows are counted aside, never mixed in.
    settled_rej = [r for r in all_settled_rej if is_rulebook((r.get("later_outcome") or {}).get("eval_spec"))]
    legacy_rej = len(all_settled_rej) - len(settled_rej)
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
            "mean_forward_return_pct": None if not fr else round(_mean(fr), 3),
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
        "rejections_legacy_excluded": legacy_rej,
        "rejection_gates": gates,
        "counterfactual_eval_spec": spec_tag(),
        "counterfactual_note": ("Rejections are hypothetical, not P&L, judged as if taken under the same rulebook "
                                f"({spec_tag()}); {legacy_rej} legacy 5-session close-to-close rows are excluded."),
    }


# ── learning influence & challengers ─────────────────────────────────────────

#: Reason codes that mean the evidence/learning policy itself refused the entry.
BINDING_POLICY_REASONS = {
    "EVIDENCE_POLICY_BLOCK": "an active learned policy blocked the setup",
    "HISTORICAL_EVIDENCE_PENDING": "the history-first system gate (setup not yet reproduced in PIT history)",
}


def _was_ranked(row: Mapping[str, Any]) -> bool:
    """True when the decision reached the selection ranking: a policy's score effect could have moved it."""
    return bool(row.get("entered")) or str(row.get("reason_code") or "").upper() in {"NO_TRADE", "ENTER_NOW"}


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
    binding: dict[str, int] = defaultdict(int)
    recorded_not_binding = 0
    rank_adjusted = 0
    exploration = 0
    from product.forward_evidence import is_forward_evidence
    # Real forward decisions only, BEFORE dedupe: a BACKTEST / WALK_FORWARD row
    # never counts as a decision learning changed (cycle-2 review QT-C2-09).
    for r in independent_decisions([row for row in forward_rows if is_forward_evidence(row)]):
        # The forward ledger persists the effect as ``policy_result``
        # (forward_evidence.freeze_observation); ``policy_effect`` is the
        # in-memory decision field. Reading only the latter counted every
        # recorded BLOCK/PENALIZE as NEUTRAL (GPT cross-review A-10).
        effect = str(r.get("policy_result") or r.get("policy_effect") or "NEUTRAL")
        effects[effect] += 1
        reason = str(r.get("reason_code") or "")
        # The policy engine's final effect is stamped on EVERY candidate,
        # including those another gate already rejected; and in production
        # the BLOCK is the history-first SYSTEM gate, not a learned policy. A
        # decision was changed by LEARNING only when a learned policy was its
        # BINDING reason, or a learned policy moved its score (cycle 2
        # learning-effectiveness audit).
        if reason in BINDING_POLICY_REASONS:
            binding[reason] += 1
        elif effect == "EXPLORATION":
            exploration += 1
        elif effect in {"PENALIZE", "SUPPORT"} and _was_ranked(r):
            # A score adjustment can change selection only for a decision that
            # reached the ranking (taken, or eligible but not top of the cycle).
            rank_adjusted += 1
        elif effect not in {"NEUTRAL", ""}:
            recorded_not_binding += 1
    changed = binding.get("EVIDENCE_POLICY_BLOCK", 0) + rank_adjusted
    return {
        "policies_by_status": dict(by_status),
        "policies_influencing_selection": influencing,
        "decisions_by_policy_effect": dict(effects),
        "decisions_by_binding_policy_reason": dict(binding),
        "decisions_blocked_by_history_first_gate": binding.get("HISTORICAL_EVIDENCE_PENDING", 0),
        "decisions_with_effect_recorded_but_rejected_by_another_gate": recorded_not_binding,
        "decisions_rank_adjusted_by_learned_policy": rank_adjusted,
        "decisions_in_exploration_lane": exploration,
        "decisions_changed_by_learning": changed,
        "limits": ("A selection change through the active challenger adjustment, or a baseline-versus-learned "
                   "ranking, is not persisted on the forward row, so it cannot be counted here."),
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
        f"NO VALIDATED EDGE · tracked BUYs {sig.get('mean_of_session_means_R')}R per session over "
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
