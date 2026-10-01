"""Per-policy scorecards and Champion-vs-Challenger paired comparison.

Reuses research.harness.evaluate() -- QuantTerm's existing, statistically
rigorous PROMOTE/REJECT/UNDERPOWERED/INCONCLUSIVE gate -- as the scientific
verdict function, and scan.ev_engine.wilson_lb for confidence-aware win
rates. This module assembles the right R-stream and hands it to those real
functions; it invents no parallel statistics engine.

Every graded shadow decision carries `counterfactual_R` (from
product.counterfactual_learning.settle(), computed the same way whether the
policy selected or rejected the candidate: move / risk against the SAME
frozen entry/stop). That one field is reused for two different questions,
filtered differently:
  - realized_r_stream(): ENTER_NOW rows only -- "what this policy's own
    equity curve would show" (the input to harness.evaluate() below).
  - paired_comparison(): every row on snapshots BOTH policies saw -- "what
    would each policy's trade have returned here" -- which is what makes
    section 15/38's incremental-value test ("Champion rejects A, Challenger
    takes A, A wins") meaningful even when the two decisions differ.
"""
from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from product import counterfactual_learning as CFL
from product.evolution import shadow_decisions as SD

MIN_PAIRED_SAMPLE = 30


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _dt(value: Any):
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _frozen_at_or_never(row: dict[str, Any]):
    return _dt(row.get("frozen_at"))


def graded_rows(
    policy_id: str, *, domain: str | None = None, path: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = SD.list_shadow_decisions(policy_id=policy_id, path=path)
    rows = [r for r in rows if r.get("outcome") is not None]
    if domain is not None:
        rows = [r for r in rows if r.get("domain") == domain]
    return rows


def realized_r_stream(
    policy_id: str, *, domain: str | None = None, path: str | Path | None = None,
) -> list[float]:
    """R this policy's own real equity curve would show: ENTER_NOW rows only."""
    out = []
    for row in graded_rows(policy_id, domain=domain, path=path):
        if row.get("decision") != "ENTER_NOW":
            continue
        r = _f(row.get("counterfactual_R"))
        if r is not None:
            out.append(r)
    return out


def _max_drawdown_r(r_stream: list[float]) -> float | None:
    if not r_stream:
        return None
    cum = 0.0
    peak = 0.0
    worst = 0.0
    for r in r_stream:
        cum += r
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return round(worst, 4)


def scorecard(
    policy_id: str, *, domain: str | None = None, path: str | Path | None = None,
) -> dict[str, Any]:
    """Section 12's per-policy scorecard: decision counts, opportunity
    capture/miss counts, and the real-equity-curve R statistics."""
    rows = graded_rows(policy_id, domain=domain, path=path)
    taken = [r for r in rows if r.get("decision") == "ENTER_NOW"]
    rejected = [r for r in rows if r.get("decision") != "ENTER_NOW"]
    classes = Counter(str(r.get("classification") or "") for r in rows)
    r_stream = realized_r_stream(policy_id, domain=domain, path=path)
    wins = [r for r in r_stream if r > 0]
    losses = [r for r in r_stream if r <= 0]

    from scan.ev_engine import wilson_lb

    win_rate = (len(wins) / len(r_stream)) if r_stream else None
    return {
        "policy_id": policy_id,
        "decision_snapshots": len(rows),
        "selected_trades": len(taken),
        "rejects": len(rejected),
        "missed_winners": classes.get(CFL.MISSED_WINNER, 0),
        "avoided_losers": classes.get(CFL.AVOIDED_LOSER, 0),
        "correct_rejections": classes.get(CFL.CORRECT_REJECTION, 0),
        "winner_taken": classes.get(CFL.WINNER_TAKEN, 0),
        "loser_taken": classes.get(CFL.LOSER_TAKEN, 0),
        "good_waits": classes.get(CFL.GOOD_WAIT, 0),
        "ran_away": classes.get(CFL.RAN_AWAY, 0),
        "r_stream_n": len(r_stream),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "wilson_lb": round(wilson_lb(win_rate, len(r_stream)), 4) if r_stream else None,
        "expectancy_R": round(statistics.mean(r_stream), 4) if r_stream else None,
        "median_R": round(statistics.median(r_stream), 4) if r_stream else None,
        "avg_win_R": round(statistics.mean(wins), 4) if wins else None,
        "avg_loss_R": round(statistics.mean(losses), 4) if losses else None,
        "profit_factor": (
            round(sum(wins) / abs(sum(losses)), 4)
            if losses and sum(losses) != 0 else None
        ),
        "max_drawdown_R": _max_drawdown_r(r_stream),
        "selection_rate": round(len(taken) / len(rows), 4) if rows else None,
        "false_positive_rate": (
            round(classes.get(CFL.LOSER_TAKEN, 0) / len(taken), 4) if taken else None
        ),
        "opportunity_capture_rate": (
            round(
                classes.get(CFL.WINNER_TAKEN, 0)
                / (classes.get(CFL.WINNER_TAKEN, 0) + classes.get(CFL.MISSED_WINNER, 0)),
                4,
            )
            if (classes.get(CFL.WINNER_TAKEN, 0) + classes.get(CFL.MISSED_WINNER, 0)) > 0
            else None
        ),
    }


def _realized_equivalent_r(row: dict[str, Any]) -> float | None:
    """The R this policy's decision actually realizes: the real
    counterfactual_R when it SELECTED (capital was at risk), or exactly 0.0
    when it REJECTED (no position, no P&L) -- never the raw hypothetical
    "what if I had taken this" R regardless of decision, which would make
    every paired diff zero (both policies share the same frozen entry/stop
    from the same snapshot, so a reject-vs-take difference only shows up
    once the decision itself gates whether that R was ever actually
    realized). This is what makes section 38's "Champion rejects A,
    Challenger takes A, A wins" test meaningful."""
    if row.get("decision") != "ENTER_NOW":
        return 0.0
    return _f(row.get("counterfactual_R"))


def paired_comparison(
    champion_policy_id: str, challenger_policy_id: str,
    *, domain: str | None = None, path: str | Path | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Direct comparison restricted to snapshots BOTH policies actually
    evaluated (section 15) -- the only fair way to isolate whether a
    challenger adds value instead of merely trading in a strong market.

    `since`: an ISO timestamp (e.g. a PROBATION checkpoint). When given, a
    shared snapshot only counts if BOTH policies' shadow decisions were
    frozen strictly after it -- i.e. this is genuinely unseen evidence, not
    a decision that already existed (and may already have been counted)
    before the checkpoint. Without this filter, a policy's pre-checkpoint
    evidence would silently keep being double-counted into every later
    promotion statistic computed from the full cumulative ledger.
    """
    champ = {r["market_snapshot_id"]: r for r in graded_rows(champion_policy_id, domain=domain, path=path)}
    chal = {r["market_snapshot_id"]: r for r in graded_rows(challenger_policy_id, domain=domain, path=path)}
    shared = sorted(set(champ) & set(chal))
    if since:
        cutoff = _dt(since)
        if cutoff is not None:
            def _after(sid: str) -> bool:
                c_at, h_at = _frozen_at_or_never(champ[sid]), _frozen_at_or_never(chal[sid])
                return c_at is not None and h_at is not None and c_at > cutoff and h_at > cutoff
            shared = [sid for sid in shared if _after(sid)]

    diffs: list[float] = []
    agreement = {"both_selected": 0, "both_rejected": 0, "champion_only": 0, "challenger_only": 0}
    examples: list[dict[str, Any]] = []
    for sid in shared:
        c, h = champ[sid], chal[sid]
        c_r, h_r = _realized_equivalent_r(c), _realized_equivalent_r(h)
        if c_r is not None and h_r is not None:
            diffs.append(h_r - c_r)
        c_sel = c.get("decision") == "ENTER_NOW"
        h_sel = h.get("decision") == "ENTER_NOW"
        if c_sel and h_sel:
            agreement["both_selected"] += 1
        elif not c_sel and not h_sel:
            agreement["both_rejected"] += 1
        elif c_sel and not h_sel:
            agreement["champion_only"] += 1
        else:
            agreement["challenger_only"] += 1
            examples.append({
                "market_snapshot_id": sid, "symbol": h.get("symbol"),
                "champion_classification": c.get("classification"),
                "challenger_classification": h.get("classification"),
            })

    return {
        "champion_policy_id": champion_policy_id,
        "challenger_policy_id": challenger_policy_id,
        "paired_snapshots": len(shared),
        "incremental_expectancy_R": round(statistics.mean(diffs), 4) if diffs else None,
        "incremental_diffs": diffs,
        "agreement": agreement,
        "challenger_only_examples": examples[:10],
    }
