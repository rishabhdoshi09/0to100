"""Does the desk have market evidence yet, and how much?

This screen exists to answer the one question the whole product rests on, and
to answer it in the awkward direction as readily as the flattering one. Right
now the honest answer is usually NO_MARKET_EVIDENCE: the machinery closes, but
no paper trade has settled, so nothing here has been earned.

A dashboard that renders an empty state as a tidy set of zeros invites the
reader to believe the zeros are measurements. So the states are named:

    NO_MARKET_EVIDENCE   nothing has settled. Not "0% win rate".
    ACCUMULATING         trades are settling, no cell has reached the floor
    MEASURED             at least one cell can carry weight in ranking

Only PAPER_FORWARD is counted. Replay and fixture rows exist in their own
cells and are reported separately, clearly, as things that prove plumbing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from product.conditional_evidence import (
    MIN_SAMPLE,
    cells,
    parse_context,
)
from product.evidence_class import (
    HISTORICAL_REPLAY,
    PAPER_FORWARD,
    TEST_FIXTURE,
)

SCHEMA_VERSION = 1

NO_MARKET_EVIDENCE = "NO_MARKET_EVIDENCE"
INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
EARLY = "EARLY"
DEVELOPING = "DEVELOPING"
MATURE = "MATURE"

#: Retained so existing callers and tests keep working: ACCUMULATING was the
#: previous name for INSUFFICIENT_SAMPLE.
ACCUMULATING = INSUFFICIENT_SAMPLE
MEASURED = EARLY

#: Sample-depth multiples of the floor. These grade how MUCH has settled, and
#: nothing else. A MATURE cell with a negative expectancy is a mature negative
#: result — maturity is never a claim that an edge exists.
DEVELOPING_MULTIPLE = 3
MATURE_MULTIPLE = 10


def _group(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Sample size and expectancy per setup, per regime, and so on."""
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = parse_context(str(row.get("context_key") or "")).get(field) or "UNKNOWN"
        bucket = buckets.setdefault(key, {
            field: key, "count": 0, "wins": 0, "losses": 0, "r_total": 0.0,
            "cells": 0,
        })
        count = int(row.get("count") or 0)
        bucket["count"] += count
        bucket["wins"] += int(row.get("wins") or 0)
        bucket["losses"] += int(row.get("losses") or 0)
        bucket["cells"] += 1
        expectancy = row.get("expectancy_R")
        if expectancy is not None:
            bucket["r_total"] += float(expectancy) * count
    out = []
    for bucket in buckets.values():
        count = bucket.pop("count")
        r_total = bucket.pop("r_total")
        bucket["count"] = count
        bucket["expectancy_R"] = (r_total / count) if count else None
        bucket["usable_for_ranking"] = count >= MIN_SAMPLE
        out.append(bucket)
    out.sort(key=lambda b: (-int(b["count"]), str(b.get(field))))
    return out


def _maturity(settled: list[dict[str, Any]], total: int) -> str:
    """How deep the evidence is. Never how good it is."""
    if total == 0:
        return NO_MARKET_EVIDENCE
    deepest = max((int(r.get("count") or 0) for r in settled), default=0)
    if deepest < MIN_SAMPLE:
        return INSUFFICIENT_SAMPLE
    if deepest >= MIN_SAMPLE * MATURE_MULTIPLE:
        return MATURE
    if deepest >= MIN_SAMPLE * DEVELOPING_MULTIPLE:
        return DEVELOPING
    return EARLY


def _max_drawdown(r_values: list[float]) -> float | None:
    """Worst peak-to-trough of the cumulative R curve, in resolution order.

    Reported only because the average tells you nothing about the path. A
    strategy whose expectancy is positive and whose drawdown is eleven R is
    not tradable by anyone who has to live through it.
    """
    if not r_values:
        return None
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for value in r_values:
        equity += float(value)
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return round(worst, 4)


def _risk_shape(settled: list[dict[str, Any]]) -> dict[str, Any]:
    """MAE, MFE, drawdown and calibration — only where they were measured."""
    maes = [float(r["mae_R"]) for r in settled if r.get("mae_R") is not None]
    mfes = [float(r["mfe_R"]) for r in settled if r.get("mfe_R") is not None]
    gaps = [float(r["calibration_gap"]) for r in settled
            if r.get("calibration_gap") is not None]
    stream: list[float] = []
    for row in settled:
        stream.extend(float(v) for v in (row.get("r_values") or []))
    return {
        "worst_mae_R": min(maes) if maes else None,
        "best_mfe_R": max(mfes) if mfes else None,
        "max_drawdown_R": _max_drawdown(stream),
        "calibration_gap": (sum(gaps) / len(gaps)) if gaps else None,
        "calibration_note": (
            "positive means the system claimed more confidence than it earned"
            if gaps else "no confidence was recorded with these outcomes"
        ),
        "measured_from": len(stream),
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Where the R outcomes actually landed, not just their average.

    An expectancy of zero built from a wall of small losses and one large win
    is a different strategy from one built of coin flips, and the average hides
    which you own.
    """
    buckets = {"<= -2R": 0, "-2R..-1R": 0, "-1R..0": 0, "0..1R": 0,
               "1R..2R": 0, ">= 2R": 0}
    for row in rows:
        for value in row.get("r_values") or []:
            r = float(value)
            if r <= -2:
                buckets["<= -2R"] += 1
            elif r <= -1:
                buckets["-2R..-1R"] += 1
            elif r < 0:
                buckets["-1R..0"] += 1
            elif r < 1:
                buckets["0..1R"] += 1
            elif r < 2:
                buckets["1R..2R"] += 1
            else:
                buckets[">= 2R"] += 1
    return buckets


def _field(position: Any, name: str, default: Any = "") -> Any:
    if isinstance(position, Mapping):
        return position.get(name, default)
    return getattr(position, name, default)


def _positions_of(book: Any) -> list[Any]:
    """Accept a live PaperBook or the JSON the desk persists between runs."""
    if book is None:
        return []
    if isinstance(book, Mapping):
        return list(book.get("open") or [])
    opens = getattr(book, "open", None)
    if isinstance(opens, Mapping):
        return list(opens.values())
    return list(opens or [])


def _open_positions(book: Any = None) -> list[dict[str, Any]]:
    """Trades still running. Unresolved is a state, not a missing number."""
    rows = []
    for position in _positions_of(book):
        rows.append({
            "symbol": _field(position, "symbol", ""),
            "entry_date": _field(position, "entry_date", ""),
            "entry_price": _field(position, "entry_price", None),
            "stop_price": _field(position, "stop_price", None),
            "target_price": _field(position, "target_price", None),
            "bars_held": _field(position, "bars_held", 0),
            "decision_id": _field(position, "decision_id", ""),
            "context_key": _field(position, "context_key", ""),
            "attributable": bool(_field(position, "decision_id", "")),
        })
    rows.sort(key=lambda r: (str(r["entry_date"]), str(r["symbol"])))
    return rows


def build_forward_evidence_board(*, book: Any = None,
                                 path: str | Path | None = None) -> dict[str, Any]:
    """The truthful state of forward evidence, empty states included."""
    settled = cells(evidence_class=PAPER_FORWARD, path=path)
    total = sum(int(row.get("count") or 0) for row in settled)
    usable = [row for row in settled if row.get("usable_for_ranking")]

    state = _maturity(settled, total)
    deepest = max((int(r.get("count") or 0) for r in settled), default=0)
    if state == NO_MARKET_EVIDENCE:
        headline = (
            "No paper trade has settled yet. The learning machinery is wired "
            "and proved, but nothing here has been earned from the market."
        )
    elif state == INSUFFICIENT_SAMPLE:
        headline = (
            f"{total} settled trade{'s' if total != 1 else ''} so far. The "
            f"richest context has {deepest} of the {MIN_SAMPLE} needed before "
            "it may affect ranking."
        )
    else:
        headline = (
            f"{total} settled trades, deepest context {deepest}. "
            f"{len(usable)} context{'s' if len(usable) != 1 else ''} carry "
            "enough sample to affect ranking. Sample depth is not an edge."
        )

    open_rows = _open_positions(book)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "headline": headline,
        "evidence_class": PAPER_FORWARD,
        "min_sample": MIN_SAMPLE,
        "settled_trades": total,
        "cells": settled,
        "cells_usable_for_ranking": len(usable),
        "by_setup": _group(settled, "setup"),
        "by_regime": _group(settled, "regime"),
        "by_sector": _group(settled, "sector"),
        "r_distribution": _distribution(settled),
        "risk_shape": _risk_shape(settled),
        "maturity_thresholds": {
            "insufficient_below": MIN_SAMPLE,
            "developing_at": MIN_SAMPLE * DEVELOPING_MULTIPLE,
            "mature_at": MIN_SAMPLE * MATURE_MULTIPLE,
            "note": "depth of evidence only; never a claim that an edge exists",
        },
        "unresolved": open_rows,
        "unresolved_count": len(open_rows),
        "unattributable_open": sum(1 for r in open_rows if not r["attributable"]),
        # Reported so nobody mistakes their absence for a gap, or their
        # presence for an edge.
        "non_market_evidence": {
            "historical_replay_cells": len(cells(evidence_class=HISTORICAL_REPLAY, path=path)),
            "test_fixture_cells": len(cells(evidence_class=TEST_FIXTURE, path=path)),
            "note": (
                "Replay and fixture rows prove the loop is wired. They are "
                "never counted toward an edge and never reach ranking."
            ),
        },
    }
