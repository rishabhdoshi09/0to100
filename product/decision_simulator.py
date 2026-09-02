"""Take-vs-skip historical decision simulator.

Replays already-recorded paper decisions against later official bars.
This is research/backtest provenance. It never writes REAL_FORWARD_MARKET
rows and cannot change today's production policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.counterfactual_learning import (
    AVOIDED_LOSER,
    CORRECT_REJECTION,
    FLAT,
    GOOD_WAIT,
    MISSED_WINNER,
    RAN_AWAY,
    classify_forward,
)
from product.forward_evidence import BACKTEST

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "decision_simulator.json"
SCHEMA_VERSION = 1

INCONCLUSIVE = "INCONCLUSIVE"

_REASON_HELPED = {
    "ENTRY_TOO_EXTENDED": "chase protection",
    "CHASE_RISK": "chase protection",
    "DD_GATE_FAILED": "due-diligence block",
    "LOW_QUALITY_SETUP": "weak setup rejection",
    "LIQUIDITY_FAILED": "liquidity filter",
    "PORTFOLIO_BLOCK": "portfolio concentration",
    "SECTOR_CAP": "sector cap",
    "CORRELATION_CAP": "correlation cap",
    "WAIT_FOR_ENTRY": "wait-for-price",
}

_REASON_HURT = {
    "SECTOR_WEAK": "sector filter",
    "EVIDENCE_POLICY_BLOCK": "evidence policy",
    "EMPIRICAL_GATE_FAILED": "empirical gate",
}


def report_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_DECISION_SIMULATOR")
    return Path(override) if override else DEFAULT_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    raw = json.dumps(
        [(r.get("symbol"), r.get("as_of"), r.get("decision"), r.get("reason_code")) for r in rows],
        default=str,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _decision_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from product.autopilot_journal import load_journal
        journal = load_journal()
    except Exception:
        journal = {}
    for cycle in list(journal.get("cycles") or [])[-80:]:
        as_of = str(cycle.get("as_of") or cycle.get("recorded_at") or "")[:10]
        for taken in list(cycle.get("taken") or []):
            if not isinstance(taken, Mapping):
                continue
            rows.append({
                "symbol": str(taken.get("symbol") or "").upper(),
                "decision": "TAKEN",
                "reason_code": str(taken.get("reason_code") or "ENTER"),
                "as_of": as_of,
                "entry": taken.get("entry") or taken.get("entry_price"),
                "stop": taken.get("stop") or taken.get("stop_price"),
                "target": taken.get("target") or taken.get("target_price"),
                "sector": taken.get("sector") or "",
                "setup": taken.get("setup_label") or taken.get("setup") or "",
            })
        for skipped in list(cycle.get("rejections") or []) + list(cycle.get("waits") or []):
            if not isinstance(skipped, Mapping):
                continue
            decision = str(skipped.get("decision") or ("WAITED" if skipped.get("reason_code") == "WAIT_FOR_ENTRY" else "REJECTED"))
            rows.append({
                "symbol": str(skipped.get("symbol") or "").upper(),
                "decision": decision,
                "reason_code": str(skipped.get("reason_code") or skipped.get("reason") or ""),
                "as_of": as_of,
                "entry": skipped.get("entry") or skipped.get("hypothetical_entry"),
                "stop": skipped.get("stop") or skipped.get("hypothetical_stop"),
                "target": skipped.get("target") or skipped.get("hypothetical_target"),
                "sector": skipped.get("sector") or "",
                "setup": skipped.get("setup_label") or skipped.get("setup") or "",
            })
    try:
        from product.counterfactual_learning import ledger_path
        path = ledger_path()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines()[-400:]:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict) or not item.get("symbol"):
                    continue
                rows.append({
                    "symbol": str(item.get("symbol") or "").upper(),
                    "decision": str(item.get("decision") or "REJECTED"),
                    "reason_code": str(item.get("reason_code") or ""),
                    "as_of": str(item.get("as_of") or "")[:10],
                    "entry": item.get("hypothetical_entry"),
                    "stop": item.get("hypothetical_stop"),
                    "target": item.get("hypothetical_target"),
                    "sector": item.get("sector") or "",
                    "setup": item.get("setup") or "",
                    "classification": item.get("classification"),
                    "forward_return_pct": (item.get("outcome") or {}).get("forward_return_pct") if isinstance(item.get("outcome"), Mapping) else None,
                })
    except Exception:
        pass
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("symbol") or not row.get("as_of"):
            continue
        key = (str(row["symbol"]), str(row["as_of"]), str(row["decision"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _forward_return(symbol: str, as_of: str, entry: Any) -> float | None:
    if not symbol or not as_of:
        return None
    try:
        from core.outcome_resolver import session_close_return
        result = session_close_return(symbol, as_of, horizon=5)
        if isinstance(result, Mapping) and result.get("return_pct") is not None:
            return float(result["return_pct"])
        if isinstance(result, (int, float)):
            return float(result)
    except Exception:
        return None
    return None


def _classify(row: Mapping[str, Any]) -> str:
    existing = str(row.get("classification") or "")
    if existing:
        return existing
    fwd = row.get("forward_return_pct")
    if fwd is None:
        fwd = _forward_return(str(row.get("symbol") or ""), str(row.get("as_of") or ""), row.get("entry"))
    if fwd is None:
        return INCONCLUSIVE
    if str(row.get("decision") or "").upper() in {"TAKEN", "ENTER"}:
        return INCONCLUSIVE if fwd is None else FLAT
    return classify_forward(
        entry=row.get("entry") if row.get("entry") is not None else None,
        stop=row.get("stop") if row.get("stop") is not None else None,
        target=row.get("target") if row.get("target") is not None else None,
        forward_return_pct=float(fwd),
        later_entered=False,
    )


def _filter_scores(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    helped: dict[str, int] = {}
    hurt: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason_code") or "")
        klass = str(row.get("classification") or "")
        label = _REASON_HELPED.get(reason)
        if label and klass in {CORRECT_REJECTION, AVOIDED_LOSER, GOOD_WAIT}:
            helped[label] = helped.get(label, 0) + 1
        hurt_label = _REASON_HURT.get(reason)
        if hurt_label and klass == MISSED_WINNER:
            hurt[hurt_label] = hurt.get(hurt_label, 0) + 1
    top_helped = [name for name, _ in sorted(helped.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    top_hurt = [name for name, _ in sorted(hurt.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    return top_helped, top_hurt


def _sector_edge(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare high vs low sector scores among already-classified rows. Not a production change."""
    high: list[Mapping[str, Any]] = []
    low: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            score = float(row.get("sector_leadership_score"))
        except (TypeError, ValueError):
            continue
        if score >= 60:
            high.append(row)
        elif score < 40:
            low.append(row)

    def _rate(items: Sequence[Mapping[str, Any]], klass: str) -> float | None:
        if len(items) < 8:
            return None
        return round(100.0 * sum(1 for r in items if r.get("classification") == klass) / len(items), 1)

    return {
        "available": bool(high or low),
        "high_n": len(high),
        "low_n": len(low),
        "high_missed_winner_pct": _rate(high, MISSED_WINNER),
        "low_missed_winner_pct": _rate(low, MISSED_WINNER),
        "high_correct_rejection_pct": _rate(high, CORRECT_REJECTION),
        "low_correct_rejection_pct": _rate(low, CORRECT_REJECTION),
        "insufficient_n": len(high) < 8 or len(low) < 8,
        "not_promotion_evidence": True,
        "note": (
            "BACKTEST only. Sector score must already be on the historical row. "
            "Missing scores stay out of this split. n<8 stays inconclusive."
        ),
    }


def load_latest(path: str | Path | None = None) -> dict[str, Any]:
    return _read_json(report_path(path))


def run_decision_simulator(*, force: bool = False, path: str | Path | None = None) -> dict[str, Any]:
    """Classify recorded take/skip decisions using later official bars only."""
    target = report_path(path)
    rows = _decision_rows()
    version = _fingerprint(rows)
    cached = _read_json(target)
    if not force and cached.get("version") == version and cached.get("decisions_tested") == len(rows):
        cached["cache_hit"] = True
        cached["provenance"] = BACKTEST
        cached["live_locked"] = True
        return cached

    classified: list[dict[str, Any]] = []
    counts = {
        CORRECT_REJECTION: 0,
        MISSED_WINNER: 0,
        AVOIDED_LOSER: 0,
        GOOD_WAIT: 0,
        RAN_AWAY: 0,
        FLAT: 0,
        INCONCLUSIVE: 0,
        "TAKEN": 0,
        "REJECTED": 0,
        "WAITED": 0,
    }
    for row in rows:
        klass = _classify(row)
        decision = str(row.get("decision") or "REJECTED").upper()
        if decision.startswith("TAKE") or decision == "ENTER":
            counts["TAKEN"] += 1
        elif decision.startswith("WAIT"):
            counts["WAITED"] += 1
        else:
            counts["REJECTED"] += 1
        if klass in counts:
            counts[klass] += 1
        classified.append({
            **row,
            "classification": klass,
            "not_pnl": True,
            "provenance": BACKTEST,
        })

    helped, hurt = _filter_scores(classified)
    simple = (
        f"{len(rows)} historical decisions tested. "
        f"Would take {counts['TAKEN']}. Rejected {counts['REJECTED']}. "
        f"Correct rejections {counts[CORRECT_REJECTION]}. "
        f"Missed winners {counts[MISSED_WINNER]}."
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "version": version,
        "provenance": BACKTEST,
        "cache_hit": False,
        "live_locked": True,
        "not_promotion_evidence": True,
        "decisions_tested": len(rows),
        "would_take": counts["TAKEN"],
        "rejected": counts["REJECTED"],
        "waited": counts["WAITED"],
        "correct_rejections": counts[CORRECT_REJECTION],
        "missed_winners": counts[MISSED_WINNER],
        "avoided_losers": counts[AVOIDED_LOSER],
        "good_waits": counts[GOOD_WAIT],
        "ran_away": counts[RAN_AWAY],
        "flat": counts[FLAT],
        "inconclusive": counts[INCONCLUSIVE],
        "filters_helped": helped,
        "filters_hurt": hurt,
        "simple": simple,
        "note": (
            "Historical simulation is BACKTEST provenance. "
            "It does not change REAL_FORWARD_MARKET promotion stats. "
            "Rejected names are not paper P&L."
        ),
        "sector_edge": _sector_edge(classified),
        "rows": classified[:250],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return payload
