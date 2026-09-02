"""Canonical production-method registry for the React desk.

Retail recommendations are NOT StrategySpec paper strategies. This module is
the only place that names what today's BUY/WATCH list actually runs:

- reco method ids from ``product.reco_methods``
- the two-family ensemble gate
- optional *related* scanner-signal calibration (never implied parity)

If a backtest cannot be proven to use the same rules_hash as the live method,
parity is UNVERIFIED. Unrelated research backtests are never attached.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from product.reco_methods import (
    CASE_MIN_N,
    CONVICTION_PASS,
    EV_MIN_N,
    METHOD_LABELS,
    METHOD_WEIGHTS,
    MIN_CONFIRMS_FOR_BUY,
    RSI_HARD,
    SEPA_PASS,
)

ROOT = Path(__file__).resolve().parents[1]
SIGNAL_BACKTEST_PATH = ROOT / "logs" / "signal_backtest.json"

ENSEMBLE_ID = "QT_RECO_ENSEMBLE"
ENSEMBLE_VERSION = 1
UNVERIFIED = "UNVERIFIED"
RELATED_NOT_PARITY = "RELATED_NOT_PARITY"


def _rules_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def ensemble_rules() -> dict[str, Any]:
    return {
        "min_confirms_for_buy": MIN_CONFIRMS_FOR_BUY,
        "sepa_pass": SEPA_PASS,
        "ev_min_n": EV_MIN_N,
        "case_min_n": CASE_MIN_N,
        "conviction_pass": CONVICTION_PASS,
        "rsi_hard": RSI_HARD,
        "method_weights": dict(METHOD_WEIGHTS),
    }


def ensemble_identity() -> dict[str, Any]:
    rules = ensemble_rules()
    return {
        "strategy_id": ENSEMBLE_ID,
        "strategy_version": ENSEMBLE_VERSION,
        "rules_hash": _rules_hash(rules),
        "active": True,
        "status": "active",
        "intended_holding_period": "setup-dependent (scan plan geometry)",
        "universe": "NSE EQ approved universe from the last saved market scan",
        "evidence_requirements": [
            "persisted market scan overlay",
            "two independent evidence families for Buy",
            f"Live EV / case memory require n≥{EV_MIN_N}",
        ],
        "entry_logic": "Saved scan entry / buy zone; ensemble does not rescore OHLCV on page open",
        "exit_logic": "Saved scan stop and target; paper GTT on autonomy fills only",
        "risk_assumptions": "Chase/extension and RSI blow-off fail tape; funds never invent a Buy",
        "backtest_parity": UNVERIFIED,
        "backtest_parity_detail": (
            "BACKTEST PARITY: UNVERIFIED. No walk-forward of QT_RECO_ENSEMBLE v1 "
            "exists. Scanner signal calibration and paper StrategySpec runs use "
            "different rules and must not be shown as this method's performance."
        ),
        "result_kind": None,
        "label": "QuantTerm recommendation ensemble",
        "role": "champion",
        "rules": rules,
    }


def method_identity(method_id: str) -> dict[str, Any]:
    mid = str(method_id or "")
    rules = {
        "method_id": mid,
        "weight": METHOD_WEIGHTS.get(mid),
        "ensemble": ensemble_rules(),
    }
    return {
        "strategy_id": f"QT_METHOD_{mid.upper()}" if mid else "QT_METHOD_UNKNOWN",
        "strategy_version": ENSEMBLE_VERSION,
        "rules_hash": _rules_hash(rules),
        "active": mid in METHOD_WEIGHTS,
        "label": METHOD_LABELS.get(mid, mid or "Unknown"),
        "family": "recommendation_method",
        "role": "supporting_check",
        "backtest_parity": UNVERIFIED,
        "backtest_parity_detail": (
            f"{METHOD_LABELS.get(mid, mid)} is a live recommendation check. "
            "It has no dedicated same-hash backtest."
        ),
        "result_kind": None,
    }


def _load_signal_backtest(path: Path | None = None) -> dict[str, Any] | None:
    target = path or SIGNAL_BACKTEST_PATH
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def related_signal_calibration(path: Path | None = None) -> dict[str, Any]:
    """Scanner weight file — related, never treated as reco-method parity."""
    raw = _load_signal_backtest(path)
    if not raw:
        return {
            "available": False,
            "parity": UNVERIFIED,
            "label": "Scanner signal calibration",
            "detail": "No signal_backtest.json on disk. Missing stays missing.",
        }
    return {
        "available": True,
        "parity": RELATED_NOT_PARITY,
        "label": "Scanner signal calibration",
        "as_of": raw.get("generated_at") or raw.get("as_of") or raw.get("finished_at"),
        "detail": (
            "This file adjusts UnifiedScanner composite weights. It is not a "
            "backtest of QT_RECO_ENSEMBLE and must not be shown as recommendation "
            "performance."
        ),
        "sample_note": raw.get("note") or raw.get("summary") or "",
    }


def annotate_method(method: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(method)
    ident = method_identity(str(row.get("id") or ""))
    row["strategy_id"] = ident["strategy_id"]
    row["strategy_version"] = ident["strategy_version"]
    row["rules_hash"] = ident["rules_hash"]
    row["backtest_parity"] = ident["backtest_parity"]
    row["backtest_parity_detail"] = ident["backtest_parity_detail"]
    return row


def fundamental_disagreement(card: Mapping[str, Any]) -> str:
    """Explain funds vs structure without calling due diligence or inventing scores."""
    methods = [m for m in (card.get("methods") or []) if isinstance(m, Mapping)]
    by_id = {str(m.get("id") or ""): m for m in methods}
    funds = by_id.get("funds") or {}
    tape = by_id.get("tape") or {}
    sepa = by_id.get("sepa") or {}
    structure_pass = str(tape.get("status")) == "pass" or str(sepa.get("status")) == "pass"
    fund_status = str(funds.get("status") or "unknown")
    if fund_status == "unknown":
        return "Fundamentals are unknown — missing evidence, not a failed business."
    if fund_status == "fail" and structure_pass:
        return (
            "Technical structure passed, but the funds family rejected or could "
            "not confirm quality. Fundamentals do not independently create a Buy."
        )
    if fund_status == "pass" and str(tape.get("status")) == "fail":
        return (
            "Business-quality overlay passed, but tape/extension rejected the "
            "setup. Quality is not a timing instruction."
        )
    conflicts = [str(x) for x in (card.get("conflicts") or []) if x]
    if conflicts:
        return "Recorded disagreement: " + " · ".join(conflicts[:3])
    return ""


def decorate_card(card: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(card)
    methods = [annotate_method(m) for m in (row.get("methods") or []) if isinstance(m, Mapping)]
    if methods:
        row["methods"] = methods
    ident = ensemble_identity()
    row["production_strategy"] = {
        "strategy_id": ident["strategy_id"],
        "strategy_version": ident["strategy_version"],
        "rules_hash": ident["rules_hash"],
        "label": ident["label"],
        "active": ident["active"],
        "role": "champion",
    }
    row["champion"] = ident["label"]
    row["challengers"] = [
        {
            "strategy_id": item.get("strategy_id"),
            "label": item.get("label"),
            "role": "challenger",
            "backtest_parity": item.get("backtest_parity") or UNVERIFIED,
        }
        for item in research_only_strategies()[:8]
    ]
    row["backtest_parity"] = UNVERIFIED
    row["backtest_parity_detail"] = ident["backtest_parity_detail"]
    disagreement = fundamental_disagreement(row)
    if disagreement:
        row["fundamental_disagreement"] = disagreement
    return row


def production_registry() -> dict[str, Any]:
    methods = [method_identity(mid) for mid in METHOD_WEIGHTS]
    return {
        "schema_version": 1,
        "role": "production_recommendations",
        "ensemble": ensemble_identity(),
        "methods": methods,
        "related_signal_calibration": related_signal_calibration(),
        "note": (
            "These ids describe the live recommendation checks. Paper "
            "StrategySpec rows belong on the research list and never rank today's BUY list."
        ),
    }


def research_only_strategies() -> list[dict[str, Any]]:
    """Registered paper/autonomy specs if a snapshot exists. Never generated here."""
    path = ROOT / "logs" / "autonomy" / "strategy_registry.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = data if isinstance(data, list) else list(data.get("strategies") or data.get("specs") or [])
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        sid = str(raw.get("strategy_id") or raw.get("id") or "")
        if not sid:
            continue
        out.append({
            "strategy_id": sid,
            "strategy_version": raw.get("version") or raw.get("strategy_version") or 1,
            "rules_hash": raw.get("rules_hash") or raw.get("config_hash") or "",
            "label": raw.get("name") or sid,
            "status": raw.get("status") or "research",
            "role": "RESEARCH_ONLY",
            "backtest_parity": UNVERIFIED,
            "backtest_parity_detail": (
                "Paper/autonomy strategy. Not the recommendation ensemble. "
                "Performance is not attached to today's BUY list."
            ),
        })
    return out
