"""Canonical production-method registry for the React desk.

Retail recommendations are NOT StrategySpec paper strategies. This module is
the only place that names what today's BUY/WATCH list actually runs.

A parity hash must identify executable decision behaviour, not merely threshold
constants. The ensemble hash therefore fingerprints the production decision
modules as well as their declared constants. Any decision-code change invalidates
old parity evidence automatically.
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
PRODUCTION_BACKTEST_PATH = ROOT / "logs" / "product" / "qt_reco_ensemble_backtest.json"

ENSEMBLE_ID = "QT_RECO_ENSEMBLE"
ENSEMBLE_VERSION = 1
UNVERIFIED = "UNVERIFIED"
VERIFIED = "VERIFIED"
RELATED_NOT_PARITY = "RELATED_NOT_PARITY"

_DECISION_CODE_PATHS = (
    "product/reco_methods.py",
    "product/reco_experts.py",
    "product/reco_ensemble.py",
    "product/recommendations_workspace.py",
    "product/breakout_quality.py",
    "product/radar_workspace.py",
)


def _rules_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return "UNREADABLE"


def decision_code_hashes() -> dict[str, str]:
    """Fingerprint files that determine recommendation nomination/tiering.

    This deliberately over-invalidates rather than under-invalidates: a harmless
    edit may require fresh evidence, but a behavioural edit can never silently
    inherit old evidence under the same rules_hash.
    """
    return {rel: _file_sha256(ROOT / rel) for rel in _DECISION_CODE_PATHS}


def ensemble_rules() -> dict[str, Any]:
    return {
        "min_confirms_for_buy": MIN_CONFIRMS_FOR_BUY,
        "sepa_pass": SEPA_PASS,
        "ev_min_n": EV_MIN_N,
        "case_min_n": CASE_MIN_N,
        "conviction_pass": CONVICTION_PASS,
        "rsi_hard": RSI_HARD,
        "method_weights": dict(METHOD_WEIGHTS),
        "decision_code_sha256": decision_code_hashes(),
    }


def current_rules_hash() -> str:
    return _rules_hash(ensemble_rules())


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def production_backtest_evidence(path: Path | None = None) -> dict[str, Any]:
    """Verify a performance artifact against today's exact executable hash."""
    target = path or PRODUCTION_BACKTEST_PATH
    raw = _load_json(target)
    expected = current_rules_hash()
    if not raw:
        return {
            "available": False,
            "parity": UNVERIFIED,
            "rules_hash": expected,
            "artifact_rules_hash": None,
            "point_in_time_verified": False,
            "metrics": None,
            "detail": (
                "BACKTEST PARITY: UNVERIFIED. No same-code production performance artifact is on disk. "
                "Generic Backtester, scanner calibration and paper StrategySpec results are not substitutes."
            ),
        }
    actual = str(raw.get("rules_hash") or "")
    pit = bool(raw.get("point_in_time_verified"))
    same = bool(actual and actual == expected)
    completed = bool(raw.get("completed"))
    # Performance attribution must also declare its scope. A decision-integrity
    # replay alone is never enough to promote historical return metrics.
    performance_scope = str(raw.get("scope") or "") in {
        "PRODUCTION_SIGNAL_OUTCOMES",
        "PRODUCTION_EXECUTION_REPLAY",
    }
    verified = same and pit and completed and performance_scope
    reasons: list[str] = []
    if not same:
        reasons.append(f"artifact hash {actual or 'missing'} != current {expected}")
    if not pit:
        reasons.append("point-in-time / leakage gate not verified")
    if not completed:
        reasons.append("replay artifact is not completed")
    if not performance_scope:
        reasons.append("artifact scope is not historical signal/execution performance")
    return {
        "available": True,
        "parity": VERIFIED if verified else UNVERIFIED,
        "rules_hash": expected,
        "artifact_rules_hash": actual or None,
        "same_rules_hash": same,
        "point_in_time_verified": pit,
        "completed": completed,
        "scope": raw.get("scope"),
        "generated_at": raw.get("generated_at"),
        "dataset": raw.get("dataset") or {},
        "metrics": raw.get("metrics") if verified else None,
        "audit_metrics": raw.get("metrics") or {},
        "walk_forward": raw.get("walk_forward") or {},
        "detail": (
            "BACKTEST PARITY: VERIFIED. Historical performance uses the same executable recommendation hash "
            "and passed point-in-time leakage guards."
            if verified else
            "BACKTEST PARITY: UNVERIFIED. " + "; ".join(reasons)
        ),
    }


def decision_replay_evidence() -> dict[str, Any]:
    """Synchronization check only — explicitly not performance evidence."""
    try:
        from product.production_replay import replay_tape_status
        return replay_tape_status(current_rules_hash=current_rules_hash())
    except Exception as exc:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "integrity_pass": False,
            "performance_evidence": False,
            "detail": f"Decision replay unavailable: {exc}",
        }


def ensemble_identity() -> dict[str, Any]:
    rules = ensemble_rules()
    evidence = production_backtest_evidence()
    replay = decision_replay_evidence()
    replay_detail = str(replay.get("detail") or "")
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
            "same executable rules_hash for production performance attribution",
            "point-in-time feature timestamps before outcome timestamps",
        ],
        "entry_logic": "Saved scan entry / buy zone; ensemble does not rescore OHLCV on page open",
        "exit_logic": "Saved scan stop and target; paper GTT on autonomy fills only",
        "risk_assumptions": "Chase/extension and RSI blow-off fail tape; funds never invent a Buy",
        "backtest_parity": evidence["parity"],
        "backtest_parity_detail": evidence["detail"] + (f" Decision replay: {replay_detail}" if replay_detail else ""),
        "backtest_evidence": evidence,
        "decision_replay": replay,
        "result_kind": "SAME_HASH_PRODUCTION_REPLAY" if evidence["parity"] == VERIFIED else None,
        "label": "QuantTerm recommendation ensemble",
        "rules": rules,
    }


def method_identity(method_id: str) -> dict[str, Any]:
    mid = str(method_id or "")
    rules = {
        "method_id": mid,
        "weight": METHOD_WEIGHTS.get(mid),
        "ensemble_rules_hash": current_rules_hash(),
    }
    return {
        "strategy_id": f"QT_METHOD_{mid.upper()}" if mid else "QT_METHOD_UNKNOWN",
        "strategy_version": ENSEMBLE_VERSION,
        "rules_hash": _rules_hash(rules),
        "active": mid in METHOD_WEIGHTS,
        "label": METHOD_LABELS.get(mid, mid or "Unknown"),
        "family": "recommendation_method",
        "backtest_parity": UNVERIFIED,
        "backtest_parity_detail": (
            f"{METHOD_LABELS.get(mid, mid)} is a component check. Production parity is evaluated at the "
            "ensemble decision layer, not inferred from this chip alone."
        ),
        "result_kind": None,
    }


def _load_signal_backtest(path: Path | None = None) -> dict[str, Any] | None:
    return _load_json(path or SIGNAL_BACKTEST_PATH)


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
            "This file adjusts UnifiedScanner composite weights. It is not a backtest of QT_RECO_ENSEMBLE "
            "and must not be shown as recommendation performance."
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
            "Technical structure passed, but the funds family rejected or could not confirm quality. "
            "Fundamentals do not independently create a Buy."
        )
    if fund_status == "pass" and str(tape.get("status")) == "fail":
        return "Business-quality overlay passed, but tape/extension rejected the setup. Quality is not a timing instruction."
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
    }
    row["backtest_parity"] = ident["backtest_parity"]
    row["backtest_parity_detail"] = ident["backtest_parity_detail"]
    disagreement = fundamental_disagreement(row)
    if disagreement:
        row["fundamental_disagreement"] = disagreement
    return row


def production_registry() -> dict[str, Any]:
    ensemble = ensemble_identity()
    methods = [method_identity(mid) for mid in METHOD_WEIGHTS]
    return {
        "schema_version": 3,
        "role": "production_recommendations",
        "ensemble": ensemble,
        "methods": methods,
        "decision_replay": ensemble.get("decision_replay") or {},
        "related_signal_calibration": related_signal_calibration(),
        "note": (
            "These ids describe the live recommendation checks. Paper StrategySpec rows belong on the research list "
            "and never rank today's BUY list."
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
                "Paper/autonomy strategy. Not the recommendation ensemble. Performance is not attached to today's BUY list."
            ),
        })
    return out
