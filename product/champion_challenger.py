"""Champion / Challenger research factory.

Production decision stack is the CHAMPION. Experimental ranking weights,
policies, entry/setup/regime rules, and ML models run as CHALLENGERS against
the same point-in-time opportunity set and never control paper capital.

Promotion is explicit: a new strategy/policy version and rules hash.
In-sample evidence alone cannot promote. A challenger never silently replaces
Champion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "challengers.json"
SCHEMA_VERSION = 1  # previous: none. Additive research ledger.

PROPOSED = "PROPOSED"
SHADOW = "SHADOW"
TESTING = "TESTING"
ELIGIBLE = "ELIGIBLE"
PROMOTED = "PROMOTED"
REJECTED = "REJECTED"
RETIRED = "RETIRED"

STATUSES = (PROPOSED, SHADOW, TESTING, ELIGIBLE, PROMOTED, REJECTED, RETIRED)

MIN_OOS_N = 30
MIN_FORWARD_N = 20
WEAK_EXPECTANCY = 0.0

# Promotion contract: OOS + forward + execution-adjusted evidence, never
# in-sample alone. Governance remains fail-closed and explicit.
PROMOTION_CONTRACT = {
    "forbid_in_sample_only": True,
    "min_oos_n": MIN_OOS_N,
    "min_forward_n": MIN_FORWARD_N,
    "require_explicit_promote": True,
    "require_adversarial_not_failed": True,
    "require_execution_adjusted_edge": True,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def store_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_CHALLENGERS")
    if override:
        return Path(override)
    return DEFAULT_PATH


def champion_identity() -> dict[str, Any]:
    from product.strategy_catalog import ensemble_identity
    ident = ensemble_identity()
    return {
        "role": "CHAMPION",
        "strategy_id": ident.get("strategy_id"),
        "version": ident.get("strategy_version"),
        "rules_hash": ident.get("rules_hash"),
        "can_execute": True,
        "controls_paper_capital": True,
    }


def empty_store() -> dict[str, Any]:
    champ = champion_identity()
    return {
        "schema_version": SCHEMA_VERSION,
        "champion": champ,
        "challengers": [],
        "promotion_log": [],
        "live_locked": True,
        "note": "Challengers cannot execute. Champion rules_hash changes only via promote().",
    }


def load_store(path: str | Path | None = None) -> dict[str, Any]:
    target = store_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return empty_store()
    if not isinstance(payload, dict):
        return empty_store()
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("challengers", [])
    payload.setdefault("promotion_log", [])
    payload["live_locked"] = True
    if not payload.get("champion"):
        payload["champion"] = champion_identity()
    return payload


def save_store(payload: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = SCHEMA_VERSION
    data["live_locked"] = True
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def _ci(mean: float, n: int, std: float | None = None) -> tuple[float | None, float | None]:
    if n < 8:
        return None, None
    se = (std if std is not None else 1.0) / math.sqrt(n)
    return round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)


def _metrics(pnls: Sequence[float]) -> dict[str, Any]:
    vals = [float(x) for x in pnls]
    n = len(vals)
    if n == 0:
        return {
            "sample_size": 0,
            "expectancy": None,
            "hit_rate": None,
            "average_win": None,
            "average_loss": None,
            "drawdown": None,
            "confidence_interval": None,
            "evidence": "INSUFFICIENT_EVIDENCE",
        }
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v < 0]
    mean = sum(vals) / n
    hit = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for v in vals:
        equity += v
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n else 0.0
    lo, hi = _ci(mean, n, std or None)
    return {
        "sample_size": n,
        "expectancy": round(mean, 6),
        "hit_rate": round(hit, 4),
        "average_win": None if avg_win is None else round(avg_win, 6),
        "average_loss": None if avg_loss is None else round(avg_loss, 6),
        "drawdown": round(dd, 6),
        "confidence_interval": None if lo is None else [lo, hi],
        "evidence": "INSUFFICIENT_EVIDENCE" if n < MIN_OOS_N else "MEASURED",
    }


class ChampionChallengerEngine:
    """Research factory. Challengers never submit orders."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = store_path(path)
        self.store = load_store(self.path)

    def champion(self) -> dict[str, Any]:
        return dict(self.store.get("champion") or champion_identity())

    def get(self, challenger_id: str) -> dict[str, Any] | None:
        for row in self.store.get("challengers") or []:
            if str(row.get("challenger_id")) == str(challenger_id):
                return dict(row)
        return None

    def register(
        self,
        *,
        challenger_id: str,
        hypothesis: str,
        changed_behavior: str,
        rules: Mapping[str, Any],
        eligible_universe: Sequence[str] | None = None,
        training_data: str = "",
        validation_data: str = "",
        oos_data: str = "",
        start_date: str = "",
        kind: str = "ranking_weights",
        version: int = 1,
    ) -> dict[str, Any]:
        row = {
            "challenger_id": str(challenger_id),
            "version": int(version),
            "kind": str(kind),
            "rules": dict(rules),
            "rules_hash": _hash(dict(rules)),
            "start_date": start_date or _now()[:10],
            "hypothesis": str(hypothesis),
            "changed_behavior": str(changed_behavior),
            "eligible_universe": list(eligible_universe or ["same_as_champion"]),
            "training_data": training_data,
            "validation_data": validation_data,
            "oos_data": oos_data,
            "forward_observations": [],
            "pit_evaluations": [],
            "status": SHADOW,
            "can_execute": False,
            "controls_paper_capital": False,
            "in_sample_only": not bool(oos_data),
            "registered_at": _now(),
        }
        others = [
            c for c in (self.store.get("challengers") or [])
            if str(c.get("challenger_id")) != str(challenger_id)
        ]
        others.append(row)
        self.store["challengers"] = others
        save_store(self.store, self.path)
        return dict(row)

    def evaluate_same_pit(
        self,
        challenger_id: str,
        opportunities: Sequence[Mapping[str, Any]],
        *,
        as_of: str,
        champion_ranking: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Score the same frozen opportunity set. Does not execute."""
        ch = self.get(challenger_id)
        if ch is None:
            raise KeyError(challenger_id)
        frozen = [dict(row) for row in opportunities]
        weights = dict((ch.get("rules") or {}).get("ranking_weights") or {})
        scored = []
        for row in frozen:
            base = float(row.get("selection_score") or row.get("score") or 0.0)
            adj = base
            setup = str(row.get("setup_label") or "")
            if setup and setup in weights:
                adj = base * float(weights[setup])
            scored.append({
                "symbol": str(row.get("symbol") or "").upper(),
                "champion_score": base,
                "challenger_score": adj,
                "tier": row.get("reco_tier"),
                "as_of": as_of,
            })
        scored.sort(key=lambda r: (-float(r["challenger_score"]), r["symbol"]))
        result = {
            "challenger_id": challenger_id,
            "as_of": as_of,
            "can_execute": False,
            "n_opportunities": len(frozen),
            "ranking": scored,
            "champion_ranking": list(champion_ranking or [str(r.get("symbol") or "").upper() for r in frozen]),
            "same_pit": True,
            "evidence_store": "challenger",
        }
        ch = self.get(challenger_id) or {}
        evals = list(ch.get("pit_evaluations") or [])
        evals.append({"as_of": as_of, "n": len(frozen), "top": [r["symbol"] for r in scored[:5]]})
        ch["pit_evaluations"] = evals[-120:]
        self._put(ch)
        return result

    def record_forward(
        self,
        challenger_id: str,
        *,
        pnl: float,
        execution_adjusted_pnl: float | None = None,
        execution_complete: bool | None = None,
        execution_source: str = "",
        regime: str = "",
        sector: str = "",
        taken_by_champion: bool = False,
        missed: bool = False,
        avoided_loss: bool = False,
        split: str = "oos",
        calibrated: float | None = None,
    ) -> dict[str, Any]:
        ch = self.get(challenger_id)
        if ch is None:
            raise KeyError(challenger_id)
        obs = list(ch.get("forward_observations") or [])
        adjusted_present = execution_adjusted_pnl is not None
        obs.append({
            "pnl": float(pnl),
            "execution_adjusted_pnl": execution_adjusted_pnl,
            # Backward compatibility: an explicitly supplied adjusted pnl is
            # treated as usable evidence unless the caller labels it incomplete.
            "execution_complete": adjusted_present if execution_complete is None else bool(execution_complete),
            "execution_source": execution_source or ("provided_execution_adjusted" if adjusted_present else "missing"),
            "regime": regime,
            "sector": sector,
            "taken_by_champion": bool(taken_by_champion),
            "missed": bool(missed),
            "avoided_loss": bool(avoided_loss),
            "split": split,
            "calibrated": calibrated,
            "recorded_at": _now(),
        })
        ch["forward_observations"] = obs
        if ch.get("status") == SHADOW and len(obs) >= 8:
            ch["status"] = TESTING
        self._put(ch)
        return self.compare(challenger_id)

    def compare(self, challenger_id: str, *, champion_pnls: Sequence[float] | None = None) -> dict[str, Any]:
        ch = self.get(challenger_id)
        if ch is None:
            raise KeyError(challenger_id)
        obs = list(ch.get("forward_observations") or [])
        oos = [o for o in obs if o.get("split") in {"oos", "forward"}]
        ins = [o for o in obs if o.get("split") == "in_sample"]
        evidence_rows = oos or obs
        pnls = [float(o["pnl"]) for o in oos] or [float(o["pnl"]) for o in obs]
        adj = [
            float(o["execution_adjusted_pnl"])
            for o in evidence_rows
            if o.get("execution_adjusted_pnl") is not None
        ]
        complete_adj = [
            float(o["execution_adjusted_pnl"])
            for o in evidence_rows
            if o.get("execution_adjusted_pnl") is not None and bool(o.get("execution_complete"))
        ]
        m = _metrics(pnls)
        champ_m = _metrics(list(champion_pnls or []))
        regimes = {}
        sectors = {}
        for o in evidence_rows:
            regimes.setdefault(str(o.get("regime") or "UNKNOWN"), []).append(float(o["pnl"]))
            sectors.setdefault(str(o.get("sector") or "UNKNOWN"), []).append(float(o["pnl"]))
        missed = sum(1 for o in obs if o.get("missed"))
        avoided = sum(1 for o in obs if o.get("avoided_loss"))
        turnover = len(obs)
        evidence_n = len(evidence_rows)
        comparison = {
            "challenger_id": challenger_id,
            "status": ch.get("status"),
            "can_execute": False,
            "controls_paper_capital": False,
            "expectancy": m["expectancy"],
            "hit_rate": m["hit_rate"],
            "average_win": m["average_win"],
            "average_loss": m["average_loss"],
            "drawdown": m["drawdown"],
            "turnover": turnover,
            "execution_adjusted_expectancy": (
                None if not adj else round(sum(adj) / len(adj), 6)
            ),
            "execution_adjusted_n": len(adj),
            "execution_adjusted_coverage": round(len(adj) / evidence_n, 4) if evidence_n else 0.0,
            "execution_complete_n": len(complete_adj),
            "execution_complete_coverage": round(len(complete_adj) / evidence_n, 4) if evidence_n else 0.0,
            "regime_stability": {
                k: round(sum(v) / len(v), 6) for k, v in regimes.items() if v
            },
            "sector_stability": {
                k: round(sum(v) / len(v), 6) for k, v in sectors.items() if v
            },
            "missed_opportunity_rate": round(missed / turnover, 4) if turnover else None,
            "avoided_loss_rate": round(avoided / turnover, 4) if turnover else None,
            "calibration": None,
            "sample_size": m["sample_size"],
            "confidence_interval": m["confidence_interval"],
            "oos_n": len(oos),
            "in_sample_n": len(ins),
            "champion_expectancy": champ_m.get("expectancy"),
            "in_sample_only": bool(ch.get("in_sample_only")) and not oos,
            "evidence_store": "challenger",
            "champion_rules_hash": self.champion().get("rules_hash"),
            "challenger_rules_hash": ch.get("rules_hash"),
        }
        ch["last_comparison"] = comparison
        # Weak challenger: measured OOS gross expectancy at or below zero with enough n.
        if comparison["oos_n"] >= MIN_OOS_N and (m["expectancy"] or 0) <= WEAK_EXPECTANCY:
            ch["status"] = REJECTED
            ch["reject_reason"] = "WEAK_OOS_EXPECTANCY"
            comparison["status"] = REJECTED
        self._put(ch)
        return comparison

    def promote(
        self,
        challenger_id: str,
        *,
        allow_in_sample: bool = False,
        adversarial_status: str = "SURVIVED",
    ) -> dict[str, Any]:
        """Explicit promotion only. Never a silent replace.

        Gross edge is insufficient. Once a challenger has promotion-sized OOS
        evidence, execution-adjusted evidence must also be sufficiently covered
        and positive. The final action is still explicit; this method never runs
        automatically from research.
        """
        if allow_in_sample:
            raise ValueError("in-sample promotion is forbidden")
        ch = self.get(challenger_id)
        if ch is None:
            raise KeyError(challenger_id)
        before = self.champion()
        comparison = self.compare(challenger_id)
        reasons: list[str] = []
        if comparison.get("in_sample_only"):
            reasons.append("IN_SAMPLE_ONLY")
        if int(comparison.get("oos_n") or 0) < MIN_OOS_N:
            reasons.append("OOS_SAMPLE_TOO_SMALL")
        if int(comparison.get("sample_size") or 0) < MIN_FORWARD_N:
            reasons.append("FORWARD_SAMPLE_TOO_SMALL")
        if adversarial_status == "FAILED":
            reasons.append("ADVERSARIAL_FAILED")
        if adversarial_status == "FRAGILE":
            reasons.append("ADVERSARIAL_FRAGILE")
        try:
            from product.promotion_governance import challenger_promotion_reasons
            reasons.extend(challenger_promotion_reasons(comparison, adversarial_status=adversarial_status))
        except Exception:
            # Governance import failure is itself not a bypass: for a promotion-sized
            # sample require execution evidence directly here.
            if int(comparison.get("oos_n") or 0) >= MIN_OOS_N:
                if int(comparison.get("execution_adjusted_n") or 0) < MIN_OOS_N:
                    reasons.append("EXECUTION_EVIDENCE_INCOMPLETE")
                elif comparison.get("execution_adjusted_expectancy") is None or float(comparison["execution_adjusted_expectancy"]) <= 0:
                    reasons.append("EXECUTION_ADJUSTED_EDGE_NON_POSITIVE")
        if ch.get("status") == REJECTED:
            reasons.append("CHALLENGER_REJECTED")
        # Stable deterministic reason list for UI/tests/audit.
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            ch["status"] = ch.get("status") if ch.get("status") == REJECTED else SHADOW
            ch["promotion_blocked"] = reasons
            ch["last_promotion_comparison"] = comparison
            self._put(ch)
            return {
                "promoted": False,
                "status": ch["status"],
                "reasons": reasons,
                "champion_rules_hash": before.get("rules_hash"),
                "champion_unchanged": True,
                "execution_adjusted_expectancy": comparison.get("execution_adjusted_expectancy"),
                "execution_adjusted_coverage": comparison.get("execution_adjusted_coverage"),
            }
        ch["status"] = ELIGIBLE
        new_version = int(before.get("version") or 1) + 1
        new_rules = {
            "promoted_from": challenger_id,
            "challenger_rules_hash": ch.get("rules_hash"),
            "champion_rules_hash_before": before.get("rules_hash"),
            "version": new_version,
            "rules": ch.get("rules"),
        }
        new_hash = _hash(new_rules)
        ch["status"] = PROMOTED
        ch["promoted_at"] = _now()
        ch["promoted_version"] = new_version
        ch["promoted_rules_hash"] = new_hash
        ch["promotion_evidence"] = {
            "gross_expectancy": comparison.get("expectancy"),
            "execution_adjusted_expectancy": comparison.get("execution_adjusted_expectancy"),
            "execution_adjusted_coverage": comparison.get("execution_adjusted_coverage"),
            "oos_n": comparison.get("oos_n"),
            "adversarial_status": adversarial_status,
        }
        # Challenger still cannot silently execute; paper capital stays on the
        # new explicit champion version only after this log entry.
        self.store["champion"] = {
            "role": "CHAMPION",
            "strategy_id": before.get("strategy_id"),
            "version": new_version,
            "rules_hash": new_hash,
            "can_execute": True,
            "controls_paper_capital": True,
            "promoted_from": challenger_id,
            "previous_rules_hash": before.get("rules_hash"),
        }
        log = list(self.store.get("promotion_log") or [])
        log.append({
            "challenger_id": challenger_id,
            "from_hash": before.get("rules_hash"),
            "to_hash": new_hash,
            "version": new_version,
            "at": _now(),
            "gross_expectancy": comparison.get("expectancy"),
            "execution_adjusted_expectancy": comparison.get("execution_adjusted_expectancy"),
            "execution_adjusted_coverage": comparison.get("execution_adjusted_coverage"),
            "adversarial_status": adversarial_status,
        })
        self.store["promotion_log"] = log
        self._put(ch)
        return {
            "promoted": True,
            "status": PROMOTED,
            "champion_rules_hash": new_hash,
            "previous_rules_hash": before.get("rules_hash"),
            "version": new_version,
            "explicit": True,
            "execution_adjusted_expectancy": comparison.get("execution_adjusted_expectancy"),
            "execution_adjusted_coverage": comparison.get("execution_adjusted_coverage"),
        }

    def reject(self, challenger_id: str, *, reason: str) -> dict[str, Any]:
        ch = self.get(challenger_id)
        if ch is None:
            raise KeyError(challenger_id)
        ch["status"] = REJECTED
        ch["reject_reason"] = reason
        ch["can_execute"] = False
        self._put(ch)
        return dict(ch)

    def retire(self, challenger_id: str) -> dict[str, Any]:
        ch = self.get(challenger_id)
        if ch is None:
            raise KeyError(challenger_id)
        ch["status"] = RETIRED
        ch["can_execute"] = False
        self._put(ch)
        return dict(ch)

    def _put(self, row: Mapping[str, Any]) -> None:
        cid = str(row.get("challenger_id"))
        others = [
            c for c in (self.store.get("challengers") or [])
            if str(c.get("challenger_id")) != cid
        ]
        others.append(dict(row))
        self.store["challengers"] = others
        save_store(self.store, self.path)
