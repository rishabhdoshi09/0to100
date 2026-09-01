"""Canonical production-strategy identity and backtest-parity contract.

This module is deliberately a bridge, not another strategy engine.  The running
Recommendations desk remains authoritative for selection.  These records give
that product surface an immutable identity that research/backtests can match.

Safety rule: historical evidence is attachable only when experiment code_hash
matches the current production rules_hash and, when present, result metadata
agrees on strategy_id/version.  Otherwise parity is UNVERIFIED and no metrics
are surfaced as evidence for today's recommendation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProductionStrategy:
    strategy_id: str
    version: int
    name: str
    category_id: str
    status: str
    holding_period: str
    universe: str
    entry_logic: tuple[str, ...]
    exit_logic: tuple[str, ...]
    risk_assumptions: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    source_files: tuple[str, ...]

    @property
    def rules_hash(self) -> str:
        payload = {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "category_id": self.category_id,
            "holding_period": self.holding_period,
            "universe": self.universe,
            "entry_logic": self.entry_logic,
            "exit_logic": self.exit_logic,
            "risk_assumptions": self.risk_assumptions,
            "evidence_requirements": self.evidence_requirements,
            "source_files": self.source_files,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def public(self) -> dict[str, Any]:
        out = asdict(self)
        out["rules_hash"] = self.rules_hash
        return out


# These are recommendation LANES, not claims that a single indicator generates
# the pick. They mirror the current mutually-exclusive product categories and
# preserve the actual ensemble/evidence gates used by Recommendations.
_STRATEGIES: tuple[ProductionStrategy, ...] = (
    ProductionStrategy(
        strategy_id="QT-WEALTH-BUILDERS",
        version=1,
        name="Wealth Builders",
        category_id="wealth_builders",
        status="ACTIVE",
        holding_period="long_term",
        universe="NSE_EQ",
        entry_logic=(
            "current long-term shortlist classification is QUALITY_COMPOUNDER, GARP_CANDIDATE, or QUALITY_BUT_EXPENSIVE",
            "fundamental evidence coverage must be at least 50% before the lane is publishable",
            "recommendation evidence/risk gates remain authoritative; this lane does not independently create a BUY",
        ),
        exit_logic=("no independent execution exit; portfolio/trade-plan risk layer remains authoritative",),
        risk_assumptions=("missing fundamentals are unknown, never pass", "quality-but-expensive remains research, not automatic buy"),
        evidence_requirements=("business/fundamental quality", "coverage >= 50%", "current long-term classification"),
        source_files=("product/recommendations_workspace.py", "scan/long_term_service.py", "product/reco_ensemble.py"),
    ),
    ProductionStrategy(
        strategy_id="QT-SUPER-TRENDS",
        version=1,
        name="Super Trends",
        category_id="super_trends",
        status="ACTIVE",
        holding_period="swing",
        universe="NSE_EQ",
        entry_logic=(
            "momentum or golden-cross evidence with actionable momentum state",
            "trend structure must not explicitly fail",
            "soft RSI/chase/volume guards apply",
            "BUY requires independent evidence-family confirmation through the recommendation ensemble",
        ),
        exit_logic=("trade-plan stop/target and risk layer remain authoritative",),
        risk_assumptions=("no chase", "known weak volume rejects", "unknown volume is disclosed rather than invented"),
        evidence_requirements=("price/structure", "momentum leadership", "independent evidence-family confirmation for BUY"),
        source_files=("product/recommendations_workspace.py", "product/reco_methods.py", "product/reco_ensemble.py"),
    ),
    ProductionStrategy(
        strategy_id="QT-MOMENTUM-BREAKOUTS",
        version=1,
        name="Momentum Breakouts",
        category_id="momentum_breakouts",
        status="ACTIVE",
        holding_period="swing",
        universe="NSE_EQ",
        entry_logic=(
            "sniper/grade A-B or current Home-visible breakout structure",
            "hard RSI/chase/known-volume guards apply",
            "BUY requires a second independent evidence family; tape/RS/breakout are not triple-counted",
        ),
        exit_logic=("trade-plan stop/target and risk layer remain authoritative",),
        risk_assumptions=("no chase", "ghost sniper hits without breakout structure reject", "missing data stays missing"),
        evidence_requirements=("breakout structure", "liquidity/volume when known", "second independent family for BUY"),
        source_files=("product/recommendations_workspace.py", "product/breakout_quality.py", "product/reco_ensemble.py"),
    ),
    ProductionStrategy(
        strategy_id="QT-RECOVERY-SETUPS",
        version=1,
        name="Recovery Setups",
        category_id="recovery_setups",
        status="ACTIVE",
        holding_period="swing",
        universe="NSE_EQ",
        entry_logic=(
            "DOUBLE_BOTTOM or ACCUMULATION turnaround evidence",
            "coils such as CUP_HANDLE/POCKET_PIVOT/NR7 are not classified as recovery",
            "confirmed breakout and chase conditions remove the name from this lane",
        ),
        exit_logic=("trade-plan stop/target and risk layer remain authoritative",),
        risk_assumptions=("soft RSI guard", "AVOID verdict rejects", "no automatic BUY from recovery label"),
        evidence_requirements=("turnaround structure", "risk gates", "independent evidence-family confirmation for BUY"),
        source_files=("product/recommendations_workspace.py", "product/reco_methods.py", "product/reco_ensemble.py"),
    ),
)

_BY_CATEGORY = {s.category_id: s for s in _STRATEGIES}


def production_strategies() -> list[dict[str, Any]]:
    return [s.public() for s in _STRATEGIES]


def strategy_for_category(category_id: str) -> ProductionStrategy | None:
    return _BY_CATEGORY.get(str(category_id or "").strip())


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
        return dict(value) if isinstance(value, Mapping) else {}
    except Exception:
        return {}


def _matching_experiments(strategy: ProductionStrategy) -> list[dict[str, Any]]:
    try:
        from research.registry import list_experiments
        rows = list_experiments()
    except Exception:
        return []
    matches: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("code_hash") or "") != strategy.rules_hash:
            continue
        result = _json_obj(row.get("result"))
        rid = result.get("strategy_id")
        rver = result.get("strategy_version")
        if rid not in (None, "", strategy.strategy_id):
            continue
        if rver not in (None, "", strategy.version, str(strategy.version)):
            continue
        matches.append(dict(row))
    return matches


def parity_for_strategy(strategy: ProductionStrategy) -> dict[str, Any]:
    matches = _matching_experiments(strategy)
    if not matches:
        return {
            "status": "UNVERIFIED",
            "reason": "No evaluated research experiment matches this exact production rules_hash/version.",
            "strategy_id": strategy.strategy_id,
            "strategy_version": strategy.version,
            "rules_hash": strategy.rules_hash,
            "evidence": None,
        }
    # Prefer an evaluated record; REGISTERED alone is not performance evidence.
    evaluated = [r for r in matches if str(r.get("status") or "") in {"PROMOTED", "REJECTED"} and r.get("result")]
    if not evaluated:
        return {
            "status": "UNVERIFIED",
            "reason": "Matching experiment is registered but has no evaluated result.",
            "strategy_id": strategy.strategy_id,
            "strategy_version": strategy.version,
            "rules_hash": strategy.rules_hash,
            "evidence": None,
        }
    row = evaluated[0]
    metrics = _json_obj(row.get("result"))
    # Return recorded metrics verbatim; this layer never manufactures win-rate,
    # expectancy, drawdown or benchmark statistics from unrelated fields.
    return {
        "status": "VERIFIED",
        "reason": "Exact production rules_hash/version matched an evaluated research experiment.",
        "strategy_id": strategy.strategy_id,
        "strategy_version": strategy.version,
        "rules_hash": strategy.rules_hash,
        "experiment_id": row.get("hypothesis_id"),
        "experiment_status": row.get("status"),
        "evaluated_at": row.get("evaluated_at"),
        "data_window": _json_obj(row.get("data_window")),
        "evidence": metrics,
    }


def strategy_reference_for_category(category_id: str) -> dict[str, Any]:
    strategy = strategy_for_category(category_id)
    if strategy is None:
        return {
            "strategy_id": None,
            "strategy_version": None,
            "rules_hash": None,
            "backtest_parity": "UNVERIFIED",
            "parity_reason": "No canonical production strategy is registered for this category.",
        }
    parity = parity_for_strategy(strategy)
    return {
        "strategy_id": strategy.strategy_id,
        "strategy_version": strategy.version,
        "rules_hash": strategy.rules_hash,
        "name": strategy.name,
        "backtest_parity": parity["status"],
        "parity_reason": parity["reason"],
        "backtest_evidence": parity.get("evidence"),
        "experiment_id": parity.get("experiment_id"),
        "evaluated_at": parity.get("evaluated_at"),
    }


def strategy_registry_contract() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for strategy in _STRATEGIES:
        item = strategy.public()
        item["backtest"] = parity_for_strategy(strategy)
        rows.append(item)
    verified = sum(1 for r in rows if (r.get("backtest") or {}).get("status") == "VERIFIED")
    return {
        "schema_version": 1,
        "production_strategy_count": len(rows),
        "verified_backtest_parity_count": verified,
        "unverified_backtest_parity_count": len(rows) - verified,
        "strategies": rows,
        "invariant": "Historical metrics may be attached to a live recommendation only after exact strategy version + rules_hash parity is verified.",
    }
