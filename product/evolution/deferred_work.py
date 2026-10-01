"""Durable, isolated Evolution work queue.

A PAPER cycle may freeze research inputs before mutation, but Challenger
research must never sit between an in-memory PAPER fill and the durable
paper-book save.  This module stores the frozen work item on disk and lets
the existing non-critical TOURNAMENT_CYCLE lane evaluate it later inside a
killable subprocess.

Safety invariants:
- no live/broker imports
- no PAPER-book mutation
- only frozen decision-time inputs are used for Challenger scoring
- a hung child is terminated by the parent timeout
- retries are idempotent because work IDs and shadow IDs are deterministic
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from core.runtime_paths import logs_dir
from product.evolution import policy_eval, tournament

SCHEMA_VERSION = 1
EVALUATOR_SCHEMA_VERSION = 1
DEFAULT_CHILD_TIMEOUT_SECONDS = float(
    os.environ.get("QT_EVOLUTION_DEFERRED_TIMEOUT_SECONDS") or 12.0
)
DEFAULT_MAX_ATTEMPTS = int(
    os.environ.get("QT_EVOLUTION_DEFERRED_MAX_ATTEMPTS") or 3
)


def _root(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_DEFERRED_ROOT")
    if override:
        return Path(override)
    return logs_dir() / "product" / "evolution_deferred"


def _item_path(work_id: str, root: str | Path | None = None) -> Path:
    return _root(root) / "items" / f"{work_id}.json"


def _result_path(work_id: str, root: str | Path | None = None) -> Path:
    return _root(root) / "results" / f"{work_id}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, dict) else None
    except Exception:
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _work_id(
    bundle: Mapping[str, Any],
    champion_policy_fingerprint: str,
    challenger_policies: Sequence[Mapping[str, Any]],
) -> str:
    material = {
        "domain": bundle.get("domain"),
        "as_of": bundle.get("as_of"),
        "champion_policy_id": bundle.get("champion_policy_id"),
        "champion_policy_fingerprint": champion_policy_fingerprint,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "snapshot_ids": sorted(
            str((row or {}).get("market_snapshot_id") or "")
            for row in dict(bundle.get("snapshots") or {}).values()
        ),
        "challengers": sorted(
            json.dumps(dict(policy), sort_keys=True, separators=(",", ":"), default=str)
            for policy in challenger_policies
        ),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:28]


def prepare_work(
    *,
    bundle: Mapping[str, Any],
    champion_policy_fingerprint: str,
    challenger_policies: Sequence[Mapping[str, Any]],
    individual_decisions_by_symbol: Mapping[str, Mapping[str, Any]],
    pre_mutation_book_snapshot: Mapping[str, Any],
    held_sector_by_symbol: Mapping[str, str],
    correlations: Mapping[str, float],
    max_new: int,
    regime: str,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist one PREPARED research item before the PAPER mutation.

    PREPARED items are never executed.  The caller promotes the item to READY
    only after the PAPER transaction has returned and its book has been
    durably saved.
    """
    work_id = _work_id(bundle, champion_policy_fingerprint, challenger_policies)
    path = _item_path(work_id, root)
    existing = _read_json(path)
    if existing is not None:
        return existing
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "work_id": work_id,
        "status": "PREPARED",
        "attempts": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_at": "",
        "last_error": "",
        "bundle": dict(bundle),
        "champion_policy_fingerprint": str(champion_policy_fingerprint or ""),
        "challenger_policies": [dict(p) for p in challenger_policies],
        "individual_decisions_by_symbol": {
            str(k).upper(): dict(v)
            for k, v in dict(individual_decisions_by_symbol or {}).items()
        },
        "pre_mutation_book_snapshot": dict(pre_mutation_book_snapshot or {}),
        "held_sector_by_symbol": {
            str(k).upper(): str(v or "")
            for k, v in dict(held_sector_by_symbol or {}).items()
        },
        "correlations": {
            str(k): float(v)
            for k, v in dict(correlations or {}).items()
        },
        "max_new": int(max_new),
        "regime": str(regime or ""),
    }
    _write_json(path, payload)
    return payload


def get_work(work_id: str, *, root: str | Path | None = None) -> dict[str, Any] | None:
    return _read_json(_item_path(work_id, root))


def mark_ready(work_id: str, *, root: str | Path | None = None) -> dict[str, Any]:
    path = _item_path(work_id, root)
    payload = _read_json(path)
    if payload is None:
        raise KeyError(f"unknown deferred Evolution work item {work_id}")
    if payload.get("status") == "SUCCEEDED":
        return payload
    payload["status"] = "READY"
    payload["ready_at"] = datetime.now(timezone.utc).isoformat()
    payload["last_error"] = ""
    _write_json(path, payload)
    return payload


def mark_abandoned(
    work_id: str,
    reason: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any] | None:
    path = _item_path(work_id, root)
    payload = _read_json(path)
    if payload is None:
        return None
    payload["status"] = "ABANDONED"
    payload["last_error"] = str(reason or "")[:500]
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, payload)
    return payload


def ready_work(
    *,
    root: str | Path | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    items_dir = _root(root) / "items"
    if not items_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(items_dir.glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        if payload.get("status") not in {"READY", "RETRYABLE", "RUNNING"}:
            continue
        if int(payload.get("attempts") or 0) >= DEFAULT_MAX_ATTEMPTS:
            continue
        rows.append(payload)
        if len(rows) >= int(limit):
            break
    return rows


def _restore_book(item: Mapping[str, Any]):
    from research.auto_research.paper_book import PaperBook

    snap = dict(item.get("pre_mutation_book_snapshot") or {})
    risk = dict(snap.get("risk_config") or {})
    book = PaperBook(
        capital=float(snap.get("capital") or 100_000.0),
        risk_per_trade_pct=float(risk.get("risk_per_trade_pct") or 0.01),
        max_position_pct=float(risk.get("max_position_pct") or 0.10),
        max_total_risk_pct=float(risk.get("max_total_risk_pct") or 0.05),
        max_positions=int(risk.get("max_positions") or 5),
        slippage_bps=float(risk.get("slippage_bps") or 0.0),
    )
    book.restore(snap)
    sectors = dict(item.get("held_sector_by_symbol") or {})
    for pos in (getattr(book, "open", {}) or {}).values():
        try:
            pos.sector = str(sectors.get(str(pos.symbol or "").upper()) or "")
        except Exception:
            pass
    return book


def _frozen_policy_batch(
    item: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate one Challenger using only decision-time frozen inputs.

    Canonical individual eligibility is inherited from the frozen production
    decision.  Evolution may make an eligible candidate more conservative or
    re-rank it, but can never resurrect a candidate the production hard gates
    rejected at decision time.
    """
    individual = dict(item.get("individual_decisions_by_symbol") or {})
    decisions: list[SimpleNamespace] = []
    ranked: list[tuple[float, SimpleNamespace]] = []

    for snap in snapshots:
        symbol = str(snap.get("symbol") or "").upper()
        frozen = dict(individual.get(symbol) or {})
        context = dict(snap.get("context") or {})
        card = dict(snap.get("card") or {})
        breakdown = dict(frozen.get("breakdown") or {})
        baseline = dict(breakdown.get("pre_evolution_decision") or {})
        original_decision = str(
            baseline.get("decision") or frozen.get("decision") or "REJECT"
        )
        reason = str(
            baseline.get("reason_code") or frozen.get("reason_code") or "NOT_EVALUATED"
        )
        detail = str(
            baseline.get("detail") or frozen.get("detail") or ""
        )

        if original_decision == "ENTER_NOW":
            base = dict(
                breakdown.get("pre_evolution_breakdown")
                or breakdown
            )
            # Never carry the previous Champion's Evolution wrapper into the
            # Challenger's base score.
            base.pop("evolution_policy", None)
            base.pop("learning_challenger", None)
            weighted = policy_eval.apply_policy_weights(base, context, policy)

            min_sample = (policy.get("weights") or {}).get("min_empirical_sample")
            sample_size = int((context.get("empirical") or {}).get("sample_size") or 0)
            if min_sample is not None and sample_size < int(float(min_sample)):
                decision = "BLOCK"
                reason = "EVOLUTION_MIN_SAMPLE_NOT_MET"
                detail = (
                    f"Evolution policy requires empirical_n>={int(float(min_sample))}; "
                    f"observed {sample_size}"
                )
            else:
                decision = "ENTER_NOW"
                reason = "ELIGIBLE"
                detail = "passed frozen production eligibility"

            learned = dict((breakdown.get("learning_challenger") or {}))
            score = float(weighted.get("selection_rank") or 0.0) + float(
                learned.get("adjustment") or 0.0
            )
            weighted["learning_challenger"] = learned
            weighted["evolution_policy"] = {
                "policy_id": str(policy.get("policy_id") or ""),
                "version": int(policy.get("version") or 1),
                "status": str(policy.get("status") or ""),
            }
        else:
            # Hard/individual production rejection is authoritative.  The
            # Challenger is not allowed to turn it into ENTER_NOW.
            decision = original_decision
            score = float(frozen.get("selection_score") or 0.0)
            weighted = breakdown

        obj = SimpleNamespace(
            symbol=symbol,
            decision=decision,
            reason_code=reason,
            detail=detail,
            card=card,
            context=context,
            selection_score=score,
            portfolio={},
            group="",
            breakdown=weighted,
        )
        decisions.append(obj)
        if decision == "ENTER_NOW":
            ranked.append((score, obj))

    ranked.sort(key=lambda pair: (-float(pair[0]), str(pair[1].symbol)))
    if ranked:
        from product.portfolio_selection_authority import apply_portfolio_authority

        book = _restore_book(item)
        kept, _diverted = apply_portfolio_authority(
            ranked,
            book=book,
            max_new=int(item.get("max_new") or 3),
            regime=str(item.get("regime") or "RISK_ON"),
            correlations=dict(item.get("correlations") or {}),
        )
        kept_ids = {id(decision) for _, decision in kept}
        for decision in decisions:
            if decision.decision == "ENTER_NOW" and id(decision) not in kept_ids:
                decision.decision = "NO_TRADE"
                decision.reason_code = "NOT_TOP_OF_PORTFOLIO"
                decision.detail = "not selected by frozen portfolio authority"

    snapshot_id_by_symbol = {
        str(s.get("symbol") or "").upper(): str(s.get("market_snapshot_id") or "")
        for s in snapshots
    }
    verdicts: list[dict[str, Any]] = []
    for decision in decisions:
        verdicts.append({
            "policy_id": str(policy.get("policy_id") or ""),
            "market_snapshot_id": snapshot_id_by_symbol.get(decision.symbol, ""),
            "domain": str((item.get("bundle") or {}).get("domain") or "EQUITY"),
            "symbol": decision.symbol,
            "decision": decision.decision,
            "reason_code": decision.reason_code,
            "detail": decision.detail,
            "adjusted_score": float(decision.selection_score or 0.0),
            "selection_score": float(decision.selection_score or 0.0),
            "breakdown": dict(getattr(decision, "breakdown", {}) or {}),
            "entry": decision.context.get("entry"),
            "stop": decision.context.get("stop"),
            "target": decision.context.get("target"),
            "sector": decision.context.get("sector"),
            "setup_label": decision.context.get("setup_label"),
        })
    return verdicts


def process_work_item(
    work_id: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Process one READY item in the current process.

    Production calls this only inside a killable child process.  Tests may
    call it directly for deterministic restart/idempotency verification.
    """
    path = _item_path(work_id, root)
    item = _read_json(path)
    if item is None:
        raise KeyError(f"unknown deferred Evolution work item {work_id}")
    if item.get("status") == "SUCCEEDED":
        existing = _read_json(_result_path(work_id, root))
        return existing or {}
    if int(item.get("evaluator_schema_version") or 0) != EVALUATOR_SCHEMA_VERSION:
        raise RuntimeError(
            f"deferred Evolution evaluator schema mismatch: "
            f"item={item.get('evaluator_schema_version')} "
            f"runtime={EVALUATOR_SCHEMA_VERSION}"
        )
    if item.get("status") not in {"READY", "RETRYABLE", "RUNNING"}:
        raise RuntimeError(
            f"deferred Evolution work {work_id} is not READY: {item.get('status')}"
        )

    policies = [dict(p) for p in item.get("challenger_policies") or []]

    def evaluator(snapshots, policy):
        return _frozen_policy_batch(item, snapshots, policy)

    result = tournament.evaluate_challengers_from_bundle(
        dict(item.get("bundle") or {}),
        max_new=int(item.get("max_new") or 3),
        challenger_policies=policies,
        challenger_batch_evaluator=evaluator,
    )
    from product.evolution.consensus_board import save_latest_consensus

    save_latest_consensus(result)
    _write_json(_result_path(work_id, root), result)
    item["status"] = "SUCCEEDED"
    item["finished_at"] = datetime.now(timezone.utc).isoformat()
    item["last_error"] = ""
    _write_json(path, item)
    return result


def _default_worker_command(root_value: str, work_id: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "product.evolution.deferred_worker",
        "--root",
        str(root_value),
        "--work-id",
        str(work_id),
    ]


def process_ready_isolated(
    *,
    root: str | Path | None = None,
    limit: int = 2,
    timeout_seconds: float = DEFAULT_CHILD_TIMEOUT_SECONDS,
    work_ids: Sequence[str] | None = None,
    command_factory=None,
) -> list[dict[str, Any]]:
    """Consume READY work in killable subprocesses.

    The parent supervisor never runs Challenger code directly.  If a child
    does not exit before the deadline it is terminated (and killed if needed),
    the work becomes RETRYABLE/FAILED, and the tournament job itself returns.
    """
    root_path = _root(root)
    factory = command_factory or _default_worker_command
    outcomes: list[dict[str, Any]] = []

    candidates = ready_work(root=root_path, limit=max(limit, 100 if work_ids else limit))
    if work_ids is not None:
        wanted = {str(value or "") for value in work_ids}
        candidates = [row for row in candidates if str(row.get("work_id") or "") in wanted]
    for payload in candidates[: int(limit)]:
        work_id = str(payload.get("work_id") or "")
        path = _item_path(work_id, root_path)
        attempts = int(payload.get("attempts") or 0) + 1
        payload["attempts"] = attempts
        payload["status"] = "RUNNING"
        payload["last_started_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(path, payload)

        command = list(factory(str(root_path), work_id))
        proc = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        timed_out = False
        try:
            proc.wait(timeout=max(0.01, float(timeout_seconds)))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)

        current = _read_json(path) or payload
        if timed_out:
            current["status"] = (
                "FAILED" if attempts >= DEFAULT_MAX_ATTEMPTS else "RETRYABLE"
            )
            current["last_error"] = (
                f"Challenger worker exceeded {float(timeout_seconds):.3f}s hard timeout"
            )
            current["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(path, current)
            outcomes.append({
                "work_id": work_id,
                "status": current["status"],
                "timed_out": True,
                "attempts": attempts,
            })
            continue

        if proc.returncode == 0 and _result_path(work_id, root_path).exists():
            current = _read_json(path) or current
            outcomes.append({
                "work_id": work_id,
                "status": current.get("status") or "SUCCEEDED",
                "timed_out": False,
                "attempts": attempts,
            })
            continue

        current["status"] = (
            "FAILED" if attempts >= DEFAULT_MAX_ATTEMPTS else "RETRYABLE"
        )
        current["last_error"] = f"Challenger worker exited with code {proc.returncode}"
        current["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(path, current)
        outcomes.append({
            "work_id": work_id,
            "status": current["status"],
            "timed_out": False,
            "attempts": attempts,
            "exitcode": proc.returncode,
        })

    return outcomes
