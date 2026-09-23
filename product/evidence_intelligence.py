"""Evidence-backed intelligence for one canonical Decision.

Conservative invariants:
- freeze decision-time features before outcomes exist;
- read only observations strictly older than the decision timestamp;
- distinguish raw sample size from effective sample size;
- keep historical priors separate from forward decision outcomes;
- never open a trade or unlock live money.

The learner is intentionally auditable: robust nearest-neighbour history plus
recency/regime/setup weighting and uncertainty-aware expectancy.
"""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from product.decision import Decision
from research import feature_schema as FS

SCHEMA_VERSION = 1
FORMULA_VERSION = "evidence_v2"
MIN_ANALOGS = 12
CLAIM_MIN_EFFECTIVE_N = 20.0
CONFIDENT_MIN_EFFECTIVE_N = 30.0
DEFAULT_K = 50
HALF_LIFE_DAYS = 365.0
SHRINKAGE_K = 8.0


_EVIDENCE_BATCH = threading.local()


@contextmanager
def evidence_read_batch():
    """Freeze prior-evidence reads for one decision-board construction.

    Decisions on the same saved scan share one point-in-time cutoff. Loading and
    JSON-decoding the identical settled corpus for every symbol is quadratic I/O
    as the evidence store grows. Within this explicit batch, cache only the raw
    rows keyed by strict cutoff; no samples are dropped or reweighted. The cache
    is discarded at exit, so later board builds observe newly settled evidence.
    """
    previous = getattr(_EVIDENCE_BATCH, "rows_by_cutoff", None)
    _EVIDENCE_BATCH.rows_by_cutoff = {}
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(_EVIDENCE_BATCH, "rows_by_cutoff")
            except AttributeError:
                pass
        else:
            _EVIDENCE_BATCH.rows_by_cutoff = previous


def _prior_observation_rows(before_ts: str) -> list[dict[str, Any]]:
    from research.feature_store import load_observations

    cache = getattr(_EVIDENCE_BATCH, "rows_by_cutoff", None)
    key = str(before_ts or "")
    if cache is not None and key in cache:
        return cache[key]
    rows = load_observations(
        kind="DECISION",
        require_outcome=True,
        before_ts=key,
    )
    if cache is not None:
        cache[key] = rows
    return rows


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        got = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return got if got.tzinfo else got.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.fromisoformat(text[:10]).replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _cat(value: Any, allowed: tuple[str, ...] | None) -> str:
    text = str(value or "UNKNOWN").upper()
    return text if not allowed or text in set(allowed) else "UNKNOWN"


def decision_features(decision: Decision) -> dict[str, Any]:
    """Canonical feature vector derived only from the Decision payload."""
    t = dict(decision.technical_evidence or {})
    entry, stop, target = _f(decision.entry), _f(decision.stop), _f(decision.target)
    rr = _f(t.get("reward_risk"))
    if rr is None and entry is not None and stop is not None and target is not None:
        risk = abs(entry - stop)
        rr = (target - entry) / risk if risk > 0 else None

    raw = {
        "rsi": _f(t.get("rsi")),
        "atr_pct": _f(t.get("atr_pct")),
        "dist_from_high_pct": _f(t.get("dist_from_high_pct")),
        "rel_strength": _f(t.get("rel_strength")),
        "clv": _f(t.get("clv")),
        "volume_z": _f(t.get("volume_z")),
        "delivery_pct": _f(t.get("delivery_pct")),
        "adr_pct": _f(t.get("adr_pct")),
        "liquidity_cr": _f(t.get("liquidity_cr")),
        "dist_20ema_pct": _f(t.get("dist_20ema_pct")),
        "dist_50dma_pct": _f(t.get("dist_50dma_pct")),
        "pct_from_pivot": _f(t.get("pct_from_pivot")),
        "base_depth_pct": _f(t.get("base_depth_pct")),
        "base_days": _f(t.get("base_days")),
        "volume_ratio": _f(t.get("volume_ratio")),
        "contraction_ratio": _f(t.get("contraction_ratio")),
        "rs_percentile": _f(t.get("rs_percentile")),
        "sector_rank_percentile": _f(t.get("sector_rank_percentile")),
        "volatility_percentile": _f(t.get("volatility_percentile")),
        "reward_risk": rr,
        "quality_score": _f(t.get("quality_score")),
        "accum_score": _f(t.get("accum_score")),
        "regime": _cat(
            t.get("regime") or decision.market_state,
            FS.FEATURE_REGISTRY["regime"].categories,
        ),
        "breadth_pct_above_50dma": _f(t.get("breadth_pct_above_50dma")),
        "sector_strength": _f(t.get("sector_strength")),
        "index_trend": _cat(
            t.get("index_trend"),
            FS.FEATURE_REGISTRY["index_trend"].categories,
        ),
        "correlation_regime": _cat(
            t.get("correlation_regime"),
            FS.FEATURE_REGISTRY["correlation_regime"].categories,
        ),
        "vix": _f(t.get("vix")),
        "usdinr": _f(t.get("usdinr")),
        "crude_usd": _f(t.get("crude_usd")),
    }
    return FS.canonicalize(raw)


def _observation_id(decision_id: str) -> str:
    return f"decision::{decision_id}"


def freeze_decision(
    decision: Decision,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write-once decision observation. Repeated reads are idempotent."""
    from product.decision_ranking import decision_context_key
    from research.feature_store import snapshot

    meta = {
        "decision_id": decision.decision_id,
        "setup": decision.setup,
        "decision_state": decision.state,
        "market_state": decision.market_state,
        "sector_state": decision.sector_state,
        "entry": decision.entry,
        "stop": decision.stop,
        "target": decision.target,
        "expected_R": decision.computed_expected_R,
        "source_scan_id": decision.source_scan_id,
        "evidence_snapshot_id": decision.evidence_snapshot_id,
        "evidence_class": decision.evidence_class,
        "context_key": decision_context_key(decision),
        "predicted_p": (evidence or {}).get("calibrated_p_positive_R"),
        "raw_predicted_p": (evidence or {}).get("p_positive_R"),
        "prediction_source": (evidence or {}).get("formula_version") or "",
        "historical_confidence": (evidence or {}).get("historical_confidence"),
        "decision_confidence": (evidence or {}).get("decision_confidence"),
        "challenger_predicted_p": (
            ((evidence or {}).get("challenger_shadow") or {}).get("p_positive_R")
        ),
        "challenger_model_version": (
            ((evidence or {}).get("challenger_shadow") or {}).get("model_version") or ""
        ),
        "challenger_status": (
            ((evidence or {}).get("challenger_shadow") or {}).get("status") or ""
        ),
        "thesis_hash": str((decision.provenance or {}).get("thesis_hash") or ""),
        "not_live": True,
    }
    return snapshot(
        _observation_id(decision.decision_id),
        decision.symbol,
        "DECISION",
        decision_features(decision),
        ts=decision.generated_at,
        reason=decision.state,
        meta=meta,
    )


def settle_decision(
    decision_id: str,
    realized_R: float,
    *,
    evidence_class: str,
    not_pnl: bool = False,
    classification: str = "",
    resolved_at: str = "",
) -> dict[str, Any]:
    """Attach an immutable labelled outcome with explicit evidence provenance."""
    from research.feature_store import set_outcome
    return set_outcome(
        _observation_id(decision_id),
        float(realized_R),
        outcome_meta={
            "evidence_class": str(evidence_class or ""),
            "not_pnl": bool(not_pnl),
            "classification": str(classification or ""),
            "resolved_at": str(resolved_at or ""),
        },
    )


def record_resolved_prediction(
    decision_id: str,
    realized_R: float,
    *,
    evidence_class: str,
    not_pnl: bool = False,
    classification: str = "",
    resolved_at: str = "",
) -> dict[str, Any]:
    """Settle the feature observation and, for real taken paper, calibration.

    Counterfactual labels improve the selection corpus but never enter the
    probability calibration ledger as if they were executed trades.
    """
    settled = settle_decision(
        decision_id,
        realized_R,
        evidence_class=evidence_class,
        not_pnl=not_pnl,
        classification=classification,
        resolved_at=resolved_at,
    )
    out: dict[str, Any] = {"feature_store": settled, "calibration": None}
    if not_pnl:
        return out
    if str(evidence_class or "").upper() not in {"PAPER_FORWARD", "REAL_FORWARD_PAPER"}:
        return out
    try:
        from research.feature_store import get_observation
        row = get_observation(_observation_id(decision_id)) or {}
        meta = dict(row.get("meta") or {})
        predicted_p = _f(meta.get("predicted_p"))
        if predicted_p is None:
            return out
        if predicted_p >= 0.70:
            tier = "high"
        elif predicted_p >= 0.55:
            tier = "good"
        else:
            tier = "watch"
        from product.decision_calibration import DecisionCalibrationEngine
        out["calibration"] = DecisionCalibrationEngine().record(
            predicted_confidence=tier,
            predicted_p=predicted_p,
            prediction_source=str(meta.get("prediction_source") or ""),
            realized_win=float(realized_R) > 0.0,
            setup=str(meta.get("setup") or ""),
            regime=str(meta.get("market_state") or ""),
            sector=str(meta.get("sector_state") or ""),
            decision_as_of=str(row.get("ts") or ""),
            outcome_as_of=str(resolved_at or datetime.now(timezone.utc).isoformat()),
        )
    except Exception as exc:
        out["calibration"] = {"error": str(exc)[:200]}
    return out


def _policy_prior(setup: str) -> dict[str, Any]:
    """Historical-replay prior. Never silently becomes forward evidence."""
    try:
        from product.learning_policy_store import load_policies
        policies = load_policies().get("policies") or []
    except Exception:
        policies = []
    try:
        from product.trading_thesis import manifest as thesis_manifest
        thesis_hash = str(thesis_manifest().get("thesis_hash") or "")
    except Exception:
        thesis_hash = ""
    wanted = f"HIST_SETUP::{thesis_hash}::{setup}" if thesis_hash else ""
    row = next(
        (dict(p) for p in policies if wanted and str(p.get("policy_id") or "") == wanted),
        {},
    )
    return {
        "policy_id": str(row.get("policy_id") or ""),
        "n": int(row.get("sample_size") or 0),
        "mean_R": _f(row.get("expectancy_R")),
        "confidence_score": _f(row.get("historical_confidence_score")) or 0.0,
        "reproduced_positive": bool(row.get("historical_reproduced_positive")),
        "generation_fingerprint": str(row.get("generation_fingerprint") or ""),
        "source": str(row.get("evidence_source") or "historical_replay"),
    }


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _robust_scales(rows: list[dict[str, Any]], names: list[str]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name in names:
        vals = [_f((r.get("features") or {}).get(name)) for r in rows]
        arr = np.asarray([v for v in vals if v is not None], dtype=float)
        if arr.size < 3:
            scales[name] = 1.0
            continue
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = float(q75 - q25)
        std = float(np.std(arr))
        scales[name] = max(iqr / 1.349 if iqr > 0 else std, std * 0.25, 1e-6)
    return scales


def _days_between(older: Any, newer: Any) -> float:
    a, b = _dt(older), _dt(newer)
    if a is None or b is None:
        return HALF_LIFE_DAYS
    return max(0.0, (b - a).total_seconds() / 86400.0)


def _analogs(decision: Decision, *, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    rows = _prior_observation_rows(str(decision.generated_at or ""))
    if not rows:
        return []

    query = decision_features(decision)
    numeric = [
        name for name in FS.FEATURE_NAMES
        if FS.FEATURE_REGISTRY[name].dtype != "categorical" and _f(query.get(name)) is not None
    ]
    if len(numeric) < 2:
        return []

    scales = _robust_scales(rows, numeric)
    setup_matches = sum(
        1 for r in rows
        if str((r.get("meta") or {}).get("setup") or "") == str(decision.setup or "")
    )
    prefer_exact = setup_matches >= MIN_ANALOGS

    ranked: list[dict[str, Any]] = []
    for row in rows:
        feats = dict(row.get("features") or {})
        shared = [
            name for name in numeric
            if _f(feats.get(name)) is not None and scales.get(name, 0) > 0
        ]
        if len(shared) < 2:
            continue
        z2 = [
            ((_f(feats.get(name)) - _f(query.get(name))) / scales[name]) ** 2
            for name in shared
        ]
        distance = math.sqrt(sum(z2) / len(z2))
        similarity = math.exp(-0.5 * distance)
        meta = dict(row.get("meta") or {})
        same_setup = str(meta.get("setup") or "") == str(decision.setup or "")
        if prefer_exact and not same_setup:
            continue
        setup_weight = 1.0 if same_setup else 0.35

        old_regime = str(feats.get("regime") or meta.get("market_state") or "")
        now_regime = str(query.get("regime") or decision.market_state or "")
        regime_weight = 1.0 if old_regime and now_regime and old_regime == now_regime else 0.70
        sector_weight = (
            1.0 if str(meta.get("sector_state") or "") == str(decision.sector_state or "")
            and str(decision.sector_state or "") else 0.85
        )
        age_days = _days_between(row.get("ts"), decision.generated_at)
        recency = 0.5 ** (age_days / HALF_LIFE_DAYS)
        problems = list(row.get("validation") or [])
        quality = max(0.55, 1.0 - min(0.45, len(problems) * 0.05))
        outcome_meta = dict(row.get("outcome_meta") or {})
        lane = str(outcome_meta.get("evidence_class") or "").upper()
        if lane in {"PAPER_FORWARD", "REAL_FORWARD_PAPER"}:
            lane_weight = 1.0
        elif lane in {"FORWARD_COUNTERFACTUAL", "COUNTERFACTUAL_FORWARD"}:
            lane_weight = 0.65
        elif lane in {"HISTORICAL_REPLAY", "BACKTEST"}:
            lane_weight = 0.35
        else:
            lane_weight = 0.50
        weight = (
            similarity * setup_weight * regime_weight * sector_weight
            * recency * quality * lane_weight
        )
        if weight <= 0:
            continue
        ranked.append({
            "observation_id": row.get("observation_id"),
            "symbol": row.get("symbol"),
            "ts": row.get("ts"),
            "outcome_R": float(row.get("outcome")),
            "similarity": similarity,
            "weight": weight,
            "same_setup": same_setup,
            "same_regime": old_regime == now_regime if old_regime and now_regime else False,
            "shared_features": len(shared),
            "evidence_class": lane,
            "not_pnl": bool(outcome_meta.get("not_pnl")),
        })

    ranked.sort(key=lambda r: (-float(r["weight"]), -float(r["similarity"]), str(r["ts"])))
    return ranked[: max(1, int(k))]


def _weighted_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "raw_n": 0, "effective_n": 0.0, "mean_R": None,
            "shrunk_mean_R": None, "median_R": None, "lower_95_R": None,
            "upper_95_R": None, "p_positive_R": None,
            "positive_rate": None, "mean_similarity": None,
        }
    w = np.asarray([max(0.0, float(r["weight"])) for r in rows], dtype=float)
    y = np.asarray([float(r["outcome_R"]) for r in rows], dtype=float)
    if w.sum() <= 0:
        return _weighted_stats([])
    w = w / w.sum()
    mean = float(np.sum(w * y))
    variance = float(np.sum(w * (y - mean) ** 2))
    sum_w2 = float(np.sum(w ** 2))
    effective_n = (1.0 / sum_w2) if sum_w2 > 0 else 0.0
    shrunk = mean * (effective_n / (effective_n + SHRINKAGE_K))
    se = math.sqrt(max(variance, 1e-12) / max(effective_n, 1.0))
    lower = shrunk - 1.96 * se
    upper = shrunk + 1.96 * se
    p_edge = _normal_cdf(shrunk / se) if se > 0 else (1.0 if shrunk > 0 else 0.0)
    positives = float(np.sum(w * (y > 0).astype(float)))
    # Individual-outcome probability is a different estimand from P(true mean R > 0).
    # Use a small symmetric Beta prior to shrink thin/weighted samples toward 50%.
    prior_alpha = 2.0
    prior_beta = 2.0
    effective_wins = positives * effective_n
    outcome_p = (effective_wins + prior_alpha) / (
        effective_n + prior_alpha + prior_beta
    )
    outcome_se = math.sqrt(
        max(outcome_p * (1.0 - outcome_p), 1e-12)
        / max(effective_n + prior_alpha + prior_beta + 1.0, 1.0)
    )
    outcome_lower = max(0.0, outcome_p - 1.96 * outcome_se)
    outcome_upper = min(1.0, outcome_p + 1.96 * outcome_se)
    sims = np.asarray([float(r["similarity"]) for r in rows], dtype=float)
    return {
        "raw_n": int(len(rows)),
        "effective_n": round(effective_n, 2),
        "mean_R": round(mean, 4),
        "shrunk_mean_R": round(shrunk, 4),
        "median_R": round(float(np.median(y)), 4),
        "lower_95_R": round(lower, 4),
        "upper_95_R": round(upper, 4),
        "p_edge_positive": round(p_edge, 4),
        "outcome_p_posterior": round(outcome_p, 4),
        "outcome_p_lower_95": round(outcome_lower, 4),
        "outcome_p_upper_95": round(outcome_upper, 4),
        "positive_rate": round(positives, 4),
        "mean_similarity": round(float(np.sum(w * sims)), 4),
    }


def _sample_cap(effective_n: float) -> float:
    if effective_n < 8:
        return 49.0
    if effective_n < CLAIM_MIN_EFFECTIVE_N:
        return 69.0
    if effective_n < CONFIDENT_MIN_EFFECTIVE_N:
        return 79.0
    return 95.0


def evidence_read(decision: Decision, *, k: int = DEFAULT_K) -> dict[str, Any]:
    """Return evidence strength, uncertainty and calibrated positive-R odds."""
    prior = _policy_prior(decision.setup)
    analogs = _analogs(decision, k=k)
    stats = _weighted_stats(analogs)
    eff = float(stats.get("effective_n") or 0.0)

    analog_score = 0.0
    edge_p = stats.get("p_edge_positive")
    if edge_p is not None:
        analog_score = min(_sample_cap(eff), max(0.0, float(edge_p) * 100.0))

    prior_score = float(prior.get("confidence_score") or 0.0)
    prior_ready = bool(prior.get("reproduced_positive"))
    if analogs:
        analog_weight = min(0.80, eff / (eff + 20.0))
        historical_confidence = (
            analog_score * analog_weight
            + (prior_score if prior_ready else 0.0) * (1.0 - analog_weight)
        )
        stage = (
            "ANALOG_CONFIRMED" if eff >= CONFIDENT_MIN_EFFECTIVE_N
            else "ANALOG_CALIBRATING" if eff >= CLAIM_MIN_EFFECTIVE_N
            else "ANALOG_EARLY"
        )
    elif prior_ready:
        historical_confidence = min(79.0, prior_score)
        stage = "HISTORICAL_POLICY_PRIOR"
    else:
        historical_confidence = min(49.0, prior_score)
        stage = "INSUFFICIENT_EVIDENCE"

    historical_confidence = round(max(0.0, min(95.0, historical_confidence)), 1)
    similarity_score = round(float(stats.get("mean_similarity") or 0.0) * 100.0, 1)

    # All PIT analogs may inform research confidence, but a historical replay or
    # counterfactual is not an executed forward trial. Keep the broad research
    # estimate separate from the narrower real-forward probability estimand.
    research_raw_p = stats.get("outcome_p_posterior")
    research_probability_estimate = (
        float(research_raw_p)
        if research_raw_p is not None and eff >= CLAIM_MIN_EFFECTIVE_N
        else None
    )
    forward_analogs = [
        row for row in analogs
        if str(row.get("evidence_class") or "").upper()
        in {"PAPER_FORWARD", "REAL_FORWARD_PAPER"}
        and not bool(row.get("not_pnl"))
    ]
    forward_stats = _weighted_stats(forward_analogs)
    forward_eff = float(forward_stats.get("effective_n") or 0.0)
    raw_forward_p = forward_stats.get("outcome_p_posterior")
    measured_probability = (
        float(raw_forward_p)
        if raw_forward_p is not None and forward_eff >= CLAIM_MIN_EFFECTIVE_N
        else None
    )
    calibrated = measured_probability
    calibration_applied = False
    calibration = {
        "status": "INSUFFICIENT_EVIDENCE",
        "sample_size": 0,
        "actual_hit_rate": None,
        "expected_p": None,
        "probability_status": "NO_EXPLICIT_PROBABILITIES",
        "probability_sample_size": 0,
        "calibration_gap": None,
        "calibration_actionable": False,
        "calibration_adjustment": 0.0,
    }
    if measured_probability is not None:
        try:
            from product.decision_calibration import DecisionCalibrationEngine
            calibration = DecisionCalibrationEngine().summary(
                setup=decision.setup,
                regime=decision.market_state,
                sector=decision.sector_state,
                prediction_source=FORMULA_VERSION,
            )
            # A setup-quality hit rate is not a probability-calibration sample.
            # Adjust a measured probability only when the historical ledger has
            # enough explicit point-in-time predicted_p observations of its own.
            if (
                calibration.get("probability_status") == "MEASURED"
                and calibration.get("calibration_actionable") is True
                and calibration.get("calibration_adjustment") is not None
            ):
                adjustment = float(calibration.get("calibration_adjustment") or 0.0)
                calibrated = max(0.01, min(0.99, measured_probability - adjustment))
                calibration_applied = bool(adjustment)
        except Exception:
            pass

    match_factor = 0.60 + 0.40 * min(1.0, max(0.0, similarity_score / 100.0))
    decision_confidence = round(historical_confidence * match_factor, 1)
    lane_counts: dict[str, int] = {}
    for row in analogs:
        lane = str(row.get("evidence_class") or "UNKNOWN")
        lane_counts[lane] = lane_counts.get(lane, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "setup": decision.setup,
        "stage": stage,
        "historical_confidence": historical_confidence,
        "setup_similarity": similarity_score,
        "decision_confidence": decision_confidence,
        "raw_n": stats.get("raw_n", 0),
        "effective_n": stats.get("effective_n", 0.0),
        "evidence_lane_counts": lane_counts,
        "mean_R": stats.get("mean_R"),
        "shrunk_mean_R": stats.get("shrunk_mean_R"),
        "median_R": stats.get("median_R"),
        "lower_95_R": stats.get("lower_95_R"),
        "upper_95_R": stats.get("upper_95_R"),
        "positive_rate": stats.get("positive_rate"),
        "p_edge_positive": stats.get("p_edge_positive"),
        "research_positive_R_estimate": research_probability_estimate,
        "research_positive_R_lower_95": (
            stats.get("outcome_p_lower_95")
            if research_probability_estimate is not None else None
        ),
        "research_positive_R_upper_95": (
            stats.get("outcome_p_upper_95")
            if research_probability_estimate is not None else None
        ),
        "research_probability_scope": "MIXED_PIT_EVIDENCE",
        "forward_probability_raw_n": int(forward_stats.get("raw_n") or 0),
        "forward_probability_effective_n": forward_stats.get("effective_n", 0.0),
        "p_positive_R": measured_probability,
        "p_positive_R_lower_95": (
            forward_stats.get("outcome_p_lower_95") if measured_probability is not None else None
        ),
        "p_positive_R_upper_95": (
            forward_stats.get("outcome_p_upper_95") if measured_probability is not None else None
        ),
        "probability_evidence_scope": (
            "REAL_FORWARD_PAPER" if measured_probability is not None
            else "INSUFFICIENT_REAL_FORWARD_PAPER"
        ),
        "calibrated_p_positive_R": None if calibrated is None else round(calibrated, 4),
        "calibration_applied": calibration_applied,
        "calibration_contract_version": "explicit_probability_only_v2",
        "calibration": calibration,
        "historical_prior": prior,
        "nearest_analogs": [
            {
                "observation_id": r["observation_id"],
                "symbol": r["symbol"],
                "ts": r["ts"],
                "outcome_R": round(float(r["outcome_R"]), 4),
                "similarity": round(float(r["similarity"]), 4),
                "weight": round(float(r["weight"]), 6),
                "same_setup": bool(r["same_setup"]),
                "same_regime": bool(r["same_regime"]),
                "evidence_class": str(r.get("evidence_class") or ""),
                "not_pnl": bool(r.get("not_pnl")),
            }
            for r in analogs[:10]
        ],
        "is_win_probability": measured_probability is not None,
        "probability_contract_version": "real_forward_only_v2",
        "affects_selection": False,
        "paper_only_learning": True,
        "live_locked": True,
        "note": (
            "Mixed historical/counterfactual evidence may strengthen research confidence, "
            "but only sufficiently sampled executed forward-paper outcomes are labelled as "
            "a win probability. This projection is shadow-only and cannot promote a scanner "
            "decision; promotion requires a separately validated policy challenger."
        ),
    }


def enrich(decision: Decision) -> Decision:
    """Attach evidence intelligence and freeze its pre-outcome prediction."""
    evidence = evidence_read(decision)
    try:
        from product.challenger_learning import score_decision
        evidence["challenger_shadow"] = score_decision(decision)
    except Exception:
        evidence["challenger_shadow"] = {
            "available": False,
            "affects_selection": False,
            "live_locked": True,
        }
    freeze_decision(decision, evidence=evidence)
    historical = dict(decision.historical_evidence or {})
    historical["evidence_intelligence"] = evidence
    provenance = dict(decision.provenance or {})
    provenance["learning_observation_id"] = _observation_id(decision.decision_id)
    provenance["evidence_intelligence_version"] = FORMULA_VERSION
    return replace(
        decision,
        historical_evidence=historical,
        provenance=provenance,
        decision_id=decision.decision_id,
    )
