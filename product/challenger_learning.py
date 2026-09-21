"""Versioned champion/challenger learner for PAPER selection.

The model cannot create a BUY.  It can only produce a bounded ordering adjustment
among candidates that the existing paper authority has already declared eligible.

Lifecycle:
  OBSERVING -> SHADOW_CANDIDATE -> PAPER_ACTIVE
                    |                   |
                    +---- REJECTED -----+

Historical evaluation uses purged/embargoed folds.  Promotion requires fresh
forward predictions from the exact challenger version.  Live execution is outside
this module and remains structurally locked.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.runtime_paths import logs_dir

SCHEMA_VERSION = 1
DEFAULT_PATH = logs_dir() / "product" / "challenger_learning.json"

OBSERVING = "OBSERVING"
SHADOW_CANDIDATE = "SHADOW_CANDIDATE"
PAPER_ACTIVE = "PAPER_ACTIVE"
DEMOTED = "DEMOTED"
REJECTED = "REJECTED"

MIN_TOTAL = 60
MIN_REAL_FORWARD = 20
PROMOTION_TOTAL = 100
PROMOTION_REAL_FORWARD = 30
MIN_FORWARD_COMPARE = 30
DEMOTION_FORWARD_COMPARE = 60
MIN_BRIER_IMPROVEMENT = 0.01

MODEL_FEATURES = (
    "rsi",
    "atr_pct",
    "dist_from_high_pct",
    "rel_strength",
    "clv",
    "volume_z",
    "delivery_pct",
    "adr_pct",
    "liquidity_cr",
    "dist_20ema_pct",
    "dist_50dma_pct",
    "pct_from_pivot",
    "base_depth_pct",
    "base_days",
    "volume_ratio",
    "contraction_ratio",
    "rs_percentile",
    "sector_rank_percentile",
    "volatility_percentile",
    "reward_risk",
    "quality_score",
    "accum_score",
    "breadth_pct_above_50dma",
    "sector_strength",
    "vix",
)

REAL_LANES = {"PAPER_FORWARD", "REAL_FORWARD_PAPER"}
COUNTERFACTUAL_LANES = {"FORWARD_COUNTERFACTUAL", "COUNTERFACTUAL_FORWARD"}
HISTORICAL_LANES = {"HISTORICAL_REPLAY", "BACKTEST"}


def store_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_CHALLENGER_LEARNING")
    return Path(override) if override else DEFAULT_PATH


def _empty() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "current": None,
        "history": [],
        "updated_at": "",
        "live_locked": True,
        "affects_selection": False,
    }


def load(path: str | Path | None = None) -> dict[str, Any]:
    target = store_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return _empty()
    if not isinstance(payload, dict):
        return _empty()
    payload.setdefault("history", [])
    payload["live_locked"] = True
    current = payload.get("current")
    payload["affects_selection"] = bool(
        isinstance(current, dict) and current.get("status") == PAPER_ACTIVE
    )
    return payload


def save(payload: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = SCHEMA_VERSION
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["live_locked"] = True
    current = data.get("current")
    data["affects_selection"] = bool(
        isinstance(current, dict) and current.get("status") == PAPER_ACTIVE
    )
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _lane_weight(row: Mapping[str, Any]) -> float:
    lane = str((row.get("outcome_meta") or {}).get("evidence_class") or "").upper()
    if lane in REAL_LANES:
        return 1.0
    if lane in COUNTERFACTUAL_LANES:
        return 0.65
    if lane in HISTORICAL_LANES:
        return 0.35
    return 0.0


def _rows() -> list[dict[str, Any]]:
    from research.feature_store import load_observations
    try:
        from product.trading_thesis import manifest as thesis_manifest
        current_thesis = str(thesis_manifest().get("thesis_hash") or "")
    except Exception:
        current_thesis = ""

    rows = load_observations(kind="DECISION", require_outcome=True)
    return [
        r for r in rows
        if _lane_weight(r) > 0
        and current_thesis
        and str((r.get("meta") or {}).get("thesis_hash") or "") == current_thesis
    ]


def _matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.full((len(rows), len(MODEL_FEATURES)), np.nan, dtype=float)
    y = np.zeros(len(rows), dtype=float)
    sw = np.ones(len(rows), dtype=float)
    for i, row in enumerate(rows):
        fv = dict(row.get("features") or {})
        for j, name in enumerate(MODEL_FEATURES):
            v = _f(fv.get(name))
            if v is not None:
                X[i, j] = v
        y[i] = 1.0 if float(row.get("outcome") or 0.0) > 0.0 else 0.0
        sw[i] = _lane_weight(row)
    return X, y, sw


def _fit_transform(
    X: np.ndarray,
    *,
    medians: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0:
        d = X.shape[1] if X.ndim == 2 else len(MODEL_FEATURES)
        return X, np.zeros(d), np.ones(d)
    if medians is None:
        # Missing optional features are legitimate in early/historical batches.
        # Avoid np.nanmedian on all-NaN columns: that warning used to let a
        # "successful" learning job finish with a hidden data-quality defect.
        finite_columns = np.any(np.isfinite(X), axis=0)
        medians = np.zeros(X.shape[1], dtype=float)
        if np.any(finite_columns):
            medians[finite_columns] = np.nanmedian(X[:, finite_columns], axis=0)
    else:
        medians = np.asarray(medians, dtype=float)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isnan(X), medians, X)
    if scales is None:
        q25 = np.nanpercentile(filled, 25, axis=0)
        q75 = np.nanpercentile(filled, 75, axis=0)
        iqr_scale = (q75 - q25) / 1.349
        std = np.std(filled, axis=0)
        scales = np.maximum(iqr_scale, std * 0.25)
        scales = np.where((~np.isfinite(scales)) | (scales < 1e-6), 1.0, scales)
    Z = (filled - medians) / scales
    return Z, medians.astype(float), scales.astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    steps: int = 600,
    lr: float = 0.04,
    l2: float = 0.25,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    Z, medians, scales = _fit_transform(X)
    n, d = Z.shape
    coef = np.zeros(d, dtype=float)
    base = float(np.average(y, weights=sample_weight)) if n else 0.5
    base = min(0.99, max(0.01, base))
    intercept = math.log(base / (1.0 - base))
    w = sample_weight / max(float(sample_weight.sum()), 1e-12)
    for _ in range(int(steps)):
        p = _sigmoid(Z @ coef + intercept)
        err = (p - y) * w
        grad = Z.T @ err + l2 * coef
        grad_i = float(err.sum())
        coef -= lr * grad
        intercept -= lr * grad_i
    return coef, float(intercept), medians, scales


def _predict(
    X: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    Z, _, _ = _fit_transform(X, medians=medians, scales=scales)
    return _sigmoid(Z @ coef + float(intercept))


def _brier(pred: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> float:
    if pred.size == 0:
        return 1.0
    losses = (pred - y) ** 2
    if weight is None:
        return float(np.mean(losses))
    return float(np.average(losses, weights=weight))


def _cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    X, y, sw = _matrix(rows)
    n = len(rows)
    if n < 10:
        return {"n": n, "challenger_brier": None, "champion_brier": None, "oof_n": 0}

    try:
        from research.harness import purged_kfold_indices
        splits = purged_kfold_indices(n, k=5, embargo=5, label_horizon=10)
    except Exception:
        indices = np.arange(n)
        splits = []
        for fold in np.array_split(indices, 5):
            test = np.asarray(fold, dtype=int)
            train = np.asarray([i for i in indices if i not in set(test.tolist())], dtype=int)
            splits.append((train, test))

    oof = np.full(n, np.nan, dtype=float)
    for train_idx, test_idx in splits:
        if len(train_idx) < 20 or len(test_idx) == 0:
            continue
        coef, intercept, med, scale = _fit(
            X[train_idx], y[train_idx], sw[train_idx]
        )
        oof[test_idx] = _predict(X[test_idx], coef, intercept, med, scale)

    mask = np.isfinite(oof)
    challenger_brier = _brier(oof[mask], y[mask], sw[mask]) if mask.any() else None

    champ_p = np.asarray([
        _f((r.get("meta") or {}).get("predicted_p"))
        if _f((r.get("meta") or {}).get("predicted_p")) is not None else np.nan
        for r in rows
    ], dtype=float)
    cmask = mask & np.isfinite(champ_p)
    champion_brier = _brier(champ_p[cmask], y[cmask], sw[cmask]) if int(cmask.sum()) >= 20 else None
    return {
        "n": n,
        "oof_n": int(mask.sum()),
        "challenger_brier": None if challenger_brier is None else round(challenger_brier, 6),
        "champion_brier": None if champion_brier is None else round(champion_brier, 6),
        "champion_compare_n": int(cmask.sum()),
    }


def _model_version(rows: list[dict[str, Any]]) -> str:
    material = "|".join(str(r.get("observation_id") or "") for r in rows)
    return "clf_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def forward_comparison(model_version: str, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(rows if rows is not None else _rows())
    chosen = []
    for row in rows:
        lane = str((row.get("outcome_meta") or {}).get("evidence_class") or "").upper()
        meta = dict(row.get("meta") or {})
        if lane not in REAL_LANES:
            continue
        if str(meta.get("challenger_model_version") or "") != str(model_version):
            continue
        cp = _f(meta.get("predicted_p"))
        xp = _f(meta.get("challenger_predicted_p"))
        if cp is None or xp is None:
            continue
        chosen.append((cp, xp, 1.0 if float(row.get("outcome") or 0.0) > 0 else 0.0))
    if not chosen:
        return {
            "n": 0,
            "champion_brier": None,
            "challenger_brier": None,
            "improvement": None,
            "improvement_lower_95": None,
            "improvement_upper_95": None,
        }
    cp = np.asarray([x[0] for x in chosen], dtype=float)
    xp = np.asarray([x[1] for x in chosen], dtype=float)
    y = np.asarray([x[2] for x in chosen], dtype=float)
    cb = _brier(cp, y)
    xb = _brier(xp, y)
    paired = (cp - y) ** 2 - (xp - y) ** 2
    improvement = float(np.mean(paired))
    se = float(np.std(paired, ddof=1) / math.sqrt(len(paired))) if len(paired) > 1 else 1.0
    lower = improvement - 1.96 * se
    upper = improvement + 1.96 * se
    return {
        "n": len(chosen),
        "champion_brier": round(cb, 6),
        "challenger_brier": round(xb, 6),
        "improvement": round(improvement, 6),
        "improvement_lower_95": round(lower, 6),
        "improvement_upper_95": round(upper, 6),
    }


def _serialize_model(
    *,
    rows: list[dict[str, Any]],
    coef: np.ndarray,
    intercept: float,
    medians: np.ndarray,
    scales: np.ndarray,
    validation: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    real_n = sum(
        1 for r in rows
        if str((r.get("outcome_meta") or {}).get("evidence_class") or "").upper() in REAL_LANES
    )
    cf_n = sum(
        1 for r in rows
        if str((r.get("outcome_meta") or {}).get("evidence_class") or "").upper() in COUNTERFACTUAL_LANES
    )
    historical_n = sum(
        1 for r in rows
        if str((r.get("outcome_meta") or {}).get("evidence_class") or "").upper() in HISTORICAL_LANES
    )
    version = _model_version(rows)
    return {
        "model_version": version,
        "status": status,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_n": len(rows),
        "real_forward_n": real_n,
        "counterfactual_n": cf_n,
        "historical_n": historical_n,
        "features": list(MODEL_FEATURES),
        "coef": [round(float(x), 10) for x in coef],
        "intercept": round(float(intercept), 10),
        "medians": [round(float(x), 10) for x in medians],
        "scales": [round(float(x), 10) for x in scales],
        "validation": dict(validation),
        "forward_validation": forward_comparison(version, rows),
        "affects_selection": status == PAPER_ACTIVE,
        "paper_only": True,
        "live_locked": True,
    }


def _eligible_status(rows: list[dict[str, Any]], validation: Mapping[str, Any]) -> str:
    real_n = sum(
        1 for r in rows
        if str((r.get("outcome_meta") or {}).get("evidence_class") or "").upper() in REAL_LANES
    )
    if len(rows) < MIN_TOTAL or real_n < MIN_REAL_FORWARD:
        return OBSERVING
    xb = validation.get("challenger_brier")
    cb = validation.get("champion_brier")
    if xb is None:
        return OBSERVING
    if cb is None:
        return SHADOW_CANDIDATE
    if float(cb) - float(xb) >= MIN_BRIER_IMPROVEMENT:
        return SHADOW_CANDIDATE
    return REJECTED


def train(*, path: str | Path | None = None, force: bool = False) -> dict[str, Any]:
    store = load(path)
    current = dict(store.get("current") or {})
    rows = _rows()

    if current and not force and current.get("status") in {SHADOW_CANDIDATE, PAPER_ACTIVE}:
        promoted = maybe_promote(path=path)
        return promoted.get("current") or current
    if current and not force and int(current.get("trained_n") or 0) == len(rows):
        return current
    if len(rows) < 2:
        return {
            "status": OBSERVING,
            "trained_n": len(rows),
            "reason": "not enough settled decision observations",
            "affects_selection": False,
            "live_locked": True,
        }

    validation = _cv(rows)
    X, y, sw = _matrix(rows)
    coef, intercept, medians, scales = _fit(X, y, sw)
    status = _eligible_status(rows, validation)
    model = _serialize_model(
        rows=rows,
        coef=coef,
        intercept=intercept,
        medians=medians,
        scales=scales,
        validation=validation,
        status=status,
    )
    history = list(store.get("history") or [])
    if current:
        history.append(current)
    store["history"] = history[-20:]
    store["current"] = model
    save(store, path)
    return model


def _remember_transition(model: Mapping[str, Any], status: str, reason: str) -> None:
    """Persist model success/failure as scientific memory; never gates runtime."""
    try:
        from research.scientific_memory import (
            ACTIVE as BELIEF_ACTIVE,
            RETIRED as BELIEF_RETIRED,
            REJECTED as BELIEF_REJECTED,
            record_belief,
        )
        mapped = {
            PAPER_ACTIVE: BELIEF_ACTIVE,
            DEMOTED: BELIEF_RETIRED,
            REJECTED: BELIEF_REJECTED,
        }.get(status)
        if mapped is None:
            return
        version = str(model.get("model_version") or "unknown")
        forward = dict(model.get("forward_validation") or {})
        n = int(forward.get("n") or 0)
        confidence = "HIGH" if n >= DEMOTION_FORWARD_COMPARE else "MEDIUM"
        record_belief(
            statement=f"Paper selection challenger {version} calibration edge",
            signal="paper_selection_classifier",
            status=mapped,
            evidence_n=n,
            confidence=confidence,
            ev_r=None,
            drift_status="DECAYING" if status == DEMOTED else "STABLE",
            dependencies=("decision_feature_store", "paper_forward_outcomes"),
            notes=reason,
        )
    except Exception:
        pass


def maybe_promote(*, path: str | Path | None = None) -> dict[str, Any]:
    store = load(path)
    current = dict(store.get("current") or {})
    if not current:
        return store
    status = str(current.get("status") or "")
    if status not in {SHADOW_CANDIDATE, PAPER_ACTIVE}:
        return store

    forward = forward_comparison(str(current.get("model_version") or ""))
    current["forward_validation"] = forward
    forward_n = int(forward.get("n") or 0)
    improvement = _f(forward.get("improvement"))
    lower = _f(forward.get("improvement_lower_95"))
    upper = _f(forward.get("improvement_upper_95"))

    if status == PAPER_ACTIVE:
        # An active learner is not immortal.  Once enough fresh exact-version
        # outcomes accumulate, robust under-performance demotes it immediately.
        if (
            forward_n >= DEMOTION_FORWARD_COMPARE
            and improvement is not None
            and improvement <= -MIN_BRIER_IMPROVEMENT
            and upper is not None
            and upper < 0.0
        ):
            current["status"] = DEMOTED
            current["affects_selection"] = False
            current["demoted_at"] = datetime.now(timezone.utc).isoformat()
            current["demotion_reason"] = (
                "Exact-version forward calibration deteriorated with a negative "
                "95% upper bound versus the champion."
            )
            _remember_transition(current, DEMOTED, current["demotion_reason"])
        store["current"] = current
        save(store, path)
        return store

    # The training set is intentionally frozen while shadowing.  Graduation
    # therefore counts that frozen evidence plus fresh exact-version forward
    # observations; requiring the frozen count itself to grow would be impossible.
    enough = (
        int(current.get("trained_n") or 0) + forward_n >= PROMOTION_TOTAL
        and int(current.get("real_forward_n") or 0) + forward_n >= PROMOTION_REAL_FORWARD
        and forward_n >= MIN_FORWARD_COMPARE
    )
    if enough and improvement is not None:
        if (
            improvement >= MIN_BRIER_IMPROVEMENT
            and lower is not None
            and lower > 0.0
        ):
            current["status"] = PAPER_ACTIVE
            current["affects_selection"] = True
            current["promoted_at"] = datetime.now(timezone.utc).isoformat()
            current["promotion_reason"] = (
                "Purged validation plus exact-version forward Brier improvement "
                "with a positive 95% lower bound."
            )
            _remember_transition(current, PAPER_ACTIVE, current["promotion_reason"])
        else:
            current["status"] = REJECTED
            current["affects_selection"] = False
            current["rejected_at"] = datetime.now(timezone.utc).isoformat()
            current["rejection_reason"] = (
                "Forward comparison reached the sample floor without a robust "
                "positive challenger improvement."
            )
            _remember_transition(current, REJECTED, current["rejection_reason"])
    store["current"] = current
    save(store, path)
    return store


def _score_features(feature_values: Mapping[str, Any], model: Mapping[str, Any]) -> float | None:
    try:
        coef = np.asarray(model.get("coef") or [], dtype=float)
        med = np.asarray(model.get("medians") or [], dtype=float)
        scale = np.asarray(model.get("scales") or [], dtype=float)
        if coef.size != len(MODEL_FEATURES) or med.size != coef.size or scale.size != coef.size:
            return None
        x = np.asarray([
            np.nan if _f(feature_values.get(name)) is None else float(_f(feature_values.get(name)))
            for name in MODEL_FEATURES
        ], dtype=float).reshape(1, -1)
        p = _predict(x, coef, float(model.get("intercept") or 0.0), med, scale)
        return float(p[0])
    except Exception:
        return None


def score_decision(decision, *, path: str | Path | None = None) -> dict[str, Any]:
    store = load(path)
    model = dict(store.get("current") or {})
    if model.get("status") not in {SHADOW_CANDIDATE, PAPER_ACTIVE}:
        return {
            "available": False,
            "status": str(model.get("status") or OBSERVING),
            "affects_selection": False,
            "live_locked": True,
        }
    try:
        from product.evidence_intelligence import decision_features
        p = _score_features(decision_features(decision), model)
    except Exception:
        p = None
    return {
        "available": p is not None,
        "model_version": str(model.get("model_version") or ""),
        "status": str(model.get("status") or ""),
        "p_positive_R": None if p is None else round(p, 6),
        "affects_selection": bool(model.get("status") == PAPER_ACTIVE),
        "live_locked": True,
    }


def score_card(card: Mapping[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    try:
        from product.decision_adapter import decision_from_card
        from product.evidence_class import PAPER_FORWARD
        decision = decision_from_card(
            card,
            market_state=str(card.get("market_state") or card.get("regime") or ""),
            sector_state=str(card.get("sector_state") or ""),
            evidence_class=PAPER_FORWARD,
            generated_at=str(card.get("scan_scanned_at") or card.get("as_of") or ""),
        )
        return score_decision(decision, path=path)
    except Exception:
        return {
            "available": False,
            "status": "ERROR",
            "affects_selection": False,
            "live_locked": True,
        }


def paper_selection_adjustment(
    card: Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Bounded reorder-only adjustment for already-eligible PAPER candidates."""
    score = score_card(card, path=path)
    if not score.get("available") or not score.get("affects_selection"):
        return {
            **score,
            "adjustment": 0.0,
            "reason": "challenger is shadow-only or unavailable",
        }
    p = float(score.get("p_positive_R") or 0.5)
    adjustment = max(-5.0, min(3.0, (p - 0.50) * 10.0))
    return {
        **score,
        "adjustment": round(adjustment, 4),
        "reason": (
            "active paper learner reorders only candidates that already passed hard gates"
        ),
    }


def dashboard(*, path: str | Path | None = None) -> dict[str, Any]:
    store = load(path)
    current = dict(store.get("current") or {})
    return {
        "available": bool(current),
        "current": current,
        "history_count": len(store.get("history") or []),
        "affects_selection": bool(current.get("status") == PAPER_ACTIVE),
        "paper_only": True,
        "live_locked": True,
    }
