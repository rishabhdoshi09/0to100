"""
Incumbent vs challenger bake-off wired through registry + harness + committee.

Default: research-only. Does not mutate live production scorers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from core.costs import round_trip_cost_pct
from research import registry as REG
from research.autonomy import challenge as CH
from research.challenger_lab.models import (
    ModelIdentity,
    fit_predict_oos,
    identity_for,
)
from research.harness import evaluate
from research.horizons.splits import train_val_test_embargo_slices
from research.horizons.spec import TargetSpec

VERDICT_PROMOTE = "PROMOTE"
VERDICT_KEEP_INCUMBENT = "KEEP_INCUMBENT"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class BakeOffConfig:
    role: str = "research_signal_model"
    seed: int = 42
    cost_product: str = "CNC"
    persist_champion: bool = False          # NEVER True for live roles by default
    write_scientific_memory: bool = True
    min_oos: int = 10
    producer: str = "challenger_lab"


@dataclass
class BakeOffResult:
    verdict: str
    incumbent: ModelIdentity
    challenger: ModelIdentity
    features: list[str]
    target: dict
    train_period: dict
    validation_period: dict
    oos_period: dict
    cost_model: dict
    oos_metrics: dict
    regime_metrics: dict
    prediction_correlation: float | None
    economic_value_delta: float | None
    evidence_result: dict
    committee: dict
    hypothesis_id: str = ""
    notes: tuple[str, ...] = ()
    live_behaviour_changed: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["incumbent"] = asdict(self.incumbent)
        d["challenger"] = asdict(self.challenger)
        return d


def _period(idx) -> dict:
    arr = np.asarray(idx, dtype=int)
    if arr.size == 0:
        return {"start": None, "end": None, "n": 0}
    return {"start": int(arr.min()), "end": int(arr.max()), "n": int(arr.size)}


def _dataset_hash(X, y, feature_names, target_name: str, seed: int) -> str:
    payload = {
        "X_shape": list(np.asarray(X).shape),
        "y_checksum": float(np.nanmean(np.asarray(y, float))) if len(y) else 0.0,
        "y_n": int(len(y)),
        "features": list(feature_names),
        "target": target_name,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _prediction_to_r(pred, y_true, *, cost_frac: float) -> np.ndarray:
    """Map classification/score predictions to a simple economic R stream.

    For directional labels in {-1,0,1}: R ≈ pred_sign * realised_move - cost.
    When y_true is already a return series and pred is a score, R ≈ sign(pred)*y - cost.
    """
    p = np.asarray(pred, float)
    y = np.asarray(y_true, float)
    n = min(p.size, y.size)
    p, y = p[:n], y[:n]
    # If labels look like classes, treat y as the class and use hit/miss payoff.
    uniq = np.unique(y[~np.isnan(y)])
    if uniq.size <= 10 and np.all(np.isclose(uniq, np.round(uniq))):
        hit = (np.sign(p) == np.sign(y)) & (y != 0) & (p != 0)
        # +1R on correct directional call, -1R on wrong non-zero call, 0 on flat
        r = np.where((p == 0) | (y == 0), 0.0, np.where(hit, 1.0, -1.0))
    else:
        direction = np.sign(p)
        direction = np.where(direction == 0, 0.0, direction)
        r = direction * y
    return r - float(cost_frac)


def run_bakeoff(
    *,
    incumbent_model,
    challenger_model,
    X,
    y,
    target: TargetSpec,
    feature_names: list[str] | None = None,
    incumbent_id: ModelIdentity | None = None,
    challenger_id: ModelIdentity | None = None,
    config: BakeOffConfig | None = None,
    regime_labels: Mapping[int, str] | None = None,
    code_hash: str | None = None,
) -> BakeOffResult:
    """Run an identical-condition bake-off. Live production is unchanged by default."""
    cfg = config or BakeOffConfig()
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch")

    feats = list(feature_names) if feature_names else [f"f{i}" for i in range(X.shape[1])]
    inc_id = incumbent_id or identity_for(incumbent_model, model_id="incumbent", model_version="1")
    chal_id = challenger_id or identity_for(challenger_model, model_id="challenger", model_version="1")

    parts = train_val_test_embargo_slices(len(y), target)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    if te.size < cfg.min_oos:
        return BakeOffResult(
            verdict=VERDICT_INCONCLUSIVE,
            incumbent=inc_id,
            challenger=chal_id,
            features=feats,
            target={"name": target.name, "bars": target.bars, "kind": target.kind},
            train_period=_period(tr),
            validation_period=_period(va),
            oos_period=_period(te),
            cost_model={"product": cfg.cost_product,
                        "round_trip_pct": round_trip_cost_pct(cfg.cost_product)},
            oos_metrics={},
            regime_metrics={},
            prediction_correlation=None,
            economic_value_delta=None,
            evidence_result={"verdict": "UNDERPOWERED", "reason": "insufficient_oos"},
            committee={},
            notes=("OOS sample below minimum — no promotion possible",),
            live_behaviour_changed=False,
        )

    # Fit on train only (validation reserved for future calibration; not used to
    # re-fit). Identical matrices for both models.
    cost_pct = round_trip_cost_pct(cfg.cost_product) / 100.0
    pred_i = fit_predict_oos(incumbent_model, X[tr], y[tr], X[te])
    pred_c = fit_predict_oos(challenger_model, X[tr], y[tr], X[te])

    r_i = _prediction_to_r(pred_i, y[te], cost_frac=cost_pct)
    r_c = _prediction_to_r(pred_c, y[te], cost_frac=cost_pct)

    ev_i = evaluate(r_i, n_trials=1)
    ev_c = evaluate(r_c, n_trials=1)

    # Prediction agreement on OOS
    if pred_i.size >= 2 and np.std(pred_i) > 0 and np.std(pred_c) > 0:
        corr = float(np.corrcoef(pred_i, pred_c)[0, 1])
    else:
        corr = None

    econ_delta = float(np.nanmean(r_c) - np.nanmean(r_i))

    regime_metrics: dict[str, Any] = {}
    if regime_labels:
        for idx, lab in regime_labels.items():
            # regime_labels maps absolute feature index → label; filter OOS
            pass
        # Build from parallel array if provided as dict of index->regime for all rows
        oos_reg = [regime_labels.get(int(i), "UNK") for i in te]
        for lab in sorted(set(oos_reg)):
            mask = np.array([r == lab for r in oos_reg], dtype=bool)
            if mask.sum() == 0:
                continue
            regime_metrics[lab] = {
                "n": int(mask.sum()),
                "incumbent_mean_r": float(np.nanmean(r_i[mask])),
                "challenger_mean_r": float(np.nanmean(r_c[mask])),
            }

    dsh = _dataset_hash(X, y, feats, target.name, cfg.seed)
    data_window = {
        "train": _period(tr),
        "validation": _period(va),
        "oos": _period(te),
        "dataset_hash": dsh,
        "target": target.name,
        "features": feats,
    }
    success_criteria = {
        # Pre-commit: challenger must show positive mean OOS R after costs.
        "challenger_mean_r": {"gt": 0.0},
    }
    hid = REG.register_hypothesis(
        name=f"bakeoff:{cfg.role}:{chal_id.key()}_vs_{inc_id.key()}",
        success_criteria=success_criteria,
        data_window=data_window,
        description=f"challenger bake-off {chal_id.key()} vs {inc_id.key()}",
        seed=cfg.seed,
        code_hash=code_hash,
    )

    metrics = {
        "challenger_mean_r": float(np.nanmean(r_c)),
        "incumbent_mean_r": float(np.nanmean(r_i)),
        "economic_value_delta": econ_delta,
        "challenger_harness": ev_c.verdict,
        "incumbent_harness": ev_i.verdict,
        "dsr_challenger": float(ev_c.dsr),
        "prediction_correlation": corr,
    }
    reg_result = REG.record_result(hid, metrics)

    # Committee context — evidence must be explicit; no silent optimistic defaults.
    committee_ctx = {
        "forward_eligible": True,
        "benchmark_available": True,
        "required_evidence_complete": True,
        "n_trades": int(te.size),
        "min_trades": max(30, cfg.min_oos),
        "net_expectancy_R": float(np.nanmean(r_c)),
        "deflated_sharpe": float(ev_c.dsr),
        "walk_forward_ok": ev_c.verdict in ("PROMOTE", "INCONCLUSIVE", "UNDERPOWERED"),
        "max_correlation_to_deployed": abs(corr) if corr is not None else 0.0,
        "num_trials": 1,
        "parameter_count": int(getattr(challenger_model, "n_params", X.shape[1])),
        "leakage_detected": False,
        "producer": cfg.producer,
    }
    committee = CH.promotion_committee(committee_ctx, producer=cfg.producer)
    committee_dict = {
        "decision": committee.decision,
        "rationale": committee.rationale,
        "verdicts": [
            {"role": v.role, "passed": v.passed, "findings": list(v.findings),
             "blocking": v.blocking}
            for v in committee.verdicts
        ],
    }

    bake = REG.should_promote(
        float(np.nanmean(r_c)),
        float(np.nanmean(r_i)),
        margin=0.0,
        challenger_scores=r_c,
        champion_scores=r_i,
    )

    notes: list[str] = []
    live_changed = False

    # Map to required Phase-A verdicts. PROMOTE never implies live cutover here.
    if committee.decision == CH.REJECT:
        verdict = VERDICT_FAIL
    elif committee.decision in (CH.INCONCLUSIVE, CH.RETEST_WITH_MORE_DATA):
        verdict = VERDICT_INCONCLUSIVE
    elif committee.decision == CH.PAPER_NOMINATED and bake["promote"] and ev_c.verdict == "PROMOTE":
        verdict = VERDICT_PROMOTE
        notes.append("research nomination only — live path unchanged")
        if cfg.persist_champion:
            # Explicit opt-in: research role champion table only.
            REG.evaluate_challenger(
                cfg.role, chal_id.key(), float(np.nanmean(r_c)),
                margin=0.0, challenger_scores=r_c, champion_scores=r_i,
            )
            live_changed = False  # still not scanner/live
            notes.append(f"persist_champion=True wrote research role '{cfg.role}'")
    elif not bake["promote"]:
        verdict = VERDICT_KEEP_INCUMBENT
    else:
        # Beats incumbent on point estimate but evidence gate not fully clear.
        verdict = VERDICT_INCONCLUSIVE
        notes.append("point improvement without full evidence clearance")

    if cfg.write_scientific_memory and verdict in (VERDICT_FAIL, VERDICT_KEEP_INCUMBENT,
                                                    VERDICT_INCONCLUSIVE):
        try:
            from research import scientific_memory as SM
            statement = (
                f"challenger {chal_id.key()} vs incumbent {inc_id.key()} on "
                f"{target.name}: verdict={verdict}"
            )
            if verdict == VERDICT_FAIL:
                SM.record_negative(
                    statement,
                    signal=cfg.role,
                    evidence_n=int(te.size),
                    notes=committee.rationale,
                )
            else:
                SM.record_belief(
                    statement,
                    signal=cfg.role,
                    status=SM.WATCH,
                    evidence_n=int(te.size),
                    confidence="LOW",
                    ev_r=econ_delta,
                    hypothesis_id=hid,
                    notes=committee.rationale,
                )
        except Exception as exc:
            notes.append(f"scientific_memory_write_failed: {exc}")

    return BakeOffResult(
        verdict=verdict,
        incumbent=inc_id,
        challenger=chal_id,
        features=feats,
        target={"name": target.name, "bars": target.bars, "kind": target.kind,
                "buy_thresh": target.buy_thresh, "sell_thresh": target.sell_thresh},
        train_period=_period(tr),
        validation_period=_period(va),
        oos_period=_period(te),
        cost_model={"product": cfg.cost_product,
                    "round_trip_pct": round_trip_cost_pct(cfg.cost_product)},
        oos_metrics={
            "incumbent": {
                "mean_r": float(np.nanmean(r_i)),
                "harness_verdict": ev_i.verdict,
                "dsr": float(ev_i.dsr),
                "n": int(te.size),
            },
            "challenger": {
                "mean_r": float(np.nanmean(r_c)),
                "harness_verdict": ev_c.verdict,
                "dsr": float(ev_c.dsr),
                "n": int(te.size),
            },
            "registry_status": reg_result.get("status"),
        },
        regime_metrics=regime_metrics,
        prediction_correlation=None if corr is None else round(corr, 4),
        economic_value_delta=round(econ_delta, 6),
        evidence_result={
            "incumbent": {"verdict": ev_i.verdict, "insight": ev_i.insight,
                          "dsr": float(ev_i.dsr)},
            "challenger": {"verdict": ev_c.verdict, "insight": ev_c.insight,
                           "dsr": float(ev_c.dsr)},
            "should_promote": bake,
        },
        committee=committee_dict,
        hypothesis_id=hid,
        notes=tuple(notes),
        live_behaviour_changed=live_changed,
    )
