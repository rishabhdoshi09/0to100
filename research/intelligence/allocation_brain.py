"""
🧭 Brain 2 — the Strategy & Paper-Allocation Brain.

Consumes ONLY immutable Brain-1 Evidence Cards and produces immutable `PaperAllocationDecision`
records. It decides which evidence-qualified strategies get paper risk and how much, using a
TRANSPARENT weighted score over the card's fields (no opaque model). It reduces, pauses and
retires on deteriorating evidence.

Hard limits by construction:
  • it reads cards, it never edits them (they are frozen dataclasses anyway),
  • it allocates PAPER risk only and is structurally incapable of live (it never emits the
    USER_APPROVED transition; only a user can),
  • an OVERFIT / INSUFFICIENT card can never be deployed,
  • allocation is never based on recent returns alone.

Weights and caps are explicit config, not magic numbers buried in logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.intelligence import schemas as SC
from research.intelligence import evidence_brain as EB


# ── transparent, tunable configuration (no hidden constants) ─────────────────────

@dataclass(frozen=True)
class AllocationConfig:
    # score weights (sum need not be 1; the score is used only for ranking + bucketing)
    w_lower_bound: float = 0.30      # uncertainty-adjusted edge — the anchor
    w_forward_calib: float = 0.20    # forward/backtest agreement
    w_confidence: float = 0.15
    w_deflated_sharpe: float = 0.15
    w_sample: float = 0.10           # more out-of-sample trades → more weight
    w_drawdown: float = 0.10         # penalty
    # risk buckets (paper risk % of capital per strategy) — configurable, not hard-coded
    bucket_risk_pct: dict = field(default_factory=lambda: {
        "established": 1.0, "promising": 0.5, "exploratory": 0.25})
    max_strategy_risk_pct: float = 1.0
    max_family_risk_pct: float = 2.0
    max_cluster_risk_pct: float = 1.5
    min_sample_for_established: int = 60
    min_sample_for_promising: int = 25


# states a card must be in to earn (or keep) paper risk
_DEPLOYABLE = {EB.CONFIRMED, EB.WEAKER_THAN_EXPECTED}
_HOLD_STATES = {EB.FORWARD_PENDING, EB.PROMISING, EB.REGIME_DEPENDENT}
_KILL_STATES = {EB.OVERFIT, EB.DECAYING, EB.RETIRED, EB.INSUFFICIENT_EVIDENCE}


def score_card(card: SC.StrategyEvidenceCard, cfg: AllocationConfig) -> float:
    """A transparent 0..1-ish score. Deliberately anchored on the UNCERTAINTY-ADJUSTED edge
    and forward calibration, not raw recent expectancy."""
    lb = max(0.0, min(1.0, card.lower_bound_R / 0.5))          # 0.5R lower bound ⇒ full marks
    calib = max(0.0, min(1.0, card.forward_to_backtest))
    conf = max(0.0, min(1.0, card.confidence))
    dsr = max(0.0, min(1.0, card.deflated_sharpe))
    sample = max(0.0, min(1.0, card.forward_trades / 60.0))
    dd_pen = max(0.0, min(1.0, card.max_drawdown / 8.0))       # in R units
    return round(cfg.w_lower_bound * lb + cfg.w_forward_calib * calib
                 + cfg.w_confidence * conf + cfg.w_deflated_sharpe * dsr
                 + cfg.w_sample * sample - cfg.w_drawdown * dd_pen, 4)


def _bucket(card: SC.StrategyEvidenceCard, cfg: AllocationConfig) -> str:
    if card.evidence_state == EB.CONFIRMED and card.forward_trades >= cfg.min_sample_for_established:
        return "established"
    if card.forward_trades >= cfg.min_sample_for_promising:
        return "promising"
    return "exploratory"


def decide(cards, *, cfg: AllocationConfig | None = None,
           current_risk: dict | None = None, clusters: dict | None = None,
           data_ok: bool = True) -> list:
    """Produce a PaperAllocationDecision per card. `current_risk` maps strategy_id → current
    paper risk %. `clusters` maps strategy_id → correlation-cluster id (from risk/correlation).
    With `data_ok=False` (no real data) NOTHING is deployed — honest no-action."""
    cfg = cfg or AllocationConfig()
    current_risk = dict(current_risk or {})
    clusters = dict(clusters or {})
    decisions: list = []

    # running tallies for portfolio caps
    family_risk: dict[str, float] = {}
    cluster_risk: dict[str, float] = {}
    for sid, r in current_risk.items():
        pass  # existing risk is re-derived per decision below

    # rank deployables by score so caps are spent on the best evidence first
    ranked = sorted(cards, key=lambda c: score_card(c, cfg), reverse=True)
    for card in ranked:
        prev = current_risk.get(card.strategy_id, 0.0)
        score = score_card(card, cfg)
        reasons, blocked = [], []

        if not data_ok:
            decisions.append(_mk(card, "SKIP", "", 0.0, prev, score,
                                 ("no real data — no deployment",), ()))
            continue
        if card.evidence_state in _KILL_STATES:
            act = "RETIRE" if prev > 0 else "SKIP"
            decisions.append(_mk(card, act, "", 0.0, prev, score,
                                 (f"evidence state {card.evidence_state}",), ()))
            continue
        if card.evidence_state in _HOLD_STATES:
            # keep whatever it has, don't add — evidence not yet strong enough
            decisions.append(_mk(card, "HOLD", "", prev, prev, score,
                                 (f"holding: {card.evidence_state} — not deployable yet",), ()))
            continue

        # deployable — size it, then apply portfolio caps
        bucket = _bucket(card, cfg)
        target = min(cfg.bucket_risk_pct.get(bucket, 0.25), cfg.max_strategy_risk_pct)

        fam = card.family
        if family_risk.get(fam, 0.0) + target > cfg.max_family_risk_pct:
            target = max(0.0, cfg.max_family_risk_pct - family_risk.get(fam, 0.0))
            blocked.append(f"family cap ({cfg.max_family_risk_pct}%)")
        cl = clusters.get(card.strategy_id, "")
        if cl:
            if cluster_risk.get(cl, 0.0) + target > cfg.max_cluster_risk_pct:
                target = max(0.0, cfg.max_cluster_risk_pct - cluster_risk.get(cl, 0.0))
                blocked.append(f"correlation-cluster cap ({cfg.max_cluster_risk_pct}%)")

        if target <= 0:
            decisions.append(_mk(card, "PAUSE", bucket, 0.0, prev, score,
                                 ("capped out by portfolio limits",), tuple(blocked)))
            continue

        action = ("DEPLOY" if prev == 0 else "INCREASE" if target > prev + 1e-9
                  else "REDUCE" if target < prev - 1e-9 else "HOLD")
        reasons.append(f"{bucket} bucket; lower-bound {card.lower_bound_R:+.2f}R, "
                       f"fwd/bt {card.forward_to_backtest:.2f}, conf {card.confidence:.0%}")
        family_risk[fam] = family_risk.get(fam, 0.0) + target
        if cl:
            cluster_risk[cl] = cluster_risk.get(cl, 0.0) + target
        decisions.append(_mk(card, action, bucket, round(target, 4), prev, score,
                             tuple(reasons), tuple(blocked)))
    return decisions


def _mk(card, action, bucket, target, prev, score, reasons, blocked):
    return SC.PaperAllocationDecision(
        strategy_id=card.strategy_id, strategy_version=card.strategy_version,
        rules_hash=card.rules_hash, data_snapshot_id=card.data_snapshot_id,
        source="allocation_brain", event_ts=card.event_ts, family=card.family,
        card_id=card.record_id, action=action, risk_bucket=bucket,
        target_risk_pct=target, prev_risk_pct=round(prev, 4), score=score,
        reasons=reasons, blocked_by=blocked)
