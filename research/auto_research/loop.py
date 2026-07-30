"""
🔁 The autonomous research cycle — one honest pass of "think for yourself".

`run_cycle()` is the brain doing a full loop with ZERO human intervention:

    observe data  →  generate ideas  →  reason through each (structural + evidence)
                  →  reject the weak ones  →  shortlist survivors  →  build the case
                  →  auto-advance the lifecycle up to ONE gate  →  propose to the human

The one gate is `AWAITING_USER_APPROVAL`. The brain drives a candidate all the way there
by itself, then STOPS. It never approves, never activates paper, never touches live, never
places an order. That single stop is the human seatbelt the user asked us to keep even
while everything before it runs on its own.

Honesty rails (inherited from the whole overhaul):
  • No research-grade data ⇒ the cycle CONCLUDES `Discovery unavailable …` and proposes
    nothing. It never invents a verdict.
  • SYNTHETIC evidence is never presented as market evidence and never produces a proposal
    to the human gate. A synthetic evaluator is only ever a labelled fixture for tests.
  • Every reasoning step is written to the append-only ResearchThread, so the human can
    watch exactly why the brain rejected or shortlisted each idea.

The cycle is deterministic: same budget seed + same injected evaluator ⇒ same thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

from research.strategy_studio import discovery as DISC
from research.strategy_studio import review as REV
from research.strategy_studio import spec as S
from research.auto_research.thread import ResearchThread


# ── the outcome of one cycle ─────────────────────────────────────────────────────

@dataclass
class Proposal:
    """A shortlisted candidate parked at the human gate — advice only, never acted on."""
    strategy_id: str
    name: str
    family: str
    config_hash: str
    net_expectancy_R: float
    n_trades: int
    recommendation: str
    lifecycle_state: str          # always AWAITING_USER_APPROVAL — the brain stops here
    is_market_evidence: bool
    reasons: tuple = ()

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CycleReport:
    cycle: int
    data_ready: bool
    generated: int = 0
    rejected_structural: int = 0
    rejected_evidence: int = 0
    survivors: int = 0
    proposals: list = field(default_factory=list)   # list[Proposal]
    conclusion: str = ""
    required_confidence: float = 0.0
    acted_on_market: bool = False                    # ALWAYS False — no LIVE order path
    approved_anything: bool = False                  # ALWAYS False — no human gate crossed
    # market-evidence survivor (spec, EvidenceReport) pairs — consumed by paper autonomy
    # when the user has engaged it. Not part of as_dict (not JSON-serialisable).
    survivors_for_paper: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["proposals"] = [p.as_dict() if isinstance(p, Proposal) else p
                          for p in self.proposals]
        d.pop("survivors_for_paper", None)
        return d


# ── the cycle ────────────────────────────────────────────────────────────────────

def run_cycle(cycle_no: int, thread: ResearchThread, *,
              dataset_status: dict | None = None,
              evaluate_fn: Callable[[S.StrategySpec, str], DISC.EvidenceReport] | None = None,
              budget: DISC.DiscoveryBudget | None = None,
              registry: DISC.AttemptRegistry | None = None,
              has_delivery: bool = False, has_fundamentals_pit: bool = False,
              max_candidates: int = 12, family_weights: dict | None = None) -> CycleReport:
    """Run ONE autonomous research cycle and record the reasoning to `thread`.

    `dataset_status` is the data-setup readiness dict; when None it is computed from the
    canonical store (`canonical_readiness()`). `evaluate_fn` is the injected point-in-time
    backtest — tests pass a deterministic synthetic evaluator (labelled synthetic); when
    None, no evidence can be produced and the cycle stays honest.
    """
    budget = budget or DISC.DiscoveryBudget()
    registry = registry if registry is not None else DISC.AttemptRegistry()
    if dataset_status is None:
        dataset_status = canonical_readiness()

    report = CycleReport(cycle=cycle_no, data_ready=DISC.data_ready(dataset_status))

    # 1 ── OBSERVE the world -------------------------------------------------------
    thread.observe(cycle_no,
                   f"Data readiness is {(dataset_status or {}).get('color', 'unknown')!r} "
                   f"(can_run={bool((dataset_status or {}).get('can_run'))}).",
                   {"dataset_status": dataset_status})

    if not report.data_ready:
        report.conclusion = DISC.DISCOVERY_UNAVAILABLE_MSG
        thread.conclude(cycle_no,
                        DISC.DISCOVERY_UNAVAILABLE_MSG + " I will not invent a verdict; "
                        "no ideas can be judged until real NSE history is loaded.",
                        {"data_ready": False})
        return report

    # 2 ── GENERATE ideas ----------------------------------------------------------
    cands = DISC.generate(budget, has_delivery=has_delivery,
                          has_fundamentals_pit=has_fundamentals_pit,
                          family_weights=family_weights)[:max_candidates]
    report.generated = len(cands)
    thread.observe(cycle_no,
                   f"Generated {len(cands)} readable candidate(s) from the approved "
                   f"grammar across {len(set(c.family for c in cands))} families.",
                   {"n_candidates": len(cands), "seed": budget.seed})

    survivors: list[tuple[S.StrategySpec, DISC.EvidenceReport]] = []
    p_values: list[float] = []

    for spec in cands:
        registry.record(spec, S.GENERATED)

        # 3 ── REASON: structural validity (leakage / complexity / PIT) ------------
        sreasons = DISC.structural_reasons(spec, budget, has_delivery=has_delivery,
                                           has_fundamentals_pit=has_fundamentals_pit)
        if sreasons:
            report.rejected_structural += 1
            registry.record(spec, S.INVALID, sreasons)
            thread.decide(cycle_no, f"Reject {spec.strategy_id} on structure — "
                          f"{sreasons[0]}.", {"strategy_id": spec.strategy_id,
                                              "reasons": sreasons})
            continue

        # 4 ── evaluate (injected). No evaluator ⇒ no evidence ⇒ honest reason ------
        if evaluate_fn is None:
            registry.record(spec, S.UNDER_REVIEW,
                            ("no backtest evaluator wired — cannot produce evidence",))
            continue
        ev = evaluate_fn(spec, budget.test_period)

        # 5 ── REASON: evidence validity (sample / concentration / cost / dd) -------
        ereasons = DISC.evidence_reasons(ev, budget)
        if ereasons:
            report.rejected_evidence += 1
            registry.record(spec, S.REJECTED, ereasons)
            thread.decide(cycle_no, f"Reject {spec.strategy_id} on evidence — "
                          f"{ereasons[0]}.", {"strategy_id": spec.strategy_id,
                                              "reasons": ereasons})
            continue

        # survivor of both gates
        registry.record(spec, S.PROMISING)
        survivors.append((spec, ev))
        p_values.append(ev.p_value)

    report.survivors = len(survivors)

    # 6 ── REASON about the whole search: multiple-testing burden ------------------
    mt = DISC.apply_multiple_testing(p_values, len(registry))
    report.required_confidence = mt["required_confidence"]
    thread.reason(cycle_no,
                  f"{report.rejected_structural} rejected on structure, "
                  f"{report.rejected_evidence} on evidence; {len(survivors)} survived. "
                  f"After {len(registry)} attempt(s) a survivor must clear "
                  f"{mt['required_confidence']:.1%} confidence (multiple-testing burden).",
                  {"multiple_testing": mt})

    # 7 ── DECIDE + PROPOSE: advance survivors to the ONE gate, stop there ---------
    for spec, ev in survivors:
        reco = REV.recommendation(ev, dataset_status)
        is_evidence = not ev.is_synthetic

        # synthetic can never be presented as market evidence → never proposed to human
        if not is_evidence:
            thread.reason(cycle_no,
                          f"{spec.strategy_id} looks interesting but its numbers are "
                          "SYNTHETIC — not market evidence. It will NOT be proposed.",
                          {"strategy_id": spec.strategy_id, "is_market_evidence": False})
            continue
        if reco in (REV.RECO_REJECT, REV.RECO_UNSUITABLE, REV.RECO_MORE_EVIDENCE):
            thread.decide(cycle_no,
                          f"Hold {spec.strategy_id}: {reco.lower()} — not ready for the "
                          "human gate yet.", {"strategy_id": spec.strategy_id,
                                              "recommendation": reco})
            continue

        # auto-advance the lifecycle: GENERATED → UNDER_REVIEW → PROMISING → AWAITING.
        # every hop is a SYSTEM transition; require_transition proves we never take the
        # user-only step. The brain parks it and stops.
        state = _advance_to_gate(spec)
        prop = Proposal(strategy_id=spec.strategy_id, name=spec.name, family=spec.family,
                        config_hash=spec.config_hash(),
                        net_expectancy_R=round(ev.net_expectancy_R, 4),
                        n_trades=ev.n_trades, recommendation=reco, lifecycle_state=state,
                        is_market_evidence=True,
                        reasons=(f"{ev.net_expectancy_R:+.2f}R/trade over {ev.n_trades} "
                                 f"trades after costs on {budget.test_period}.",))
        report.proposals.append(prop)
        report.survivors_for_paper.append((spec, ev))
        thread.propose(cycle_no,
                       f"PROPOSE {spec.strategy_id} ({spec.family}) to the human gate: "
                       f"{reco}. It is now {state}. I will NOT approve it — a person must.",
                       prop.as_dict())

    # 8 ── CONCLUDE ----------------------------------------------------------------
    if report.proposals:
        report.conclusion = (f"{len(report.proposals)} candidate(s) advanced to "
                             f"{S.AWAITING_USER_APPROVAL} for a human decision. Nothing was "
                             "approved or traded.")
    else:
        report.conclusion = ("No candidate earned a place at the human gate this cycle. "
                             "Nothing was approved or traded.")
    thread.conclude(cycle_no, report.conclusion,
                    {"proposals": len(report.proposals), "acted_on_market": False,
                     "approved_anything": False})
    return report


def _advance_to_gate(spec: S.StrategySpec) -> str:
    """Walk the lifecycle from GENERATED to the human gate using only SYSTEM transitions.
    Raises via require_transition if any hop were ever mis-actored — a structural proof
    that the autonomous path can never take the user-only approval step."""
    state = S.GENERATED
    for nxt in (S.UNDER_REVIEW, S.PROMISING, S.AWAITING_USER_APPROVAL):
        S.require_transition(state, nxt, actor="system")
        state = nxt
    return state


# ── canonical readiness (autonomous data check, no network) ──────────────────────

def canonical_readiness(logs_root=None) -> dict:
    """Judge the CANONICAL on-disk dataset (logs/) with the same rules the data-setup page
    uses. Pure, network-free. Returns the readiness dict (red/amber/green). If anything
    goes wrong it fails CLOSED to red so the brain never runs on unverifiable data."""
    try:
        from pathlib import Path
        from research.momentum_breakout import data_setup as D
        root = Path(logs_root) if logs_root else (
            Path(__file__).resolve().parent.parent.parent / "logs")
        v = D.validate_dataset(root)
        return D.readiness(v)
    except Exception as e:
        return {"color": "red", "can_run": False,
                "label": "Data could not be verified",
                "reasons": [f"readiness check failed: {e}"]}
