"""
🔬 Constrained autonomous strategy-discovery engine.

Generates READABLE strategy candidates from the approved grammar within a recorded
search budget, logs EVERY attempt (including rejects), applies the honest evidence
controls (validity, minimum sample, concentration, turnover, cost > edge,
multiple-testing burden, complexity penalty, simpler-baseline, untouched-test isolation),
and NEVER presents synthetic results as market evidence.

It cannot place an order. It cannot approve a strategy. Without a research-grade dataset
it reports `Discovery unavailable — historical research data is not ready.` and produces
candidates only as fixtures explicitly labelled non-evidence.

Evidence is produced by an injected `evaluate_fn(spec, split) -> EvidenceReport` so this
module stays pure and deterministic; production wires a real point-in-time backtest, and
tests inject a deterministic synthetic evaluator (labelled synthetic).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

from research.strategy_studio import grammar as G
from research.strategy_studio import spec as S

DISCOVERY_UNAVAILABLE_MSG = "Discovery unavailable — historical research data is not ready."


# ── search budget (every discovery run records this) ────────────────────────────

@dataclass(frozen=True)
class DiscoveryBudget:
    families: tuple = S.FAMILIES
    feature_groups: tuple = ("price", "participation", "context")
    max_rule_count: int = 4
    max_complexity: int = 8
    max_search_attempts: int = 200
    max_runtime_s: float = 60.0
    min_trade_count: int = 30
    max_turnover: float = 3.0
    max_drawdown: float = 0.35
    allowed_holding_days: tuple = (5, 10, 20, 60)
    max_concentration: float = 0.25          # max share of one symbol
    seed: int = 1
    train_period: str = "development"
    validation_period: str = "validation"
    test_period: str = "untouched_test"
    walk_forward: str = "purged_embargoed"
    benchmark: str = "^NSEI"
    cost_model: str = "india_cash_equity"

    def as_dict(self) -> dict:
        return asdict(self)


# ── evidence report (produced by the injected evaluator) ────────────────────────

@dataclass
class EvidenceReport:
    n_trades: int = 0
    n_symbols: int = 0
    gross_expectancy_R: float = 0.0
    net_expectancy_R: float = 0.0
    cost_drag_R: float = 0.0
    p_value: float = 1.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    max_symbol_share: float = 0.0
    regime_consistency: float = 0.0          # 0..1
    sector_consistency: float = 0.0
    worst_period: str = ""
    depends_on_few_trades: bool = False
    depends_on_one_period: bool = False
    invalid_data: bool = False
    is_synthetic: bool = True                # TRUE unless produced on RESEARCH_GRADE data
    verdict: str = "INCONCLUSIVE"

    def as_dict(self) -> dict:
        return asdict(self)


# ── attempt registry (append-only; nothing hidden) ──────────────────────────────

@dataclass
class Attempt:
    attempt_no: int
    strategy_id: str
    family: str
    status: str                              # GENERATED / INVALID / REJECTED / UNDER_REVIEW / PROMISING
    reasons: tuple = ()
    config_hash: str = ""

    def as_dict(self):
        return asdict(self)


class AttemptRegistry:
    """Every candidate attempted is recorded — generated AND rejected. Append-only."""

    def __init__(self):
        self._rows: list[Attempt] = []

    def record(self, spec: S.StrategySpec, status: str, reasons=()) -> Attempt:
        a = Attempt(attempt_no=len(self._rows) + 1, strategy_id=spec.strategy_id,
                    family=spec.family, status=status, reasons=tuple(reasons),
                    config_hash=spec.config_hash())
        self._rows.append(a)
        return a

    def __len__(self):
        return len(self._rows)

    def all(self):
        return list(self._rows)

    def rejected(self):
        return [a for a in self._rows if a.status in (S.INVALID, S.REJECTED)]

    def families_seen(self):
        return {a.family for a in self._rows}


# ── data-readiness gate ─────────────────────────────────────────────────────────

def data_ready(dataset_status: dict | None) -> bool:
    """True only when a RESEARCH-GRADE dataset is present. `dataset_status` is the
    data-setup readiness (e.g. {'color':'green'/'amber','can_run':True}). Amber counts as
    'ready enough to discover' but evidence stays limitation-bound; None/red = not ready."""
    if not dataset_status:
        return False
    return bool(dataset_status.get("can_run")) and dataset_status.get("color") in ("green", "amber")


# ── generation (Stage A) — seeded, family-diverse, budgeted ─────────────────────

def generate(budget: DiscoveryBudget, has_delivery=False, has_fundamentals_pit=False,
             family_weights: dict | None = None) -> list[S.StrategySpec]:
    """Create readable candidates from the grammar. Deterministic for a given seed.

    Diversity vs focus: by default families are visited round-robin (guaranteed diversity).
    When `family_weights` is given (a learned bias, e.g. from forward-test trust), the visit
    sequence is drawn proportional to those weights — so the search spends more attempts on
    families that have proven themselves out-of-sample and fewer on chronic decayers, WITHOUT
    ever fully starving a family (it keeps exploring). This is how the search itself gets
    smarter over time. Respects max_search_attempts and max_complexity."""
    rng = random.Random(budget.seed)
    out: list[S.StrategySpec] = []
    families = [f for f in budget.families if f in G.FAMILY_BLOCKS]
    n = min(budget.max_search_attempts, len(families) * 8)
    visit = _family_sequence(families, family_weights, n, rng)
    for i in range(n):
        fam = visit[i]
        blocks = [b for b in G.FAMILY_BLOCKS[fam]
                  if G.is_pit_available(b, has_fundamentals_pit, has_delivery)]
        if not blocks:
            continue
        k = min(len(blocks), rng.randint(1, budget.max_rule_count))
        chosen = blocks[:k]
        sid = f"STR-{i+1:04d}"
        spec = S.StrategySpec(
            strategy_id=sid, name=f"{fam.replace('_',' ').title()} #{i+1}", version=1,
            hypothesis=_hypothesis(fam, chosen), family=fam,
            entry_rules=tuple(f"require:{b}" for b in chosen),
            exit_rules=("trend_exit",), stop_rules=("structural_stop",),
            feature_defs=tuple(chosen),
            thresholds={"entry_timing": "next_bar_open"},
            max_holding_days=rng.choice(budget.allowed_holding_days),
            benchmark=budget.benchmark, cost_model=budget.cost_model,
            generation_method="autonomous_grammar", search_attempt=i + 1, seed=budget.seed,
        )
        spec = _with_complexity(spec)
        if S.complexity_of(spec) <= budget.max_complexity:
            out.append(spec)
    return out


def _family_sequence(families: list[str], weights: dict | None, n: int,
                     rng: random.Random) -> list[str]:
    """The order in which families are visited across `n` attempts. No weights ⇒ round-robin.
    With weights ⇒ a deterministic weighted draw with a floor, so strong families get more
    attempts but weak ones are still explored (never a weight of exactly zero)."""
    if not weights:
        return [families[i % len(families)] for i in range(n)]
    floor = 0.10                                    # every family keeps at least this share
    w = {f: max(floor, float(weights.get(f, 1.0))) for f in families}
    total = sum(w.values()) or 1.0
    # allocate integer counts proportional to weight, remainder by largest fractional part
    raw = {f: (w[f] / total) * n for f in families}
    counts = {f: int(raw[f]) for f in families}
    rem = n - sum(counts.values())
    for f in sorted(families, key=lambda x: raw[x] - counts[x], reverse=True)[:max(0, rem)]:
        counts[f] += 1
    # interleave deterministically so diversity is preserved within the weighting
    pools = {f: counts[f] for f in families}
    seq: list[str] = []
    order = sorted(families, key=lambda f: -counts[f])
    while len(seq) < n:
        for f in order:
            if pools[f] > 0:
                seq.append(f); pools[f] -= 1
                if len(seq) >= n:
                    break
    return seq


def _hypothesis(fam: str, blocks) -> str:
    labels = ", ".join(G.block_label(b) for b in blocks)
    return f"Buy stocks that show {labels} (a {fam.replace('_',' ')} idea)."


def _with_complexity(spec: S.StrategySpec):
    import dataclasses
    return dataclasses.replace(spec, complexity_score=S.complexity_of(spec))


# ── Stage B: structural + evidence validity ─────────────────────────────────────

def structural_reasons(spec: S.StrategySpec, budget: DiscoveryBudget,
                       has_delivery=False, has_fundamentals_pit=False) -> list[str]:
    reasons = []
    timing = (spec.thresholds or {}).get("entry_timing", "next_bar_open")
    if timing in ("same_bar_close", "future"):
        reasons.append("future leakage: entry would use information not available yet")
    if timing == "same_bar_close":
        reasons.append("impossible entry: assumes a fill at a price not yet known")
    if S.complexity_of(spec) > budget.max_complexity:
        reasons.append("too complex (over the complexity budget)")
    for b in spec.feature_defs:
        if not G.is_pit_available(b, has_fundamentals_pit, has_delivery):
            reasons.append(f"unsupported point-in-time input: {G.block_label(b)}")
    return reasons


def evidence_reasons(ev: EvidenceReport, budget: DiscoveryBudget) -> list[str]:
    reasons = []
    if ev.invalid_data:
        reasons.append("invalid data")
    if ev.n_trades < budget.min_trade_count:
        reasons.append(f"too few trades ({ev.n_trades} < {budget.min_trade_count})")
    if ev.n_symbols <= 1:
        reasons.append("depends on a single symbol")
    if ev.max_symbol_share > budget.max_concentration:
        reasons.append(f"too concentrated (one name is {ev.max_symbol_share:.0%})")
    if ev.turnover > budget.max_turnover:
        reasons.append("turnover too high")
    if ev.net_expectancy_R <= 0 <= ev.gross_expectancy_R and ev.cost_drag_R > 0:
        reasons.append("costs are larger than the gross edge")
    if ev.depends_on_one_period:
        reasons.append("depends on one short period")
    if ev.max_drawdown > budget.max_drawdown:
        reasons.append("drawdown beyond the budget")
    return reasons


# ── overfitting controls (Stages F, G, H) ───────────────────────────────────────

def evidence_burden(n_attempts: int, base_promote_p: float = 0.95) -> float:
    """The more strategies attempted, the higher the confidence a survivor must clear.
    A single pre-registered idea faces `base_promote_p`; 50,000 attempts face a much
    stronger burden. Reuses the same idea as the harness's deflation, made explicit."""
    import math
    extra = min(0.049, 0.01 * math.log10(max(1, n_attempts)))
    return min(0.999, base_promote_p + extra)


def apply_multiple_testing(p_values, n_total_attempts: int) -> dict:
    """Reuse the repository's Benjamini-Hochberg FDR across the survivors, and report the
    total family size so a lucky survivor of a big search is visible."""
    from research import harness as H
    bh = H.benjamini_hochberg(list(p_values)) if p_values else {"n_significant": 0}
    return {"family_size": n_total_attempts, "bh": bh,
            "required_confidence": evidence_burden(n_total_attempts)}


class UntouchedTestSet:
    """The final test period may be evaluated ONCE. A second touch raises — you cannot
    optimise against the untouched test."""

    def __init__(self, name="untouched_test"):
        self.name = name
        self._used = False

    def evaluate_once(self, fn):
        if self._used:
            raise RuntimeError("untouched test set already evaluated — re-touching it "
                               "would invalidate the result")
        self._used = True
        return fn()


def simpler_baseline_comparison(strategy: EvidenceReport, baseline: EvidenceReport,
                                strat_complexity: int, baseline_complexity: int,
                                min_edge_gain: float = 0.05) -> dict:
    """Stage H — a complex strategy must beat a simpler baseline by a meaningful margin.
    Prefer the simpler one when results are materially similar."""
    gain = strategy.net_expectancy_R - baseline.net_expectancy_R
    extra_complexity = max(0, strat_complexity - baseline_complexity)
    meaningful = gain >= min_edge_gain and extra_complexity >= 0
    return {"edge_gain_R": round(gain, 4), "extra_complexity": extra_complexity,
            "complexity_justified": bool(meaningful and gain > 0),
            "prefer_simpler": not meaningful,
            "note": ("The extra rules earn their keep." if meaningful else
                     "The simpler idea does about as well — prefer it.")}
