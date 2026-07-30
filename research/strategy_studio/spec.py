"""
🧬 Canonical strategy specification + lifecycle.

ONE versioned representation for every strategy — autonomous, manually-created, or
user-tweaked. It is a description, not code: it can place no order. Its `config_hash`
is computed only from the RESULT-DETERMINING fields (rules/thresholds/universe/costs/…)
so a cosmetic rename never changes the evidence identity, while any material rule change
does. The lifecycle state machine makes the human-in-the-loop rule explicit: only the
USER may approve a strategy for paper.

Pure: no streamlit, no network, no clock-in-identity, no order path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, replace

SPEC_VERSION = 1   # bumps if the SHAPE of the spec changes

# ── strategy families (diversity — not everything is EXP-006) ───────────────────
FAMILIES = (
    "cross_sectional_momentum", "trend_following", "breakout", "pullback",
    "mean_reversion", "sector_rotation", "volatility_contraction",
    "quality_plus_momentum", "event_driven", "regime_specific", "rule_combo",
)

# ── lifecycle states + allowed transitions ──────────────────────────────────────
GENERATED = "GENERATED"
INVALID = "INVALID"
REJECTED = "REJECTED"
UNDER_REVIEW = "UNDER_REVIEW"
PROMISING = "PROMISING"
AWAITING_USER_APPROVAL = "AWAITING_USER_APPROVAL"
APPROVED_FOR_PAPER = "APPROVED_FOR_PAPER"
PAPER_EVALUATION = "PAPER_EVALUATION"
ELIGIBLE_FOR_LIVE_REVIEW = "ELIGIBLE_FOR_LIVE_REVIEW"
RETIRED = "RETIRED"
DECAYED = "DECAYED"

LIFECYCLE = (GENERATED, INVALID, REJECTED, UNDER_REVIEW, PROMISING,
             AWAITING_USER_APPROVAL, APPROVED_FOR_PAPER, PAPER_EVALUATION,
             ELIGIBLE_FOR_LIVE_REVIEW, RETIRED, DECAYED)

# who may drive each transition. "system" = research code; "user" = a human only;
# "paper_autopilot" = the OPT-IN autonomous PAPER driver — it may cross the PAPER gates on
# its own BUT is structurally forbidden from the only transition that leads toward live
# (PAPER_EVALUATION -> ELIGIBLE_FOR_LIVE_REVIEW stays user-only). A transition's value may
# be a single actor or a tuple of allowed actors.
PAPER_AUTOPILOT = "paper_autopilot"

_TRANSITIONS: dict[str, dict] = {
    GENERATED: {INVALID: "system", UNDER_REVIEW: "system", REJECTED: "system"},
    UNDER_REVIEW: {REJECTED: "system", PROMISING: "system", INVALID: "system"},
    PROMISING: {REJECTED: "system", AWAITING_USER_APPROVAL: "system"},
    # a USER may always approve for paper; the opt-in paper autopilot may ALSO approve for
    # paper (never live) once a human has engaged full paper autonomy:
    AWAITING_USER_APPROVAL: {APPROVED_FOR_PAPER: ("user", PAPER_AUTOPILOT),
                             REJECTED: ("user", PAPER_AUTOPILOT)},
    APPROVED_FOR_PAPER: {PAPER_EVALUATION: ("user", PAPER_AUTOPILOT),
                         RETIRED: ("user", PAPER_AUTOPILOT)},
    # the paper autopilot may retire/decay its own paper strategies, but ONLY a user may
    # move anything toward LIVE review — this is the hard money boundary:
    PAPER_EVALUATION: {ELIGIBLE_FOR_LIVE_REVIEW: "user",
                       DECAYED: ("system", PAPER_AUTOPILOT),
                       RETIRED: ("user", PAPER_AUTOPILOT)},
}


class LifecycleError(Exception):
    """An illegal or unauthorised lifecycle transition was attempted."""


def _allowed_actors(src: str, dst: str) -> tuple:
    who = _TRANSITIONS.get(src, {}).get(dst)
    if who is None:
        return ()
    return who if isinstance(who, tuple) else (who,)


def can_transition(src: str, dst: str, actor: str) -> bool:
    return actor in _allowed_actors(src, dst)


def require_transition(src: str, dst: str, actor: str) -> None:
    if not can_transition(src, dst, actor):
        who = _allowed_actors(src, dst)
        if who:
            raise LifecycleError(
                f"{src}->{dst} requires actor {' or '.join(who)!r}, not {actor!r} "
                "(research code may not perform user approval)")
        raise LifecycleError(f"illegal transition {src}->{dst}")


# ── the canonical spec ──────────────────────────────────────────────────────────

# fields that DETERMINE results (drive the config hash / evidence identity)
_RESULT_FIELDS = (
    "family", "eligible_universe", "required_data", "entry_rules", "exit_rules",
    "stop_rules", "position_sizing", "max_holding_days", "regime_conditions",
    "sector_conditions", "liquidity_conditions", "feature_defs", "thresholds",
    "cost_model", "slippage_pct", "benchmark", "max_positions", "rebalance_days",
    "turnover_cap",
)
# fields that are DISPLAY/provenance only (never change evidence identity)
#   name, hypothesis, generation_method, search_attempt, parent_id, complexity_score,
#   version, strategy_id, dataset_snapshot, code_commit, seed


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    name: str
    version: int
    hypothesis: str                       # plain-language
    family: str
    # rules / conditions
    eligible_universe: str = "liquid_nse"
    required_data: tuple = ("daily_ohlcv", "nifty_benchmark")
    entry_rules: tuple = ()
    exit_rules: tuple = ()
    stop_rules: tuple = ()
    position_sizing: str = "risk_1pct"
    max_holding_days: int = 60
    regime_conditions: tuple = ()
    sector_conditions: tuple = ()
    liquidity_conditions: tuple = ("min_turnover_cr>=5",)
    feature_defs: tuple = ()
    thresholds: dict = field(default_factory=dict)
    cost_model: str = "india_cash_equity"
    slippage_pct: float = 0.10
    benchmark: str = "^NSEI"
    max_positions: int = 5
    rebalance_days: int = 0               # 0 = event-driven, else periodic
    turnover_cap: float = 1.0
    # provenance / display
    dataset_snapshot: str = ""
    code_commit: str = ""
    seed: int = 0
    generation_method: str = "manual"
    search_attempt: int = 0
    parent_id: str | None = None
    complexity_score: int = 0
    spec_version: int = SPEC_VERSION

    def result_fingerprint(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in _RESULT_FIELDS}

    def config_hash(self) -> str:
        """Hash of the RESULT-DETERMINING fields only. A cosmetic rename does NOT change
        it; any material rule/threshold change does."""
        blob = json.dumps(self.result_fingerprint(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["config_hash"] = self.config_hash()
        return d

    def bump_version(self, **result_changes) -> "StrategySpec":
        """Create a NEW version with material changes applied. Preserves the parent
        (this spec) and gets a new version number + parent_id. The caller must re-run
        the full evidence gate — old evidence does NOT transfer."""
        return replace(self, **result_changes, version=self.version + 1,
                       parent_id=self.strategy_id, generation_method="tweak")


def is_material_change(old: StrategySpec, new: StrategySpec) -> bool:
    """True when the change alters the evidence identity (config hash), i.e. touches a
    result-determining field. A pure display change (name/hypothesis) is NOT material."""
    return old.config_hash() != new.config_hash()


def complexity_of(spec: StrategySpec) -> int:
    """A transparent complexity score = number of rules + conditions + thresholds.
    Simpler is preferred; the simplicity challenge uses this."""
    return (len(spec.entry_rules) + len(spec.exit_rules) + len(spec.stop_rules)
            + len(spec.regime_conditions) + len(spec.sector_conditions)
            + len(spec.liquidity_conditions) + len(spec.thresholds))
