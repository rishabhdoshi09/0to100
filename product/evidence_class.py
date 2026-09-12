"""The one vocabulary for how a piece of evidence was produced.

Every number the desk shows about its own edge came from somewhere, and the
somewheres are not interchangeable. A backtest is a claim about a fitted past.
A replay is a claim about plumbing. A paper trade is a claim about what the
system would have done with real, unseen prices. Only real money is a claim
about money. Mixing them is how a system talks itself into being ready.

    BACKTEST            fitted over history the strategy has already seen
    WALK_FORWARD        refit forward over history, still history
    HISTORICAL_REPLAY   the live code path re-run over point-in-time data.
                        Proves the PLUMBING works. Never market evidence.
    PAPER_FORWARD       decided before the bar existed, settled on real
                        completed sessions. No money at risk.
    REAL_FORWARD        real orders, real fills, real money.
    COUNTERFACTUAL      what the rejected candidate went on to do
    TEST_FIXTURE        invented bars. Proves code behaviour, never edge.

REAL_FORWARD is unreachable by construction while the live-execution interlock
is locked, which it is. Nothing in this repository may produce it, and the fact
that the ledger can name it is not evidence that any exists.

Historical note on the legacy name: the ledger's original REAL_FORWARD_MARKET
meant "paper trade taken forward against the real market" — PAPER_FORWARD in
this vocabulary, not real money. The name overclaimed. It stays readable as an
alias so old rows keep their meaning, but new code names the class it means.
"""
from __future__ import annotations

BACKTEST = "BACKTEST"
WALK_FORWARD = "WALK_FORWARD"
HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
PAPER_FORWARD = "PAPER_FORWARD"
REAL_FORWARD = "REAL_FORWARD"
COUNTERFACTUAL = "COUNTERFACTUAL"
TEST_FIXTURE = "TEST_FIXTURE"

#: The legacy ledger spelling of PAPER_FORWARD. Rows on disk still carry it.
LEGACY_PAPER_FORWARD = "REAL_FORWARD_MARKET"

EVIDENCE_CLASSES = (
    BACKTEST,
    WALK_FORWARD,
    HISTORICAL_REPLAY,
    PAPER_FORWARD,
    REAL_FORWARD,
    COUNTERFACTUAL,
    TEST_FIXTURE,
)

#: Classes that may be counted as evidence that the system has an edge in the
#: market. Everything else proves something about code, not about money.
MARKET_EVIDENCE = frozenset({PAPER_FORWARD, REAL_FORWARD})

#: Classes produced by running the machinery over data it could have seen.
#: Useful, necessary, and never promotable.
PLUMBING_EVIDENCE = frozenset({BACKTEST, WALK_FORWARD, HISTORICAL_REPLAY, TEST_FIXTURE})


def normalise(value: str | None) -> str:
    """Map any spelling (including the legacy one) onto this vocabulary."""
    name = str(value or "").strip().upper()
    if name == LEGACY_PAPER_FORWARD:
        return PAPER_FORWARD
    return name if name in EVIDENCE_CLASSES else ""


def is_market_evidence(value: str | None) -> bool:
    """True only for classes that say something about real market outcomes."""
    return normalise(value) in MARKET_EVIDENCE


def may_promote(value: str | None) -> bool:
    """Whether this class may move a strategy toward more capital.

    Deliberately identical to :func:`is_market_evidence`: a replay that proves
    the loop is wired correctly is still not a reason to size up.
    """
    return is_market_evidence(value)
