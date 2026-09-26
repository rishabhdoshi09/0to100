"""One paper-cycle job run must invoke the cycle exactly once.

research.autonomy.jobs.run_paper_cycle resolved the injected dependency's arity
by calling it and catching TypeError:

    try:
        result = ctx.deps.run_paper_cycle(entries_ok, reason, phase, failures)
    except TypeError:            # legacy injected fakes
        result = ctx.deps.run_paper_cycle(entries_ok)

The canonical cycle persists real paper positions before it can raise, and a
TypeError raised from inside the cycle -- for example out of the notification
step, which runs after the executor -- is indistinguishable from a signature
mismatch under that probe. The fallback therefore re-ran the entire cycle:
management pass and canonical executor both executed twice, producing a second
paper position from a single job run.

Arity is now resolved by inspection, which has no side effects.
"""
from __future__ import annotations

from research.autonomy import jobs as JOBS


class _BaseDeps:
    live_feed = None

    def now_ist(self):
        from core.market_clock import now_ist
        return now_ist()

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return "snapshot-1"


class _Ctx:
    active_failures: set = set()
    owner_paused = False

    def __init__(self, deps):
        self.deps = deps


def test_internal_type_error_does_not_re_run_the_cycle():
    calls: list[int] = []
    positions: list[dict] = []

    class Deps(_BaseDeps):
        def run_paper_cycle(self, entries_allowed, entry_block_reason="",
                            session_phase="intraday", capability_failures=()):
            calls.append(len(calls) + 1)
            positions.append({"symbol": f"SYM{len(calls)}"})   # a real fill
            raise TypeError("not an arity problem")

    result = JOBS.run_paper_cycle(_Ctx(Deps()))

    assert len(calls) == 1, f"cycle ran {len(calls)} times from one job"
    assert len(positions) == 1, f"{len(positions)} paper positions from one job run"
    assert result.status == JOBS.JS.PERMANENT_FAILED
    assert result.error_code == "CYCLE_ERROR"


def test_legacy_single_argument_dependency_is_still_supported():
    """The compatibility the old probe existed for must survive the fix."""
    seen: list[tuple] = []

    class LegacyDeps(_BaseDeps):
        def run_paper_cycle(self, entries_allowed):
            seen.append((entries_allowed,))
            return {"eligibility": "NO_ELIGIBLE_TRADE"}

    result = JOBS.run_paper_cycle(_Ctx(LegacyDeps()))

    assert len(seen) == 1, "legacy dependency must be called exactly once"
    assert result.status == JOBS.JS.SUCCEEDED


def test_full_signature_dependency_receives_all_four_arguments():
    seen: list[tuple] = []

    class FullDeps(_BaseDeps):
        def run_paper_cycle(self, entries_allowed, entry_block_reason="",
                            session_phase="intraday", capability_failures=()):
            seen.append((entries_allowed, entry_block_reason, session_phase))
            return {"eligibility": "NO_ELIGIBLE_TRADE"}

    result = JOBS.run_paper_cycle(_Ctx(FullDeps()))

    assert len(seen) == 1
    assert len(seen[0]) == 3, "reason and phase must reach the canonical cycle"
    assert result.status == JOBS.JS.SUCCEEDED


def test_var_positional_dependency_gets_the_full_call():
    seen: list[tuple] = []

    class StarDeps(_BaseDeps):
        def run_paper_cycle(self, *args):
            seen.append(args)
            return {"eligibility": "NO_ELIGIBLE_TRADE"}

    JOBS.run_paper_cycle(_Ctx(StarDeps()))
    assert seen and len(seen[0]) == 4, "*args must receive the canonical four arguments"


def test_non_type_errors_still_fail_the_job_once():
    calls: list[int] = []

    class Deps(_BaseDeps):
        def run_paper_cycle(self, entries_allowed, entry_block_reason="",
                            session_phase="intraday", capability_failures=()):
            calls.append(1)
            raise RuntimeError("PAPER_EXECUTION_FAILED: executor crashed")

    result = JOBS.run_paper_cycle(_Ctx(Deps()))
    assert len(calls) == 1
    assert result.status == JOBS.JS.PERMANENT_FAILED
