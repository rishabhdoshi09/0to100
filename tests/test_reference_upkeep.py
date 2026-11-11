"""Corporate actions and the index store are refreshed once per EOD again
(owner's Mac 2026-11-11: CA last refreshed 2026-09-18, indices 2026-10-04)."""
from __future__ import annotations

from types import SimpleNamespace

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor


def test_reference_upkeep_enqueues_once_per_session(tmp_path):
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: 1000.0)
    stub = SimpleNamespace(jobs=store)
    Supervisor._enqueue_reference_upkeep(stub, "2026-11-11")
    Supervisor._enqueue_reference_upkeep(stub, "2026-11-11")
    keys = sorted(j.idempotency_key for j in store.list())
    assert keys == ["corporate_actions:2026-11-11", "index_warmup:2026-11-11"]
    types = {j.job_type for j in store.list()}
    assert types == {SCH.CORPORATE_ACTIONS, SCH.INDEX_WARMUP}
    Supervisor._enqueue_reference_upkeep(stub, "2026-11-12")
    assert len(store.list()) == 4


def test_a_failing_store_never_breaks_the_tick():
    class Boom:
        def enqueue(self, *a, **k):
            raise RuntimeError("locked")

    Supervisor._enqueue_reference_upkeep(SimpleNamespace(jobs=Boom()), "2026-11-11")
