"""
Pytest collection policy.

The CANONICAL network-free unit suite is simply:

    python -m pytest

`tests/integration/` is EXCLUDED from that default run by classification (not an ad-hoc
`--ignore`): it holds tests whose import chain reaches heavy, environment-dependent
operational modules (e.g. `scan/*`, which make lazy data/network calls at import time
and stall without network). Those are integration tests, not deterministic unit tests.

To run the integration suite explicitly (may be slow / need network):

    QT_INTEGRATION=1 python -m pytest tests/integration

`collect_ignore` prevents pytest from even importing the integration directory during
the default run, so the network-free suite cannot stall on their import chain.
"""
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests import network_policy

# ---------------------------------------------------------------------------
# Runtime isolation. This block runs before any QuantTerm module is imported,
# which matters because several of them resolve durable paths into module-level
# constants at import time.
#
# Without it the suite wrote real artifacts into the checkout: a long-term
# shortlist of the fixture symbols AAA and BBB reached
# logs/product/latest_long_term_scan.json, research evidence for the invented
# ticker QTTRUTHA reached logs/research_evidence/, and the running desk then
# read them as genuine and told the operator a scan was available.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_LOGS = _REPO_ROOT / "logs"

if not os.environ.get("QT_RUNTIME_ROOT"):
    os.environ["QT_RUNTIME_ROOT"] = tempfile.mkdtemp(prefix="quantterm-test-runtime-")

_RUNTIME_ROOT = Path(os.environ["QT_RUNTIME_ROOT"])
(_RUNTIME_ROOT / "logs").mkdir(parents=True, exist_ok=True)

# During the default (network-free) run, do not collect/import tests/integration.
collect_ignore = [] if os.getenv("QT_INTEGRATION") else ["integration"]

_LONG_TERM_PROJECTOR = None


def _runtime_tree_fingerprint(root: Path) -> str:
    """Content hash of every file under ``root``.

    Directory existence is deliberately not part of the fingerprint: an empty
    ``logs/`` created by a mkdir is not contamination, a file written into it
    is. Absent and empty therefore hash the same.
    """
    if not root.exists():
        return "EMPTY"
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        return "EMPTY"
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        try:
            digest.update(str(path.stat().st_size).encode("utf-8"))
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def _runtime_tree_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


_REAL_LOGS_BEFORE: tuple[str, set[str]] = ("", set())


def pytest_configure(config):
    """Fingerprint the production runtime tree, and close the network."""
    global _REAL_LOGS_BEFORE
    # Installed before collection: a module that opens a socket at import time
    # is an uncontrolled external dependency like any other.
    network_policy.install()
    _REAL_LOGS_BEFORE = (
        _runtime_tree_fingerprint(_REAL_LOGS),
        _runtime_tree_files(_REAL_LOGS),
    )


def pytest_sessionfinish(session, exitstatus):
    """Fail the run if the suite touched production runtime state.

    This is the invariant, not a lint: tests may not write into the real
    ``logs/`` tree by any route, whichever module resolved the path.
    """
    before_hash, before_files = _REAL_LOGS_BEFORE
    after_hash = _runtime_tree_fingerprint(_REAL_LOGS)
    if after_hash == before_hash:
        try:
            shutil.rmtree(_RUNTIME_ROOT, ignore_errors=True)
        except Exception:
            pass
        return

    after_files = _runtime_tree_files(_REAL_LOGS)
    added = sorted(after_files - before_files)
    removed = sorted(before_files - after_files)
    modified = sorted(f for f in (after_files & before_files))

    detail = [
        "",
        "=" * 78,
        "TEST CONTAMINATION: the suite changed production runtime state.",
        f"  runtime root under test : {_RUNTIME_ROOT}",
        f"  protected tree          : {_REAL_LOGS}",
    ]
    if added:
        detail.append(f"  files created ({len(added)}):")
        detail += [f"    + {name}" for name in added[:40]]
        if len(added) > 40:
            detail.append(f"    … and {len(added) - 40} more")
    if removed:
        detail.append(f"  files deleted ({len(removed)}):")
        detail += [f"    - {name}" for name in removed[:20]]
    if not added and not removed and modified:
        detail.append("  existing files were modified in place")
    detail += [
        "",
        "  Route every durable path through core.runtime_paths (logs_dir /",
        "  logs_path) so QT_RUNTIME_ROOT redirects it. A module that resolves",
        "  its own path from __file__ writes into the developer's checkout and",
        "  the product then reads the fixture back as real evidence.",
        "=" * 78,
    ]
    message = "\n".join(detail)
    print(message)
    if hasattr(session, "exitstatus"):
        session.exitstatus = 1
    raise pytest.UsageError(message)


def pytest_sessionstart(session):
    """Freeze the canonical saved-scan projector identity for leak detection.

    Tests may monkeypatch it through pytest's monkeypatch fixture, but the fixture
    must restore it at teardown. A direct assignment leak is a test-isolation bug
    because later tests (and long-lived processes) would see altered behavior.
    """
    global _LONG_TERM_PROJECTOR
    from scan import long_term_service
    _LONG_TERM_PROJECTOR = long_term_service.technical_rows_from_market_scan


def pytest_collection_modifyitems(config, items):
    """Keep the canonical run hermetic by construction, not by discipline.

    A test that needs a real upstream is not skipped quietly — it is not part
    of this gate at all. It runs in its own named gate:

        QT_LIVE_SOURCE=1 python -m pytest -m live_source
    """
    if os.getenv("QT_LIVE_SOURCE"):
        return
    deselected = []
    kept = []
    for item in items:
        if item.get_closest_marker("live_source") or item.get_closest_marker("network"):
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Open the network only for tests that are explicitly allowed to use it."""
    network_policy.take_attempts()
    allowed = any(item.get_closest_marker(name) for name in network_policy.ALLOWED_MARKERS)
    network_policy.allow(bool(allowed))
    try:
        yield
    finally:
        network_policy.allow(False)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Fail at the test that leaves the canonical projector mutated."""
    yield
    attempts = network_policy.take_attempts()
    if attempts:
        pytest.fail(network_policy.failure_message(attempts), pytrace=False)
    if _LONG_TERM_PROJECTOR is None:
        return
    from scan import long_term_service
    current = long_term_service.technical_rows_from_market_scan
    if current is not _LONG_TERM_PROJECTOR:
        pytest.fail(
            "test leaked scan.long_term_service.technical_rows_from_market_scan "
            f"after teardown: {item.nodeid}",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def isolate_mutable_runtime_state(tmp_path_factory, monkeypatch, request):
    """Hermetic suite: never inherit warmed bhavcopy, analog corpus, or paper memory."""
    from data.bhavcopy_store import reset_in_memory_store
    from research.market_memory import reset_analog_corpus_cache

    reset_in_memory_store()
    reset_analog_corpus_cache()
    paper_mem = tmp_path_factory.mktemp("paper_memory") / "paper_memory.json"
    monkeypatch.setenv("QT_PAPER_MEMORY", str(paper_mem))
    auto_journal = tmp_path_factory.mktemp("autopilot_journal") / "journal.json"
    monkeypatch.setenv("QT_PAPER_AUTOPILOT_JOURNAL", str(auto_journal))
    policies = tmp_path_factory.mktemp("learning_policies") / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policies))
    counter = tmp_path_factory.mktemp("counterfactuals") / "cf.jsonl"
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(counter))
    taken = tmp_path_factory.mktemp("taken_evidence") / "taken.jsonl"
    monkeypatch.setenv("QT_TAKEN_EVIDENCE", str(taken))
    ingested = tmp_path_factory.mktemp("learning_ingested") / "ingested.json"
    monkeypatch.setenv("QT_LEARNING_INGESTED", str(ingested))
    challengers = tmp_path_factory.mktemp("challengers") / "challengers.json"
    monkeypatch.setenv("QT_CHALLENGERS", str(challengers))
    calibration = tmp_path_factory.mktemp("calibration") / "calibration.json"
    monkeypatch.setenv("QT_CALIBRATION", str(calibration))
    forward_ledger = tmp_path_factory.mktemp("forward_ledger") / "forward_evidence.jsonl"
    monkeypatch.setenv("QT_FORWARD_LEDGER", str(forward_ledger))
    forward_journey = tmp_path_factory.mktemp("forward_journey") / "forward_journey.json"
    monkeypatch.setenv("QT_FORWARD_JOURNEY", str(forward_journey))
    forward_daily = tmp_path_factory.mktemp("forward_daily")
    monkeypatch.setenv("QT_FORWARD_DAILY", str(forward_daily))
    scan_path = tmp_path_factory.mktemp("scan") / "latest_momentum_scan.json"
    monkeypatch.setenv("QT_SCAN_PATH", str(scan_path))
    reco_path = tmp_path_factory.mktemp("reco") / "latest_recommendations.json"
    monkeypatch.setenv("QT_RECO_PATH", str(reco_path))
    soak_verify = tmp_path_factory.mktemp("soak_verify") / "forward_soak_verify.json"
    monkeypatch.setenv("QT_FORWARD_SOAK_VERIFY", str(soak_verify))
    cap_state = tmp_path_factory.mktemp("capability_state") / "capability_runtime.json"
    monkeypatch.setenv("QT_CAPABILITY_STATE", str(cap_state))
    ops_runtime = tmp_path_factory.mktemp("market_ops") / "runtime.json"
    monkeypatch.setenv("QT_MARKET_OPS_RUNTIME", str(ops_runtime))
    auto_status = tmp_path_factory.mktemp("autonomy") / "status.json"
    monkeypatch.setenv("QT_AUTONOMY_STATUS", str(auto_status))
    auto_runtime = tmp_path_factory.mktemp("autonomy_runtime") / "runtime.json"
    monkeypatch.setenv("QT_AUTONOMY_RUNTIME", str(auto_runtime))
    pit_wh = tmp_path_factory.mktemp("pit_warehouse") / "pit_warehouse.db"
    monkeypatch.setenv("QT_PIT_WAREHOUSE", str(pit_wh))
    pipeline_snap = tmp_path_factory.mktemp("desk_pipeline") / "desk_pipeline.json"
    monkeypatch.setenv("QT_DESK_PIPELINE_SNAPSHOT", str(pipeline_snap))
    evo_state = tmp_path_factory.mktemp("autonomous_evolution") / "evolution.json"
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION", str(evo_state))
    evo_dir = tmp_path_factory.mktemp("autonomous_evolution_runs")
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION_DIR", str(evo_dir))
    evo_identity = tmp_path_factory.mktemp("autonomous_evolution_identity") / "identity.json"
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION_IDENTITY", str(evo_identity))

    # This legacy smart-acquire test intentionally writes an Aug-26 cache and
    # asserts that the 3-day filings lane is still fresh. Without an explicit
    # clock it becomes date-dependent and started failing on Aug-29 even though
    # production correctly treats >72h exchange data as stale. Freeze ONLY that
    # test; never weaken the live freshness policy to satisfy a calendar test.
    if request.node.name == "test_smart_acquire_skips_fresh_lanes":
        import product.due_diligence.acquire as acquire_module

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                fixed = cls(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
                return fixed if tz is not None else fixed.replace(tzinfo=None)

        monkeypatch.setattr(acquire_module, "datetime", _FrozenDateTime)

    # The money-path journal test intentionally stores decisions on Aug-01 and
    # exercises official-session outcome resolution, not the passage of real
    # wall-clock time. Freeze only that test at the original 40-day boundary so
    # the canonical retention policy remains strict and the test stays stable.
    if request.node.name == "test_outcomes_and_gate_audit":
        import core.decision_journal as decision_journal_module

        monkeypatch.setattr(
            decision_journal_module,
            "_now",
            lambda: datetime(2026, 9, 10, 12, 0, 0),
        )

    yield
    reset_in_memory_store()
    reset_analog_corpus_cache()
