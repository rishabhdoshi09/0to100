import json
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from research import registry
from research.autonomy import job_store, jobs, outcome_discovery as study, schedules


@pytest.fixture
def checkpoint(monkeypatch):
    with TemporaryDirectory(dir="/tmp") as scratch:
        root = Path(scratch).resolve()
        monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
        monkeypatch.setattr(registry, "_DB_PATH", root / "experiments.sqlite")
        source = root / "gauntlet.sqlite"
        days = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(31)]
        days = [study.START, *days, study.END]
        names = [f"EQ{i:02}" for i in range(50)]
        binding = {"start": study.START, "sessions": days, "symbols": names, "observations": True,
                   "exploratory": True, "archive_sha256": dict.fromkeys(days[:-1], "fixture")}
        with closing(sqlite3.connect(source)) as db, db:
            db.executescript("""
                CREATE TABLE geometry_config (id INTEGER, spec TEXT);
                CREATE TABLE geometry_done (symbol TEXT);
                CREATE TABLE geometry_observations (symbol TEXT, day TEXT, features TEXT, outcome TEXT);
            """)
            db.execute("INSERT INTO geometry_config VALUES (1,?)", (json.dumps(
                ["2025-01-01", study.END, {"scanner": [study.HORIZON]}, 2.0, binding]),))
            db.executemany("INSERT INTO geometry_done VALUES (?)", [(s,) for s in names])
            db.executemany("INSERT INTO geometry_observations VALUES (?,?,?,?)", [
                (s, days[1], json.dumps({"momentum_20": i/100, "turnover_20": 1e9, "volatility_20": 0.02, "signals": []}),
                 json.dumps({"open_to_close_return": (49-i)/100, "next_open_gap": 0.01}))
                for i, s in enumerate(names)
            ])
        yield source


def test_scheduler_discovery_is_once_only_and_cannot_promote(checkpoint, monkeypatch):
    monkeypatch.setenv("QT_OUTCOME_DISCOVERY_CHECKPOINT", str(checkpoint))
    ctx = SimpleNamespace(job=SimpleNamespace(idempotency_key=schedules.OUTCOME_DISCOVERY_KEY))
    first = jobs.run_research_cycle(ctx)
    assert first.status == job_store.SUCCEEDED
    assert first.clears == set()
    result = first.metadata
    assert result["status"] == "INCONCLUSIVE"
    assert result["metrics"]["eligible_observations_measured"] == 50
    assert result["metrics"]["without_scanner_signals"] == 50
    assert result["metrics"]["expected_net_return"] == pytest.approx(0.4418)
    assert registry.record_result(result["hypothesis_id"], {})["error"] == "discovery_only"
    monkeypatch.setattr(study, "_measure", lambda *_args: pytest.fail("repeated completed experiment"))
    assert jobs.run_research_cycle(ctx).metadata["decision"] == "DISCOVERY_BUDGET_EXHAUSTED"


def test_refuses_contaminated_checkpoint_and_escaping_lock(checkpoint):
    lock = checkpoint.parent / "logs/outcome_discovery.lock"
    lock.parent.mkdir()
    lock.symlink_to("/tmp/outside-quantterm-discovery.lock")
    with pytest.raises(ValueError, match="scratch runtime"):
        study.run(checkpoint)
    lock.unlink()
    with closing(sqlite3.connect(checkpoint)) as db, db:
        config = json.loads(db.execute("SELECT spec FROM geometry_config").fetchone()[0])
        config[-1]["archive_sha256"][study.END] = "reserved"
        db.execute("UPDATE geometry_config SET spec=?", (json.dumps(config),))
    with pytest.raises(ValueError, match="discovery-only"):
        study.run(checkpoint)
    assert not registry._DB_PATH.exists()


def test_programme_continues_after_interrupt_and_data_block(checkpoint, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from research.autonomy import discovery_programme as programme

    baseline = study.run(checkpoint)
    original = programme._measure

    def interrupted(*args):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(programme, "_measure", interrupted)
        with pytest.raises(KeyboardInterrupt):
            study.run(checkpoint, programme_slot=1)

    [(row, spec)] = programme.rows()
    assert row["status"] == "RUNNING" and row["result"] is None
    assert programme.next_slot() == 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = pool.map(lambda _: study.run(checkpoint, programme_slot=1), range(2))
    assert {left["decision"], right["decision"]} == {"DISCOVERY_COMPLETE", "DISCOVERY_ALREADY_COMPLETED"}
    assert left["hypothesis_id"] == right["hypothesis_id"] == row["hypothesis_id"]
    assert left["blocked_dependencies"] == {"sector": "historical_sector_membership"}

    with monkeypatch.context() as patch:
        patch.setattr(programme, "_measure", lambda *_: (_ for _ in ()).throw(ValueError("invalid fixture")))
        failed = study.run(checkpoint, programme_slot=2)
    assert failed["decision"] == "DISCOVERY_FAILED"
    assert failed["budget_used"] == 2
    assert programme._measure is original
    last = study.run(checkpoint, programme_slot=3)
    assert last["decision"] == "DISCOVERY_COMPLETE"
    assert programme.scheduler_key() is None
    assert registry.record_result(last["hypothesis_id"], {})["error"] == "discovery_only"
    assert registry.get_experiment(baseline["hypothesis_id"])["status"] == "INCONCLUSIVE"


def test_programme_refuses_duplicate_slot_records(checkpoint):
    from research.autonomy import discovery_programme as programme

    for family in ("continuation", "liquidity"):
        registry.register_hypothesis(family, {}, {
            "stage": "DISCOVERY", "programme": programme.PROGRAMME, "slot": 1, "family": family,
        })
    with pytest.raises(ValueError, match="conflicting spend"):
        programme.next_slot()
