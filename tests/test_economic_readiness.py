from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from research import registry
from research.autonomy import economic_readiness as readiness


def filing(received, published):
    return {"symbol": "SANOFI", "period_end": "2024-12-31", "received_at": received,
            "disseminated_at": published, "source_url": (
                "https://www.nseindia.com/companies-listing/corporate-filings-financial-results?symbol=SANOFI"
            )}


def test_dissemination_controls_next_open_and_gap():
    sessions = ["2025-02-26", "2025-02-27", "2025-02-28", "2025-03-03"]
    row = filing("2025-02-26T22:00:00+05:30", "2025-02-28T16:23:13+05:30")
    outcome = readiness.first_daily_entry(row, sessions, set())
    assert outcome["formation_session"] == "2025-02-28"
    assert outcome["earliest_actionable_daily_session"] == "2025-03-03"
    assert outcome["reprice_gap_observable"]
    row["disseminated_at"] = "2025-02-28T14:23:13+05:30"
    assert not readiness.first_daily_entry(row, sessions, set())["reprice_gap_observable"]
    assert readiness.first_daily_entry(row, sessions, {"2025-03-03"})["exclusion"] == "NEXT_SESSION_SPECIAL_HOURS_UNVERIFIED"


@pytest.mark.parametrize("timestamp", ["2025-02-28", "2025-02-28T16:23:13", "2025-11-01T01:00:00+05:30"])
def test_refuses_date_only_naive_and_reserved_publication(timestamp):
    row = filing("2025-02-26T22:00:00+05:30", timestamp)
    with pytest.raises(ValueError):
        readiness.first_daily_entry(row, ["2025-02-26", "2025-03-03"], set())


def test_audit_refuses_overwriting_a_receipt(monkeypatch):
    with TemporaryDirectory(dir="/tmp") as scratch:
        root = Path(scratch).resolve()
        monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
        monkeypatch.setattr(registry, "_DB_PATH", root / "experiments.sqlite")
        receipt = root / "logs" / readiness.OUTPUT_NAME
        receipt.parent.mkdir()
        receipt.write_text("[]")
        base, frontier = root / "base.sqlite", root / "frontier.sqlite"
        base.touch()
        frontier.touch()
        with pytest.raises(ValueError, match="escapes isolated runtime"):
            readiness.audit(base, frontier, receipt)
        assert receipt.read_text() == "[]"


def test_scheduled_audit_bound_to_receipt_generation(monkeypatch):
    from types import SimpleNamespace
    from research.autonomy import jobs, job_store

    with TemporaryDirectory(dir="/tmp") as scratch:
        root = Path(scratch).resolve()
        monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
        monkeypatch.setattr(registry, "_DB_PATH", root / "experiments.sqlite")
        base, frontier, receipt = (root / name for name in ("base", "frontier", "receipt"))
        for path in (base, frontier, receipt):
            path.touch()
        monkeypatch.setenv("QT_OUTCOME_DISCOVERY_CHECKPOINT", str(base))
        monkeypatch.setenv("QT_ECONOMIC_FRONTIER", str(frontier))
        monkeypatch.setenv("QT_ECONOMIC_RECEIPT", str(receipt))
        key, *_ = readiness.configured_job(base)
        monkeypatch.setattr(readiness, "audit", lambda *_a, **_k: {"predictive_looks_used": 0})
        ctx = SimpleNamespace(job=SimpleNamespace(idempotency_key=key))
        assert jobs.run_research_cycle(ctx).status == job_store.SUCCEEDED
        receipt.write_text("new generation")
        assert jobs.run_research_cycle(ctx).status == job_store.PERMANENT_FAILED
