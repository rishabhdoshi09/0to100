"""Official-data autonomous loop: no Kite required for post-market work."""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from product import candidate_lifecycle as CL
from product import readiness as RDY
from research.autonomy import jobs as JOBS
from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH


IST = ZoneInfo("Asia/Kolkata")


def test_readiness_separates_official_from_broker():
    matrix = RDY.inspect_readiness()
    assert RDY.OFFICIAL_MARKET_DATA_READY in matrix["capabilities"]
    assert RDY.BROKER_LIVE_DATA_READY in matrix["capabilities"]
    assert RDY.OUTCOME_DATA_READY in matrix["capabilities"]
    assert "OUTCOME_RESOLUTION" in matrix["allowed_without_kite"]
    assert "PAPER_ENTRY" in matrix["blocked_without_kite"]
    assert "MARKET_SCAN_COMPLETED_SESSION" in matrix["allowed_without_kite"]


def test_candidate_lifecycle_persists(tmp_path):
    db = tmp_path / "c.db"
    row = CL.upsert(
        symbol="INFY", session_date="2026-09-02", state=CL.SCREENED,
        reason="scan", scan_run_id="scan-1", path=db, trigger="scan",
    )
    CL.upsert(
        symbol="INFY", session_date="2026-09-02", state=CL.QUALIFIED,
        reason="reco", recommendation_id_value="scan-1:INFY:high_conviction",
        path=db, trigger="recommendation",
    )
    got = CL.get(CL.candidate_id("INFY", "2026-09-02"), path=db)
    assert got["state"] == CL.QUALIFIED
    assert got["scan_run_id"] == "scan-1"
    assert got["recommendation_id"] == "scan-1:INFY:high_conviction"
    hist = CL.transitions_for(got["candidate_id"], path=db)
    assert [h["to_state"] for h in hist] == [CL.SCREENED, CL.QUALIFIED]


def test_official_outcome_settles_without_kite(tmp_path, monkeypatch):
    from product import autonomous_loop as LOOP
    from product import counterfactual_learning as CF

    cf = tmp_path / "cf.jsonl"
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(cf))
    monkeypatch.setenv("QT_FORWARD_LEDGER", str(tmp_path / "fe.jsonl"))
    CF.freeze_decision(
        symbol="HAL", reason_code="WAIT_FOR_ENTRY", decision="WAIT",
        entry=100.0, stop=90.0, target=120.0, as_of="2026-08-03",
        evidence={"scan_run_id": "s1", "decision_id": "d1"},
        path=cf,
    )
    monkeypatch.setattr(
        "core.outcome_resolver.session_close_return",
        lambda symbol, day, horizon=5: (110.0, 10.0),
    )
    out = LOOP.settle_official_outcomes("2026-08-03")
    assert out["n_settled"] >= 1, out
    assert out["settled"][0]["classification"]
    assert out["settled"][0]["symbol"] == "HAL"


def test_paper_consume_keeps_broker_on_execution_only(monkeypatch, tmp_path):
    from product import autonomous_loop as LOOP
    from product import decision_journal as DJ
    from product import opportunity_memory as OM

    monkeypatch.setattr(CL, "DB_PATH", tmp_path / "c.db")
    monkeypatch.setattr(DJ, "DB_PATH", tmp_path / "d.db")
    monkeypatch.setattr(DJ, "JSONL_PATH", tmp_path / "d.jsonl")
    monkeypatch.setattr(OM, "DB_PATH", tmp_path / "o.db")
    monkeypatch.setattr(
        RDY, "inspect_readiness",
        lambda: {"capabilities": {RDY.BROKER_LIVE_DATA_READY: False}},
    )
    monkeypatch.setattr("product.forward_evidence.freeze_observation", lambda *a, **k: None)
    monkeypatch.setattr("product.counterfactual_learning.freeze_decision", lambda *a, **k: None)
    rec = {
        "symbol": "INFY",
        "decision": "BUY",
        "candidate_state": "READY",
        "entry_state": "ENTER_NOW",
        "execution_state": "BLOCKED_BROKER_AUTH",
        "reason_code": "COMMITTEE_BUY",
        "reason": "families justify taking risk",
        "tier": "high_conviction",
        "vetoes": [],
        "wait_trigger": {},
    }
    paper = LOOP._consume_paper(
        [{"symbol": "INFY", "reco_tier": "high_conviction", "entry": 100, "stop": 90, "target": 120}],
        {}, "2026-09-02", "scan-1", committee=[rec],
    )
    assert paper["broker_ok"] is False
    assert paper["intents"]
    assert paper["intents"][0]["decision"] == "BUY"
    assert paper["intents"][0]["reason_code"] == "COMMITTEE_BUY"
    assert paper["intents"][0]["execution_state"] == "BLOCKED_BROKER_AUTH"
    assert paper["eligibility"] == "BLOCKED_BROKER"
    got = CL.get(CL.candidate_id("INFY", "2026-09-02"), path=tmp_path / "c.db")
    assert got["state"] == "READY"
    assert got["decision"] == "BUY"
    assert got["execution_state"] == "BLOCKED_BROKER_AUTH"
    assert got["reason"] != "BROKER_LOGIN_REQUIRED"


def test_acquire_without_download_marks_wait_evidence(monkeypatch, tmp_path):
    from product import autonomous_loop as LOOP

    monkeypatch.setattr(CL, "DB_PATH", tmp_path / "c.db")
    monkeypatch.setattr(LOOP, "_facts_present", lambda symbol: {})
    monkeypatch.setattr(CL, "upsert", lambda **k: {"candidate_id": k.get("symbol"), "state": k.get("state")})
    out = LOOP._acquire(["INFY"], "2026-09-02", "scan-1", download=False)
    assert out["n_waiting"] == 1
    assert out["downloaded"] is False
    assert out["n_ok"] == 0


def test_lineage_ids_are_stable():
    scan_run = "2026-09-02T18:30:37.723153+00:00"
    cid = CL.candidate_id("INFY", "2026-09-02")
    rid = CL.recommendation_id(scan_run, "INFY", "high_conviction")
    assert cid == "2026-09-02:INFY"
    assert rid.startswith(scan_run)
    assert rid.endswith(":INFY:high_conviction")


def test_consume_learning_memory_does_not_change_policy(tmp_path, monkeypatch):
    from product import autonomous_loop as LOOP
    from product import counterfactual_learning as CF

    cf = tmp_path / "cf.jsonl"
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(cf))
    monkeypatch.setattr(LOOP, "MEMORY_PATH", tmp_path / "memory.json")
    monkeypatch.setattr(LOOP, "LEARNING_PATH", tmp_path / "learn.jsonl")
    CF.freeze_decision(
        symbol="HAL", reason_code="WAIT_FOR_ENTRY", decision="WAIT",
        entry=100.0, stop=90.0, target=120.0, as_of="2026-08-03",
        evidence={"setup": "breakout"}, path=cf,
    )
    row = json.loads(cf.read_text().strip())
    row["classification"] = "MISSED_WINNER"
    cf.write_text(json.dumps(row) + "\n")
    memory = LOOP.consume_learning_memory("2026-09-02")
    assert memory["updates_policy"] is False
    assert memory["observations"] == 1
    assert memory["classification_counts"]["MISSED_WINNER"] == 1


def test_last_completed_session_uses_holidays():
    # Thursday 00:05 after Wednesday session.
    assert SCH.last_completed_session_date(datetime(2026, 9, 3, 0, 5, tzinfo=IST), holidays=set()) == "2026-09-02"
    # Saturday after Friday.
    assert SCH.last_completed_session_date(datetime(2026, 9, 5, 12, 0, tzinfo=IST), holidays=set()) == "2026-09-04"
    # Monday after a Friday holiday — walk back past Sunday/Saturday to Thursday.
    assert SCH.last_completed_session_date(
        datetime(2026, 9, 7, 8, 0, tzinfo=IST),
        holidays={datetime(2026, 9, 4).date()},
    ) == "2026-09-03"


def test_outcome_job_succeeds_on_official_without_snapshot():
    class Deps:
        def now_ist(self):
            return datetime(2026, 9, 2, 23, 50, tzinfo=IST)

        def holidays(self):
            return set()

        def active_snapshot_id(self):
            return None

        def official_history(self):
            return {"current": True, "available_session": "2026-09-02", "latest_date": "2026-09-02"}

        def resolve_outcomes(self, session_date, failures=()):
            return {"positions_closed": [], "outcomes_recorded": []}

    result = JOBS.run_outcome_resolution(JOBS._Ctx(Deps()))
    assert result.status == JS.SUCCEEDED
    assert "official" in result.summary
