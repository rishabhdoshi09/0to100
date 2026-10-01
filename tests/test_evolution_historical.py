"""Historical Evolution is a weak PIT diagnostic lane, never promotion evidence."""
from __future__ import annotations

from product.evolution import historical_priors as HP
from product.evolution import policy_registry as PR
from product.evolution import promotion as PROMO


def _card(symbol: str, score: float, rs: float) -> dict:
    return {
        "symbol": symbol,
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 95.0,
        "target": 110.0,
        "score": score,
        "rs_percentile": rs,
        "sector": "Technology",
        "setup_label": "VCP",
        "dd_status": "PASS",
        "family_confirms": 3,
        "methods": [
            {"id": "tape", "status": "pass", "points": 90},
            {"id": "sepa", "status": "pass", "points": 80},
            {"id": "funds", "status": "pass", "points": 80},
            {"id": "trend", "status": "pass", "points": 80},
            {"id": "rs", "status": "pass", "points": 80},
            {"id": "sector", "status": "pass", "points": 80},
        ],
    }


def _row(symbol: str, day: str, *, decision: str, r: float, score: float, rs: float, klass: str) -> dict:
    return {
        "canonical_decision_id": f"{symbol}:{day}",
        "symbol": symbol,
        "as_of": day,
        "decision": decision,
        "raw_decision": "ENTER_NOW" if decision == "BUY" else "BLOCK",
        "reason_code": "ELIGIBLE" if decision == "BUY" else "HARD_REJECT",
        "regime": "TRENDING_BULL",
        "pit_grade": "PIT_STRONG",
        "pit": {"future_evidence_used": False},
        "selection_card": _card(symbol, score, rs),
        "outcome_status": "MATURED",
        "r_multiple": r,
        "classification": klass,
    }


def test_historical_policy_diagnostics_are_pit_and_not_promotion_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    rows = [
        _row("AAA", "2026-01-05", decision="BUY", r=-1.0, score=100, rs=20, klass="LOSER_TAKEN"),
        _row("BBB", "2026-01-05", decision="BUY", r=2.0, score=80, rs=95, klass="WINNER_TAKEN"),
        _row("CCC", "2026-01-05", decision="REJECT", r=2.0, score=99, rs=99, klass="MISSED_WINNER"),
    ]
    result = HP.evaluate_historical_policies(rows, path=tmp_path / "hist.jsonl")
    assert result["not_promotion_evidence"] is True
    assert result["historical_rows_evaluated"] == 3

    evidence = HP._read(tmp_path / "hist.jsonl")
    assert evidence
    assert all(row["evidence_class"] == "HISTORICAL_REPLAY" for row in evidence)
    assert all(row["not_promotion_evidence"] is True for row in evidence)
    ccc = [row for row in evidence if row["symbol"] == "CCC"]
    assert ccc and all(row["canonical_eligible"] is False for row in ccc)
    assert all(row["policy_selected"] is False for row in ccc)

    batch = PROMO.evaluate_promotion_batch(PR.EQUITY, persist_proofs=False)
    assert batch
    assert all(row["status"] == PROMO.NOT_ELIGIBLE for row in batch)
    assert all(int(row.get("paired_snapshots") or 0) == 0 for row in batch)


def test_historical_policy_output_does_not_read_future_market_data(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    rows = [
        _row("AAA", "2026-01-05", decision="BUY", r=1.0, score=90, rs=80, klass="WINNER_TAKEN"),
        _row("BBB", "2026-01-05", decision="BUY", r=-1.0, score=85, rs=30, klass="LOSER_TAKEN"),
    ]
    path = tmp_path / "hist.jsonl"
    first = HP.evaluate_historical_policies(rows, path=path)

    monkeypatch.setattr(
        "data.bhavcopy_store.get_ohlcv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("future data read")),
    )
    second = HP.evaluate_historical_policies(rows, path=path)
    assert first["policy_summaries"] == second["policy_summaries"]


def test_future_tainted_historical_rows_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    row = _row("AAA", "2026-01-05", decision="BUY", r=2.0, score=90, rs=90, klass="WINNER_TAKEN")
    row["pit"]["future_evidence_used"] = True
    result = HP.evaluate_historical_policies([row], path=tmp_path / "hist.jsonl")
    assert result["historical_rows_evaluated"] == 0
    assert HP._read(tmp_path / "hist.jsonl") == []



def test_fno_historical_evolution_is_underlying_only_and_never_promotion_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    rows = [
        {
            "historical_decision_id": "FNO:AAA:2026-01-05:LONG",
            "symbol": "AAA",
            "as_of": "2026-01-05",
            "direction": "LONG",
            "base_score": 78.0,
            "canonical_eligible": True,
            "setup": {
                "direction": "LONG",
                "score": 78.0,
                "breakout_distance_pct": 1.4,
                "components": {
                    "nifty_alignment": 0.0,
                    "sector_strength": 0.0,
                },
            },
            "realized_R": 1.5,
            "classification": "WINNER_TAKEN",
            "future_evidence_used": False,
        },
        {
            "historical_decision_id": "FNO:BBB:2026-01-05:SHORT",
            "symbol": "BBB",
            "as_of": "2026-01-05",
            "direction": "SHORT",
            "base_score": 70.0,
            "canonical_eligible": False,
            "setup": {
                "direction": "SHORT",
                "score": 70.0,
                "breakout_distance_pct": 0.9,
                "components": {
                    "nifty_alignment": 0.0,
                    "sector_strength": 0.0,
                },
            },
            "realized_R": -1.0,
            "classification": "CORRECT_REJECTION",
            "future_evidence_used": False,
        },
    ]
    path = tmp_path / "hist.jsonl"
    result = HP.evaluate_fno_underlying_historical(rows, path=path)
    assert result["domain"] == PR.FNO_UNDERLYING
    assert result["option_evidence_status"] == "UNDERLYING_ONLY_COUNTERFACTUAL"
    assert result["not_promotion_evidence"] is True

    evidence = HP._read(path)
    fno_rows = [row for row in evidence if row["domain"] == PR.FNO_UNDERLYING]
    assert fno_rows
    assert all(row["not_promotion_evidence"] is True for row in fno_rows)
    assert all(
        "option_chain" in row["unavailable_historical_dimensions"]
        for row in fno_rows
    )
    rejected = [row for row in fno_rows if row["symbol"] == "BBB"]
    assert rejected and all(row["policy_selected"] is False for row in rejected)

    batch = PROMO.evaluate_promotion_batch(
        PR.FNO_UNDERLYING, persist_proofs=False,
    )
    assert batch
    assert all(int(row.get("paired_snapshots") or 0) == 0 for row in batch)
    assert all(row["status"] == PROMO.NOT_ELIGIBLE for row in batch)
