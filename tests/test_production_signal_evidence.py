from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from product.production_signal_evidence import (
    MIN_VERIFIED_SAMPLE,
    MIN_VERIFIED_SCAN_DAYS,
    build_production_signal_evidence,
    save_production_signal_evidence,
)


def _candidate(symbol: str, *, allows: bool = True, geometry: bool = True, future_event: str = "") -> dict:
    raw = {
        "symbol": symbol,
        "category_id": "momentum_breakouts",
        "status": "Ready to trade",
        "entry": 100.0 if geometry else None,
        "stop": 95.0 if geometry else None,
        "target": 110.0 if geometry else None,
    }
    if future_event:
        raw["material_events"] = [{"published_at": future_event, "headline": "future"}]
    return {
        "input": raw,
        "expert_snapshot": [
            {
                "id": "xs_momentum", "family": "price_leadership", "status": "pass",
                "eligible": True, "thesis": "Cross-sectional momentum", "horizon": "swing",
                "evidence": ["ranked leadership"],
            },
            {
                "id": "vcp", "family": "structure", "status": "pass",
                "eligible": True, "thesis": "VCP", "horizon": "swing",
                "evidence": ["tight base"],
            },
            {
                "id": "quality", "family": "business_quality", "status": "pass",
                "eligible": True, "thesis": "Quality", "horizon": "position",
                "evidence": ["quality measured"],
            },
        ],
        "decision_at_capture": {
            "symbol": symbol,
            "reco_tier": "high_conviction" if allows else "watch",
            "allows_recommend": allows,
            "entry_state": "ready" if allows else "watch",
            "primary_thesis_id": "vcp" if allows else "",
            "family_confirms": 3 if allows else 1,
            "category_id": "momentum_breakouts",
        },
    }


def _record(day: int, candidates: list[dict], *, rules_hash: str = "same-hash", captured_live: bool = True) -> dict:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "schema_version": 2,
        "captured_at": (base + timedelta(minutes=2)).isoformat(),
        "scan_scanned_at": base.isoformat(),
        "captured_live": captured_live,
        "replay_scope": "FROZEN_EXPERTS_TO_ENSEMBLE_DECISION",
        "production_strategy": {
            "strategy_id": "QT_RECO_ENSEMBLE",
            "strategy_version": 1,
            "rules_hash": rules_hash,
        },
        "candidates": candidates,
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def test_same_hash_live_forward_requires_sample_and_date_diversity(tmp_path):
    replay = tmp_path / "replay.jsonl"
    records = []
    for day in range(MIN_VERIFIED_SCAN_DAYS):
        records.append(_record(day, [_candidate(f"S{day}_{j}") for j in range(3)]))
    _write(replay, records)

    calls = []
    def resolver(symbol, opened_at, entry, stop, target):
        calls.append((symbol, opened_at, entry, stop, target))
        # deterministic mix of target and stop outcomes
        if len(calls) % 3 == 0:
            return stop, (stop - entry) / entry * 100.0, 0
        return target, (target - entry) / entry * 100.0, 1

    payload = build_production_signal_evidence(
        replay_path=replay,
        current_rules_hash="same-hash",
        resolver=resolver,
    )
    assert payload["point_in_time_verified"] is True
    assert payload["evidence_ready"] is True
    assert payload["metrics"]["sample_size"] == MIN_VERIFIED_SAMPLE
    assert payload["metrics"]["distinct_scan_dates"] == MIN_VERIFIED_SCAN_DAYS
    assert payload["metrics"]["wins"] == 20
    assert payload["metrics"]["losses"] == 10
    assert payload["metrics"]["costs_included"] is False
    assert payload["metrics"]["portfolio_drawdown_pct"] is None
    assert payload["walk_forward"]["out_of_sample"] is True
    assert payload["walk_forward"]["historical_reconstruction"] is False


def test_one_day_or_small_sample_cannot_become_ready(tmp_path):
    replay = tmp_path / "replay.jsonl"
    # 30 names from one scan day are intentionally not enough.
    _write(replay, [_record(0, [_candidate(f"ONE{j}") for j in range(MIN_VERIFIED_SAMPLE)])])

    payload = build_production_signal_evidence(
        replay_path=replay,
        current_rules_hash="same-hash",
        resolver=lambda *_args: (110.0, 10.0, 1),
    )
    assert payload["metrics"]["sample_size"] == MIN_VERIFIED_SAMPLE
    assert payload["metrics"]["distinct_scan_dates"] == 1
    assert payload["evidence_ready"] is False
    assert any("distinct scan dates" in reason for reason in payload["blockers"])


def test_hash_mismatch_is_excluded_not_borrowed(tmp_path):
    replay = tmp_path / "replay.jsonl"
    _write(replay, [_record(0, [_candidate("OLD")], rules_hash="old-hash")])
    payload = build_production_signal_evidence(
        replay_path=replay,
        current_rules_hash="new-hash",
        resolver=lambda *_args: (110.0, 10.0, 1),
    )
    assert payload["dataset"]["same_hash_records"] == 0
    assert payload["metrics"]["sample_size"] == 0
    assert payload["evidence_ready"] is False
    assert any("current executable rules hash" in reason for reason in payload["blockers"])


def test_future_dated_evidence_fails_point_in_time_gate(tmp_path):
    replay = tmp_path / "replay.jsonl"
    future = datetime(2026, 1, 3, 10, 0, tzinfo=timezone.utc).isoformat()
    _write(replay, [_record(0, [_candidate("FUTURE", future_event=future)])])
    payload = build_production_signal_evidence(
        replay_path=replay,
        current_rules_hash="same-hash",
        resolver=lambda *_args: (110.0, 10.0, 1),
    )
    assert payload["point_in_time_verified"] is False
    assert payload["metrics"]["sample_size"] == 0
    assert payload["dataset"]["pit_violations"]


def test_no_fill_pending_missing_geometry_and_watch_are_not_losses(tmp_path):
    replay = tmp_path / "replay.jsonl"
    candidates = [
        _candidate("NOFILL"),
        _candidate("PENDING"),
        _candidate("NOGEO", geometry=False),
        _candidate("WATCH", allows=False),
        _candidate("WIN"),
    ]
    _write(replay, [_record(0, candidates)])

    def resolver(symbol, *_args):
        if symbol == "NOFILL":
            return 0.0, 0.0, -1
        if symbol == "PENDING":
            return None
        if symbol == "WIN":
            return 110.0, 10.0, 1
        raise AssertionError(f"resolver should not be called for {symbol}")

    payload = build_production_signal_evidence(
        replay_path=replay,
        current_rules_hash="same-hash",
        resolver=resolver,
    )
    metrics = payload["metrics"]
    assert metrics["recommended_captures"] == 4
    assert metrics["sample_size"] == 1
    assert metrics["wins"] == 1
    assert metrics["losses"] == 0
    assert metrics["no_fill"] == 1
    assert metrics["pending"] == 1
    assert metrics["missing_geometry"] == 1


def test_duplicate_same_day_symbol_category_is_counted_once(tmp_path):
    replay = tmp_path / "replay.jsonl"
    first = _record(0, [_candidate("DUP")])
    second = _record(0, [_candidate("DUP")])
    _write(replay, [first, second])
    calls = 0
    def resolver(*_args):
        nonlocal calls
        calls += 1
        return 110.0, 10.0, 1

    payload = build_production_signal_evidence(
        replay_path=replay,
        current_rules_hash="same-hash",
        resolver=resolver,
    )
    assert calls == 1
    assert payload["metrics"]["sample_size"] == 1


def test_saved_artifact_is_plain_auditable_json(tmp_path):
    artifact = tmp_path / "evidence.json"
    payload = {
        "rules_hash": "abc",
        "scope": "COLLECTING_PRODUCTION_SIGNAL_OUTCOMES",
        "completed": True,
        "evidence_ready": False,
        "metrics": {"sample_size": 0},
    }
    saved = save_production_signal_evidence(payload, artifact_path=artifact)
    assert saved == artifact
    assert json.loads(artifact.read_text(encoding="utf-8"))["scope"] == "COLLECTING_PRODUCTION_SIGNAL_OUTCOMES"
