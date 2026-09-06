from __future__ import annotations

from research.autonomy import job_store as JS
from research.autonomy import jobs as J
from research.autonomy import outcome_integrity as OI


def test_partial_official_settlement_is_retryable_not_success(monkeypatch):
    base = J.JobResult(
        JS.SUCCEEDED,
        "outcomes resolved",
        unblocks=("OUTCOMES_RESOLVED:2026-09-04",),
        metadata={
            "official_settlement": {
                "n_settled": 2,
                "failed": [{"symbol": "AAA", "error": "official bar decoder failed"}],
            }
        },
    )
    monkeypatch.setattr(OI, "_BASE_HANDLER", lambda _ctx: base)

    result = OI.strict_outcome_resolution(object())

    assert result.status == JS.RETRYABLE_FAILED
    assert result.error_code == "OFFICIAL_SETTLEMENT_PARTIAL"
    assert "AAA" in result.error_message
    assert result.unblocks == ()
    assert result.metadata["official_settlement"]["n_settled"] == 2


def test_clean_official_settlement_preserves_success(monkeypatch):
    base = J.JobResult(
        JS.SUCCEEDED,
        "outcomes resolved",
        unblocks=("OUTCOMES_RESOLVED:2026-09-04",),
        metadata={"official_settlement": {"n_settled": 2, "failed": []}},
    )
    monkeypatch.setattr(OI, "_BASE_HANDLER", lambda _ctx: base)

    result = OI.strict_outcome_resolution(object())

    assert result is base
    assert result.status == JS.SUCCEEDED
    assert result.unblocks == ("OUTCOMES_RESOLVED:2026-09-04",)
