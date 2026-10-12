"""Worker logs never stringify full persisted scanner results."""
from __future__ import annotations


class ForbiddenPayload:
    def __str__(self):
        raise AssertionError("large payload stringification must never occur")

    def __repr__(self):
        raise AssertionError("large payload repr must never occur")


def test_completion_log_keeps_only_short_metadata():
    from operations.market_ops import _operation_log_summary

    sample = {
        "status": "SUCCEEDED",
        "records": 419,
        "as_of_session": "2026-10-09",
        "source_snapshot_id": "abc123",
        "payload": {
            "scanned": 2179,
            "qualified_rows": 419,
            "records": [ForbiddenPayload() for _ in range(419)],
            "raw_provider_data": ForbiddenPayload(),
        },
        "other_sensitive_data": ForbiddenPayload(),
    }
    summary = _operation_log_summary(sample)
    assert "status=SUCCEEDED" in summary
    assert "records=419" in summary
    assert "scanned=2179" in summary
    assert "qualified=419" in summary
    assert "as_of_session=2026-10-09" in summary
    assert "abc123" in summary
    assert len(summary) < 250


def test_completion_log_handles_non_dict_and_large_nested_dict_without_dumping():
    from operations.market_ops import _operation_log_summary

    assert _operation_log_summary(None) == "result=unavailable"
    assert _operation_log_summary({}) == "result=persisted"
    summary = _operation_log_summary({
        "status": "SUCCEEDED",
        "payload": {"records": [ForbiddenPayload()], "metadata": ForbiddenPayload()},
    })
    assert summary == "status=SUCCEEDED"
