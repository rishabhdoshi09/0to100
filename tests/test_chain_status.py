"""The chain diagnostic must name the first real stop, never guess past it.

An intentionally gated desk and a broken desk both look identical from the UI:
empty. This diagnostic exists so the operator can tell them apart, so its own
honesty matters more than its coverage -- a probe that fails must say UNKNOWN
rather than silently report the link as flowing.
"""
from __future__ import annotations

import product.chain_status as CS


def _stub(name, state):
    return lambda: CS._link(name, state, f"{name} is {state}")


def test_first_stop_is_the_earliest_non_flowing_link(monkeypatch):
    monkeypatch.setattr(CS, "_official_history", _stub("Official history", CS.FLOWING))
    monkeypatch.setattr(CS, "_raw_sessions_for_replay", _stub("Replay sessions on disk", CS.FLOWING))
    monkeypatch.setattr(CS, "_latest_scan", _stub("Whole-market scan", CS.BLOCKED))
    monkeypatch.setattr(CS, "_candidates", _stub("Candidates", CS.BLOCKED))
    monkeypatch.setattr(CS, "_historical_gate", _stub("Historical reproduction", CS.FLOWING))
    monkeypatch.setattr(CS, "_paper_capability", _stub("Paper entry capability", CS.FLOWING))
    monkeypatch.setattr(CS, "_open_positions", _stub("Open paper positions", CS.FLOWING))
    monkeypatch.setattr(CS, "_forward_evidence", _stub("Forward evidence", CS.FLOWING))

    out = CS.build_chain_status()
    assert out["first_stop"] == "Whole-market scan"
    assert out["chain_complete"] is False
    assert out["flowing"] == 6


def test_a_probe_that_raises_reports_unknown_not_flowing():
    def boom():
        raise RuntimeError("subsystem exploded")

    row = CS._safe(boom, "Some link")
    assert row["state"] == CS.UNKNOWN
    assert "RuntimeError" in row["detail"]
    assert row["state"] != CS.FLOWING


def test_every_non_flowing_link_carries_an_unblock_instruction(monkeypatch):
    out = CS.build_chain_status()
    for row in out["links"]:
        if row["state"] in {CS.BLOCKED, CS.WAITING}:
            assert row["unblock"], f"{row['link']} gives no way forward"


def test_complete_chain_reports_no_first_stop(monkeypatch):
    for attr, name in (
        ("_official_history", "Official history"),
        ("_raw_sessions_for_replay", "Replay sessions on disk"),
        ("_latest_scan", "Whole-market scan"),
        ("_candidates", "Candidates"),
        ("_historical_gate", "Historical reproduction"),
        ("_paper_capability", "Paper entry capability"),
        ("_open_positions", "Open paper positions"),
        ("_forward_evidence", "Forward evidence"),
    ):
        monkeypatch.setattr(CS, attr, _stub(name, CS.FLOWING))
    out = CS.build_chain_status()
    assert out["chain_complete"] is True
    assert out["first_stop"] == ""
    assert "Every link is flowing." in CS.render_text(out)


def test_render_marks_a_stop_distinctly(monkeypatch):
    monkeypatch.setattr(CS, "_latest_scan", _stub("Whole-market scan", CS.BLOCKED))
    text = CS.render_text(CS.build_chain_status())
    assert "[STOP]" in text
    assert "FIRST STOP:" in text
