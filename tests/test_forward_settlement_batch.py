"""Forward-ledger settlement must not rewrite the whole ledger once per row.

Owner's Mac 2026-11-18: after a 3-day outage ~4,000 decisions matured at once.
attach_settlement re-read and atomically rewrote the 11 MB ledger for every row,
so the outcome job ran ~1 row/s on the USB HDD for over an hour and held every
other EOD job behind it. Settlement is now batched; these tests pin the write
count and the crash-ordering guarantees.
"""

from __future__ import annotations

import pytest

from product import forward_evidence as FE
from product import forward_soak as FS
from tests.evidence_fixtures import rulebook_outcome


def _freeze_rejections(n: int, *, as_of: str = "2026-09-01", cycle_id: str = "batch-1") -> None:
    FE.freeze_cycle({
        "as_of": as_of,
        "cycle_id": cycle_id,
        "rules_hash": "hash",
        "rejections": [
            {
                "symbol": f"SYM{i:04d}",
                "reason_code": "ENTRY_TOO_EXTENDED",
                "decision": "WAIT",
                "entry": 100,
                "stop": 94,
                "target": 115,
                "setup_label": "VCP",
                "regime": "RISK_ON",
            }
            for i in range(n)
        ],
    })


def _count_writes(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = FE._write_jsonl

    def counting(path, rows):
        calls.append(len(rows))
        return real(path, rows)

    monkeypatch.setattr(FE, "_write_jsonl", counting)
    return calls


def _open_ids() -> list[str]:
    return [str(r["decision_id"]) for r in FE.load_ledger() if r.get("later_outcome") is None]


def test_attach_settlements_settles_many_rows_with_one_rewrite(monkeypatch):
    _freeze_rejections(40)
    ids = _open_ids()
    assert len(ids) == 40
    writes = _count_writes(monkeypatch)

    found = FE.attach_settlements(
        [{"decision_id": d, "classification": "GOOD_REJECT", "forward_return_pct": -3.0} for d in ids]
        + [{"decision_id": "no-such-id", "classification": "X"}]
    )

    assert len(writes) == 1
    assert set(found) == set(ids)
    assert _open_ids() == []
    rows = FE.load_ledger()
    assert all(r["later_outcome"]["forward_return_pct"] == -3.0 for r in rows)
    assert all(r["counterfactual_classification"] == "GOOD_REJECT" for r in rows)


def test_attach_settlements_never_resettles_and_first_duplicate_wins():
    _freeze_rejections(2)
    first, second = _open_ids()
    FE.attach_settlement(first, classification="MISSED_WINNER", forward_return_pct=9.0)

    FE.attach_settlements([
        {"decision_id": first, "classification": "GOOD_REJECT", "forward_return_pct": -1.0},
        {"decision_id": second, "classification": "GOOD_REJECT", "forward_return_pct": -2.0},
        {"decision_id": second, "classification": "MISSED_WINNER", "forward_return_pct": 7.0},
    ])

    by_id = {r["decision_id"]: r for r in FE.load_ledger()}
    assert by_id[first]["later_outcome"]["forward_return_pct"] == 9.0
    assert by_id[first]["counterfactual_classification"] == "MISSED_WINNER"
    assert by_id[second]["later_outcome"]["forward_return_pct"] == -2.0
    assert by_id[second]["counterfactual_classification"] == "GOOD_REJECT"


def test_attach_settlement_single_row_still_returns_the_row_or_none():
    _freeze_rejections(1)
    (only,) = _open_ids()
    assert FE.attach_settlement("missing", classification="X") is None
    row = FE.attach_settlement(only, classification="GOOD_REJECT", forward_return_pct=-4.0)
    assert row is not None and row["later_outcome"]["forward_return_pct"] == -4.0


def test_counterfactual_settlement_rewrites_ledger_per_chunk_not_per_row(monkeypatch):
    monkeypatch.setattr(FS, "_SETTLE_FLUSH_EVERY", 10)
    _freeze_rejections(25)
    writes = _count_writes(monkeypatch)

    out = FS.settle_pending_from_market(outcome_fn=lambda _symbol, _as_of, _row: rulebook_outcome(-0.4, "FLAT", -2.0))

    assert out["updated"] == 25
    assert out["pending"] == 0
    assert len(writes) == 3  # 10 + 10 + 5, not 25
    assert _open_ids() == []
    again = FS.settle_pending_from_market(outcome_fn=lambda _symbol, _as_of, _row: rulebook_outcome(-0.4, "FLAT", -2.0))
    assert again["updated"] == 0


def test_counterfactual_crash_mid_chunk_is_replayed_exactly_once(monkeypatch):
    """Cross-review A-3: the outbox replays what a crash stranded; nothing is lost or counted twice."""
    from product import paper_learning_loop as PLL

    monkeypatch.setattr(FS, "_SETTLE_FLUSH_EVERY", 5)
    _freeze_rejections(5)
    ingested: list[str] = []
    raised: list[bool] = []
    real_ingest = PLL.ingest_counterfactual

    def flaky(settled, **kwargs):
        if len(ingested) == 2 and not raised:
            raised.append(True)
            raise RuntimeError("power cut")
        ingested.append(str(settled.get("symbol")))
        return real_ingest(settled, **kwargs)

    monkeypatch.setattr(PLL, "ingest_counterfactual", flaky)
    with pytest.raises(RuntimeError):
        FS.settle_pending_from_market(outcome_fn=lambda _symbol, _as_of, _row: rulebook_outcome(-0.4, "FLAT", -2.0))

    # The whole chunk is settled on the ledger; the failed event stays queued.
    assert _open_ids() == []
    rerun = FS.settle_pending_from_market(outcome_fn=lambda _symbol, _as_of, _row: rulebook_outcome(-0.4, "FLAT", -2.0))
    assert rerun["updated"] == 0
    assert len(ingested) == len(set(ingested)) == 5          # every observation exactly once


def test_missing_bars_stay_pending_and_cost_no_rewrite(monkeypatch):
    _freeze_rejections(4)
    writes = _count_writes(monkeypatch)
    out = FS.settle_pending_from_market(outcome_fn=lambda _symbol, _as_of, _row: None)
    assert out == {"updated": 0, "pending": 4, "classifications": {}}
    assert writes == []


def test_official_settlement_rewrites_ledger_per_chunk_not_per_row(monkeypatch):
    from product import autonomous_loop as LOOP

    monkeypatch.setattr(LOOP, "_SETTLE_FLUSH_EVERY", 4)
    _freeze_rejections(9)
    monkeypatch.setattr(
        "core.outcome_resolver.counterfactual_outcome",
        lambda *a, **k: rulebook_outcome(0.1, "FLAT", 1.0),
    )
    writes = _count_writes(monkeypatch)

    out = LOOP.settle_official_outcomes("2026-09-08")

    assert out["n_settled"] == 9
    assert len(writes) == 3  # 4 + 4 + 1, not 9
    assert _open_ids() == []
    assert LOOP.settle_official_outcomes("2026-09-08")["n_settled"] == 0


def test_official_settlement_keeps_resolved_rows_when_a_later_row_raises(monkeypatch):
    from product import autonomous_loop as LOOP
    from product import counterfactual_learning as CF

    _freeze_rejections(3)
    monkeypatch.setattr(
        "core.outcome_resolver.counterfactual_outcome",
        lambda *a, **k: rulebook_outcome(0.1, "FLAT", 1.0),
    )
    real_classify = CF.classify_rulebook
    seen: list[int] = []

    def classify(*args, **kwargs):
        seen.append(1)
        if len(seen) == 3:
            raise RuntimeError("bad row")
        return real_classify(*args, **kwargs)

    monkeypatch.setattr(CF, "classify_rulebook", classify)
    with pytest.raises(RuntimeError):
        LOOP.settle_official_outcomes("2026-09-08")
    assert len(_open_ids()) == 1
