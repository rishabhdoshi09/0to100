from __future__ import annotations


def test_default_coverage_observer_never_runs_synchronous_repair(monkeypatch):
    import scan.bulk_fetcher as bulk_fetcher
    from scan.scan_coverage import NO_OHLCV, QUALIFIED, observe_scanner

    def forbidden_repair(_symbols):
        raise AssertionError("whole-market coverage must not synchronously repair history")

    monkeypatch.setattr(bulk_fetcher, "backfill_missing", forbidden_repair)

    class Result:
        symbol = "AAA"
        signals = ["MOMENTUM"]

    class Scanner:
        def _analyze(self, symbol, _df):
            assert symbol == "AAA"
            return Result()

    scanner = Scanner()
    with observe_scanner(scanner, ["AAA", "BBB"]) as probe:
        result = scanner._analyze("AAA", None)

    audit = probe.finalize([result], cached=["AAA"], walked_total=1)
    rows = {row["symbol"]: row for row in audit["ledger"]}
    repair = audit["summary"]["history_repair"]

    assert rows["AAA"]["status"] == QUALIFIED
    assert rows["BBB"]["status"] == NO_OHLCV
    assert repair["skipped"] is True
    assert repair["reason"] == "whole_market_scan_nonblocking"
    assert repair["attempted"] == 0
    assert repair["loaded"] == 0


def test_targeted_caller_can_explicitly_opt_into_history_repair(monkeypatch):
    import scan.bulk_fetcher as bulk_fetcher
    from scan.scan_coverage import observe_scanner

    calls = []

    def repair(symbols):
        calls.append(list(symbols))
        return {
            "requested": len(symbols),
            "attempted": 1,
            "loaded": 1,
            "unresolved": 0,
        }

    monkeypatch.setattr(bulk_fetcher, "backfill_missing", repair)

    class Scanner:
        def _analyze(self, _symbol, _df):
            return None

    with observe_scanner(Scanner(), ["AAA"], repair_missing=True) as probe:
        pass

    assert calls == [["AAA"]]
    assert probe.history_repair["attempted"] == 1
    assert probe.history_repair["loaded"] == 1
