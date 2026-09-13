"""A scan that evaluated nothing must not replace the last real scan.

A background pass started before the OHLCV cache was warm reported
scan_status=SUCCEEDED with scanned=0 and records=[], and save_scan() wrote that
empty artifact over a full 234-row scan. The desk then showed an empty
Opportunities page with no indication that real results had just been
destroyed. A genuine zero-setup scan (symbols evaluated, none qualified) is a
different thing and must still persist.
"""
from __future__ import annotations

import scan.market_scan_service as MSS


class _Scanner:
    """Evaluates nothing, mimicking a cold cache."""
    def __init__(self, results=()):
        self._results = list(results)

    def scan(self, *a, **k):
        return list(self._results)

    def __call__(self, *a, **k):
        return list(self._results)


def test_zero_evaluated_symbols_is_data_unavailable_not_success(monkeypatch, tmp_path):
    saved: list = []
    monkeypatch.setattr("product.scan_store.save_scan", lambda p, *a, **k: saved.append(p))

    report = MSS.run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha", "BBB": "Beta"},
        prefetch_fn=lambda *a, **k: 2,
        scanner=_Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    # Either the cache guard or the zero-evaluated guard must stop it; what
    # matters is that it is NOT reported as a successful scan, and nothing
    # was written over the previous artifact.
    assert report.status != MSS.SUCCEEDED
    assert report.error_code in {"NO_SYMBOL_EVALUATED", "OHLCV_CACHE_EMPTY"}
    assert not saved, "an empty scan must never overwrite the last real one"


def test_error_message_says_the_previous_scan_was_kept(monkeypatch):
    monkeypatch.setattr("product.scan_store.save_scan",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not save")))
    report = MSS.run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda *a, **k: 1,
        scanner=_Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert "kept" in (report.error_message or "").lower()
