# Opportunities fixture

Produced by the real production pipeline, not hand-written:
`scan.market_scan_service.run_whole_market_scan(save=True)` with the real
`UnifiedScanner`, coverage ledger, long-term overlay and recommendation
publication, in a disposable runtime root.

Only the OHLCV input is synthetic (seeded random walks, 240 symbols x 500
sessions, a subset shaped into volume breakouts) because NSE archives are not
reachable from CI. Result: 100 scan records, 22 BUY verdicts, and the
recommendation projection the scan service published for that exact scan.

These files are test evidence only. They are never loaded by the product.
