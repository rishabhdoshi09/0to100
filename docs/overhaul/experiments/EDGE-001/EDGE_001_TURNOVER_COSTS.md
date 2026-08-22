# EDGE-001 — Turnover and Costs

Primary book is equal-weight Top 20. One-way turnover = `(|added| + |removed|) / (2N)`, 100% on the first deploy. CNC round-trip from `core.costs` is applied as `one_way × rt_pct / 100`.

- Average one-way turnover per year: **472.29%**
- Average cost drag per year: **1.51%**
- CAGR gross → net: 31.98% → 30.04%
- Cost vs EW: the edge must survive this drag. Failure flag `costs_destroy_edge` is evaluated in the decision file.

Per-rebalance added / removed / retained / cost live in `logs/edge001/portfolio_periods.json` and `transaction_ledger.csv`.
