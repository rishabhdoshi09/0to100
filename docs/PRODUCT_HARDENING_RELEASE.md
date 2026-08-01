# QuantTerm Product Hardening Release

## Why this release exists

QuantTerm already had real Python engines and persisted stores, but the product layer did not consistently turn them into a usable retail workflow. A technically honest empty card is still a poor product when the user cannot see what the feature means, why it is empty, how old the data is, or which action repairs it.

The hardening audit found five structural problems:

1. The interface read saved state but did not offer one product-level preparation workflow.
2. Stock Intelligence was too thin: chart, scores and labels existed, but the actual technical and fundamental metrics were not explained together.
3. Search was restricted to symbols already present in saved scanner, long-term or F&O records.
4. F&O was presented like a desk even though the current data only supports instrument eligibility, expiry and lot-size coverage.
5. Paper/autonomy status occupied too much product attention relative to research and data quality.

## What changed

### Product readiness contract

The Command Center now scores the actual working product, not the number of rendered sections. Each data lane is classified as `FRESH`, `STALE`, `MISSING` or `UNKNOWN_DATE` and includes:

- the purpose of the lane;
- the strict source date;
- its age;
- the current record count or coverage;
- the owner action that repairs it.

The weighted lanes are market operations, official history, whole-market scanner, long-term research, news and filings, F&O coverage, and market regime/breadth.

### One-click product preparation

`Make QuantTerm ready` queues independent operations for:

- official data preparation and F&O instrument refresh;
- news and filing refresh;
- whole-market technical scan;
- long-term scan with fundamental refresh.

The operations remain durable and visible. They do not run scanner logic inside React, and paper execution cannot block them.

### Explainable single-stock workspace

Any syntactically valid NSE symbol can now open Stock Intelligence, even when it is not already in a shortlist. The workspace combines available persisted evidence and calculates technicals from official daily history:

- EMA 20, EMA 50 and EMA 200;
- RSI 14;
- ATR 14 and ATR as a percentage of price;
- 1-month, 3-month and 12-month returns;
- 52-week high/low distances;
- volume relative to the 20-day average;
- trend state and a plain-language explanation.

Current fundamental metrics are shown with definitions and interpretations rather than unexplained abbreviations:

- market capitalisation and P/E;
- ROE and ROCE;
- three-year sales and profit growth;
- debt/equity, interest coverage and cash conversion;
- promoter holding and pledge;
- FII and DII holding when available.

Every stock workspace exposes source status, as-of date, age and missing-data actions. Missing metrics remain unavailable; they are never replaced with estimates.

### Clearer feature boundaries

- **News & Events** explains that news is dated context, not an order signal, and derives only the category tabs that are actually present.
- **F&O Coverage** explicitly states that the current lane provides contract eligibility, nearest future, expiry and lot size. It does not claim option-chain OI, IV, PCR, Greeks or directional signals.
- **Paper Portfolio** and **System Health** remain available but are secondary to research, data and stock intelligence.
- The previously omitted Research Data stylesheet is now loaded by the frontend.

## Acceptance criteria

This release is acceptable only when:

- an empty panel is not counted as a working product lane;
- every material data lane has a purpose, freshness state, source date and repair action;
- a valid NSE symbol can open a stock workspace without first appearing in a shortlist;
- technical and fundamental metrics include both a definition and an interpretation;
- missing fundamentals have an explicit refresh or evidence-completion path;
- unparsed documents do not create analytical coverage;
- F&O coverage is not marketed as a strategy engine;
- frontend build, canonical tests and import-safety compilation pass.

## What still separates QuantTerm from an institutional terminal

This release makes the existing system compound into a coherent product; it does not manufacture institutional data that the repository does not own. The next real engineering frontier is:

- point-in-time historical fundamentals and restatement handling;
- split, bonus, dividend and merger-adjusted research series;
- peer and sector-normalised valuation/quality models;
- exchange-grade symbol and corporate-action lineage;
- historical derivatives chains with OI, IV surfaces, Greeks and expiry-aware backtests;
- automatic, source-traced extraction of annual reports and earnings transcripts;
- portfolio construction and live execution only after the evidence standards justify them.

The operating rule remains: remove or rename a surface when its data contract is not strong enough. The software must explain itself to the user, not require the user to defend the software.
