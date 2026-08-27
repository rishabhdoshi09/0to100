# QuantTerm Terminal Branch Reconciliation

## Branch of record

`agent/quantterm-terminal-ui` remains the primary retail product branch. The dedicated React terminal is the only product UI. Streamlit is not started by the local stack and is not a fallback.

## What was selectively merged

### From `overhaul/evidence-lab`

- pure command-center projection;
- unified scanner-mode projection;
- separation of presentation from scanner/trading logic;
- one coherent workspace built from persisted scan, long-term, market, paper and autonomy state.

The pure projection now lives in `product/workspace.py` and is consumed by `terminal_product_api.py`. It is not an unused block.

### From `overhaul/simple-mode`

- Simple versus Professional presentation depth;
- plain-language page purpose;
- questions each page should answer;
- glossary and safety boundaries;
- explicit statements about what a feature does not mean.

The useful concepts were ported to `frontend/src/productLanguage.ts` and the React help drawer. Old Streamlit routing was not copied into the primary terminal.

### From the visual references

Adopted:

- dense three-column research board;
- Momentum and Long-Term lanes visible together;
- central chart workspace;
- compact data/freshness strip;
- professional dark terminal hierarchy;
- filterable scanner and long-term research surfaces;
- visible backend operation progress.

Rejected:

- invented Sharpe, CAGR, projections or portfolio curves;
- fake real-time labels;
- decorative percentages without a data source;
- full options intelligence claims without OI/IV/Greeks history.

## New user experience

Primary navigation:

1. Home
2. Discover
3. Stock Intelligence
4. Long-Term Research
5. Research Data

Secondary operations and evidence:

- Market & Breadth
- News & Events
- F&O Coverage
- System Health
- Paper Portfolio

## Product rules preserved

- no scanner logic duplicated in React;
- every displayed date comes from persisted state;
- scan controls dispatch real market operations;
- performance cards require recorded closed trades;
- at least 20 closed trades are required before stable win-rate/profit-factor cards appear;
- paper trading is a secondary evidence layer;
- LIVE broker orders remain locked.

## Acceptance gates

- `tests/test_product_workspace.py` validates command-center and scanner projections;
- Terminal UI CI builds React and compiles terminal/product APIs;
- canonical Python CI remains the final network-free regression gate;
- local browser acceptance must confirm data, operation progress, filters and responsive layout with real persisted state.
