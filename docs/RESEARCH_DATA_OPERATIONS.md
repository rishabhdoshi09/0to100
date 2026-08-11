# QuantTerm Research Data Operations

QuantTerm separates **data present**, **data fresh**, **source attached**, and **structured evidence usable**. These are not interchangeable states.

## Research Data workspace

Select a stock, then open **Research Data** in the dedicated terminal. The workspace shows:

- exact as-of date for every runtime and research dataset;
- age in days and the freshness limit used by QuantTerm;
- `FRESH`, `STALE`, `MISSING`, `UNKNOWN_DATE`, or `SOURCE_ATTACHED_UNPARSED`;
- official NSE/BSE pages and company investor-relations discovery;
- accepted file formats and downloadable CSV templates;
- upload controls requiring a source URL and source-data date;
- a checksum-backed local source ledger.

## Automatic preparation

QuantTerm can initiate these itself:

- deep fundamentals and raw financial/shareholding tables;
- official price-history preparation;
- news and exchange-filing refresh;
- current F&O instrument refresh.

A provider failure is shown to the user. It is never replaced with a synthetic number.

## Manual evidence

The following commonly require the user to select an official company document:

- annual report;
- segment or product mix;
- earnings-call transcript and exact management quotations;
- order-book and forward-guidance disclosure;
- exchange shareholding files where automatic extraction is incomplete.

Every upload requires:

1. the original source URL;
2. the data/document as-of date;
3. an accepted file type;
4. for CSV/JSON, the required QuantTerm schema and at least one complete row.

## When exchange pages are blocked (Mac / offline / sandbox)

Official NSE/BSE links still appear so you can fetch the real filing when the network allows.
When those pages are unreachable, QuantTerm also exposes a **worked-example** path that is
schema-valid and useful for verifying download → upload → analysis:

1. Open **Research Data** for any symbol.
2. Click **Download worked example** on a requirement (pre-filled CSV), *or*
3. Click **Auto-install worked example** — QuantTerm generates the CSVs, uploads them, and
   refreshes coverage so the research dossier can use the structured rows.
4. Uploaded rows keep `https://example.com/quantterm/worked-example/...` as provenance —
   they are **not** live Screener/NSE fundamentals.

API equivalents (report API on `:8766`):

- `GET /evidence/examples/{kind}.csv` — download filled CSV
- `POST /evidence/{symbol}/actions/install-worked-example` — one-click install
- `GET /evidence/{symbol}` — coverage after install

## Extraction rule

- Validated CSV/JSON: usable structured evidence.
- PDF/XLS/XML/TXT/VTT/SRT: source attached, extraction pending unless a dedicated parser has produced structured rows.
- An unparsed document does not improve analytical coverage and cannot create a report claim.
- The annual-report requirement itself can be satisfied by an original PDF, but its contents do not automatically satisfy business, segment, financial or management sections.

## Local persistence

Uploaded sources are written under:

```text
logs/research_evidence/<SYMBOL>/
```

Generated reports are written under:

```text
logs/reports/
```

Both are local runtime artifacts and are excluded from Git.

## Publication rule

The PDF source ledger records each source's status and as-of date. Missing or stale sections contain retrieval instructions. QuantTerm does not use model memory to fill absent financial history, management quotes, ownership changes, business segments, or guidance.
