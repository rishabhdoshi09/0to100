# QuantTerm News Curator — Automation Flow

## Product goal

Give a retail investor one high-coverage, continuously refreshed view of the news that can affect the Indian economy, cash equities, sectors, and current stock/index futures—without turning headlines into automatic trade instructions.

## End-to-end flow

```mermaid
flowchart TD
    A[Official sources\nNSE announcements · RBI · SEBI · PIB] --> F[Concurrent source fetch]
    B[Business media RSS\nET · Mint · Business Standard · CNBC-TV18 · configured feeds] --> F
    C[Discovery feeds\nGoogle News: markets · economy · F&O · results · global] --> F
    D[Optional paid enrichment\nMarketaux / future providers] --> F

    F --> G{Source response valid?}
    G -- No --> H[Record source health\nerror · latency · last checked]
    G -- Yes --> I[Parse headline · summary · URL · published time]

    I --> J[Validate age and URL]
    J --> K[Headline fingerprint + story clustering]
    K --> L[Remove duplicate copies\nretain corroboration count]

    L --> M[Entity resolution]
    M --> M1[Map full NSE cash universe]
    M --> M2[Mark current F&O underlyings]
    M --> M3[Detect sectors and macro topics]

    M1 --> N[Event classification]
    M2 --> N
    M3 --> N
    N --> N1[Results · orders · M&A · capital actions]
    N --> N2[RBI · SEBI · policy · inflation · GDP · rupee]
    N --> N3[Futures · options · OI · expiry]
    N --> N4[Global · crude · Fed · tariffs · geopolitics]

    N --> O[Impact scoring]
    O --> O1[Recency]
    O --> O2[Official-source trust]
    O --> O3[Stock / F&O relevance]
    O --> O4[Event severity]
    O --> O5[Multiple-source confirmation]

    O --> P[Retail explanation]
    P --> P1[Potential direction\npositive · negative · mixed · unclear]
    P --> P2[Why it matters]
    P --> P3[Stocks · F&O · sectors · event tags]

    P --> Q[(Durable SQLite news store)]
    H --> R[(Source-health store)]

    Q --> S[Market News page]
    S --> S1[Important Now]
    S --> S2[Stocks & F&O]
    S --> S3[Economy & Policy]
    S --> S4[Sectors]
    S --> S5[All News]
    R --> S6[Source Health]

    Q --> T[Compatibility context API]
    T --> T1[LLM / JARVIS context]
    T --> T2[Momentum and watchlist explanation]
    T --> T3[Future high-impact alerts]
    T --> T4[PAPER_AUTO risk context only]

    T4 --> U{Can news place an order?}
    U -- Never --> V[Price · evidence · liquidity · portfolio safety remain authoritative]
```

## Automation schedule

```mermaid
flowchart LR
    A[Retail app starts] --> B[Start one idempotent news-curator worker]
    B --> C{NSE market open?}
    C -- Yes --> D[Refresh every 5 minutes]
    C -- No --> E[Refresh every 20 minutes]
    D --> F[Fetch · curate · persist]
    E --> F
    F --> G[Prune history older than configured retention]
    G --> C
    H[User presses Refresh now] --> I{Refresh already running?}
    I -- Yes --> J[Return current status; do not start duplicate work]
    I -- No --> F
```

## Trust hierarchy

1. **Tier 1 — official:** NSE corporate announcements, RBI, SEBI, PIB.
2. **Tier 2 — established business media:** existing configured RSS publishers.
3. **Tier 3 — discovery:** Google News searches used to increase coverage.

A Tier 3 headline cannot outrank a fresh Tier 1 filing merely because several websites repeated it.

## Retail doctrine

- Display a lot of news, but collapse repeated copies.
- Keep the original source link and publication time.
- Explain relevance in plain language.
- Show source outages rather than silently shrinking the feed.
- Treat direction as a text-derived cue, not a forecast.
- News may add context or block unsafe confidence; it may never bypass the existing evidence, price, liquidity, and risk gates.
