# My Trading Terminal Vision

```mermaid
flowchart TD
    %% ─── DATA INGESTION ───────────────────────────────────────────
    subgraph DATA["🔌 DATA LAYER"]
        D1["📡 Kite WebSocket\nLive Ticks"]
        D2["📊 Kite Historical\nOHLCV 1m/5m/1d"]
        D3["📰 News Pipeline\nRSS + Marketaux"]
        D4["🏦 Fundamentals\nScreener.in scraper"]
        D5["🎙️ Earnings Intelligence\nNSE Announcements\n+ YouTube Transcripts"]
        D6["📈 Options Chain\nPCR + OI Buildup"]
    end

    %% ─── PROCESSING ───────────────────────────────────────────────
    subgraph PROC["⚙️ PROCESSING ENGINES"]
        P1["🔬 Technical Engine\nXGBoost + 40 indicators\nChart Pattern Recognition\nMulti-timeframe confluence"]
        P2["💰 Fundamental Engine\nPE · PB · ROE · Debt\nPromoter holding trend\nEarnings growth quality"]
        P3["🧠 Sentiment Engine\nVADER scoring\nDeepSeek news summary\nSector-level mood"]
        P4["🎯 Earnings Agent\nTranscript → Key metrics\nGuidance vs Delivery score\nManagement credibility"]
        P5["🐋 Options Flow\nInstitutional direction\nUnusual OI activity\nPCR momentum"]
    end

    %% ─── AGENT LAYER ───────────────────────────────────────────────
    subgraph AGENTS["🤖 PARALLEL AGENTS  (ThreadPool)"]
        A1["Technical Agent\nScore: 0–100"]
        A2["Fundamental Agent\nScore: 0–100"]
        A3["Sentiment Agent\nScore: 0–100"]
        A4["Earnings Agent\nCredibility: 0–100"]
        A5["Options Agent\nFlow Score: 0–100"]
    end

    %% ─── SIGNAL BRAIN ──────────────────────────────────────────────
    subgraph BRAIN["🧬 SIGNAL BRAIN"]
        B1["Weighted Ensemble\nTech 35% · Fund 25%\nSent 20% · Earn 15%\nOpts 5%"]
        B2["⚡ DeepSeek V3\nFast preliminary signal\nJSON structured output"]
        B3["🔴 DeepSeek R1\nDevil's Advocate\nChain-of-thought challenge\n'Why should we NOT trade?'"]
        B4{"Signal\nSurvives\nChallenge?"}
        B5["🟢 CONFIRMED\nSIGNAL"]
        B6["❌ REJECTED\n— log reason —"]
    end

    %% ─── RISK GATE ──────────────────────────────────────────────────
    subgraph RISK["🛡️ RISK GATE  (non-negotiable)"]
        R1["Position Sizing\nKelly + ATR-based"]
        R2["Exposure Check\nMax 20% capital"]
        R3["Daily Loss Limit\n2% drawdown → kill"]
        R4["Kill Switch\nImmediate halt"]
    end

    %% ─── EXECUTION ──────────────────────────────────────────────────
    subgraph EXEC["⚡ EXECUTION"]
        E1["Paper Trade\n(always on)"]
        E2["Zerodha Kite API\nLive order placement"]
        E3["Order Monitor\nSL · Target · Trailing"]
    end

    %% ─── UI ─────────────────────────────────────────────────────────
    subgraph UI["🖥️ TERMINAL UI"]
        U1["Left Panel\nWatchlist + Live LTP\nColor-coded signals"]
        U2["Center\nTradingView Chart\nSignal overlays"]
        U3["Right Panel\nAgent Reasoning\nLive stream —\nwhat AI is thinking"]
        U4["Bottom\nPortfolio P&L\nTrade Journal\nEquity Curve"]
    end

    %% ─── MEMORY ─────────────────────────────────────────────────────
    subgraph MEM["💾 PERSISTENT MEMORY"]
        M1["Trade Outcomes DB\nWin/loss per setup"]
        M2["Agent learns:\nwhat signals worked\nwhat failed"]
    end

    %% ─── CONNECTIONS ────────────────────────────────────────────────
    D1 & D2 --> P1
    D3 --> P3
    D4 --> P2
    D5 --> P4
    D6 --> P5

    P1 --> A1
    P2 --> A2
    P3 --> A3
    P4 --> A4
    P5 --> A5

    A1 & A2 & A3 & A4 & A5 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4

    B4 -->|YES| B5
    B4 -->|NO| B6

    B5 --> R1 --> R2 --> R3
    R3 -->|Pass| E1
    R3 -->|Pass| E2
    R3 -->|Breach| R4

    E2 --> E3
    E3 -->|outcome| M1
    M1 --> M2
    M2 -->|feedback loop| A1 & A2 & A3

    E1 & E2 --> U4
    A1 & A2 & A3 & A4 & A5 --> U3
    D1 --> U1
    D2 --> U2
```

## The Key Idea — Devil's Advocate

Sabse important cheez jo maine add ki:

**DeepSeek R1 ko deliberately ulti role diya.**

Pehle V3 bolega "BUY karo" — R1 ka kaam sirf yahi hai:
"10 reasons batao kyun ye trade FAIL ho sakta hai."

Agar R1 ke arguments weak hain → signal confirm.
Agar R1 strong counter hai → trade reject.

Ye ek extra filter hai jo overconfidence hatata hai.

## Management Credibility Score (Earnings Agent)

Har company ka ek score track karo:
- Q1 guidance diya tha 20% growth → actual kya aaya?
- Promoter holding badh rahi hai ya ghatt rahi?
- Concall mein positive words hain lekin numbers nahi?

Ye score fundamentals se zyada powerful hota hai.

## Options Flow (missing in current system)

PCR + Unusual OI = institutional bets dikh jaate hain.
Retail sab chart dekh raha, FII options mein khel raha hai.
