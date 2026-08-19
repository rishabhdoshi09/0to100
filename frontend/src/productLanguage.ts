export type DisplayDepth = 'simple' | 'professional'

export type PageGuide = {
  title: string
  purpose: string
  questions: string[]
  action: string
  doesNot: string
}

export const PAGE_GUIDE: Record<string, PageGuide> = {
  'Command Center': {
    title: 'Home',
    purpose: 'The tape first — names that cleared the scan, plus Internals for the weather they sit in.',
    questions: [
      'Is the market-data pipeline fresh?',
      'Did the current scan actually finish?',
      'Which names qualify for technical or long-term review?',
      'What should be refreshed before trusting the screen?',
    ],
    action: 'Click Start here on Home. The system fills the next missing layer. Search a name only when you want one stock.',
    doesNot: 'A green market or high score is not an instruction to buy.',
  },
  Scanner: {
    title: 'Ideas · Table',
    purpose: 'Ranks saved whole-market research by momentum, breakout, conviction or avoidance state.',
    questions: [
      'When was the universe last scanned?',
      'How many symbols were processed?',
      'Why did a stock qualify?',
      'What invalidates or disqualifies it?',
    ],
    action: 'Run a fresh scan, filter the result, then inspect one stock in Stock Intelligence.',
    doesNot: 'A scanner match is a research candidate, not a guaranteed trade.',
  },
  Recommendations: {
    title: 'Ideas · Categories',
    purpose: 'Best Setups ranks the last scan on Minervini\'s 7-rule Stage-2 template (near 52-week highs, rising 200-DMA). Other tabs are Wealth Builders, Super Trends, Breakouts and Recovery. Table, Long-term and F&O are the other Ideas floors.',
    questions: [
      'Which names clear SEPA (≥40/100), and which of the 7 rules failed?',
      'Is CMP live or EOD, and how old is the scan?',
      'What is upside from entry vs room to target?',
      'Which tracked picks are still open vs resolved?',
    ],
    action: 'Open Best Setups first. Tap a name for the full 7-rule monitor in Stock Intelligence. Reload keeps the last ranking and the name you had open.',
    doesNot: 'A Buy badge is research classification — not a broker order or return promise.',
  },
  'Market Reports': {
    title: 'Context · Pulse',
    purpose: 'Daily Market Pulse assembled from live scanners — movers, breadth notes and breakout context. News is the source list; Learn teaches the same flow.',
    questions: [
      'What changed in the market today?',
      'Which names are buzzing or near breakouts?',
      'Is this report from today IST or an older saved pulse?',
    ],
    action: 'Read takeaways, then open Ideas for the names mentioned.',
    doesNot: 'A pulse summary is context, not a trade ticket.',
  },
  'Stock Intelligence': {
    title: 'Stock Intelligence',
    purpose: 'SEPA setup monitor (7 Stage-2 rules, score /100) plus price structure, fundamentals, risks, news and source dates.',
    questions: [
      'What is the current trend and volatility?',
      'Are fundamentals complete and fresh?',
      'Which evidence supports the shortlist?',
      'Which missing fact could change the conclusion?',
    ],
    action: 'Refresh missing fundamentals or open System → Data to close one evidence gap.',
    doesNot: 'Current fundamentals are not point-in-time historical evidence unless explicitly labelled.',
  },
  'Long-Term': {
    title: 'Ideas · Long-term',
    purpose: 'Filters technically eligible companies using current quality, growth, leverage and valuation evidence.',
    questions: [
      'Is fundamental coverage adequate?',
      'Is the company quality-backed or only technically strong?',
      'Is price timing favourable or extended?',
      'Which risk flag needs manual review?',
    ],
    action: 'Run the long-term refresh, filter by quality and coverage, then inspect the company dossier.',
    doesNot: 'A long-term classification is not a promise of compounding or future returns.',
  },
  'Research Data': {
    title: 'System · Data',
    purpose: 'The file layer under a stock — which datasets are fresh, stale, missing, uploaded or awaiting extraction. Served from the same terminal API as the rest of the desk. Lives next to System Health because both answer “can I trust this machine?”',
    questions: [
      'What period does each number represent?',
      'Which official source supports it?',
      'Can QuantTerm fetch it automatically?',
      'Is an uploaded document structured and usable or only attached?',
    ],
    action: 'Open a stock first, then use the official link, template and upload flow to close one evidence gap.',
    doesNot: 'Attaching a PDF does not automatically make its claims available to the model.',
  },
  'News & Events': {
    title: 'Context · News',
    purpose: 'The source list — dated company, regulatory and macro items with source-by-source health. Learn is the same flow, taught.',
    questions: [
      'Is the item official or editorial?',
      'When was it published and fetched?',
      'Which company or derivative is actually linked?',
      'Why might it matter relative to company size or fundamentals?',
    ],
    action: 'Open the original source, or switch to Learn if you want the teach point.',
    doesNot: 'News is never a standalone order signal.',
  },
  Education: {
    title: 'Context · Learn',
    purpose: 'The news flow, taught — curated items become micro/macro/policy/F&O cards plus evergreen concepts. Not a second news feed and never invented articles.',
    questions: [
      'Is this macro weather or company micro?',
      'What is the teach point, not just the headline?',
      'Is there an original source URL to verify?',
      'Am I treating education as context instead of a trade tip?',
    ],
    action: 'Read the teach point, open the source when present, then open the linked stock.',
    doesNot: 'Education never invents blogs and never places or recommends orders.',
  },
  'F&O Desk': {
    title: 'Ideas · F&O',
    purpose: 'The derivatives floor of a name. Maps which stocks have current futures, then shows live nearest-expiry OI, IV, PCR and max pain. The same chain also lives on that stock’s Options tab.',
    questions: [
      'Is the cash symbol mapped to the current instrument master?',
      'What is PCR / ATM IV / max pain on the nearest expiry?',
      'Do saved EOD snapshots show a multi-day shift?',
      'Which underlyings were excluded and why?',
    ],
    action: 'Pick a mapped stock (indices stay on this floor), read PCR as positioning context, then Open stock · Options for the same chain on the company workspace.',
    doesNot: 'Black-Scholes Greeks and buy/sell trade direction are not calculated — this is not an F&O signal desk.',
  },
  'Market Internals': {
    title: 'Home · Internals',
    purpose: 'The weather the desk sits in — regime, breadth, VIX and FII/DII.',
    questions: ['Is breadth supportive?', 'Which sectors lead or lag?', 'Is volatility expanding?', 'Does the regime support new risk?'],
    action: 'Use the market state as a portfolio constraint, not as a stock recommendation.',
    doesNot: 'A healthy index does not make every individual setup safe.',
  },
  Portfolio: {
    title: 'Holdings',
    purpose: 'Records simulated positions, risk and outcomes so the system can be judged honestly.',
    questions: ['How much open risk exists?', 'Are exits protected?', 'How many closed trades exist?', 'Is the sample large enough for performance statistics?'],
    action: 'Review position-level risk and refusals before looking at headline P&L.',
    doesNot: 'Paper equity is evidence plumbing, not the brain or heart of QuantTerm.',
  },
  Automation: {
    title: 'System · Health',
    purpose: 'The production ladder: subsystem jobs, handshake, paper sample, and live lock. Then worker liveness. Data is the file layer under a stock.',
    questions: ['Which rung are we on (observe / research / paper / transition / live)?', 'Is live locked, and why?', 'Is the paper supervisor alive?', 'What exact blocker requires action?'],
    action: 'Run paper. Do not arm live until the ladder is unlocked. Resolve the oldest critical blocker.',
    doesNot: 'A live PID, or QT_LIVE_ENABLED, does not graduate a strategy to real money.',
  },
}

export const GLOSSARY: Record<string, string> = {
  Candidate: 'A stock worth reviewing; not an instruction to trade.',
  Conviction: 'A combined evidence score, not certainty.',
  Expectancy: 'Average result per trade after the stated assumptions.',
  Drawdown: 'The fall from a previous portfolio high.',
  Slippage: 'Difference between the expected and actual fill price.',
  'Point-in-time safe': 'Uses only information that was available at that historical moment.',
  'Fundamental coverage': 'How much of the required quality data is actually present.',
  'Chase risk': 'Price is extended enough that a fresh entry may have poor risk/reward.',
  'Market regime': 'The broad trend, breadth and volatility environment.',
  'Market condition': 'The broad trend, breadth and volatility environment.',
  'Source date': 'The period or timestamp the displayed fact actually represents.',
  'Portfolio overlap': 'How much a new stock moves like names you already own.',
  'Research confidence': 'How strongly historical evidence supports the claim.',
  'Data quality': 'Whether the numbers are good enough for charts, trading, or scientific tests.',
  'Historical data quality': 'Whether past prices and memberships reflect only what was known then.',
}

/** Internal enum → Layer-1 copy. Canonical codes stay unchanged in APIs. */
export type PlainTrustCopy = {
  label: string
  state: string
  explanation: string
  implication: string
  technical: string
}

export const TRUST_CLASS_PLAIN: Record<string, PlainTrustCopy> = {
  RESEARCH_GRADE: {
    label: 'Research quality',
    state: 'PROVEN',
    explanation: 'This dataset passed the checks required for scientific historical testing.',
    implication: 'Safe to use when judging whether a strategy deserves real money.',
    technical: 'trust_class=RESEARCH_GRADE',
  },
  OPERATIONAL_ONLY: {
    label: 'Data quality',
    state: 'CAUTION',
    explanation: "Good enough for today's trading view, not fully reconstructible history.",
    implication: 'Use for live/paper decisions; do not treat results as scientific proof.',
    technical: 'trust_class=OPERATIONAL_ONLY',
  },
  DISPLAY_ONLY: {
    label: 'Data quality',
    state: 'UNPROVEN',
    explanation: 'Good enough for charts and exploration, not for proving a strategy.',
    implication: 'QuantTerm will not promote a strategy on this data alone.',
    technical: 'trust_class=DISPLAY_ONLY',
  },
  NOT_PIT_SAFE: {
    label: 'Historical data quality',
    state: 'RISKY',
    explanation:
      'This historical data may contain information that was not actually known at the time. Do not use it for serious backtesting.',
    implication: 'Treat any backtest on this series as exploratory only.',
    technical: 'pit_state=NOT_PIT_SAFE',
  },
}

export const RESEARCH_VERDICT_PLAIN: Record<string, PlainTrustCopy> = {
  INCONCLUSIVE: {
    label: 'Past test result',
    state: 'UNPROVEN',
    explanation:
      'We do not have enough trustworthy data yet to say whether this strategy works. The result remains unproven.',
    implication: 'Do not promote and do not expand models on this result alone.',
    technical: 'verdict=INCONCLUSIVE',
  },
  FAIL: {
    label: 'Past test result',
    state: 'FAILED',
    explanation:
      'We tested this idea properly. So far, it does not show a reliable advantage after trading costs. QuantTerm will not use it for real trades.',
    implication: 'Do not promote this idea; keep or archive as a negative result.',
    technical: 'verdict=FAIL',
  },
  REJECTED: {
    label: 'Past test result',
    state: 'FAILED',
    explanation: 'The idea failed its pre-registered success checks.',
    implication: 'QuantTerm will not use it for live decisions.',
    technical: 'verdict=REJECTED',
  },
}

/** Internal metric key → friendly UI label (do not rename API fields). */
export const FIELD_LABELS: Record<string, string> = {
  regime: 'Market condition',
  network_concentration_score: 'Portfolio overlap risk',
  betweenness_centrality: 'Portfolio dependency',
  evidence_level: 'Research confidence',
  trust_class: 'Data quality',
  pit_state: 'Historical data quality',
  expectancy: 'How much can we expect?',
  calibration_score: 'How reliable is this?',
  gauntlet_verdict: 'Past test result',
}

export const depthLabel = (depth: DisplayDepth) => depth === 'simple' ? 'Simple desk' : 'Pro desk'
